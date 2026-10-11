#!/usr/bin/env python3
"""One-shot login-only quota controller; never uses SSH or downloads assets."""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'cache', 'expectations', 'package-sha256', 'job-id'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    from package_check import verify
    verify(Path(__file__).resolve().parent, args.package_sha256)
    import quota_grants as q
    q.require(platform.system() == 'Linux' and socket.gethostname().split('.')[:2] == ['csis', 'mn1'], 'Use the mapped login node')
    out, cache = Path(args.out).resolve(), Path(args.cache).resolve()
    q.require(out.is_dir() and cache.is_dir(), 'Existing acquisition/cache directories required')
    q.require(q.mount_identity(cache) == q.LOGIN_MOUNT, 'Login cache filesystem mapping differs')
    q.require(q.sha(args.expectations) == q.EXPECTATIONS_SHA, 'Expectations hash mismatch')
    files = q.ordered_files(q.strict(Path(args.expectations).read_bytes()))
    pending = [p for p in sorted((out / 'quota_requests').glob('*.request.json'))
               if not p.with_name(p.name.replace('.request.json', '.grant.json')).exists()]
    q.require(len(pending) <= 1, 'More than one pending quota request')
    if not pending:
        print(json.dumps({'status': 'idle', 'completed_receipt_present': (out / 'result.json').exists()}))
        return 0
    request_path = pending[0]
    request = q.strict(request_path.read_bytes())
    q.validate_request(request, files, args.job_id, cache, args.package_sha256, time.time())
    q.require(request_path.name == f'{request["index"] + 1:02d}.request.json', 'Unexpected request filename')
    request_sha = q.sha(request_path)
    observed = time.time()  # Conservative freshness starts before the quota query.
    raw = subprocess.check_output(['quota', '-v', '-w'], text=True, timeout=30)
    quota = q.parse_login_quota(raw)
    q.require(quota['available_soft_bytes'] >= request['size_bytes'] + q.RESERVE_BYTES, 'Insufficient login quota; no grant published')
    q.require(q.sha(request_path) == request_sha, 'Quota request changed during observation')
    os.umask(0o077)
    evidence = request_path.with_name(request_path.name.replace('.request.json', '.quota.json'))
    q.atomic_new(evidence, {'quota_stdout': raw, 'observed_unix': observed, 'request_sha256': request_sha,
                            'login_mount': q.LOGIN_MOUNT, 'quota_units': 'KiB from quota -v -w'})
    grant = {'schema': 'third-quota-grant-v2', 'request_sha256': request_sha, 'request': request,
             'observed_unix': observed, 'published_unix': time.time(), 'quota': quota,
             'quota_raw_sha256': q.sha(evidence), 'quota_raw_file': evidence.name,
             'login_mount': q.LOGIN_MOUNT, 'package_sha256': args.package_sha256}
    q.validate_grant(grant, request, request_sha, time.time())
    q.atomic_new(request_path.with_name(request_path.name.replace('.request.json', '.grant.json')), grant)
    print(json.dumps({'status': 'granted', 'job_id': args.job_id, 'index': request['index'],
                      'relative_path': request['relative_path'], 'quota_available_soft_bytes': quota['available_soft_bytes']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
