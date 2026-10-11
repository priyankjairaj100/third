"""Fresh login-node quota observations for an NFS download worker; no SSH."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

MODEL_ID = 'mistralai/Mistral-Nemo-Instruct-2407'
REVISION = '04d8a90549d23fc6bd7f642064003592df51e9b3'
EXPECTATIONS_SHA = 'e79a30e87ece097cea63dd1e3af8e977970f7715abf55c3b858b4180146460b5'
TOTAL_BYTES = 24510683806
RESERVE_BYTES = 8 * 1024**3
GRANT_MAX_AGE = 120
REQUEST_WAIT = 300
CLOCK_TOLERANCE = 5
WORKER_MOUNT = {'source': 'csis.mn1:/nfs_home', 'target': '/nfs_home', 'fstype': 'nfs'}
LOGIN_MOUNT = {'source': '/dev/sdb1', 'target': '/nfs_home', 'fstype': 'ext4'}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def atomic_new(path, value):
    """Publish only a complete new file; hard-link creation cannot overwrite."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def ordered_files(pins):
    files = pins['files']
    require(pins['model_id'] == MODEL_ID and pins['revision'] == REVISION, 'Unexpected model identity')
    require(len(files) == 13 and len({x['relative_path'] for x in files}) == 13
            and sum(x['size_bytes'] for x in files) == pins['total_bytes'] == TOTAL_BYTES,
            'Unexpected file inventory or bytes')
    require(all(Path(x['relative_path']).name == x['relative_path'] and x['relative_path'] not in ('', '.', '..')
                and type(x['size_bytes']) is int and x['size_bytes'] > 0 for x in files), 'Unsafe asset entry')
    return sorted(files, key=lambda item: (item['relative_path'].endswith('.safetensors'), item['relative_path']))


def mount_identity(path):
    raw = subprocess.check_output(['findmnt', '--json', '--target', str(path), '--output', 'SOURCE,TARGET,FSTYPE'], text=True)
    entries = strict(raw)['filesystems']
    require(len(entries) == 1, 'Ambiguous filesystem mapping')
    return {key: entries[0][key] for key in ('source', 'target', 'fstype')}


def parse_login_quota(raw):
    rows = [line.split() for line in raw.splitlines() if line.strip().startswith('/dev/sdb1 ')]
    require(len(rows) == 1, 'Expected exactly one /dev/sdb1 user quota row')
    fields = rows[0]
    require(len(fields) >= 4 and all(re.fullmatch(r'[0-9]+\*?', x) for x in fields[1:4]), 'Malformed KiB quota numbers')
    used, soft, hard = (int(x.rstrip('*')) for x in fields[1:4])
    require(0 < soft <= hard, 'Unexpected quota limits')
    return {'used_kib': used, 'soft_kib': soft, 'hard_kib': hard,
            'available_soft_bytes': (soft - used) * 1024}


def request_spec(index, files):
    if index == -1:
        return {'stage': 'initial_total', 'index': -1, 'relative_path': None, 'size_bytes': TOTAL_BYTES}
    require(type(index) is int and 0 <= index < len(files), 'Invalid request file index')
    item = files[index]
    return {'stage': 'file', 'index': index, 'relative_path': item['relative_path'], 'size_bytes': item['size_bytes']}


def validate_request(request, files, job_id, cache, package_sha, now):
    expected_keys = {'schema', 'job_id', 'nonce', 'created_unix', 'cache', 'package_sha256',
                     'expectations_sha256', 'reserve_bytes', 'stage', 'index', 'relative_path', 'size_bytes'}
    require(set(request) == expected_keys and request['schema'] == 'third-quota-request-v2', 'Unexpected quota request fields')
    require(isinstance(job_id, str) and job_id.isdecimal() and request['job_id'] == job_id, 'Wrong request job identity')
    require(isinstance(request['nonce'], str) and re.fullmatch('[0-9a-f]{32}', request['nonce']), 'Invalid quota nonce')
    require(request['cache'] == str(Path(cache).resolve()) and Path(cache).name == 'mistral-nemo-hub-v2', 'Wrong quota cache')
    require(request['package_sha256'] == package_sha and request['expectations_sha256'] == EXPECTATIONS_SHA
            and request['reserve_bytes'] == RESERVE_BYTES, 'Request binding differs')
    require(type(request['index']) is int, 'Request index is not an integer')
    require(all(request[key] == value for key, value in request_spec(request['index'], files).items()), 'Wrong quota scope/amount')
    stamp = request['created_unix']
    require(type(stamp) in (int, float) and math.isfinite(stamp)
            and -CLOCK_TOLERANCE <= now - stamp <= REQUEST_WAIT, 'Stale or future quota request')


def validate_grant(grant, request, request_sha, now):
    require(set(grant) == {'schema', 'request_sha256', 'request', 'observed_unix', 'published_unix',
                          'quota', 'quota_raw_sha256', 'quota_raw_file', 'login_mount', 'package_sha256'}, 'Unexpected grant fields')
    require(grant['schema'] == 'third-quota-grant-v2' and grant['request_sha256'] == request_sha
            and grant['request'] == request and grant['package_sha256'] == request['package_sha256'], 'Grant request binding differs')
    require(grant['login_mount'] == LOGIN_MOUNT, 'Grant login filesystem differs')
    observed, published = grant['observed_unix'], grant['published_unix']
    require(all(type(t) in (int, float) and math.isfinite(t) for t in (observed, published)), 'Invalid grant timestamps')
    require(-CLOCK_TOLERANCE <= now - observed <= GRANT_MAX_AGE
            and observed <= published <= now + CLOCK_TOLERANCE
            and observed >= request['created_unix'] - CLOCK_TOLERANCE, 'Stale or future quota grant')
    quota = grant['quota']
    require(set(quota) == {'used_kib', 'soft_kib', 'hard_kib', 'available_soft_bytes'}
            and all(type(value) is int for value in quota.values())
            and quota['used_kib'] >= 0 and 0 < quota['soft_kib'] <= quota['hard_kib']
            and quota['available_soft_bytes'] == (quota['soft_kib'] - quota['used_kib']) * 1024,
            'Invalid quota arithmetic')
    require(quota['available_soft_bytes'] >= request['size_bytes'] + RESERVE_BYTES, 'Insufficient freshly observed quota')
    require(isinstance(grant['quota_raw_sha256'], str) and re.fullmatch('[0-9a-f]{64}', grant['quota_raw_sha256']), 'Invalid raw quota binding')
    require(grant['quota_raw_file'] == f'{request["index"] + 1:02d}.quota.json', 'Wrong quota evidence path')


def validate_raw_evidence(evidence, grant):
    require(set(evidence) == {'quota_stdout', 'observed_unix', 'request_sha256', 'login_mount', 'quota_units'},
            'Raw quota evidence fields differ')
    require(isinstance(evidence['quota_stdout'], str)
            and evidence['observed_unix'] == grant['observed_unix']
            and evidence['request_sha256'] == grant['request_sha256']
            and evidence['login_mount'] == grant['login_mount'] == LOGIN_MOUNT
            and evidence['quota_units'] == 'KiB from quota -v -w', 'Raw quota provenance differs from grant')
    require(parse_login_quota(evidence['quota_stdout']) == grant['quota'], 'Raw login quota differs from grant')


def wait_for_grant(out, index, files, cache, package_sha, job_id):
    request = {'schema': 'third-quota-request-v2', 'job_id': job_id, 'nonce': uuid.uuid4().hex,
               'created_unix': time.time(), 'cache': str(cache.resolve()), 'package_sha256': package_sha,
               'expectations_sha256': EXPECTATIONS_SHA, 'reserve_bytes': RESERVE_BYTES, **request_spec(index, files)}
    path = out / 'quota_requests' / f'{index + 1:02d}.request.json'
    atomic_new(path, request)
    request_sha = sha(path)
    print(json.dumps({'waiting_for_login_quota_grant': str(path), 'index': index}), flush=True)
    grant_path = path.with_name(path.name.replace('.request.json', '.grant.json'))
    deadline = time.monotonic() + REQUEST_WAIT
    while True:
        if grant_path.exists():
            grant = strict(grant_path.read_bytes())
            validate_request(request, files, job_id, cache, package_sha, time.time())
            require(sha(path) == request_sha, 'Quota request changed while waiting')
            validate_grant(grant, request, request_sha, time.time())
            raw_path = grant_path.with_name(grant['quota_raw_file'])
            require(sha(raw_path) == grant['quota_raw_sha256'], 'Raw login quota evidence changed')
            raw_quota = strict(raw_path.read_bytes())
            validate_raw_evidence(raw_quota, grant)
            return grant, sha(grant_path)
        require(time.monotonic() < deadline, 'Login quota grant timed out after 300 seconds')
        time.sleep(2)
