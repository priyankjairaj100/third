"""Submit once only after completed positive calibration and a fresh quota gate."""
import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import time
import admission
import common as c

RESERVE = 8 * 1024**3
ALLOWANCE = 256 * 1024**2
QUOTA_PIN = 'dc7b31d9ed6a4b5dc7a8a5a8e66ffeaad31e057747832641694234c0ed4093bc'


def command(args):
    return subprocess.check_output(args, text=True, timeout=15)


def empty_project_queue(raw):
    lines = [line.split('|') for line in raw.splitlines() if line]
    c.require(all(len(row) == 3 for row in lines), 'queue_parse')
    c.require(not any(row[1].startswith('third-') for row in lines), 'active_project_stage_no_duplicate')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package-sha256', required=True)
    p.add_argument('--admission-sha256', required=True); a = p.parse_args()
    c.verify_package(a.package_sha256)
    c.require(not os.path.lexists(c.ROOT), 'occupied_slot_no_retry')
    approved = admission.check_bundle(c.BUNDLE, a.admission_sha256, remote=True)
    c.require(command(['git', '-C', str(c.REPO), 'rev-parse', 'HEAD']).strip() ==
              'de9de39a1a72db40485878452561affe2fbd24aa' and
              not command(['git', '-C', str(c.REPO), 'status', '--porcelain']).strip(), 'clean_frozen_repo')
    e = c.strict((c.HERE/'expectations.json').read_bytes())
    c.require(c.binding(c.FRAME/'result.json') == e['frame_result'], 'actual_frame_pin')
    for name, wanted in e['repository_files'].items():
        c.require(c.binding(c.REPO/name) == wanted, 'frozen_source_pin')
    qpath = c.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    c.require(c.binding(qpath)['sha256'] == QUOTA_PIN, 'quota_source')
    spec = importlib.util.spec_from_file_location('guard_quota', qpath)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    started = time.monotonic(); raw = command(['quota', '-v', '-w']); quota = q.parse_login_quota(raw)
    c.require(q.mount_identity(c.W) == {'source': '/dev/sdb1', 'target': '/nfs_home', 'fstype': 'ext4'}
              and quota['available_soft_bytes'] >= RESERVE + ALLOWANCE, 'fresh_storage_gate')
    queue = command(['squeue', '-h', '-u', 'poonam', '-o', '%i|%j|%T']); empty_project_queue(queue)
    c.require(time.monotonic()-started <= 120, 'fresh_launch_snapshot')
    cmd = ['sbatch', '--parsable', '--job-name=third-development-guard', '--output', str(c.ROOT/'slurm-%j.out'),
           '--error', str(c.ROOT/'slurm-%j.err'), str(c.HERE/'guard.sbatch'), str(c.HERE),
           a.package_sha256, a.admission_sha256]
    intent = {'schema': 'third-full-development-guard-launch-v1',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': cmd,
              'package_sha256': a.package_sha256, 'admission_sha256': a.admission_sha256,
              'replayed_positive_calibration': approved, 'fresh_queue': queue, 'quota': quota, 'quota_raw': raw,
              'reserve_bytes': RESERVE, 'stage_allowance_bytes': ALLOWANCE,
              'quota_is_snapshot_not_reservation': True, 'automatic_retry': False,
              'effect_experiments_started': False, 'human_annotation_calls': 0}
    os.umask(0o077); c.ROOT.mkdir(mode=0o700, exist_ok=False); c.publish(c.ROOT/'launch_intent.json', intent)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        receipt = {**intent, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        if r.returncode == 0:
            job = r.stdout.strip().split(';')[0]; c.require(job.isdigit(), 'ambiguous_submission')
            receipt['job_id'] = job
    except Exception as error:
        c.publish(c.ROOT/'launch_receipt.json', {**intent, 'submission_outcome': 'unknown_reconcile_without_retry',
                                               'error': str(error)})
        raise
    c.publish(c.ROOT/'launch_receipt.json', receipt); print(json.dumps(receipt)); raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
