#!/usr/bin/env python3
"""One explicit assembly submission after reviewed metadata and fresh capacity.

Run only on the existing authenticated cluster session. No SSH, automatic retry,
model load or payload hashing. Never execute this file as a local test launch.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time

REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
WORKSPACE = REPO.parent / 'local_workspace'
ROOT = WORKSPACE / 'qualification/e5-full-sharded-v2-20261013-a01'
AUDIT = ROOT / 'terminal-metadata-audit-v1-a01'
CORE = REPO / 'local_run/e5_sharded_cpu_v2_20261013'
DISPATCH = REPO / 'local_run/e5_sharded_dispatch_v2_20261013'
CORE_SHA = '0eb7309ea3bc4e92d37ac155eef03ce2a59a5daffefc3e3725f511de557cfa81'
DISPATCH_SHA = 'd68d07cae7c8d5c1640650a9464156ec9cdcd396e8f41c6af9f881e6417e3697'
PLAN_SHA = '857fcf65cf5a8a2e55d6abbc5b8c96335c5aca3905f701b37d9b2a51aae116f3'
QUAL_SHA = 'bb84fc18556e6a7d57eda5fa1fffe2c9e2ca5a3f7da5e85f2b96fac11d2e484f'
REPAIR_SHA = 'fc1c54310d6258b39179a350c6ef4e66fdd466c45cec20ca76b85c6682d47af6'
QUOTA_HELPER_SHA = 'dc7b31d9ed6a4b5dc7a8a5a8e66ffeaad31e057747832641694234c0ed4093bc'
GIB = 1024**3
RESERVE = 8 * GIB
REMAINING_ALLOWANCE = 5 * GIB // 4
ORIGINAL_ALLOWANCE = 9 * GIB // 4
DIRECTORY_SLACK = 16 * 1024**2
NODE = 'csis.cn1'
JOB_NAME = 'third-e5-assembly-13498'
MAX_JSON = 25 * 1024**2


def require(value, why):
    if not value:
        raise ValueError(why)


def binding(path, cap=MAX_JSON):
    path = Path(path); info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, 'regular_bounded_metadata_required')
    raw = path.read_bytes()
    require(len(raw) == info.st_size, 'metadata_size_changed')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    binding(path)
    return json.loads(Path(path).read_bytes())


def verify_package(folder, wanted):
    require(re.fullmatch('[0-9a-f]{64}', wanted) is not None, 'invalid_package_SHA')
    manifest = folder / 'package_manifest.json'
    require(binding(manifest)['sha256'] == wanted, 'package_manifest_changed')
    files = load(manifest)['files']; require(bool(files), 'empty_package')
    for name, expected in files.items():
        require(Path(name).name == name and name not in ('.', '..'), 'unsafe_package_member')
        require(binding(folder / name) == expected, 'package_member_changed')
    return binding(manifest)


def command(args, timeout=10, cwd=None):
    return subprocess.check_output(args, stdin=subprocess.DEVNULL, text=True, timeout=timeout, cwd=cwd)


def fresh_terminal(raw):
    wanted = {f'13498_{i}' for i in range(32)}; found = {}
    for line in raw.splitlines():
        fields = line.split('|')
        if fields[0] in wanted:
            require(len(fields) >= 4 and fields[0] not in found, 'duplicate_or_short_task_accounting')
            require(fields[1:4] == ['COMPLETED', '0:0', NODE], 'array_task_not_successful_terminal')
            found[fields[0]] = fields[:4]
    require(set(found) == wanted, 'incomplete_32_task_accounting')
    return [found[f'13498_{i}'] for i in range(32)]


def refuse_previous():
    for name in ('assembly_launch_intent.json', 'assembly_launch_receipt.json', 'assembly-run'):
        require(not os.path.lexists(ROOT / name), 'prior_assembly_attempt_requires_reconciliation')


def materialized_budget(root):
    """Metadata-only stat accounting; no reads/hashes of NPY or JSONL data."""
    total = logical = count = 0
    require(root.is_dir() and not root.is_symlink(), 'regular_run_root_required')
    for path in root.rglob('*'):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), 'symlink_in_output_root')
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), 'nonregular_output_file')
        logical += info.st_size
        total += max(info.st_size, getattr(info, 'st_blocks', 0) * 512)
        count += 1
    require(total + DIRECTORY_SLACK + REMAINING_ALLOWANCE <= ORIGINAL_ALLOWANCE,
            'remaining_allowance_would_exceed_original_2_25GiB_output_envelope')
    return {'regular_files': count, 'logical_bytes': logical,
            'sum_max_logical_or_allocated_bytes': total,
            'directory_and_stat_slack_bytes': DIRECTORY_SLACK,
            'remaining_allowance_bytes': REMAINING_ALLOWANCE,
            'original_output_allowance_bytes': ORIGINAL_ALLOWANCE}


def write_exclusive(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-result-sha256', required=True)
    parser.add_argument('--gate-package-sha256', required=True)
    args = parser.parse_args()
    require(platform.system() == 'Linux' and REPO.is_dir() and WORKSPACE.is_dir(), 'cluster_paths_and_Linux_required')
    require(re.fullmatch('[0-9a-f]{64}', args.metadata_result_sha256) is not None, 'reviewed_metadata_SHA_required')
    os.umask(0o077)
    refuse_previous()
    package = Path(__file__).resolve().parent
    gate_package = verify_package(package, args.gate_package_sha256)
    require('assembly_launch_remote.py' in load(package / 'package_manifest.json')['files'], 'launcher_not_in_seal')
    result_path = AUDIT / 'metadata_gate_result.json'
    require(binding(result_path)['sha256'] == args.metadata_result_sha256, 'reviewed_metadata_result_changed')
    result = load(result_path)
    require(result.get('schema') == 'third-e5-full-array-metadata-gate-v1' and result.get('passed') is True,
            'passed_reviewed_terminal_metadata_gate_required')
    require(str(result.get('array_job_id')) == '13498' and result.get('root') == str(ROOT), 'wrong_metadata_run')
    for key, expected in [('terminal_task_count',32),('dispatch_receipt_count',32),('child_process_receipt_count',64),('completion_receipt_count',64)]:
        require(type(result.get(key)) is int and result[key] == expected, 'incomplete_metadata_gate')
    require(result.get('heavy_vector_or_log_hashes_verified') is False
            and result.get('requires_unchanged_slurm_assembly_validation') is True, 'metadata_scope_changed')
    # The sibling gate verifies the exported source-bound metadata, then rechecks
    # only metadata hashes and payload lstat identities on the login node.
    spec = importlib.util.spec_from_file_location('assembly_metadata_gate', package / 'gate.py')
    gate = importlib.util.module_from_spec(spec); sys.modules[spec.name] = gate; spec.loader.exec_module(gate)
    bundle_path = AUDIT / 'bundle.json'
    require(result['bundle_file'].get('name') == 'bundle.json', 'wrong_bundle_name')
    bundle_binding = {key: result['bundle_file'][key] for key in ('bytes', 'sha256')}
    require(binding(bundle_path) == bundle_binding, 'reviewed_bundle_changed')
    bundle = load(bundle_path)
    checked = gate.verify_bundle(bundle)
    require(checked.get('passed') is True and all(result.get(key) == value for key, value in checked.items()),
            'stored_gate_result_differs_from_verified_bundle')
    gate.recheck_observations(bundle)
    prior = load(ROOT / 'plan_launch_receipt.json'); array = load(ROOT / 'array_launch_receipt.json')
    require(str(prior.get('job_id')) == '13497' and prior.get('returncode') == 0
            and str(array.get('job_id')) == '13498' and array.get('returncode') == 0, 'wrong_prior_launches')
    view = WORKSPACE / 'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu'
    source_replay = WORKSPACE / 'qualification/source-replay-v1-20261011-a01/replay/result.json'
    census = WORKSPACE / 'qualification/e5-token-census-v1-20261011-a01/census'
    budget = WORKSPACE / 'qualification/e5-forward-budget-v1-20261011-a01/budget'
    qual = WORKSPACE / 'qualification/e5-sharded-cpu-v2-20261013-a01/qualification/result.json'
    repair = WORKSPACE / 'qualification/e5-scipy-import-repair-v2-20261011-a01/repair/result.json'
    plan = ROOT / 'plan-run/output/plan.json'
    common = [str(REPO),str(WORKSPACE),str(view),str(source_replay),str(census),str(budget),CORE_SHA,str(qual),QUAL_SHA,str(ROOT)]
    require(prior['common_arguments'] == common, 'prior_common_arguments_changed')
    for path, wanted in ((plan,PLAN_SHA),(qual,QUAL_SHA),(repair,REPAIR_SHA)):
        require(binding(path)['sha256'] == wanted, 'immutable_launch_input_changed')
    require(result['full_plan_file'] == binding(plan) and result['qualification_result'] == binding(qual)
            and result['scipy_repair_result'] == binding(repair), 'metadata_gate_launch_pin_mismatch')
    core = verify_package(CORE, CORE_SHA); dispatch = verify_package(DISPATCH, DISPATCH_SHA)
    require(result['core_package'] == core and result['dispatch_package'] == dispatch, 'metadata_gate_package_mismatch')
    require(not command(['git','status','--porcelain'],cwd=REPO).strip(), 'dirty_cluster_repository')
    commit = command(['git','rev-parse','HEAD'],cwd=REPO).strip()
    raw_accounting = command(['sacct','-j','13498','-P','-n','--format=JobID,State,ExitCode,NodeList'])
    tasks = fresh_terminal(raw_accounting)
    queue = command(['squeue','-u','poonam','-h','-o','%i|%j'])
    for line in queue.splitlines():
        fields = line.split('|',1); require(len(fields)==2,'unexpected_queue_format')
        require(not (fields[0] == '13498' or fields[0].startswith('13498_')) and not fields[1].startswith('third-e5-assembly'),
                'active_array_or_assembly_requires_reconciliation')
    materialized = materialized_budget(ROOT)
    quota_helper = REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    require(binding(quota_helper)['sha256'] == QUOTA_HELPER_SHA, 'quota_helper_changed')
    spec=importlib.util.spec_from_file_location('assembly_quota_parser',quota_helper)
    quota_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=quota_module;spec.loader.exec_module(quota_module)
    mount=quota_module.mount_identity(WORKSPACE);require(mount==quota_module.LOGIN_MOUNT,'quota_filesystem_mismatch')
    quota_started=time.monotonic();quota_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_quota=command(['quota','-v','-w'],timeout=30);quota=quota_module.parse_login_quota(raw_quota)
    require(quota['available_soft_bytes'] >= RESERVE+REMAINING_ALLOWANCE, 'need_1_25GiB_remaining_output_plus_8GiB_reserve')
    refuse_previous()
    require(binding(result_path)['sha256']==args.metadata_result_sha256 and binding(bundle_path)==bundle_binding, 'audit_changed_before_launch')
    require(time.monotonic()-quota_started <=120, 'quota_snapshot_stale')
    cmd=['sbatch','--parsable','--nodelist='+NODE,'--job-name='+JOB_NAME,'--output',str(ROOT/'assembly-%j.out'),'--error',str(ROOT/'assembly-%j.err'),str(DISPATCH/'assembly.sbatch'),*common,str(plan),PLAN_SHA,DISPATCH_SHA]
    intent={'schema':'third-canonical-assembly-launch-v1','source_array_job_id':'13498','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo_commit':commit,'command':cmd,
            'reviewed_metadata_result':binding(result_path),'reviewed_metadata_bundle':binding(bundle_path),'gate_package_manifest':gate_package,'core_package':core,'dispatch_package':dispatch,
            'full_plan':binding(plan),'qualification_result':binding(qual),'scipy_repair_result':binding(repair),'fresh_accounting':raw_accounting,'terminal_task_count':len(tasks),'fresh_queue':queue,
            'materialized_output_budget':materialized,'quota_before_launch':{'observed_utc':quota_utc,'workspace_mount':mount,'raw_stdout':raw_quota,'parsed':quota,'remaining_output_allowance_bytes':REMAINING_ALLOWANCE,'reserve_bytes':RESERVE,'snapshot_not_reservation':True},
            'actual_payload_validation_deferred_to_sealed_Slurm_assembly_before_publication':True,'full_population_cache_acceptance':False,'confirmatory_study_ready':False,'automatic_retry':False}
    write_exclusive(ROOT/'assembly_launch_intent.json',intent)
    receipt = dict(intent)
    try:
        submitted=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        receipt={**intent,'returncode':submitted.returncode,'stdout':submitted.stdout,'stderr':submitted.stderr}
        if submitted.returncode==0:
            job=submitted.stdout.strip().split(';')[0]
            require(re.fullmatch('[0-9]+',job) is not None,'ambiguous_sbatch_success_requires_reconciliation')
            receipt['job_id']=job
    except Exception as error:
        receipt.update(submission_outcome='unknown_reconcile_without_retry',error_type=type(error).__name__,error=str(error))
        for key in ('stdout', 'stderr'):
            value = getattr(error, key, None)
            if value is not None:
                receipt[key] = value.decode('utf-8', 'replace') if isinstance(value, bytes) else value
        write_exclusive(ROOT/'assembly_launch_receipt.json',receipt)
        raise
    write_exclusive(ROOT/'assembly_launch_receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True));require(submitted.returncode==0,'sbatch_failed_no_retry')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'error':str(error),'automatic_retry':False,'full_population_cache_acceptance':False}))
        raise SystemExit(2)
