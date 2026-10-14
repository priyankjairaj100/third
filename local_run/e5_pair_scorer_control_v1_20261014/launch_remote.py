#!/usr/bin/env python3
"""One explicitly reviewed benchmark submission; no SSH, retry or payload scans."""
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
PACKAGE = WORKSPACE / 'tools/e5-pair-scorer-benchmark-v1-20261014'
ROOT = WORKSPACE / 'qualification/e5-pair-scorer-benchmark-v1-20261014-a01'
ASSEMBLY = WORKSPACE / 'qualification/e5-full-sharded-v2-20261013-a01'
PACKAGE_SHA = 'ce03a07c9458e23c33d3f47df5f5208dbdf53b8eac36737a22783b1d39255c84'
CORE_SHA = '0eb7309ea3bc4e92d37ac155eef03ce2a59a5daffefc3e3725f511de557cfa81'
DISPATCH_SHA = 'd68d07cae7c8d5c1640650a9464156ec9cdcd396e8f41c6af9f881e6417e3697'
QUAL_SHA = 'bb84fc18556e6a7d57eda5fa1fffe2c9e2ca5a3f7da5e85f2b96fac11d2e484f'
REPAIR_SHA = 'fc1c54310d6258b39179a350c6ef4e66fdd466c45cec20ca76b85c6682d47af6'
PLAN_SHA = '857fcf65cf5a8a2e55d6abbc5b8c96335c5aca3905f701b37d9b2a51aae116f3'
QUOTA_HELPER_SHA = 'dc7b31d9ed6a4b5dc7a8a5a8e66ffeaad31e057747832641694234c0ed4093bc'
NODE, USER, JOB_NAME = 'csis.cn1', 'poonam', 'third-e5-scorebench-v1'
RESERVE, OUTPUT_ALLOWANCE = 8*1024**3, 32*1024**2
CAP = 8*1024**2


def require(value, why):
    if not value: raise ValueError(why)


def sha_argument(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid_SHA256')
    return value


def strict(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'duplicate_JSON_key');result[key]=value
        return result
    def invalid(value): raise ValueError('nonfinite_JSON')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid)


def read_small(path, cap=CAP):
    path=Path(path);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'regular_bounded_metadata_required')
    raw=path.read_bytes();after=path.lstat()
    fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(len(raw)==before.st_size and all(getattr(before,k)==getattr(after,k) for k in fields),'file_changed_during_read')
    return raw


def binding(path):
    raw=read_small(path);return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def load(path): return strict(read_small(path))


def verify_file(path,wanted):
    actual=binding(path);require(actual['sha256']==sha_argument(wanted),'pinned_file_changed:'+str(path));return actual


def verify_package(folder,wanted):
    manifest=verify_file(folder/'package_manifest.json',wanted)
    files=load(folder/'package_manifest.json')['files']
    require(isinstance(files,dict) and files,'empty_package')
    for name,expected in files.items():
        require(Path(name).name==name and name not in ('.','..','package_manifest.json'),'unsafe_package_member')
        require(binding(folder/name)==expected,'package_member_changed')
    return manifest


def fixed_inputs():
    return {
        'view':WORKSPACE/'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu',
        'replay_result':WORKSPACE/'qualification/source-replay-v1-20261011-a01/replay/result.json',
        'census_dir':WORKSPACE/'qualification/e5-token-census-v1-20261011-a01/census',
        'budget_dir':WORKSPACE/'qualification/e5-forward-budget-v1-20261011-a01/budget',
        'qualification_result':WORKSPACE/'qualification/e5-sharded-cpu-v2-20261013-a01/qualification/result.json',
        'repair_result':WORKSPACE/'qualification/e5-scipy-import-repair-v2-20261011-a01/repair/result.json'}


def source_check():
    require(PACKAGE.is_dir() and not PACKAGE.is_symlink(),'staged_private_package_required')
    require(PACKAGE.stat().st_mode & 0o077 == 0,'private_package_permissions_required')
    package=verify_package(PACKAGE,PACKAGE_SHA)
    files=load(PACKAGE/'package_manifest.json')['files']
    require({p.name for p in PACKAGE.iterdir()}==set(files)|{'package_manifest.json'},'unexpected_staged_package_file')
    pins=load(PACKAGE/'expectations.json')
    require(pins['core_package_sha256']==CORE_SHA and pins['dispatch_package_sha256']==DISPATCH_SHA
            and pins['qualification_result']['sha256']==QUAL_SHA and pins['repair_result']['sha256']==REPAIR_SHA,'unexpected_benchmark_input_pins')
    evidence={'benchmark_package':package,
        'core_package':verify_package(REPO/pins['core_folder'],CORE_SHA),
        'dispatch_package':verify_package(REPO/pins['dispatch_folder'],DISPATCH_SHA)}
    for name,expected in pins['repository_files'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'unsafe_repository_pin')
        require(binding(REPO/relative)==expected,'frozen_repository_pin_changed')
    paths=fixed_inputs()
    for key in ('qualification_result','repair_result'):
        require(binding(paths[key])==pins[key],'fixed_metadata_input_changed')
        evidence[key]=binding(paths[key])
    require(binding(paths['qualification_result'].parent/'assembled/vectors.npy')==pins['qualification_vectors'],'qualification_vectors_changed')
    evidence['qualification_vectors']=pins['qualification_vectors']
    # Tiny source receipts only. Full records/ledger/weights are checked by Slurm.
    core_pins=load(REPO/pins['core_folder']/'expectations.json')
    verify_file(paths['replay_result'],core_pins['replay_result_sha256'])
    for key,directory in (('census_files','census_dir'),('budget_files','budget_dir')):
        expected=core_pins[key]['result.json']
        require(binding(paths[directory]/'result.json')==expected,'source_result_pin_changed')
    evidence['repository_pin_count']=len(pins['repository_files'])
    return evidence


def command(args,timeout=10,cwd=None):
    raw=subprocess.check_output(args,stdin=subprocess.DEVNULL,text=True,timeout=timeout,cwd=cwd)
    require(len(raw.encode())<=CAP,'command_output_too_large')
    return raw


def terminal_parent(raw,job_id,success=True):
    rows=[]
    for line in raw.splitlines():
        fields=line.split('|')
        if fields[0]==job_id:
            require(len(fields)>=6,'short_job_accounting');rows.append(fields[:6])
    require(len(rows)==1,'missing_or_duplicate_parent_accounting')
    row=rows[0]
    if success:require(row[1]=='COMPLETED' and row[2]=='0:0' and row[5]==NODE,'assembly_not_successful_terminal')
    else:require(row[1].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'},'job_not_terminal')
    return dict(zip(('JobIDRaw','State','ExitCode','ElapsedRaw','MaxRSS','NodeList'),row))


def queue_clear(raw,assembly_job):
    for line in raw.splitlines():
        fields=line.split('|',1);require(len(fields)==2,'unexpected_queue_format')
        job,name=fields
        require(job!=assembly_job and not(job=='13498' or job.startswith('13498_'))
                and not name.startswith(('third-e5-assembly','third-e5-score','third-score-bench')),
                'active_array_assembly_or_benchmark_requires_reconciliation')


def assembly_check(reviewed_sha):
    path=ASSEMBLY/'assembly-run/dispatch_result.json';checked=verify_file(path,reviewed_sha)
    report=load(path);launch=load(ASSEMBLY/'assembly_launch_receipt.json')
    job=launch.get('job_id');require(isinstance(job,str) and job.isdecimal() and launch.get('returncode')==0,'successful_assembly_submission_required')
    require(report.get('schema')=='third-e5-sharded-dispatch-result-v1' and report.get('passed') is True
            and report.get('mode')=='assembly' and report.get('shard_index') is None,'passed_reviewed_assembly_required')
    require(report['bindings']['core_package']['sha256']==CORE_SHA
            and report['bindings']['dispatch_package']['sha256']==DISPATCH_SHA
            and report['bindings']['qualification_result']['sha256']==QUAL_SHA
            and report['bindings']['full_plan']['sha256']==PLAN_SHA,'assembly_source_binding_mismatch')
    require(len(report['children'])==1 and report['children'][0]['name']=='assembly','single_assembly_child_required')
    child=report['children'][0];stage=path.parent
    stage_launch=load(stage/'launch.json')
    require(stage_launch.get('slurm_job_id')==job and stage_launch.get('mode')=='assembly'
            and stage_launch.get('bindings')==report['bindings'],'assembly_job_stage_identity_mismatch')
    require(binding(stage/'assembly.process.json')==child['process_receipt']
            and load(stage/'assembly.process.json')['returncode']==0,'assembly_child_receipt_mismatch')
    require(binding(stage/'cache/manifest.json')==child['artifact'],'assembly_cache_manifest_changed')
    manifest=load(stage/'cache/manifest.json')
    require(manifest.get('passed') is True and manifest.get('rows')==140250
            and manifest.get('shape')==[140250,768]
            and manifest.get('all_rows_fresh_process_replayed_under_this_version') is True,
            'complete_assembled_manifest_required')
    return job,{'dispatch_result':checked,'launch_receipt':binding(ASSEMBLY/'assembly_launch_receipt.json'),
                'cache_manifest':child['artifact'],'stage_launch':binding(stage/'launch.json'),
                'payload_hashes_rechecked_by_login_helper':False}


def refuse_previous():
    require(not os.path.lexists(ROOT),'existing_benchmark_root_requires_reconciliation')


def write_exclusive(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def benchmark_command():
    p=fixed_inputs()
    answer=[str(WORKSPACE/'envs/encoder-cpu/bin/python'),str(PACKAGE/'benchmark.py')]
    for name,value in [('repo',REPO),('workspace',WORKSPACE),*[(k.replace('_','-'),v) for k,v in p.items()]]:
        answer += ['--'+name,str(value)]
    return answer+['--out',str(ROOT/'run/benchmark'),'--benchmark-package-sha256',PACKAGE_SHA]


def quota_gate(quota,age,mount,expected_mount):
    require(mount==expected_mount,'wrong_quota_filesystem')
    require(0<=age<=120,'stale_quota_snapshot')
    require(quota['available_soft_bytes']>=RESERVE+OUTPUT_ALLOWANCE,'need_32MiB_output_plus_8GiB_reserve')


def sbatch_command():
    p=fixed_inputs()
    positional=[str(PACKAGE),str(REPO),str(WORKSPACE),str(p['view']),str(p['replay_result']),str(p['census_dir']),str(p['budget_dir']),str(p['qualification_result']),str(p['repair_result']),str(ROOT/'run'),PACKAGE_SHA]
    require(len(positional)==11,'benchmark_argument_count')
    return ['sbatch','--parsable','--nodelist='+NODE,'--job-name='+JOB_NAME,
            '--output',str(ROOT/'slurm-%j.out'),'--error',str(ROOT/'slurm-%j.err'),str(PACKAGE/'benchmark.sbatch'),*positional]


def submit_once(cmd,intent):
    write_exclusive(ROOT/'launch_intent.json',intent)
    receipt=dict(intent)
    try:
        process=subprocess.run(cmd,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        receipt.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
        if process.returncode==0:
            job=process.stdout.strip().split(';')[0]
            require(re.fullmatch('[0-9]+',job) is not None,'ambiguous_sbatch_success_requires_reconciliation')
            receipt['job_id']=job
    except Exception as error:
        receipt.update(submission_outcome='unknown_reconcile_without_retry',error_type=type(error).__name__,error=str(error))
        for name in ('stdout','stderr'):
            raw=getattr(error,name,None)
            if raw is not None:receipt[name]=raw.decode(errors='replace') if isinstance(raw,bytes) else raw
        write_exclusive(ROOT/'launch_receipt.json',receipt);raise
    write_exclusive(ROOT/'launch_receipt.json',receipt)
    require(process.returncode==0,'sbatch_failed_no_retry')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-package-sha256',required=True)
    parser.add_argument('--reviewed-assembly-dispatch-sha256',required=True)
    args=parser.parse_args()
    require(platform.system()=='Linux' and REPO.is_dir() and WORKSPACE.is_dir(),'cluster_paths_and_Linux_required')
    control=Path(__file__).resolve().parent
    code=verify_package(control,sha_argument(args.control_package_sha256))
    require('launch_remote.py' in load(control/'package_manifest.json')['files'],'launcher_not_in_reviewed_package')
    refuse_previous()
    evidence=source_check()
    assembly_job,assembly=assembly_check(sha_argument(args.reviewed_assembly_dispatch_sha256))
    accounting=command(['sacct','-j',assembly_job,'-P','-n','--format=JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    terminal=terminal_parent(accounting,assembly_job)
    queue=command(['squeue','-u',USER,'-h','-o','%i|%j']);queue_clear(queue,assembly_job)
    require(not command(['git','status','--porcelain'],cwd=REPO).strip(),'dirty_cluster_repository')
    commit=command(['git','rev-parse','HEAD'],cwd=REPO).strip()
    qpath=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    verify_file(qpath,QUOTA_HELPER_SHA)
    spec=importlib.util.spec_from_file_location('benchmark_verified_quota',qpath)
    quota_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(quota_module)
    mount_raw=command(['findmnt','--json','--target',str(WORKSPACE),'--output','SOURCE,TARGET,FSTYPE'])
    mounts=quota_module.strict(mount_raw)['filesystems'];require(len(mounts)==1,'ambiguous_mount')
    mount={k:mounts[0][k] for k in ('source','target','fstype')}
    require(mount==quota_module.LOGIN_MOUNT,'wrong_quota_filesystem')
    quota_start=time.monotonic();quota_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_quota=command(['quota','-v','-w'],timeout=30);quota=quota_module.parse_login_quota(raw_quota)
    require(quota['available_soft_bytes']>=RESERVE+OUTPUT_ALLOWANCE,'need_32MiB_output_plus_8GiB_reserve')
    refuse_previous()
    require(source_check()==evidence and assembly_check(args.reviewed_assembly_dispatch_sha256)[1]==assembly,'inputs_changed_before_intent')
    verify_package(control,args.control_package_sha256)
    queue=command(['squeue','-u',USER,'-h','-o','%i|%j']);queue_clear(queue,assembly_job)
    quota_gate(quota,time.monotonic()-quota_start,mount,quota_module.LOGIN_MOUNT)
    verify_file(qpath,QUOTA_HELPER_SHA)
    cmd=sbatch_command()
    # Everything above is read-only. The initial code/review/input record is
    # fully constructed before the first filesystem mutation and sbatch call.
    intent={'schema':'third-pair-scorer-benchmark-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root':str(ROOT),'repo_commit':commit,'command':cmd,'control_package_manifest':code,
        'initial_launcher_file':binding(Path(__file__)),
        'review_basis':'Caller-supplied reviewed control-manifest SHA and reviewed assembly dispatch-result SHA; no human scientific attestation implied',
        'initial_input_bindings':evidence,'assembly':assembly,'assembly_job_id':assembly_job,
        'fresh_assembly_accounting':accounting,'assembly_terminal_parent':terminal,'fresh_queue':queue,
        'quota_before_launch':{'observed_utc':quota_utc,'mount':mount,'mount_raw':mount_raw,'raw_stdout':raw_quota,'parsed':quota,
            'reserve_bytes':RESERVE,'output_allowance_bytes':OUTPUT_ALLOWANCE,'snapshot_not_reservation':True},
        'benchmark_package_sha256':PACKAGE_SHA,'quota_parser':binding(qpath),'expected_benchmark_command':benchmark_command(),'automatic_retry':False,'human_responses':0,'confirmatory_study_ready':False}
    os.umask(0o077);ROOT.mkdir(mode=0o700,exist_ok=False)
    receipt=submit_once(cmd,intent)
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'passed':False,'error_type':type(error).__name__,'error':str(error),'automatic_retry':False}));sys.exit(2)
