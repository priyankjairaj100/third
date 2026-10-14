"""Bounded metadata-only preflight and exactly one frame-admission submission."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
WORKSPACE = REPO.parent / 'local_workspace'
ROOT = WORKSPACE / 'qualification/e5-calibration-frame-v1-20261014-a01'
PACKAGE = WORKSPACE / 'tools/e5-calibration-frame-v1-20261014'
PRIOR = WORKSPACE / 'tools/e5-pair-scorer-control-v1-20261014'
PRIOR_SHA = 'd89117b3dd623573574091c6289efb890365d9dc255b1efeedcde5af18268bc9'
ASSEMBLY = WORKSPACE / 'qualification/e5-full-sharded-v2-20261013-a01'
ASSEMBLY_SHA = '112b9851c0bc1e2b908e08db485d5ba1b5b9e258cb3b8694a70184bcbe089595'
QUAL_SHA = 'bb84fc18556e6a7d57eda5fa1fffe2c9e2ca5a3f7da5e85f2b96fac11d2e484f'
RESERVE, ALLOWANCE = 8*1024**3, 256*1024**2
JOB_NAME = 'third-e5-frame-v1'


def require(ok, why):
    if not ok: raise ValueError(why)


def verified_prior(folder=PRIOR):
    raw=(folder/'package_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PRIOR_SHA, 'prior_control_manifest_changed')
    for name, wanted in json.loads(raw)['files'].items():
        require(Path(name).name==name and name!='package_manifest.json','unsafe_prior_member')
        path=folder/name
        require(not path.is_symlink() and path.is_file(),'prior_member_not_regular')
        data=path.read_bytes()
        require({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==wanted,'prior_control_source_changed')
    spec=importlib.util.spec_from_file_location('reviewed_frame_metadata_helpers',folder/'launch_remote.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def queue_clear(raw):
    for line in raw.splitlines():
        fields=line.split('|')
        require(len(fields)==2,'invalid_queue_row')
        job,name=fields
        require(not (job=='13498' or job.startswith('13498_') or job in {'13546','13547'}
                     or name.startswith(('third-e5-frame','third-e5-assembly','third-e5-score','third-score-bench'))),
                'prior_or_frame_work_active_reconcile')


def quota_gate(available, age, mount):
    require(type(available) is int and available>=RESERVE+ALLOWANCE,'need_256MiB_output_plus_8GiB_reserve')
    require(0<=age<=120,'stale_quota')
    require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'},'wrong_quota_filesystem')


def frame_command(package_sha, output):
    args={'repo':REPO,'workspace':WORKSPACE,
      'view':WORKSPACE/'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu',
      'replay-result':WORKSPACE/'qualification/source-replay-v1-20261011-a01/replay/result.json',
      'census-dir':WORKSPACE/'qualification/e5-token-census-v1-20261011-a01/census',
      'budget-dir':WORKSPACE/'qualification/e5-forward-budget-v1-20261011-a01/budget',
      'qualification-result':WORKSPACE/'qualification/e5-sharded-cpu-v2-20261013-a01/qualification/result.json',
      'qualification-sha256':QUAL_SHA,
      'plan':ASSEMBLY/'plan-run/output/plan.json','assembly-root':ASSEMBLY,
      'repair-result':WORKSPACE/'qualification/e5-scipy-import-repair-v2-20261011-a01/repair/result.json',
      'frame-package-sha256':package_sha,'out':output}
    command=[str(WORKSPACE/'envs/encoder-cpu/bin/python'),str(PACKAGE/'frame.py')]
    for name,value in args.items(): command+=['--'+name,str(value)]
    return command


def submit_once(helper, command, intent):
    helper.write_exclusive(ROOT/'launch_intent.json',intent)
    receipt=dict(intent)
    try:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode==0:
            job=result.stdout.strip().split(';')[0]
            require(re.fullmatch('[0-9]+',job) is not None,'ambiguous_sbatch_success')
            receipt['job_id']=job
    except Exception as error:
        receipt.update(submission_outcome='unknown_reconcile_without_retry',error_type=type(error).__name__,error=str(error))
        helper.write_exclusive(ROOT/'launch_receipt.json',receipt)
        raise
    helper.write_exclusive(ROOT/'launch_receipt.json',receipt)
    require(result.returncode==0,'submission_failed_no_retry')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-package-sha256',required=True)
    parser.add_argument('--frame-package-sha256',required=True)
    parser.add_argument('--expected-tests',required=True,type=int)
    args=parser.parse_args()
    h=verified_prior()
    h.sha_argument(args.control_package_sha256);h.sha_argument(args.frame_package_sha256)
    require(args.expected_tests>0,'positive_test_count_required')
    control=Path(__file__).resolve().parent
    code=h.verify_package(control,args.control_package_sha256)
    frame=h.verify_package(PACKAGE,args.frame_package_sha256)
    require(not os.path.lexists(ROOT),'existing_stage_requires_reconciliation')
    require(not h.command(['git','status','--porcelain'],cwd=REPO).strip(),'dirty_cluster_repository')
    commit=h.command(['git','rev-parse','HEAD'],cwd=REPO).strip()
    assembly_job,assembly=h.assembly_check(ASSEMBLY_SHA)
    accounting=h.command(['sacct','-j',assembly_job+',13547','-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,assembly_job);h.terminal_parent(accounting,'13547')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j']);queue_clear(queue)
    qpath=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA)
    spec=importlib.util.spec_from_file_location('frame_quota',qpath)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    mount=q.mount_identity(WORKSPACE)
    started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    quota_raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(quota_raw)
    quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    require(h.verify_package(control,args.control_package_sha256)==code and h.verify_package(PACKAGE,args.frame_package_sha256)==frame,'package_changed_before_launch')
    require(h.assembly_check(ASSEMBLY_SHA)[1]==assembly,'assembly_metadata_changed')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j']);queue_clear(queue)
    quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name='+JOB_NAME,
       '--output',str(ROOT/'slurm-%j.out'),'--error',str(ROOT/'slurm-%j.err'),
       str(control/'frame.sbatch'),str(control),str(ROOT),args.control_package_sha256,
       args.frame_package_sha256,str(args.expected_tests)]
    intent={'schema':'third-calibration-frame-launch-v1','created_utc':stamp,'repo_commit':commit,
       'control_package':code,'frame_package':frame,'prior_control_sha256':PRIOR_SHA,
       'command':command,'expected_frame_command':frame_command(args.frame_package_sha256,ROOT/'run/frame'),
       'assembly':assembly,'fresh_accounting':accounting,'fresh_queue':queue,
       'quota':quota,'quota_raw':quota_raw,'mount':mount,'reserve_bytes':RESERVE,'output_allowance_bytes':ALLOWANCE,
       'quota_snapshot_not_reservation':True,'automatic_retry':False,'model_forwards':0,
       'human_annotation_required':False,'planned_human_reviews':0,'scientific_activation':False}
    os.umask(0o077);ROOT.mkdir(mode=0o700,exist_ok=False)
    print(json.dumps(submit_once(h,command,intent),sort_keys=True))


if __name__=='__main__': main()
