"""Single submissions for bounded qualification and a separately locked census."""
import argparse, datetime, hashlib, importlib.util, json, os, re, subprocess, sys, time
from pathlib import Path

REPO=Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W=REPO.parent/'local_workspace'
PRIOR=W/'tools/e5-calibration-frame-control-v1-20261014'
PRIOR_SHA='b118394966d36b9783dd209d9f1c494e03a38618dba114c0ff0347ed445d35bd'
SAMPLER=W/'tools/llm-selection-sampler-v1-20261014'
LOCK_HELPER=W/'tools/llm-sampling-lock-v1-20261014'
PROTOCOL=W/'tools/model-annotation-v3-20261014'
PROTOCOL_SHA='56e97fe1b55f4047f4b8ddac1c7db00be22d69d45b270c8b31beab33d685aa01'
EVIDENCE=W/'tools/model-annotation-v3-evidence-v1-20261014'
EVIDENCE_SHA='25a0dae8bad0023ec1183b0ca4978ee1b2e64aedf20f7b0fe1cc3aad8810b754'
FRAME_PACKAGE=W/'tools/e5-calibration-frame-v1-20261014'
FRAME_PACKAGE_SHA='015ea8122ba5b7a366eba6f96fc3e4d64767bb7d8425c48b3b047ff5fad87fa8'
FRAME=W/'qualification/e5-calibration-frame-v1-20261014-a01/run/frame'
FRAME_SHA='d4fc45ec0d38ddc94c2a9c82ee3948020d0acd1f47fd84582b87ad5e273aedb1'
QUAL_ROOT=W/'qualification/llm-selection-sampler-v1-20261014-qualify-a01'
FULL_ROOT=W/'qualification/llm-selection-sampler-v1-20261014-selection-a01'
LOCK=W/'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json'
RESERVE,ALLOWANCE=8*1024**3,128*1024**2

def require(ok,why):
    if not ok:raise ValueError(why)

def helpers():
    raw=(PRIOR/'package_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PRIOR_SHA,'prior_control_seal_changed')
    for name,wanted in json.loads(raw)['files'].items():
        require(Path(name).name==name and name!='package_manifest.json','unsafe_prior_member')
        path=PRIOR/name;require(path.is_file() and not path.is_symlink(),'prior_member_not_regular')
        data=path.read_bytes()
        require({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==wanted,'prior_control_changed')
    spec=importlib.util.spec_from_file_location('sampler_frame_control',PRIOR/'launch.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    return base.verified_prior()

def queue_clear(raw):
    for line in raw.splitlines():
        parts=line.split('|');require(len(parts)==2,'invalid_queue_row')
        job,name=parts
        require(job!='13556' and not name.startswith(('third-llm-sampl','third-e5-frame','third-e5-assembly')),
                'relevant_job_active_reconcile')

def quota_gate(available,age,mount):
    require(type(available) is int and available>=RESERVE+ALLOWANCE,'need_128MiB_output_plus_8GiB_reserve')
    require(0<=age<=120,'stale_quota')
    require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'},'wrong_quota_mount')

def root_for(mode):
    require(mode in ('qualification','selection'),'unknown_mode')
    return QUAL_ROOT if mode=='qualification' else FULL_ROOT

def child_command(args):
    command=[str(W/'envs/encoder-cpu/bin/python'),str(SAMPLER/'sampler.py'),args.mode]
    fields={'repo':REPO,'workspace':W,'frame-dir':FRAME,'frame-result-sha256':FRAME_SHA,
      'protocol-package':PROTOCOL,'frame-package':FRAME_PACKAGE,
      'protocol-evidence':EVIDENCE,'protocol-evidence-sha256':EVIDENCE_SHA,
      'package-sha256':args.sampler_package_sha256,'out':root_for(args.mode)/'run/output'}
    if args.mode=='selection':
        fields.update({'lock-package':LOCK_HELPER,'lock-package-sha256':args.lock_package_sha256,
          'sampling-lock':LOCK,'sampling-lock-sha256':args.sampling_lock_sha256,
          'qualification-result':QUAL_ROOT/'run/output/result.json',
          'qualification-result-sha256':args.qualification_result_sha256,
          'qualification-review':QUAL_ROOT/'independent_review.json',
          'qualification-review-sha256':args.qualification_review_sha256})
    for name,value in fields.items():command+=['--'+name,str(value)]
    return command

def source_check(h,args):
    bindings={'sampler':h.verify_package(SAMPLER,args.sampler_package_sha256),
      'lock_helper':h.verify_package(LOCK_HELPER,args.lock_package_sha256),
      'protocol':h.verify_package(PROTOCOL,PROTOCOL_SHA),
      'frame_package':h.verify_package(FRAME_PACKAGE,FRAME_PACKAGE_SHA),
      'frame_result':h.verify_file(FRAME/'result.json',FRAME_SHA),
      'protocol_evidence':h.verify_file(EVIDENCE/'snapshot_manifest.json',EVIDENCE_SHA)}
    evidence=h.load(EVIDENCE/'snapshot_manifest.json')
    plan=h.load(PROTOCOL/'PLAN.json')
    require(evidence['files']==plan['bound_files'],'evidence_inventory_differs')
    for name,wanted in evidence['files'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'unsafe_evidence_path')
        require(h.binding(EVIDENCE/path)==wanted,'historical_protocol_evidence_changed')
    result=h.load(FRAME/'result.json')
    require(result['passed'] is True and result['stage']=='complete' and result['rows']==20996
            and result['unordered_pair_count']==220405510
            and result['within_calibration_same_source_exclusion'] is False,'full_frame_required')
    for name,wanted in result['files'].items():
        p=FRAME/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size==wanted['bytes'],'frame_payload_stat_changed')
    if args.mode=='selection':
        bindings.update({'sampling_lock':h.verify_file(LOCK,args.sampling_lock_sha256),
          'qualification_result':h.verify_file(QUAL_ROOT/'run/output/result.json',args.qualification_result_sha256),
          'qualification_review':h.verify_file(QUAL_ROOT/'independent_review.json',args.qualification_review_sha256)})
        qualification=h.load(QUAL_ROOT/'run/output/result.json')
        review=h.load(QUAL_ROOT/'independent_review.json')
        require(qualification['schema']=='third-llm-selection-qualification-v1' and qualification['passed'] is True,
                'passed_bounded_qualification_required')
        require(review['passed'] is True,'independent_qualification_review_required')
        receipt=h.load(QUAL_ROOT/'run/process_completion.json')
        require(receipt['returncode']==0 and receipt['console_cap_exceeded'] is False,'qualification_child_not_successful')
        job=h.load(QUAL_ROOT/'launch_receipt.json')['job_id']
        raw=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
        h.terminal_parent(raw,job)
        bindings['qualification_terminal_accounting']=raw
    return bindings

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('qualification','selection'),required=True)
    for name in ('control-package-sha256','sampler-package-sha256','lock-package-sha256'):
        p.add_argument('--'+name,required=True)
    for name in ('sampling-lock-sha256','qualification-result-sha256','qualification-review-sha256'):
        p.add_argument('--'+name)
    p.add_argument('--expected-tests',type=int,required=True)
    p.add_argument('--time-limit-minutes',type=int,choices=(20,60,120,240),required=True)
    return p

def validate_args(args):
    require(args.expected_tests>0,'positive_test_count_required')
    if args.mode=='qualification':
        require(args.time_limit_minutes==20,'bounded_qualification_time_required')
        require(not any((args.sampling_lock_sha256,args.qualification_result_sha256,args.qualification_review_sha256)),
                'qualification_must_not_use_production_lock')
    else:
        require(args.time_limit_minutes in (60,120,240),'full_time_allowance_required')
        require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in
            (args.sampling_lock_sha256,args.qualification_result_sha256,args.qualification_review_sha256)),
            'independently_reviewed_lock_qualification_required')

def submit_once(h,root,command,intent):
    h.write_exclusive(root/'launch_intent.json',intent);receipt=dict(intent)
    try:
        r=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        receipt.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        if r.returncode==0:
            job=r.stdout.strip().split(';')[0]
            require(re.fullmatch('[0-9]+',job) is not None,'ambiguous_submission')
            receipt['job_id']=job
    except Exception as error:
        receipt.update(submission_outcome='unknown_reconcile_without_retry',error_type=type(error).__name__,error=str(error))
        h.write_exclusive(root/'launch_receipt.json',receipt);raise
    h.write_exclusive(root/'launch_receipt.json',receipt)
    require(r.returncode==0,'submission_failed_no_retry')
    return receipt

def main():
    args=parser().parse_args();validate_args(args);h=helpers()
    for value in (args.control_package_sha256,args.sampler_package_sha256,args.lock_package_sha256):h.sha_argument(value)
    control=Path(__file__).resolve().parent;code=h.verify_package(control,args.control_package_sha256)
    root=root_for(args.mode);require(not os.path.lexists(root),'existing_stage_reconcile')
    require(not h.command(['git','status','--porcelain'],cwd=REPO).strip(),'dirty_executing_checkout')
    initial=source_check(h,args)
    accounting=h.command(['sacct','-j','13556','-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,'13556')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j']);queue_clear(queue)
    qpath=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA)
    spec=importlib.util.spec_from_file_location('sampler_fresh_quota',qpath)
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    mount=q.mount_identity(W);started=time.monotonic()
    quota_raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(quota_raw)
    quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    require(source_check(h,args)==initial,'metadata_inputs_changed')
    require(h.verify_package(control,args.control_package_sha256)==code,'control_changed')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j']);queue_clear(queue)
    quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-llm-sampler-'+args.mode,
      '--time='+str(args.time_limit_minutes),'--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),
      str(control/'sampler.sbatch'),str(control),*sys.argv[1:]]
    intent={'schema':'third-llm-sampler-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'mode':args.mode,'root':str(root),'command':command,'expected_child_command':child_command(args),
      'control_package':code,'initial_bindings':initial,'fresh_frame_accounting':accounting,
      'fresh_queue':queue,'quota':quota,'quota_raw':quota_raw,'mount':mount,
      'reserve_bytes':RESERVE,'output_allowance_bytes':ALLOWANCE,'quota_snapshot_not_reservation':True,
      'automatic_retry':False,'human_annotation_required':False,'human_reviews_planned':0,
      'model_inference_calls':0,'inference_activation':False,'source_authenticity_verified':False}
    os.umask(0o077);root.mkdir(mode=0o700,exist_ok=False)
    print(json.dumps(submit_once(h,root,command,intent),sort_keys=True))

if __name__=='__main__':main()
