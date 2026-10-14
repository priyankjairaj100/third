"""Bounded one-shot CPU qualification and serial family selection submissions."""
import argparse,datetime,hashlib,importlib.util,json,os,socket,sys,time
from pathlib import Path
REPO=Path('/nfs_home/users/poonam/ccu-acl2027/repo');W=REPO.parent/'local_workspace'
PRIOR=W/'tools/llm-sampler-control-v1-20261014';PRIOR_SHA='a1779a1ea98b0968ab46680be6a0249983381bcd25a02a2c405877a32b237cf9'
HERE=Path(__file__).resolve().parent
PACKAGES={k:W/'tools'/v for k,v in {'component':'llm-annotation-inference-v1-20261014','runner':'llm-annotation-execution-v1-20261014','integration':'llm-annotation-integration-v1-20261014','statistics':'llm-annotation-statistics-v1-20261014','protocol':'model-annotation-v3-20261014','preparation':'llm-annotation-requests-v2-20261014','control':'llm-annotation-control-v1-20261014'}.items()}
ADMISSION=W/'qualification/llm-only-v3-selection-admission-a01';ACTIVATION=W/'qualification/llm-only-v3-annotation-activation-a01'
REQUEST_ROOT=W/'qualification/llm-annotation-requests-v2-20261014-selection-a01';REQUESTS=REQUEST_ROOT/'run/output'
QUAL_ROOT=W/'qualification/llm-only-v3-execution-qualification-a01'
PYTHON=W/'envs/judge-qwen-v3-20261011/bin/python'
MODELS={'qwen':Path('/nfs_home/users/poonam/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'),'mistral':W/'models/mistral-nemo-hub-v2/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3'}
WEIGHTS={'qwen':W/'annotation_v5/qwen-pilot-v5-20261011-a01/preparation/weights_manifest.json','mistral':W/'annotation_mistral_v1/mistral-pilot-v1-20261011-a01/preparation/weights_manifest.json'}
ENGINEERING_SHA='76c229739d0c6d1a012b9d922b5504999df919e592a607c4a7700a15d87b9c16'
RESERVE=8*1024**3;FAMILY_ALLOWANCE=96*1024**2;QUAL_ALLOWANCE=32*1024**2

def require(ok,why):
    if not ok:raise ValueError(why)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def helpers():
    raw=(PRIOR/'package_manifest.json').read_bytes();require(hashlib.sha256(raw).hexdigest()==PRIOR_SHA,'prior_manifest')
    for n,b in json.loads(raw)['files'].items():
        require(Path(n).name==n and n!='package_manifest.json','prior_member_name');p=PRIOR/n
        require(p.is_file() and not p.is_symlink(),'prior_file');x=p.read_bytes();require({'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}==b,'prior_member')
    prior=module('llm_control_prior',PRIOR/'control.py');return prior,prior.helpers()
def root_for(mode):
    require(mode in ('qualification','qwen','mistral'),'unknown_mode')
    return QUAL_ROOT if mode=='qualification' else W/('qualification/llm-only-v3-selection-'+mode+'-a01')
def queue_clear(raw):
    for line in raw.splitlines():
        fields=line.split('|');require(len(fields)==2,'queue_shape')
        require(not fields[1].startswith('third-'),'existing_project_job_reconcile')
def quota_gate(available,age,mount,allowance):
    require(type(available) is int and available>=RESERVE+allowance,'quota_reserve_and_commitment')
    require(0<=age<=120,'stale_quota')
    require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'},'quota_mount')
def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('qualification','qwen','mistral'),required=True)
    for n in ('control-package-sha256','resource-plan-sha256'):p.add_argument('--'+n,required=True)
    for n in ('admission-sha256','activation-sha256'):p.add_argument('--'+n)
    return p
def check_args(a):
    for k,v in vars(a).items():
        if k.endswith('sha256') and v is not None:require(len(v)==64 and all(c in '0123456789abcdef' for c in v),'sha256_argument')
    if a.mode=='qualification':require(a.admission_sha256 is None and a.activation_sha256 is None,'qualification_cannot_activate')
    else:require(a.admission_sha256 and a.activation_sha256,'production_activation_required')
def source_check(h,a,terminal=True):
    code=h.verify_package(HERE,a.control_package_sha256)
    rb=h.verify_file(ADMISSION/'resource_plan.json',a.resource_plan_sha256);r=h.load(ADMISSION/'resource_plan.json')
    require(set(r['packages'])==set(PACKAGES),'exact_package_roles')
    for name,b in r['packages'].items():require(h.verify_package(PACKAGES[name],b['sha256'])==b,'package_identity:'+name)
    require(r['packages']['control']==code,'executed_control_identity')
    sys.path.insert(0,str(PACKAGES['integration']));integration=module('llm_control_activation',PACKAGES['integration']/'activation.py')
    integration.validate_resource_plan(r)
    for alias,f in r['families'].items():require(Path(f['output_dir'])==root_for(alias)/'run/output','fixed_family_slot')
    request=h.load(REQUESTS/'result.json');require(h.binding(REQUESTS/'result.json')==r['request_manifest'],'request_binding')
    for n,b in request['files'].items():
        require(Path(n).name==n,'request_member_name');require(h.binding(REQUESTS/n)==b,'request_artifact')
    review=h.load(ADMISSION/'request_review.json');require(h.binding(ADMISSION/'request_review.json')==r['request_review'],'request_review_binding')
    require(review['passed'] is True and review['request_result']==r['request_manifest'] and review['all_input_token_IDs_and_rendered_prompt_hashes_independently_replayed'] is True,'actual_request_token_review')
    state={'control':code,'resource_plan':rb,'packages':r['packages'],'request_result':r['request_manifest'],'request_review':r['request_review']}
    if terminal:
        job=h.load(REQUEST_ROOT/'launch_receipt.json')['job_id'];require(job=='13572','request_job')
        accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList']);h.terminal_parent(accounting,job);state['request_terminal']=accounting
    if a.mode=='qualification':
        state['engineering_replay_manifest']=h.verify_file(ADMISSION/'engineering_replay_manifest.json',ENGINEERING_SHA)
        require(not os.path.lexists(ACTIVATION),'qualification_after_activation_refused')
    else:
        ab=h.verify_file(ADMISSION/'admission.json',a.admission_sha256);vb=h.verify_file(ACTIVATION/'activation.json',a.activation_sha256)
        activated,admitted=integration.read_activation(ADMISSION/'admission.json',ab,ACTIVATION,vb)
        require(admitted['resource_plan']==r,'activation_resources');state.update(admission=ab,activation=vb)
        if terminal:
            job=h.load(QUAL_ROOT/'launch_receipt.json')['job_id'];accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList']);h.terminal_parent(accounting,job)
            proc=h.load(QUAL_ROOT/'run/process_completion.json');require(proc['returncode']==0 and proc['console_cap_exceeded'] is False,'qualification_process_failed')
            state['qualification_terminal']=accounting
        if a.mode=='mistral' and terminal:
            qroot=root_for('qwen');job=h.load(qroot/'launch_receipt.json')['job_id'];accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList']);h.terminal_parent(accounting,job)
            proc=h.load(qroot/'run/process_completion.json');require(proc['returncode']==0 and proc['console_cap_exceeded'] is False,'qwen_process_needs_review')
            state['qwen_terminal']=accounting
    return r,state

def child_command(a,r):
    common={'repo':REPO,'workspace':W,'protocol-package':PACKAGES['protocol'],'component-package':PACKAGES['component'],'integration-package':PACKAGES['integration'],'requests-dir':REQUESTS,'package-sha256':r['packages']['runner']['sha256'],'request-result-sha256':r['request_manifest']['sha256']}
    if a.mode=='qualification':
        script='qualify.py';common.update({'engineering-replay-manifest':ADMISSION/'engineering_replay_manifest.json','engineering-replay-sha256':ENGINEERING_SHA,'control-package':PACKAGES['control'],'statistics-package':PACKAGES['statistics'],'resource-plan':ADMISSION/'resource_plan.json','resource-plan-sha256':a.resource_plan_sha256,'qwen-model-dir':MODELS['qwen'],'qwen-weights-manifest':WEIGHTS['qwen'],'mistral-model-dir':MODELS['mistral'],'mistral-weights-manifest':WEIGHTS['mistral'],'output-dir':root_for(a.mode)/'run/output'})
    else:
        script='run_model.py';common.update({'admission':ADMISSION/'admission.json','activation-dir':ACTIVATION,'model-alias':a.mode,'model-dir':MODELS[a.mode],'weights-manifest':WEIGHTS[a.mode],'output-dir':root_for(a.mode)/'run/output','admission-sha256':a.admission_sha256,'activation-sha256':a.activation_sha256})
    cmd=[str(PYTHON),'-B',str(PACKAGES['runner']/script)]
    for k,v in common.items():cmd.extend(['--'+k,str(v)])
    return cmd

def main():
    a=parser().parse_args();check_args(a);prior,h=helpers();root=root_for(a.mode)
    require(not os.path.lexists(root),'existing_stage_reconcile_no_retry')
    require(h.command(['git','rev-parse','HEAD'],cwd=REPO).strip()=='de9de39a1a72db40485878452561affe2fbd24aa' and not h.command(['git','status','--porcelain'],cwd=REPO).strip(),'frozen_executing_repo')
    r,initial=source_check(h,a);queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']))
    qpath=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py';h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=module('llm_control_quota',qpath)
    mount=q.mount_identity(W);started=time.monotonic();raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(raw)
    allowance=QUAL_ALLOWANCE+r['quota']['total_output_commitment_bytes'] if a.mode=='qualification' else (r['quota']['total_output_commitment_bytes'] if a.mode=='qwen' else FAMILY_ALLOWANCE)
    quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,allowance)
    require(source_check(h,a)==(r,initial),'inputs_changed_before_submission');queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']));quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,allowance)
    flags=['--partition=cpu-short','--nodelist=csis.cn1','--cpus-per-task=4','--mem=8G','--time=20'] if a.mode=='qualification' else ['--partition=gpu-short','--gres=gpu:a100-80gb:1','--cpus-per-task=4','--mem='+str(r['families'][a.mode]['host_memory_bytes']//1024**3)+'G','--time='+str(r['families'][a.mode]['slurm_time_seconds']//60)]
    command=['sbatch','--parsable',*flags,'--job-name=third-llm-infer-'+a.mode,'--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),str(HERE/'execute.sbatch'),str(HERE),*sys.argv[1:]]
    intent={'schema':'third-llm-execution-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':a.mode,'root':str(root),'command':command,'child_command':child_command(a,r),'bindings':initial,'quota':quota,'quota_raw':raw,'mount':mount,'reserve_bytes':RESERVE,'admitted_remaining_output_commitment_bytes':allowance,'quota_snapshot_not_reservation':True,'human_annotation_calls':0,'automatic_retry':False,'sampling_entropy_draws':0}
    os.umask(0o077);root.mkdir(mode=0o700,exist_ok=False);print(json.dumps(prior.submit_once(h,root,command,intent),sort_keys=True))
if __name__=='__main__':main()
