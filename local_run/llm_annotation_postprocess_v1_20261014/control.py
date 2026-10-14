"""One-shot CPU artifact audits and statistics; never generates or retries labels."""
import argparse, datetime, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=Path('/nfs_home/users/poonam/ccu-acl2027/repo');W=REPO.parent/'local_workspace'
EXECUTION_CONTROL=W/'tools/llm-annotation-control-v1-20261014'
EXECUTION_CONTROL_SHA='8999a6328c13dff99cea662944b31a10f5a6f9d50c37cfd0d91e5cfb4c715a6c'
INTEGRATION_SHA='6baf3ffaa5d64304293e382913db8b2943ea2de17fb142a50c54659697d2fedb'
ALLOWANCE=32*1024**2;RESERVE=8*1024**3
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def require(ok,why):
    if not ok:raise ValueError(why)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def helpers():
    raw=(EXECUTION_CONTROL/'package_manifest.json').read_bytes();require(hashlib.sha256(raw).hexdigest()==EXECUTION_CONTROL_SHA,'execution_control_seal')
    for name,want in json.loads(raw)['files'].items():
        require(Path(name).name==name and name!='package_manifest.json','package_member_name');p=EXECUTION_CONTROL/name
        require(p.is_file() and not p.is_symlink(),'package_member_type');x=p.read_bytes();require({'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}==want,'execution_control_member')
    ec=module('postprocess_execution_control',EXECUTION_CONTROL/'control.py');prior,h=ec.helpers();return ec,prior,h
def root_for(mode):
    require(mode in ('qwen','mistral','statistics'),'unknown_mode')
    return W/('qualification/llm-only-v3-selection-statistics-a01' if mode=='statistics' else 'qualification/llm-only-v3-output-audit-'+mode+'-a01')
def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('qwen','mistral','statistics'),required=True)
    for key in ('package-sha256','resource-plan-sha256','admission-sha256','activation-sha256'):p.add_argument('--'+key,required=True)
    for key in ('result-sha256','qwen-result-sha256','mistral-result-sha256','qwen-audit-sha256','mistral-audit-sha256'):p.add_argument('--'+key)
    return p
def check_args(a):
    for k,v in vars(a).items():
        if k.endswith('sha256') and v is not None:
            require(v=='absent' and k=='result_sha256' or len(v)==64 and all(c in '0123456789abcdef' for c in v),'invalid_sha256')
    both=[a.qwen_result_sha256,a.mistral_result_sha256,a.qwen_audit_sha256,a.mistral_audit_sha256]
    require((a.mode=='statistics' and a.result_sha256 is None and all(both)) or (a.mode!='statistics' and a.result_sha256 is not None and not any(both)),'exact_mode_pins')
def execution_args(a,alias):
    return argparse.Namespace(mode=alias,control_package_sha256=EXECUTION_CONTROL_SHA,resource_plan_sha256=a.resource_plan_sha256,admission_sha256=a.admission_sha256,activation_sha256=a.activation_sha256)
def child_command(ec,a):
    command=[str(ec.PYTHON),'-B',str(ec.PACKAGES['integration']/('ingest.py' if a.mode=='statistics' else 'audit.py'))]
    fields={'admission':ec.ADMISSION/'admission.json','admission-sha256':a.admission_sha256,'activation-dir':ec.ACTIVATION,'activation-sha256':a.activation_sha256,'output':root_for(a.mode)/'run/report.json'}
    if a.mode=='statistics':
        for alias in ('qwen','mistral'):fields.update({alias+'-audit':root_for(alias)/'run/report.json',alias+'-audit-sha256':getattr(a,alias+'_audit_sha256'),alias+'-run':ec.root_for(alias)/'run/output'})
    else:fields.update({'root':ec.root_for(a.mode)/'run/output','repo':REPO,'alias':a.mode,'tokenizer-path':ec.MODELS[a.mode]})
    for k,v in fields.items():command.extend(['--'+k,str(v)])
    return command
def terminal_parent(raw,job):
    rows=[x.split('|') for x in raw.splitlines() if x.strip()]
    require(rows and all(len(x)==6 for x in rows),'accounting_shape')
    parents=[x for x in rows if x[0]==job];require(len(parents)==1,'unique_terminal_parent');r=parents[0]
    require(r[1].split()[0] in TERMINAL and r[3].isdigit(),'nonterminal_parent')
    require(r[5] and 'mn1' not in r[5],'compute_node_required')
    require(all(x[0]==job or x[0].startswith(job+'.') for x in rows),'unrelated_accounting')
    require(all(x[1].split()[0] in TERMINAL for x in rows),'nonterminal_step')
    return r
def accepted_terminal(row,process):
    if process is None:return False
    wanted={('COMPLETED','0:0'):0,('FAILED','2:0'):2}.get((row[1].split()[0],row[2]))
    return wanted is not None and process.get('returncode')==wanted and process.get('console_cap_exceeded') is False
def process_check(process,command,job,mode):
    require(process['schema']=='third-llm-execution-process-v1' and process['command']==command and str(process['slurm_job_id'])==job and process['mode']==mode,'family_process_identity')
    t=process['child_wall_seconds_through_exit'];require(type(t) in (float,int) and math.isfinite(t) and t>0,'family_process_timing')
    require(process['retry_attempts']==0 and process['human_annotation_calls']==0,'family_process_scope')
def accounting(h,root,job,saved=None):
    return saved if saved is not None else h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
def family_evidence(h,ec,a,r,alias,pin,saved=None):
    root=ec.root_for(alias);intent=h.load(root/'launch_intent.json');launch=h.load(root/'launch_receipt.json')
    require(all(launch.get(k)==v for k,v in intent.items()),'family_launch_intent_changed')
    job=launch.get('job_id');require(type(job) is str and job.isdigit() and launch['returncode']==0,'family_submission_receipt')
    command=ec.child_command(execution_args(a,alias),r)
    require(intent['schema']=='third-llm-execution-launch-v1' and intent['mode']==alias and intent['root']==str(root) and intent['child_command']==command,'family_launch_identity')
    require(intent['bindings']['resource_plan']['sha256']==a.resource_plan_sha256 and intent['bindings']['admission']['sha256']==a.admission_sha256 and intent['bindings']['activation']['sha256']==a.activation_sha256,'family_launch_admission')
    raw=accounting(h,root,job,saved);row=terminal_parent(raw,job);procpath=root/'run/process_completion.json';proc=h.load(procpath) if procpath.exists() else None
    if proc is not None:
        process_check(proc,command,job,alias)
        require(proc['bindings']['resource_plan']['sha256']==a.resource_plan_sha256 and proc['bindings']['admission']['sha256']==a.admission_sha256 and proc['bindings']['activation']['sha256']==a.activation_sha256,'family_process_admission')
    result=root/'run/output/result.json'
    if pin=='absent':require(not os.path.lexists(result),'expected_missing_result');rb=None
    else:rb=h.verify_file(result,pin)
    return {'job_id':job,'terminal_accounting':raw,'launch_intent':h.binding(root/'launch_intent.json'),'launch_receipt':h.binding(root/'launch_receipt.json'),'process_receipt':h.binding(procpath) if proc is not None else None,'execution_result':rb,'eligible_terminal_process':accepted_terminal(row,proc)}
def check_seal(value):
    obj=dict(value);want=obj.pop('sha256');require(hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()==want,'internal_result_seal')
def post_evidence(h,ec,a,alias,family,saved=None):
    root=root_for(alias);launch=h.load(root/'launch_receipt.json');intent=h.load(root/'launch_intent.json');require(all(launch.get(k)==v for k,v in intent.items()),'audit_launch_intent')
    job=launch['job_id'];raw=accounting(h,root,job,saved);parent=terminal_parent(raw,job)
    require(parent[1]=='COMPLETED' and parent[2]=='0:0' and parent[5] in ('csis.cn1','cn1'),'audit_job_success')
    proc=h.load(root/'run/process_completion.json');result=h.load(root/'run/postprocess_result.json')
    fa=argparse.Namespace(**vars(a));fa.mode=alias;fa.result_sha256=getattr(a,alias+'_result_sha256')
    command=child_command(ec,fa)
    require(intent['schema']=='third-llm-postprocess-launch-v1' and intent['mode']==alias and intent['root']==str(root),'audit_launch_role')
    require(proc['schema']=='third-llm-postprocess-process-v1' and proc['mode']==alias and launch['child_command']==command and proc['command']==command and proc['slurm_job_id']==job and proc['returncode']==0 and proc['console_cap_exceeded'] is False,'audit_child_process')
    elapsed=proc['child_wall_seconds_through_exit'];require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'audit_process_timing')
    require(proc['model_inference_calls']==0 and proc['retry_attempts']==0,'audit_process_scope')
    require(result['schema']=='third-llm-postprocess-result-v1' and result['passed'] is True and result['mode']==alias and result['execution_eligible'] is True,'accepted_audit_wrapper')
    require(result['bindings']['families'][alias]==family and result['bindings']['admission']['sha256']==a.admission_sha256 and result['bindings']['activation']['sha256']==a.activation_sha256,'audit_provenance')
    require(result['package']['sha256']==a.package_sha256 and proc['package']==result['package'],'same_postprocess_package')
    require(result['child_returncode']==0 and result['console_cap_exceeded'] is False and result['integrity_error'] is None and result['source_and_result_bytes_initial_final_equal'] is True,'audit_wrapper_integrity')
    rb=h.verify_file(root/'run/report.json',getattr(a,alias+'_audit_sha256'));report=h.load(root/'run/report.json');check_seal(report)
    require(result['report']==rb and report['passed'] is True and report['alias']==alias and report['execution_result']==family['execution_result'],'exact_passed_family_audit')
    return {'job_id':job,'terminal_accounting':raw,'launch_intent':h.binding(root/'launch_intent.json'),'launch_receipt':h.binding(root/'launch_receipt.json'),'process_receipt':h.binding(root/'run/process_completion.json'),'postprocess_result':h.binding(root/'run/postprocess_result.json'),'report':rb}
def source_check(h,ec,a,saved=None):
    code=h.verify_package(HERE,a.package_sha256);r,base=ec.source_check(h,execution_args(a,'qwen'),terminal=False)
    require(r['packages']['integration']['sha256']==INTEGRATION_SHA,'sealed_integration_required')
    state={'package':code,'resource_plan':base['resource_plan'],'admission':base['admission'],'activation':base['activation'],'packages':base['packages'],'families':{},'audits':{}}
    aliases=('qwen','mistral') if a.mode=='statistics' else (a.mode,)
    for alias in aliases:
        pin=getattr(a,alias+'_result_sha256') if a.mode=='statistics' else a.result_sha256
        old=saved['families'][alias]['terminal_accounting'] if saved is not None else None
        state['families'][alias]=family_evidence(h,ec,a,r,alias,pin,old)
        if a.mode=='statistics':
            require(state['families'][alias]['eligible_terminal_process'],'family_terminal_not_admitted')
            old=saved['audits'][alias]['terminal_accounting'] if saved is not None else None
            state['audits'][alias]=post_evidence(h,ec,a,alias,state['families'][alias],old)
    return r,state
def main():
    a=parser().parse_args();check_args(a);ec,prior,h=helpers();root=root_for(a.mode);require(not os.path.lexists(root),'existing_postprocess_stage_reconcile_no_retry')
    require(h.command(['git','rev-parse','HEAD'],cwd=REPO).strip()=='de9de39a1a72db40485878452561affe2fbd24aa' and not h.command(['git','status','--porcelain'],cwd=REPO).strip(),'frozen_executing_repo')
    r,initial=source_check(h,ec,a);ec.queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']))
    qp=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py';h.verify_file(qp,h.QUOTA_HELPER_SHA);q=module('postprocess_quota',qp)
    mount=q.mount_identity(W);started=time.monotonic();raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(raw)
    ec.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,ALLOWANCE)
    require(source_check(h,ec,a,initial)==(r,initial),'postprocess_inputs_changed');ec.queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']));ec.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,ALLOWANCE)
    command=['sbatch','--parsable','--job-name=third-llm-postprocess-'+a.mode,'--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),str(HERE/'postprocess.sbatch'),str(HERE),*sys.argv[1:]]
    intent={'schema':'third-llm-postprocess-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':a.mode,'root':str(root),'command':command,'child_command':child_command(ec,a),'bindings':initial,'quota':quota,'quota_raw':raw,'mount':mount,'reserve_bytes':RESERVE,'output_allowance_bytes':ALLOWANCE,'quota_snapshot_not_reservation':True,'automatic_retry':False,'model_inference_calls':0,'human_annotation_calls':0}
    os.umask(0o077);root.mkdir(mode=0o700,exist_ok=False);print(json.dumps(prior.submit_once(h,root,command,intent),sort_keys=True))
if __name__=='__main__':main()
