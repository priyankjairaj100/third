"""Finite continuation from completed Qwen through corrected audits and validation."""
import hashlib,importlib.util,json,os
from pathlib import Path
TASK=Path(os.environ['THIRD_TASK_ROOT']);REPO=Path(os.environ['THIRD_LOCAL_REPO'])
WORK=TASK/'work/resume-20261015';OUT=WORK/'repair_pipeline_a01'
helper=REPO/'local_run/llm_node_pool_controller_v2_20261014/continue_pipeline.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='e5c632d822e2a410938ff5c171533b77c21ddef8f06996f9c585ee3757913225'
spec=importlib.util.spec_from_file_location('prior_finite_controller',helper);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
p=n.p;p.OUT=OUT;p.REMOTE=WORK/'remote_fresh_session.py';n.OUT=OUT
R_PIN='d498d2d0dc5fe30b81912d8804e0e402b1b65f9263ac0d6b8b69e7bf38ee1f0e'
R_FOLDER=p.W+'/tools/llm-v4-audit-repair-v1-20261015'
R_ROOT=p.W+'/qualification/llm-v4-audit-repair-v1-20261015-a01'
SEL_BASE=['--package-sha256',n.PINS['selection_inference'],'--request-result-sha256','334e193f0cac3aa0d36bac510d06719b98efe5f715b031b7944c1c1ec7757db4','--activation-sha256','d2a61920214a8291976d537f442d73aabf74e542d29c8aa9343fe20856b36436','--resource-plan-sha256','a154ba396b72d340421a8b7da9226202144ef8c3c5c5791346cb048e7eb5f16d']

def rcli(label,scope,script,args):
 return p.run_cli(label,R_FOLDER,R_PIN,script,['--scope',scope,'--repair-package-sha256',R_PIN,*args])
def adopt_qwen_audit():
 local=json.loads((WORK/'qwen_audit_repair_launch_receipt.json').read_text())
 job=local['job_id'];p.require(job.isdigit() and local['repair_package_sha256']==R_PIN,'known_corrected_audit_launch')
 script='''import hashlib,json
from pathlib import Path
root=Path(ROOT);job=JOB
receipt=json.loads((root/'launch_receipt.json').read_text());assert receipt['job_id']==job and receipt['repair_package_sha256']==PIN
old=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace/qualification/llm-only-v4-inference-v2-a01')
assert hashlib.sha256((old/'qwen/output/result.json').read_bytes()).hexdigest()=='ba716a2588a174cf0968c60e3a8aa659b47b9e806b1d94534988539fb7b25011'
assert hashlib.sha256((old/'audit_qwen/child.stderr.txt').read_bytes()).hexdigest()=='f23819738ab590c58396c8a27a1d9a1d4d330e3419d234ca889a8114af17a9b3'
assert not (old/'mistral').exists() and not (old/'statistics').exists(),'reconcile_unexpected_downstream_stage'
print(json.dumps({'passed':True,'adopted_job':job,'original_Qwen_result_unchanged':True,'original_failure_preserved':True}))
'''.replace('ROOT',repr(R_ROOT+'/selection/audit_qwen')).replace('JOB',repr(job)).replace('PIN',repr(R_PIN))
 p.write('adoption.json',p.remote('adopt_existing_corrected_audit',script));return job

def audit(scope,alias,base,adopted_job=None):
 label=scope+'_repaired_audit_'+alias
 job=adopted_job
 if job is None:job=rcli(label+'_launch',scope,'audit_launch.py',[*base,'--model-alias',alias])['job_id']
 n.await_success(label,R_ROOT+'/'+scope+'/audit_'+alias,job,'run/result.json','csis.cn1')
 bridge=rcli(label+'_publication',scope,'publish_bridge.py',['--model-alias',alias]);p.require(bridge['published'] is True,'accepted_audit_bridge')

def remaining(scope,root,base):
 launch=rcli(scope+'_mistral_launch',scope,'continue_launch.py',['--mode','mistral',*base])
 n.await_success(scope+'_mistral',root+'/mistral',launch['job_id'],'output/result.json','gpu')
 audit(scope,'mistral',base)
 launch=rcli(scope+'_statistics_launch',scope,'continue_launch.py',['--mode','statistics',*base])
 n.await_success(scope+'_statistics',root+'/statistics',launch['job_id'],'run/result.json','csis.cn1')

def validation(selected):
 samplepin=n.PINS['validation_sampler'];folder=p.W+'/tools/llm-v4-validation-sampler-v2-20261014'
 launch=p.run_cli('validation_sampling_launch',folder,samplepin,'launch.py',['--package-sha256',samplepin,'--selection-statistics-sha256',selected['bindings']['statistics']['sha256']])
 sample=n.await_success('validation_sampling',n.SAMPLING,launch['job_id'],'run/result.json','csis.cn1')
 requestpin=n.PINS['validation_requests'];folder=p.W+'/tools/llm-v4-validation-requests-v2-20261014'
 launch=p.run_cli('validation_requests_launch',folder,requestpin,'launch.py',['--package-sha256',requestpin,'--selection-result-sha256',sample['result_binding']['sha256']])
 prepared=n.await_success('validation_requests',p.W+'/qualification/llm-v4-validation-requests-v2-20261014-a01',launch['job_id'],'run/result.json','csis.cn1')
 p.require(0<prepared['result']['planned_requests']<=440,'validation_call_cap')
 pin=n.PINS['validation_inference'];folder=p.W+'/tools/llm-v4-validation-inference-v2-20261014'
 base=['--package-sha256',pin,'--request-result-sha256',prepared['result_binding']['sha256']]
 launch=p.run_cli('validation_qualification_launch',folder,pin,'launch.py',['--mode','qualification',*base])
 qualified=n.await_success('validation_qualification',p.W+'/qualification/llm-v4-validation-inference-qualification-v2-a01',launch['job_id'],'run/result.json','csis.cn1')
 activated=p.run_cli('validation_activation',folder,pin,'activate.py',[*base,'--qualification-result-sha256',qualified['result_binding']['sha256']])
 base+=['--activation-sha256',activated['activation']['sha256'],'--resource-plan-sha256',activated['resource_plan']['sha256']]
 launch=p.run_cli('validation_qwen_launch',folder,pin,'launch.py',['--mode','qwen',*base])
 n.await_success('validation_qwen',n.VALROOT+'/qwen',launch['job_id'],'output/result.json','gpu')
 audit('validation','qwen',base);remaining('validation',n.VALROOT,base)
 return n.v.independent_review()

def main():
 os.umask(0o077);OUT.mkdir(exist_ok=False)
 p.write('intent.json',{'created_utc':p.now(),'pid':os.getpid(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'repair_package_sha256':R_PIN,'original_execution_packages':n.PINS,'scope':'Adopt corrected audit of preserved Qwen; continue Mistral, selection review and conditional fresh validation',
  'model_calls_repeated':0,'automatic_retry':False,'human_annotation_calls':0,'project_complete':False})
 try:
  job=adopt_qwen_audit();audit('selection','qwen',SEL_BASE,job);remaining('selection',n.SELROOT,SEL_BASE)
  selected=p.independent_statistics_review();p.write('selection_completion.json',selected)
  if not selected['selection_passed']:
   p.write('completion.json',{'completed_utc':p.now(),'selection_pipeline_complete':True,'selection_passed':False,'validation_run':False,'no_retuning':True,'project_complete':False});p.event('selection_failed_validation_not_launched');return
  reviewed=validation(selected)
  p.write('completion.json',{'completed_utc':p.now(),'selection_pipeline_complete':True,'selection_passed':True,'validation_run':True,'independent_validation_review':reviewed,'project_complete':False})
  p.event('calibration_pipeline_complete',validation_passed=reviewed['model_validation_passed'])
 except BaseException as error:
  p.write('stopped.json',{'utc':p.now(),'error_type':type(error).__name__,'error':str(error),'automatic_retry':False,'reconcile_before_resume':True});p.event('stopped',error=str(error));raise
if __name__=='__main__':main()
