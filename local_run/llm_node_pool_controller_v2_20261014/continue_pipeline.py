"""One finite node-pool v2 pipeline; preserves v4 samples and zero-human rules."""
import hashlib,importlib.util,json,os,time
from pathlib import Path
TASK=Path(os.environ['THIRD_TASK_ROOT'])
REPO=Path(os.environ['THIRD_LOCAL_REPO'])
WORK=TASK/'work/resume-20261014';HERE=WORK/'node_pool_v2';OUT=WORK/'node_pool_v2_pipeline_a01'
def module(name,path,pin):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin,'controller_source_pin'
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=module('node_pool_selection_helpers',Path(__file__).with_name('selection_helpers.py'),'720a783a2d034b351452f6089329a9086f3dbec849a00c6bda166544dcae56e7')
v=module('node_pool_validation_review_helpers',Path(__file__).with_name('validation_helpers.py'),'f3cda717f5e6b93e48fbb20c1cf1abda6cc531f8439aab4c56e318d800ca8505')
p.OUT=OUT;p.REPO=REPO
W=p.W
PINS={'selection_inference':'adcd4704a99bb6d3eabfbd8e5409453a96faa891b00aef93e399e2a23d1034fe',
      'validation_sampler':'1abac2e6d14f38c55e78b32031040e7ee387bfe92b8234998187ee1b91e21ad2',
      'validation_requests':'8721e0499db68998ef509f9f4263410f786ec2e7c9118f970270965b4ee68b67',
      'validation_inference':'3b5d8a811909d33cefd9fd6b7c5dbb0e8d17fd1facca447fa0717f1f2ce438c7'}
SELROOT=W+'/qualification/llm-only-v4-inference-v2-a01'
VALROOT=W+'/qualification/llm-only-v4-validation-inference-v2-a01'
SAMPLING=W+'/qualification/llm-only-v4-validation-sampling-v2-a01'
p.ROOT=SELROOT;v.p=p;v.ROOT=VALROOT;v.SAMPLING=SAMPLING;v.OLD=SELROOT;v.INFER=PINS['validation_inference'];v.REPO=REPO

def await_success(label,root,job,result_relative,node):
    index=0;last=None
    while True:
        r=p.status(label,root,job,result_relative,index);index+=1
        if r['state']!=last:p.event('stage_state',stage=label,job=job,state=r['state']);last=r['state']
        if r['state'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'):
            time.sleep(60);continue
        correct_node=r['node'] in ('csis.cn1','csis.cn3','csis.cn4') if node=='gpu' else r['node']==node
        p.require(r['state']=='COMPLETED' and r['exit']=='0:0' and correct_node,'stage_failed:'+label)
        p.require(r['result'] is not None and r['result']['passed'] is True,'no_accepted_result:'+label)
        p.write(label+'_terminal.json',r);p.event('stage_complete',stage=label,job=job,result=r['result_binding']);return r

def reconcile():
    result=p.remote('reconcile_migration','''import hashlib,json,subprocess
from pathlib import Path
w=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
raw=subprocess.check_output(['sacct','-j','13641,13659','-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=10)
rows={r.split('|')[0]:r.split('|') for r in raw.splitlines()}
assert rows['13641'][1].startswith('CANCELLED') and rows['13641'][3]=='0','old_model_job_must_be_cancelled_unstarted'
assert rows['13659'][1:3]==['COMPLETED','0:0'] and rows['13659'][4]=='csis.cn4','capability_probe_not_successful'
assert not (w/'qualification/llm-only-v4-inference-a01/qwen/output').exists(),'no_original_model_outputs'
probe=w/'qualification/llm-node-pool-probe-v1-20261014-a01/run/result.json';data=probe.read_bytes();result=json.loads(data)
assert result['passed'] is True and result['host']=='csis.cn4' and result['model_inference_calls']==0
print(json.dumps({'passed':True,'accounting':raw,'probe_binding':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()},'probe':result,'prior_v4_model_calls':0}))
''')
    p.write('migration_reconciliation.json',result)

def model_sequence(label,folder,pin,root,qualroot,request_sha):
    base=['--package-sha256',pin,'--request-result-sha256',request_sha]
    launch=p.run_cli(label+'_qualification_launch',folder,pin,'launch.py',['--mode','qualification',*base])
    qualified=await_success(label+'_qualification',qualroot,launch['job_id'],'run/result.json','csis.cn1')
    activation=p.run_cli(label+'_activation',folder,pin,'activate.py',
        base+['--qualification-result-sha256',qualified['result_binding']['sha256']])
    base+=['--activation-sha256',activation['activation']['sha256'],'--resource-plan-sha256',activation['resource_plan']['sha256']]
    for mode in ('qwen','audit_qwen','mistral','audit_mistral','statistics'):
        launch=p.run_cli(label+'_'+mode+'_launch',folder,pin,'launch.py',['--mode',mode,*base])
        gpu=mode in ('qwen','mistral')
        await_success(label+'_'+mode,root+'/'+mode,launch['job_id'],'output/result.json' if gpu else 'run/result.json',
                      'gpu' if gpu else 'csis.cn1')

def main():
    os.umask(0o077);OUT.mkdir(exist_ok=False)
    p.write('intent.json',{'created_utc':p.now(),'pid':os.getpid(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'packages':PINS,'scope':'Same v4 selection and conditional fresh validation, amended compute-node scheduling',
        'allowed_gpu_compute_nodes':['csis.cn1','csis.cn3','csis.cn4'],'human_annotation_calls':0,
        'automatic_retry':False,'sample_or_prompt_changes':False,'project_complete':False})
    try:
        reconcile()
        model_sequence('selection',W+'/tools/llm-v4-inference-v2-20261014',PINS['selection_inference'],SELROOT,
            W+'/qualification/llm-v4-inference-qualification-v2-a01','334e193f0cac3aa0d36bac510d06719b98efe5f715b031b7944c1c1ec7757db4')
        selected=p.independent_statistics_review();p.write('selection_completion.json',selected)
        if not selected['selection_passed']:
            p.write('completion.json',{'completed_utc':p.now(),'selection_pipeline_complete':True,'selection_passed':False,
                'validation_run':False,'no_retuning':True,'project_complete':False})
            p.event('selection_failed_validation_not_launched');return
        samplepin=PINS['validation_sampler'];samplefolder=W+'/tools/llm-v4-validation-sampler-v2-20261014'
        launch=p.run_cli('validation_sampling_launch',samplefolder,samplepin,'launch.py',
            ['--package-sha256',samplepin,'--selection-statistics-sha256',selected['bindings']['statistics']['sha256']])
        sample=await_success('validation_sampling',SAMPLING,launch['job_id'],'run/result.json','csis.cn1')
        requestpin=PINS['validation_requests'];folder=W+'/tools/llm-v4-validation-requests-v2-20261014'
        launch=p.run_cli('validation_requests_launch',folder,requestpin,'launch.py',
            ['--package-sha256',requestpin,'--selection-result-sha256',sample['result_binding']['sha256']])
        prepared=await_success('validation_requests',W+'/qualification/llm-v4-validation-requests-v2-20261014-a01',
            launch['job_id'],'run/result.json','csis.cn1')
        p.require(0<prepared['result']['planned_requests']<=440,'validation_call_cap')
        model_sequence('validation',W+'/tools/llm-v4-validation-inference-v2-20261014',PINS['validation_inference'],VALROOT,
            W+'/qualification/llm-v4-validation-inference-qualification-v2-a01',prepared['result_binding']['sha256'])
        reviewed=v.independent_review()
        p.write('completion.json',{'completed_utc':p.now(),'selection_pipeline_complete':True,'selection_passed':True,
            'validation_run':True,'independent_validation_review':reviewed,'project_complete':False,
            'next_stage':'Model-defined effect study if validation passed; preserve failure without retuning otherwise'})
        p.event('calibration_pipeline_complete',validation_passed=reviewed['model_validation_passed'])
    except BaseException as error:
        p.write('stopped.json',{'utc':p.now(),'error_type':type(error).__name__,'error':str(error),'automatic_retry':False,
            'reconcile_before_resume':True});p.event('stopped',error=str(error));raise

if __name__=='__main__':main()
