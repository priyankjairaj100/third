"""One-shot local coordinator for this already activated study; no model retries."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

REPO=Path('/Users/priyankjairaj/Downloads/ACL/Three')
TASK=Path('/Users/priyankjairaj/Documents/Codex/2026-10-11/see-i-am-working-on-a')
WORK=TASK/'work/resume-20261014/sampling_v1'
REPORTS=REPO/'local_workspace/reports'
REMOTE=TASK/'work/resume-20261014/remote_existing_session.py'
HERE=Path(__file__).resolve()
def require(ok,why):
    if not ok:raise ValueError(why)
def bind(path):
    data=path.read_bytes();return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(path):return json.loads(path.read_bytes())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def new_json(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

class Coordinator:
    def __init__(self,state,poll_seconds=120,deadline_seconds=10*3600):
        self.state=state;self.poll=poll_seconds;self.deadline=time.monotonic()+deadline_seconds
        self.step=0;self.pins={};self.last_state={}
    def event(self,stage,**fields):
        obj={'utc':now(),'stage':stage,**fields}
        with (self.state/'events.jsonl').open('a') as f:f.write(json.dumps(obj,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(obj,sort_keys=True),flush=True)
    def check_time(self):require(time.monotonic()<self.deadline,'coordinator_deadline_reached_no_cancellation_or_retry')
    def recheck(self):
        self.check_time()
        require(all(bind(Path(p))==b for p,b in self.pins.items()),'coordinator_input_or_source_changed')
    def local(self,name,args):
        self.recheck();self.step+=1
        with (self.state/f'{self.step:03d}-{name}.stdout.txt').open('xb') as out,(self.state/f'{self.step:03d}-{name}.stderr.txt').open('xb') as err:
            result=subprocess.run([sys.executable,*map(str,args)],cwd=TASK,stdout=out,stderr=err,timeout=180)
        require(result.returncode==0,'local_step_failed:'+name)
    def remote(self,name,script,output):
        self.recheck();self.step+=1
        # An occupied receipt or unknown submission is never overwritten/retried.
        with output.open('xb') as out,(self.state/f'{self.step:03d}-{name}.stderr.txt').open('xb') as err:
            result=subprocess.run([sys.executable,str(REMOTE)],input=script.read_bytes(),cwd=TASK,stdout=out,stderr=err,timeout=70)
        require(result.returncode==0,'remote_step_failed_stop_no_retry:'+name+':'+str(result.returncode))
        return read(output)
    def prepare(self,name,builder,args,destination):
        if destination.exists():
            require(str(destination) in self.pins,'unregistered_existing_generated_script:'+str(destination))
        else:
            self.local('prepare-'+name,[builder,*args]);require(destination.is_file(),'missing_prepared_script')
            self.pins[str(destination)]=bind(destination)
            new_json(self.state/(name+'-script-binding.json'),{'path':str(destination),'binding':self.pins[str(destination)]})
    def wait_export(self,kind,mode,script):
        index=0
        while True:
            index+=1;path=self.state/f'{kind}-{mode}-observation-{index:04d}.json'
            export=self.remote('observe-'+kind+'-'+mode,script,path)
            status=(export['job_id'],export['queue'].split('|')[1] if export['queue'] else 'terminal_or_accounting_pending')
            if self.last_state.get((kind,mode))!=status:
                self.event('observed_'+kind,mode=mode,job_id=export['job_id'],state=status[1]);self.last_state[kind,mode]=status
            if export.get('files'):
                require(not export['queue'].strip(),'export_during_active_job')
                return path,export
            # Local waiting only. No login-node inference or recurring scheduler.
            end=time.monotonic()+self.poll
            while time.monotonic()<end:
                self.check_time();time.sleep(min(30,max(0,end-time.monotonic())))
    def family(self,alias):
        script=WORK/('export_inference_'+alias+'_remote.py')
        self.prepare('export-inference-'+alias,WORK/'build_inference_export.py',[alias],script)
        path,export=self.wait_export('inference',alias,script)
        root=REPORTS/('llm-only-v3-selection-'+alias+'-a01')
        self.local('extract-'+alias,[WORK/'extract_private_export.py','--export',path,'--out',root])
        review=WORK/('inference_'+alias+'_local_terminal_review.json')
        self.local('review-'+alias,[WORK/'review_family_terminal.py',alias,'--output',review])
        record=read(review);require(record['passed'] is True,'family_metadata_review_failed:'+alias)
        self.event('family_terminal_review_passed',alias=alias,job_id=export['job_id'],status_counts=record['status_counts'],review=bind(review))
        return record
    def postprocess(self,mode):
        launch=WORK/('launch_postprocess_'+mode+'_remote.py')
        self.prepare('launch-postprocess-'+mode,WORK/'build_postprocess_operation.py',['launch',mode],launch)
        receipt=WORK/('postprocess_'+mode+'_launch_receipt.json')
        result=self.remote('launch-postprocess-'+mode,launch,receipt)
        require(result['returncode']==0 and str(result['job_id']).isdigit(),'postprocess_submission_receipt')
        self.event('postprocess_submitted_once',mode=mode,job_id=result['job_id'],receipt=bind(receipt))
        script=WORK/('export_postprocess_'+mode+'_remote.py')
        self.prepare('export-postprocess-'+mode,WORK/'build_postprocess_operation.py',['export',mode],script)
        path,export=self.wait_export('postprocess',mode,script)
        name='llm-only-v3-selection-statistics-a01' if mode=='statistics' else 'llm-only-v3-output-audit-'+mode+'-a01'
        self.local('extract-postprocess-'+mode,[WORK/'extract_private_export.py','--export',path,'--out',REPORTS/name])
        review=WORK/('postprocess_'+mode+'_local_terminal_review.json')
        self.local('review-postprocess-'+mode,[WORK/'review_postprocess_terminal.py',mode,'--output',review])
        record=read(review);require(record['passed'] is True,'postprocess_review_failed:'+mode)
        self.event('postprocess_review_passed',mode=mode,job_id=export['job_id'],review=bind(review),**({k:record[k] for k in ('selection_passed','selected_tau','conservative_positives_of90')} if mode=='statistics' else {}))
        return record
    def run(self):
        # The existing Qwen job is consumed, never submitted by this coordinator.
        qwen=read(WORK/'inference_qwen_launch_receipt.json');require(qwen['job_id']=='13574' and qwen['returncode']==0,'expected_existing_qwen')
        qwen_review=self.family('qwen');self.postprocess('qwen')
        require(qwen_review['execution_passed'] is True,'qwen_failure_audited_but_mistral_launch_held')
        receipt=WORK/'inference_mistral_launch_receipt.json'
        result=self.remote('launch-mistral',WORK/'launch_inference_mistral_remote.py',receipt)
        require(result['returncode']==0 and str(result['job_id']).isdigit(),'mistral_submission_receipt')
        self.event('mistral_submitted_once',job_id=result['job_id'],receipt=bind(receipt))
        self.family('mistral');self.postprocess('mistral');result=self.postprocess('statistics')
        final={'schema':'third-llm-local-one-shot-pipeline-completion-v1','completed_utc':now(),'passed':True,'planned_model_calls':198,'human_annotation_calls':0,'model_retries':0,'selection_passed':result['selection_passed'],'selected_tau':result['selected_tau'],'conservative_positives_of90':result['conservative_positives_of90'],'statistics_review':bind(WORK/'postprocess_statistics_local_terminal_review.json'),'independent_validation_performed':False,'semantic_validity_established':False,'confirmatory_study_ready':False,'new_codex_automation_created':False}
        new_json(self.state/'completion.json',final);self.event('pipeline_finished',**final)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',action='store_true',required=True);a=p.parse_args()
    os.umask(0o077);state=WORK/'remaining_pipeline_a01';state.mkdir(mode=0o700,exist_ok=False)
    runner=Coordinator(state)
    names=['build_inference_export.py','build_postprocess_operation.py','extract_private_export.py','review_family_terminal.py','review_postprocess_terminal.py','export_inference_qwen_remote.py','export_postprocess_qwen_remote.py','launch_inference_mistral_remote.py','inference_resource_plan.json','inference_admission.json','inference_activation_receipt.json','inference_qwen_launch_receipt.json']
    paths=[HERE,REMOTE,*[WORK/name for name in names]]
    for folder in ('llm_annotation_execution_v1_20261014','llm_annotation_integration_v1_20261014','llm_annotation_control_v1_20261014','llm_annotation_postprocess_v1_20261014','llm_annotation_statistics_v1_20261014'):
        root=REPO/'local_run'/folder;manifest=read(root/'package_manifest.json');paths.extend([root/'package_manifest.json',*[root/n for n in manifest['files']]])
    runner.pins={str(p):bind(p) for p in paths}
    new_json(state/'intent.json',{'schema':'third-llm-local-one-shot-pipeline-intent-v1','created_utc':now(),'pid':os.getpid(),'deadline_seconds':10*3600,'poll_seconds':120,'existing_qwen_job':'13574','source_and_input_bindings':runner.pins,'model_retries':0,'human_annotation_calls':0,'new_activation_or_entropy':False,'recurring_automation':False,'private_outputs_only':True})
    try:
        runner.event('coordinator_started',pid=os.getpid(),existing_qwen_job='13574',human_annotation_calls=0)
        runner.run()
    except BaseException as exc:
        failure={'schema':'third-llm-local-one-shot-pipeline-failure-v1','failed_utc':now(),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False,'active_cluster_jobs_not_cancelled':True,'next_step':'Inspect the recorded receipts and error before any new remote action; no automatic reconnect or duplicate submission.'}
        new_json(state/'failure.json',failure);runner.event('coordinator_stopped',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        raise
if __name__=='__main__':main()
