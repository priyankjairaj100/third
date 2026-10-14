"""Continue only the unstarted Mistral stage after explicit GPU-node guard repair."""
import argparse, os, traceback
from pathlib import Path
from run_remaining_pipeline import Coordinator, REPO, WORK, REPORTS, REMOTE, bind, read, now, new_json, require

STATE=WORK/'mistral_resume_v3_a01'
RECEIPT=WORK/'inference_mistral_launch_receipt_adapter_v2.json'
QWEN_REVIEW_SHA='dd20af7bf2a13c8af32c45207e260fe77e04d3cc322b6db1936dbfa6b3c2f26b'

def validate(snapshot):
    require(not snapshot['project_queue'].strip(),'active_project_job')
    require(all(snapshot['next_stage_slots']['qwen_audit'].values()),'missing_completed_qwen_audit_slot')
    require(all(not any(snapshot['next_stage_slots'][k].values()) for k in ('mistral','mistral_audit','statistics')),'occupied_remaining_slot')
    require(snapshot['quota']['available_soft_bytes']>=snapshot['mistral_required_bytes'],'quota_not_restored')

class MistralResume(Coordinator):
    def family(self,alias):
        require(alias=='mistral','no_repeat_qwen')
        script=WORK/'export_inference_mistral_remote.py'
        self.prepare('export-inference-mistral',WORK/'build_inference_export.py',['mistral'],script)
        path,export=self.wait_export('inference','mistral',script)
        self.local('extract-mistral',[WORK/'extract_private_export.py','--export',path,'--out',REPORTS/'llm-only-v3-selection-mistral-a01'])
        review=WORK/'inference_mistral_local_terminal_review.json'
        self.local('review-mistral',[WORK/'review_family_terminal_v2.py','mistral','--launch-receipt',RECEIPT,'--output',review])
        record=read(review);require(record['passed'] is True,'family_metadata_review_failed:mistral')
        self.event('family_terminal_review_passed',alias=alias,job_id=export['job_id'],status_counts=record['status_counts'],review=bind(review))
        return record

    def run(self):
        review=WORK/'postprocess_qwen_local_terminal_review.json'
        require(bind(review)['sha256']==QWEN_REVIEW_SHA and read(review)['passed'] is True,'accepted_qwen_audit')
        result=self.remote('launch-mistral-role-adapter',WORK/'launch_inference_mistral_adapter_v2_remote.py',RECEIPT)
        require(result['returncode']==0 and str(result['job_id']).isdigit(),'mistral_submission_receipt')
        self.event('mistral_submitted_once',job_id=result['job_id'],receipt=bind(RECEIPT))
        self.family('mistral');self.postprocess('mistral');result=self.postprocess('statistics')
        final={'schema':'third-llm-mistral-resumption-completion-v3','completed_utc':now(),'passed':True,
               'planned_model_calls':198,'human_annotation_calls':0,'model_retries':0,
               'selection_passed':result['selection_passed'],'selected_tau':result['selected_tau'],
               'conservative_positives_of90':result['conservative_positives_of90'],
               'statistics_review':bind(WORK/'postprocess_statistics_local_terminal_review.json'),
               'independent_validation_performed':False,'semantic_validity_established':False,'confirmatory_study_ready':False,
               'new_codex_automation_created':False}
        new_json(self.state/'completion.json',final);self.event('pipeline_finished',**final)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',action='store_true',required=True);p.parse_args()
    old=WORK/'storage_resume_v2_a01'
    require(read(old/'failure.json')['error']=='remote_step_failed_stop_no_retry:launch-mistral:1','expected_preflight_stop')
    require('assembly_not_successful_terminal' in (old/'008-launch-mistral.stderr.txt').read_text(),'expected_node_guard_failure')
    require((WORK/'inference_mistral_launch_receipt.json').read_bytes()==b'','preserved_failed_empty_receipt')
    try:os.kill(read(old/'intent.json')['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('prior_controller_pid_exists')
    for path in [RECEIPT,WORK/'inference_mistral_local_terminal_review.json',REPORTS/'llm-only-v3-selection-mistral-a01']:
        require(not path.exists(),'occupied_resumption_destination:'+str(path))
    os.umask(0o077);STATE.mkdir(mode=0o700,exist_ok=False);runner=MistralResume(STATE)
    names=['run_mistral_resume_v3.py','run_remaining_pipeline.py','review_family_terminal_v2.py',
           'review_family_terminal.py','review_postprocess_terminal.py','build_inference_export.py',
           'build_postprocess_operation.py','extract_private_export.py','launch_inference_mistral_adapter_v2_remote.py',
           'inference_resource_plan.json','inference_admission.json','inference_activation_receipt.json',
           'postprocess_qwen_local_terminal_review.json','postprocess_qwen_launch_receipt_after_storage_cleanup.json',
           'inference_mistral_launch_receipt.json','status_quota_after_controller_stop_remote.py']
    paths=[REMOTE,*[WORK/n for n in names],old/'intent.json',old/'failure.json',old/'008-launch-mistral.stderr.txt']
    for folder in ('llm_annotation_execution_v1_20261014','llm_annotation_integration_v1_20261014',
                   'llm_annotation_control_v1_20261014','llm_annotation_postprocess_v1_20261014',
                   'llm_annotation_statistics_v1_20261014','llm_mistral_launch_adapter_v2_20261014'):
        root=REPO/'local_run'/folder;manifest=read(root/'package_manifest.json');paths.extend([root/'package_manifest.json',*[root/n for n in manifest['files']]])
    runner.pins={str(p):bind(p) for p in paths}
    new_json(STATE/'intent.json',{'schema':'third-llm-mistral-resumption-v3','created_utc':now(),'pid':os.getpid(),
             'source_and_input_bindings':runner.pins,'qwen_model_and_audit_reused':True,'old_controllers_and_empty_receipts_preserved':True,
             'new_mistral_launch_receipt':str(RECEIPT),'model_retries':0,'human_annotation_calls':0,'new_activation_or_entropy':False,
             'poll_seconds':120,'deadline_seconds':36000,'recurring_automation':False})
    try:
        snapshot=runner.remote('reconcile-before-first-mistral',WORK/'status_quota_after_controller_stop_remote.py',STATE/'resumption_preflight.json')
        validate(snapshot);runner.event('mistral_resumption_started',pid=os.getpid(),quota_available_bytes=snapshot['quota']['available_soft_bytes'])
        runner.run()
    except BaseException as exc:
        failure={'schema':'third-llm-mistral-resumption-failure-v3','failed_utc':now(),'error_type':type(exc).__name__,'error':str(exc),
                 'traceback':traceback.format_exc(),'automatic_retry':False,'active_cluster_jobs_not_cancelled':True}
        new_json(STATE/'failure.json',failure);runner.event('coordinator_stopped',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        raise

if __name__=='__main__':main()
