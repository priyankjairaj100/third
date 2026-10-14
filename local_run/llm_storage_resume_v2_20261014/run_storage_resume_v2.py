"""One-shot continuation after reviewed quota cleanup; preserves stopped v1."""
import argparse, os, traceback
from pathlib import Path
from run_remaining_pipeline import Coordinator, REPO, WORK, REPORTS, REMOTE, bind, read, now, new_json, require

HERE = Path(__file__).resolve()
STATE = WORK / 'storage_resume_v2_a01'
NEW_RECEIPT = WORK / 'postprocess_qwen_launch_receipt_after_storage_cleanup.json'
OLD_REVIEW_SHA = 'ab88f6feb94e2c6be44ad326aeda5379fc89d1f971052ed31652c5ce735f8859'

def validate_resumption(snapshot):
    require(not snapshot['project_queue'].strip(), 'active_project_job')
    require(all(not any(v.values()) for v in snapshot['next_stage_slots'].values()), 'occupied_next_stage')
    require(snapshot['quota']['available_soft_bytes'] >= snapshot['mistral_required_bytes'], 'quota_not_restored')
    require('13574|COMPLETED|0:0|' in snapshot['qwen_accounting'], 'prior_qwen_not_complete')

class StorageResume(Coordinator):
    def family(self, alias):
        if alias != 'qwen':
            return super().family(alias)
        path = WORK / 'inference_qwen_local_terminal_review.json'
        require(bind(path)['sha256'] == OLD_REVIEW_SHA, 'changed_saved_qwen_review')
        value = read(path)
        require(value['passed'] is True and value['execution_passed'] is True
                and value['job_id'] == '13574' and value['status_counts'] == {'valid': 99},
                'saved_qwen_review_not_accepted')
        self.event('reused_completed_qwen_review', job_id='13574', model_calls_repeated=0, review=bind(path))
        return value

    def postprocess(self, mode):
        if mode != 'qwen':
            return super().postprocess(mode)
        # This is the first actual audit launch. V1 stopped before remote intent creation.
        result = self.remote('launch-postprocess-qwen-after-cleanup',
                             WORK/'launch_postprocess_qwen_remote.py', NEW_RECEIPT)
        require(result['returncode'] == 0 and str(result['job_id']).isdigit(), 'postprocess_submission_receipt')
        self.event('postprocess_submitted_once', mode=mode, job_id=result['job_id'], receipt=bind(NEW_RECEIPT))
        path, export = self.wait_export('postprocess', mode, WORK/'export_postprocess_qwen_remote.py')
        self.local('extract-postprocess-qwen', [WORK/'extract_private_export.py', '--export', path,
                   '--out', REPORTS/'llm-only-v3-output-audit-qwen-a01'])
        review = WORK/'postprocess_qwen_local_terminal_review.json'
        self.local('review-postprocess-qwen', [WORK/'review_postprocess_terminal_v2.py', mode,
                   '--launch-receipt', NEW_RECEIPT, '--output', review])
        record = read(review); require(record['passed'] is True, 'qwen_audit_review_failed')
        self.event('postprocess_review_passed', mode=mode, job_id=export['job_id'], review=bind(review))
        return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',action='store_true',required=True);p.parse_args()
    old=WORK/'remaining_pipeline_a01'
    require(read(old/'failure.json')['error']=='remote_step_failed_stop_no_retry:launch-postprocess-qwen:1', 'unexpected_v1_stop')
    require('quota_reserve_and_commitment' in (old/'015-launch-postprocess-qwen.stderr.txt').read_text(), 'not_storage_stop')
    require((WORK/'postprocess_qwen_launch_receipt.json').read_bytes()==b'', 'old_empty_receipt_changed')
    old_pid=read(old/'intent.json')['pid']
    try:
        os.kill(old_pid,0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('prior_controller_pid_exists')
    for path in [NEW_RECEIPT, WORK/'inference_mistral_launch_receipt.json',
                 WORK/'postprocess_qwen_local_terminal_review.json',
                 REPORTS/'llm-only-v3-output-audit-qwen-a01']:
        require(not path.exists(), 'occupied_resumption_destination:'+str(path))
    os.umask(0o077);STATE.mkdir(mode=0o700,exist_ok=False)
    runner=StorageResume(STATE)
    names=['run_remaining_pipeline.py','run_storage_resume_v2.py','review_postprocess_terminal_v2.py',
           'build_inference_export.py','build_postprocess_operation.py','extract_private_export.py',
           'review_family_terminal.py','review_postprocess_terminal.py','export_inference_qwen_remote.py',
           'export_postprocess_qwen_remote.py','launch_postprocess_qwen_remote.py','launch_inference_mistral_remote.py',
           'inference_resource_plan.json','inference_admission.json','inference_activation_receipt.json',
           'inference_qwen_launch_receipt.json','inference_qwen_local_terminal_review.json',
           'status_quota_after_controller_stop_remote.py','postprocess_qwen_launch_receipt.json']
    paths=[REMOTE,*[WORK/name for name in names],old/'intent.json',old/'failure.json',
           old/'015-launch-postprocess-qwen.stderr.txt',WORK.parent/'cache_cleanup_terminal.json']
    for folder in ('llm_annotation_execution_v1_20261014','llm_annotation_integration_v1_20261014',
                   'llm_annotation_control_v1_20261014','llm_annotation_postprocess_v1_20261014',
                   'llm_annotation_statistics_v1_20261014'):
        root=REPO/'local_run'/folder;manifest=read(root/'package_manifest.json')
        paths.extend([root/'package_manifest.json',*[root/n for n in manifest['files']]])
    runner.pins={str(path):bind(path) for path in paths}
    new_json(STATE/'intent.json',dict(schema='third-llm-storage-resumption-v2',created_utc=now(),pid=os.getpid(),
             source_and_input_bindings=runner.pins,existing_qwen_job='13574',prior_controller_preserved=True,
             failed_empty_receipt_preserved=True,new_qwen_audit_receipt=str(NEW_RECEIPT),
             model_retries=0,human_annotation_calls=0,new_activation_or_entropy=False,
             poll_seconds=120,deadline_seconds=36000,recurring_automation=False))
    try:
        snapshot=runner.remote('reconcile-after-cleanup',WORK/'status_quota_after_controller_stop_remote.py',STATE/'resumption_preflight.json')
        validate_resumption(snapshot)
        runner.event('storage_resumption_started',pid=os.getpid(),quota_available_bytes=snapshot['quota']['available_soft_bytes'])
        runner.run()
    except BaseException as exc:
        failure=dict(schema='third-llm-storage-resumption-failure-v2',failed_utc=now(),error_type=type(exc).__name__,
                     error=str(exc),traceback=traceback.format_exc(),automatic_retry=False,active_cluster_jobs_not_cancelled=True)
        new_json(STATE/'failure.json',failure)
        runner.event('coordinator_stopped',error_type=type(exc).__name__,error=str(exc),automatic_retry=False)
        raise

if __name__ == '__main__':
    main()
