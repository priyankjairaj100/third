"""Fresh quota and terminal reconciliation before one tokenizer-only Slurm job."""
import argparse
from datetime import datetime,timezone
import json
import os
import time
import prepare as x

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-sha256',required=True)
    parser.add_argument('--selection-result-sha256',required=True)
    args=parser.parse_args()
    x.bootstrap(x.HERE,args.package_sha256);x.bootstrap(x.OLD_CONTROL,x.OLD_CONTROL_SHA)
    x.bootstrap(x.V4,x.V4_SHA);x.bootstrap(x.OLD,x.OLD_SHA)
    controls=x.load('v4_request_launch_prior',x.OLD_CONTROL/'control.py');prior,h=controls.helpers()
    x.require(not os.path.lexists(x.ROOT),'occupied_request_stage_reconcile')
    x.require(not h.command(['git','status','--porcelain'],cwd=x.REPO).strip(),'dirty_frozen_repo')
    rb=h.verify_file(x.SELECTION/'run/result.json',args.selection_result_sha256)
    result=h.load(x.SELECTION/'run/result.json')
    x.admit_selection(result,h.binding(x.V4/'PLAN.json')['sha256'],
                      h.load(x.OLD/'expectations.json')['sampler_expectations']['frame_result'])
    x.require(h.load(x.SELECTION/'run/process_completion.json')['passed'] is True,'selection_process')
    job=h.load(x.SELECTION/'launch_receipt.json')['job_id']
    accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,job)
    qp=x.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qp,h.QUOTA_HELPER_SHA);q=x.load('v4_request_quota',qp)
    started=time.monotonic();raw=h.command(['quota','-v','-w'],timeout=30)
    quota=q.parse_login_quota(raw);mount=q.mount_identity(x.W)
    x.require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'} and
              quota['available_soft_bytes']>=8*1024**3+256*1024**2,'quota_256MiB_plus_8GiB')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j'])
    x.require(all(not row.split('|')[1].startswith('third-') for row in queue.splitlines()),'active_project_job')
    x.bootstrap(x.HERE,args.package_sha256)
    x.require(h.binding(x.SELECTION/'run/result.json')==rb and time.monotonic()-started<=120,'fresh_inputs_and_quota')
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-v4-requests','--time=20',
        '--output',str(x.ROOT/'slurm-%j.out'),'--error',str(x.ROOT/'slurm-%j.err'),
        str(x.HERE/'requests.sbatch'),str(x.HERE),args.package_sha256,args.selection_result_sha256]
    intent={'schema':'third-v4-request-launch','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'selection_result':rb,'root':str(x.ROOT),'command':command,
        'fresh_accounting':accounting,'quota':quota,'quota_raw':raw,'mount':mount,'fresh_queue':queue,
        'reserve_bytes':8*1024**3,'output_allowance_bytes':256*1024**2,'automatic_retry':False,
        'model_inference_calls':0,'human_annotation_calls':0}
    os.umask(0o077);x.ROOT.mkdir(mode=0o700,exist_ok=False)
    print(json.dumps(prior.submit_once(h,x.ROOT,command,intent),sort_keys=True))

if __name__=='__main__':main()
