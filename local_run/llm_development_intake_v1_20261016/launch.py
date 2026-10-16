"""One bounded CPU development-intake job alongside the identified Mistral selection job."""
import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import time
import intake as t

RESERVE=8*1024**3
ALLOWANCE=128*1024**2
MISTRAL_ALLOWANCE=288*1024**2
QUOTA_PIN='dc7b31d9ed6a4b5dc7a8a5a8e66ffeaad31e057747832641694234c0ed4093bc'


def command(args):return subprocess.check_output(args,text=True,timeout=15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-sha256',required=True);p.add_argument('--paused-controller-receipt-sha256',required=True);a=p.parse_args()
    t.require(len(a.paused_controller_receipt_sha256)==64 and all(c in '0123456789abcdef' for c in a.paused_controller_receipt_sha256),'local_controller_coordination_receipt')
    t.verify_package(a.package_sha256)
    t.require(not os.path.lexists(t.ROOT),'occupied_slot_no_retry')
    t.require(command(['git','-C',str(t.REPO),'rev-parse','HEAD']).strip()=='de9de39a1a72db40485878452561affe2fbd24aa' and not command(['git','-C',str(t.REPO),'status','--porcelain']).strip(),'clean_frozen_repo')
    expected=json.loads((t.HERE/'expectations.json').read_bytes())
    t.require(t.binding(t.FRAME/'result.json')==expected['frame_result'],'calibration_frame_result_pin')
    for name,wanted in expected['repository_files'].items():t.require(t.binding(t.REPO/name)==wanted,'frozen_source_pin')
    qpath=t.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    t.require(t.binding(qpath)['sha256']==QUOTA_PIN,'quota_source')
    spec=importlib.util.spec_from_file_location('task_quota',qpath);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    started=time.monotonic();raw=command(['quota','-v','-w']);quota=q.parse_login_quota(raw);mount=q.mount_identity(t.W)
    t.require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'} and quota['available_soft_bytes']>=RESERVE+ALLOWANCE+MISTRAL_ALLOWANCE,'task_storage_gate')
    # The local transition controller is paused while this bounded independent job
    # runs. Already submitted Mistral continues; no GPU/model worker is changed.
    queue=command(['squeue','-h','-u','poonam','-o','%i|%j|%T'])
    project=[line.split('|') for line in queue.splitlines() if line.split('|')[1].startswith('third-')]
    t.require(len(project)==1 and project[0][0]=='15242' and project[0][1]=='third-v4-byte-repair-mistral' and project[0][2]=='RUNNING','only_known_Mistral_selection_active')
    t.require(time.monotonic()-started<=120,'fresh_quota')
    cmd=['sbatch','--parsable','--job-name=third-development-intake','--output',str(t.ROOT/'slurm-%j.out'),'--error',str(t.ROOT/'slurm-%j.err'),str(t.HERE/'intake.sbatch'),str(t.HERE),a.package_sha256]
    intent={'schema':'third-development-intake-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'package_sha256':a.package_sha256,'paused_local_controller_receipt_sha256':a.paused_controller_receipt_sha256,'concurrent_Mistral_selection_job':'15242','fresh_queue':queue,'quota':quota,'quota_raw':raw,'reserve_bytes':RESERVE,'task_allowance_bytes':ALLOWANCE,'concurrent_mistral_allowance_bytes':MISTRAL_ALLOWANCE,'quota_is_snapshot_not_reservation':True,'model_inference_calls':0,'human_annotation_calls':0,'automatic_retry':False}
    os.umask(0o077);t.ROOT.mkdir(mode=0o700,exist_ok=False);t.publish(t.ROOT/'launch_intent.json',intent)
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        receipt={**intent,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        if r.returncode==0:
            job=r.stdout.strip().split(';')[0];t.require(job.isdigit(),'ambiguous_submission');receipt['job_id']=job
    except Exception as e:
        t.publish(t.ROOT/'launch_receipt.json',{**intent,'submission_outcome':'unknown_reconcile_without_retry','error':str(e)});raise
    t.publish(t.ROOT/'launch_receipt.json',receipt);print(json.dumps(receipt));raise SystemExit(r.returncode)


if __name__=='__main__':main()
