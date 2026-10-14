"""Commit four seeds exactly once, then submit one v4 sampling job."""
import argparse
from datetime import datetime, timezone
import json
import os
import secrets
import sys
import time
import protocol as p
import select_worker as worker

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-sha256',required=True)
    parser.add_argument('--tail-result-sha256',required=True)
    args=parser.parse_args()
    worker.bootstrap(worker.FOLDER,args.package_sha256);worker.bootstrap(worker.TAIL,worker.TAIL_SHA)
    sys.path.insert(0,str(worker.TAIL));import census as t
    t.verify(t.CONTROL,t.CONTROL_SHA);t.verify(t.SAMPLER,t.SAMPLER_SHA)
    c=t.module('v4_launch_control',t.CONTROL/'control.py');h=c.helpers()
    p.require(not os.path.lexists(worker.ROOT),'occupied_v4_sampling_stage_reconcile')
    p.require(not h.command(['git','status','--porcelain'],cwd=t.REPO).strip(),'dirty_executing_repo')
    p.require(h.command(['git','rev-parse','HEAD'],cwd=t.REPO).strip()==
              'de9de39a1a72db40485878452561affe2fbd24aa','frozen_execution_commit')
    plan=json.loads((worker.FOLDER/'PLAN.json').read_bytes());p.validate(plan,worker.FOLDER)
    tail_binding=h.verify_file(worker.TAIL_RESULT,args.tail_result_sha256)
    tail=h.load(worker.TAIL_RESULT)
    p.require(tail['passed'] is True and tail['all_220405510_pair_scores_replayed_exactly'] is True
              and tail['observed']['pair_count']==220405510,'full_census_pass')
    tail_job=h.load(worker.TAIL_RESULT.parent.parent/'launch_receipt.json')['job_id']
    accounting=h.command(['sacct','-j',tail_job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,tail_job)
    p.require(h.load(worker.TAIL_RESULT.parent/'process_completion.json')['passed'] is True,'tail_process_pass')
    qpath=t.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=t.module('v4_selection_quota',qpath)
    mount=q.mount_identity(t.W);started=time.monotonic()
    quota_raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(quota_raw)
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j'])
    p.require(all(not line.split('|')[1].startswith('third-') for line in queue.splitlines()),'active_project_job')
    worker.bootstrap(worker.FOLDER,args.package_sha256)
    p.require(p.binding(worker.TAIL_RESULT)==tail_binding,'tail_changed')
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    os.umask(0o077);worker.ROOT.mkdir(mode=0o700,exist_ok=False)
    # Intent precedes entropy. Any failure from here occupies the slot permanently.
    h.write_exclusive(worker.ROOT/'sampling_intent.json',{'package_sha256':args.package_sha256,
        'tail_result':tail_binding,'created_utc':datetime.now(timezone.utc).isoformat(),
        'fresh_queue':queue,'fresh_accounting':accounting,'quota':quota,'quota_raw':quota_raw,'mount':mount})
    s=t.module('v4_lock_serialization',t.SAMPLER/'sampler.py')
    lock=s.seal({'schema':'third-llm-v4-sampling-lock','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'plan_sha256':p.binding(worker.FOLDER/'PLAN.json')['sha256'],
        'root':str(worker.ROOT),'tail_result':tail_binding,
        'seeds':{name:secrets.token_hex(32) for name in plan['randomness']['seeds']},
        'entropy':'OS via secrets.token_hex; single committed draw per stage; no seed search',
        'prior_v3_exposure_disclosed':True,'human_annotation_calls':0,'inference_activation':False})
    s.publish(worker.ROOT/'sampling_lock.json',lock)
    lock_binding=p.binding(worker.ROOT/'sampling_lock.json')
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-v4-selection',
        '--time=60','--output',str(worker.ROOT/'slurm-%j.out'),'--error',str(worker.ROOT/'slurm-%j.err'),
        str(worker.FOLDER/'selection.sbatch'),str(worker.FOLDER),args.package_sha256,lock_binding['sha256']]
    intent={'schema':'third-v4-selection-launch','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'sampling_lock':lock_binding,'root':str(worker.ROOT),
        'command':command,'tail_result':tail_binding,'fresh_accounting':accounting,
        'quota':quota,'reserve_bytes':c.RESERVE,'allowance_bytes':c.ALLOWANCE,
        'automatic_retry':False,'human_annotation_calls':0,'model_inference_calls':0}
    print(json.dumps(c.submit_once(h,worker.ROOT,command,intent),sort_keys=True))

if __name__=='__main__':main()
