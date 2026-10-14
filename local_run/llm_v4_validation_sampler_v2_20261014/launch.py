"""One validation sample only after a successful, terminal v4 selection analysis."""
import argparse
from datetime import datetime,timezone
import json
import os
import time
import worker as x
import primitives as v

def main():
    parser=argparse.ArgumentParser()
    for name in ('package-sha256','selection-statistics-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p,original,t,c,s,plan=x.setup(args.package_sha256);h=c.helpers()
    v.require(not os.path.lexists(x.ROOT),'occupied_validation_stage_reconcile')
    v.require(not h.command(['git','status','--porcelain'],cwd=x.REPO).strip(),'dirty_frozen_execution_repo')
    v.require(h.command(['git','rev-parse','HEAD'],cwd=x.REPO).strip()==
              'de9de39a1a72db40485878452561affe2fbd24aa','frozen_execution_commit')
    rb=h.verify_file(x.STATISTICS,args.selection_statistics_sha256)
    report=h.load(x.STATISTICS);s.verify_seal(report,True)
    v.require(report['passed'] is True and report['package_sha256']==x.INFERENCE_SHA and
              report['statistics']['selection_passed'] is True,'positive_selection_required')
    job=h.load(x.SELECTION_INFERENCE/'statistics/launch_receipt.json')['job_id']
    accounting=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(accounting,job)
    process=h.load(x.SELECTION_INFERENCE/'statistics/process_completion.json')
    v.require(process['returncode']==0 and not process['console_cap_exceeded'],'successful_selection_process')
    sb=h.binding(x.SELECTION/'sampling_lock.json');seedlock=h.load(x.SELECTION/'sampling_lock.json');s.verify_seal(seedlock,True)
    v.require(seedlock['package_sha256']==x.V4_SHA and
        seedlock['plan_sha256']==p.binding(x.V4/'PLAN.json')['sha256'],'committed_v4_seeds')
    qpath=x.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=t.module('v4_validation_quota',qpath)
    mount=q.mount_identity(x.W);started=time.monotonic();raw=h.command(['quota','-v','-w'],timeout=30)
    quota=q.parse_login_quota(raw);c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j'])
    v.require(all(not row.split('|')[1].startswith('third-') for row in queue.splitlines()),'active_project_job')
    x.bootstrap(x.HERE,args.package_sha256)
    v.require(h.binding(x.STATISTICS)==rb and h.binding(x.SELECTION/'sampling_lock.json')==sb,'unchanged_inputs')
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount)
    os.umask(0o077);x.ROOT.mkdir(mode=0o700,exist_ok=False)
    lock=s.seal({'schema':'third-v4-validation-sampling-lock-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'root':str(x.ROOT),'plan_sha256':seedlock['plan_sha256'],
        'selection_statistics':rb,'sampling_lock':sb,'locked_tau':report['statistics']['selected_tau'],
        'new_entropy_draws':0,'sampling_roles':['validation','validation_swap'],
        'human_annotation_calls':0,'inference_activated':False})
    s.publish(x.ROOT/'validation_lock.json',lock);lb=h.binding(x.ROOT/'validation_lock.json')
    command=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-v4-validation-sample','--time=60',
        '--output',str(x.ROOT/'slurm-%j.out'),'--error',str(x.ROOT/'slurm-%j.err'),
        str(x.HERE/'sample.sbatch'),str(x.HERE),args.package_sha256,lb['sha256']]
    intent={'schema':'third-v4-validation-sampling-launch','created_utc':datetime.now(timezone.utc).isoformat(),
        'package_sha256':args.package_sha256,'validation_lock':lb,'root':str(x.ROOT),'command':command,
        'fresh_accounting':accounting,'fresh_queue':queue,'quota':quota,'quota_raw':raw,'mount':mount,
        'reserve_bytes':c.RESERVE,'allowance_bytes':c.ALLOWANCE,'automatic_retry':False,
        'new_entropy_draws':0,'human_annotation_calls':0,'model_inference_calls':0}
    print(json.dumps(c.submit_once(h,x.ROOT,command,intent),sort_keys=True))

if __name__=='__main__':main()
