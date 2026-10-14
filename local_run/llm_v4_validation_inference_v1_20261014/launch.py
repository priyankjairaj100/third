"""Exclusive staged v4 submissions with fresh quota and predecessor receipts."""
import argparse
from datetime import datetime,timezone
import json
import os
import time
import common as x
import contract as C

MODES=('qualification','qwen','audit_qwen','mistral','audit_mistral','statistics')

def root_for(mode):
    C.require(mode in MODES,'known_mode')
    return x.QUAL if mode=='qualification' else x.ROOT/mode

def terminal(h,root,relative,node):
    receipt=h.load(root/'launch_receipt.json');job=receipt['job_id']
    raw=h.command(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    rows=[line.split('|') for line in raw.splitlines() if line.split('|')[0]==job]
    C.require(len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0'] and rows[0][5]==node,'successful_predecessor')
    # Request preparation predates this package's general process wrapper.
    if (root/'process_completion.json').exists():
        process=h.load(root/'process_completion.json')
        C.require(process['returncode']==0 and not process['console_cap_exceeded'],'predecessor_process')
    result=h.load(root/relative);C.require(result['passed'] is True,'predecessor_passed')
    return {'job_id':job,'accounting':raw,'result':h.binding(root/relative)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True)
    for name in ('package-sha256','request-result-sha256'):p.add_argument('--'+name,required=True)
    for name in ('activation-sha256','resource-plan-sha256'):p.add_argument('--'+name)
    args=p.parse_args();pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    root=root_for(args.mode);C.require(not os.path.lexists(root),'occupied_stage_reconcile_no_retry')
    C.require(not h.command(['git','status','--porcelain'],cwd=x.REPO).strip(),'dirty_execution_checkout')
    C.require(h.command(['git','rev-parse','HEAD'],cwd=x.REPO).strip()==
              'de9de39a1a72db40485878452561affe2fbd24aa','execution_commit')
    _,request=x.request_contract(args.request_result_sha256,pins)
    predecessors={}
    if args.mode=='qualification':
        predecessors['requests']=terminal(h,x.REQUESTS.parent,'run/result.json','csis.cn1')
        C.require(args.activation_sha256 is None and args.resource_plan_sha256 is None,'qualification_precedes_activation')
    else:
        C.require(args.activation_sha256 and args.resource_plan_sha256,'activated_source_pins_required')
        C.require(h.binding(x.ROOT/'activation/activation.json')['sha256']==args.activation_sha256 and
                  h.binding(x.ROOT/'resource_plan.json')['sha256']==args.resource_plan_sha256,'activation_resource_pins')
        dependencies={'qwen':[('qualification',x.QUAL,'run/result.json','csis.cn1')],
            'audit_qwen':[('qwen',x.ROOT/'qwen','output/result.json','csis.cn3')],
            'mistral':[('audit_qwen',x.ROOT/'audit_qwen','run/result.json','csis.cn1')],
            'audit_mistral':[('mistral',x.ROOT/'mistral','output/result.json','csis.cn3')],
            'statistics':[('audit_qwen',x.ROOT/'audit_qwen','run/result.json','csis.cn1'),
                          ('audit_mistral',x.ROOT/'audit_mistral','run/result.json','csis.cn1')]}
        for key,folder,name,node in dependencies[args.mode]:predecessors[key]=terminal(h,folder,name,node)
    is_gpu=args.mode in ('qwen','mistral')
    allowance=(288 if is_gpu else 64)*1024**2
    qpath=x.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
    h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=x.module('v4_stage_quota',qpath)
    mount=q.mount_identity(x.W);started=time.monotonic();qraw=h.command(['quota','-v','-w'],timeout=30)
    quota=q.parse_login_quota(qraw)
    C.require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'} and
              quota['available_soft_bytes']>=8*1024**3+allowance,'fresh_storage_gate')
    queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j'])
    C.require(all(not row.split('|')[1].startswith('third-') for row in queue.splitlines()),'active_project_job')
    x.bootstrap(x.HERE,args.package_sha256)
    C.require(time.monotonic()-started<=120,'fresh_quota')
    if is_gpu:
        resources=h.load(x.ROOT/'resource_plan.json');family=resources['families'][args.mode]
        flags=['--partition=gpu-short','--nodelist=csis.cn3','--gres=gpu:a100-80gb:1','--cpus-per-task=4',
            '--mem='+str(family['host_memory_bytes']//1024**3)+'G','--time='+str(family['slurm_time_seconds']//60)]
    else:flags=['--partition=cpu-short','--nodelist=csis.cn1','--cpus-per-task=2','--mem=8G','--time=20']
    command=['sbatch','--parsable',*flags,'--job-name=third-v4-val-'+args.mode,
        '--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),
        str(x.HERE/'execute.sbatch'),str(x.HERE),'--mode',args.mode,
        '--package-sha256',args.package_sha256,'--request-result-sha256',args.request_result_sha256]
    if not args.mode=='qualification':
        command+=['--activation-sha256',args.activation_sha256,'--resource-plan-sha256',args.resource_plan_sha256]
    intent={'schema':'third-v4-validation-stage-launch','created_utc':datetime.now(timezone.utc).isoformat(),
        'mode':args.mode,'root':str(root),'package_sha256':args.package_sha256,'command':command,
        'predecessors':predecessors,'quota':quota,'quota_raw':qraw,'mount':mount,'fresh_queue':queue,
        'reserve_bytes':8*1024**3,'stage_allowance_bytes':allowance,'automatic_retry':False,'human_annotation_calls':0}
    os.umask(0o077);root.mkdir(mode=0o700,exist_ok=False)
    print(json.dumps(prior.submit_once(h,root,command,intent),sort_keys=True))

if __name__=='__main__':main()
