"""Explicit scheduler-role adapter for the first Mistral launch; v1 remains sealed."""
import argparse, datetime, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
BASE=W/'tools/llm-annotation-control-v1-20261014'
BASE_SHA='8999a6328c13dff99cea662944b31a10f5a6f9d50c37cfd0d91e5cfb4c715a6c'
HERE=Path(__file__).resolve().parent
RESOURCE_SHA='5d5337cbf90119b87760c5f2feca1a8ef2ed5536a88c1953bacabcf68af54b9e'
ADMISSION_SHA='2a39e6f877cca49f823265a4c6589b013aecde342df776fdff14c6d9404e5df7'
ACTIVATION_SHA='2de33cf3a91c51d8ce219df92c0e7b77678d5002f3aaa621850ccb53345620bf'
AUDIT=W/'qualification/llm-only-v3-output-audit-qwen-a01'
AUDIT_PINS={'run/report.json':'c58c3279179f27190f309e2f5c19ebce4cb9ead991e4a6afa34c8781b3e6f1d4',
            'run/postprocess_result.json':'2bc82df0044471dfffc0a2e17dadfc85ee0c6eb46ef2c5f427d6392798de1bd9',
            'run/process_completion.json':'e07a60305779f867cc53f04179069ff5a72697d12d244e4bf4c4fc7c23ee55d8'}

def require(ok, reason):
    if not ok:raise ValueError(reason)

def gpu_qwen_terminal(raw, job):
    require(job=='13574','exact_completed_qwen_job')
    rows=[s.split('|') for s in raw.splitlines() if s.strip()]
    require(rows and all(len(r)==6 for r in rows),'accounting_shape')
    require(all(r[0]==job or r[0].startswith(job+'.') for r in rows),'unrelated_job')
    parent=[r for r in rows if r[0]==job]
    require(len(parent)==1,'unique_parent')
    require(all(r[1]=='COMPLETED' and r[2]=='0:0' and r[5]=='csis.cn3' for r in rows),'successful_GPU_job_and_steps')
    require(parent[0][3].isdigit() and int(parent[0][3])==2634,'completed_qwen_runtime_identity')
    return dict(zip(('JobIDRaw','State','ExitCode','ElapsedRaw','MaxRSS','NodeList'),parent[0]))

class RoleAwareHelpers:
    def __init__(self, original):self.original=original;self.qwen_checks=0
    def __getattr__(self, name):return getattr(self.original,name)
    def terminal_parent(self, raw, job, success=True):
        if job!='13574':return self.original.terminal_parent(raw,job,success=success)
        require(success is True,'no_failure_override');self.qwen_checks+=1
        return gpu_qwen_terminal(raw,job)

def verify_package(path, pin):
    raw=(path/'package_manifest.json').read_bytes();require(hashlib.sha256(raw).hexdigest()==pin,'manifest_pin')
    for name,want in json.loads(raw)['files'].items():
        p=path/name;require(Path(name).name==name and name!='package_manifest.json' and p.is_file() and not p.is_symlink(),'member_name_type')
        x=p.read_bytes();require({'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}==want,'member_pin')
    return {'bytes':len(raw),'sha256':pin}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--adapter-sha256',required=True);args=parser.parse_args()
    adapter=verify_package(HERE,args.adapter_sha256);verify_package(BASE,BASE_SHA)
    spec=importlib.util.spec_from_file_location('original_annotation_control',BASE/'control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    prior,h=c.helpers();proxy=RoleAwareHelpers(h)
    a=argparse.Namespace(mode='mistral',control_package_sha256=BASE_SHA,resource_plan_sha256=RESOURCE_SHA,admission_sha256=ADMISSION_SHA,activation_sha256=ACTIVATION_SHA)
    c.check_args(a);root=c.root_for('mistral');require(not os.path.lexists(root),'existing_mistral_slot_reconcile_no_retry')
    require(h.command(['git','rev-parse','HEAD'],cwd=c.REPO).strip()=='de9de39a1a72db40485878452561affe2fbd24aa' and not h.command(['git','status','--porcelain'],cwd=c.REPO).strip(),'frozen_executing_repo')
    # Require the exact completed Qwen CPU audit in addition to every original gate.
    audit_files={name:h.verify_file(AUDIT/name,pin) for name,pin in AUDIT_PINS.items()}
    audit=h.load(AUDIT/'run/report.json');wrapped=h.load(AUDIT/'run/postprocess_result.json');proc=h.load(AUDIT/'run/process_completion.json')
    require(audit['passed'] is True and audit['alias']=='qwen' and audit['status_counts']=={'valid':99},'audited_qwen')
    require(wrapped['passed'] is True and wrapped['execution_eligible'] is True and proc['returncode']==0 and proc['slurm_job_id']=='13623','audit_success')
    audit_accounting=h.command(['sacct','-j','13623','-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList'])
    h.terminal_parent(audit_accounting,'13623')  # Original strict CPU-worker gate remains.
    r,initial=c.source_check(proxy,a);c.queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']))
    qpath=c.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py';h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=c.module('adapter_quota',qpath)
    mount=q.mount_identity(W);started=time.monotonic();raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(raw)
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,c.FAMILY_ALLOWANCE)
    require(c.source_check(proxy,a)==(r,initial),'inputs_changed_before_submission')
    require(proxy.qwen_checks==2,'exact_two_role_aware_qwen_checks')
    require(verify_package(HERE,args.adapter_sha256)==adapter,'adapter_changed')
    require({name:h.verify_file(AUDIT/name,pin) for name,pin in AUDIT_PINS.items()}==audit_files,'audit_changed')
    c.queue_clear(h.command(['squeue','-h','-u','poonam','-o','%i|%j']))
    c.quota_gate(quota['available_soft_bytes'],time.monotonic()-started,mount,c.FAMILY_ALLOWANCE)
    child_args=['--mode','mistral','--control-package-sha256',BASE_SHA,'--resource-plan-sha256',RESOURCE_SHA,'--admission-sha256',ADMISSION_SHA,'--activation-sha256',ACTIVATION_SHA]
    flags=['--partition=gpu-short','--gres=gpu:a100-80gb:1','--cpus-per-task=4','--mem='+str(r['families']['mistral']['host_memory_bytes']//1024**3)+'G','--time='+str(r['families']['mistral']['slurm_time_seconds']//60)]
    command=['sbatch','--parsable',*flags,'--job-name=third-llm-infer-mistral','--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),str(BASE/'execute.sbatch'),str(BASE),*child_args]
    intent={'schema':'third-llm-execution-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'mistral','root':str(root),'command':command,'child_command':c.child_command(a,r),'bindings':initial,'quota':quota,'quota_raw':raw,'mount':mount,'reserve_bytes':c.RESERVE,'admitted_remaining_output_commitment_bytes':c.FAMILY_ALLOWANCE,'quota_snapshot_not_reservation':True,'human_annotation_calls':0,'automatic_retry':False,'sampling_entropy_draws':0,
            'launch_adapter':{'package':adapter,'role_fix':'Only completed Qwen13574 uses its exact GPU worker csis.cn3; request/qualification/audit CPU checks unchanged.','qwen_checks':proxy.qwen_checks,'qwen_cpu_audit_files':audit_files,'qwen_cpu_audit_accounting':audit_accounting,'unchanged_model_child_and_resource_plan':True}}
    os.umask(0o077);root.mkdir(mode=0o700,exist_ok=False);print(json.dumps(prior.submit_once(h,root,command,intent),sort_keys=True))

if __name__=='__main__':main()
