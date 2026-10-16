"""Submit a fresh read-only audit after verifying the original model completion."""
import repair_common as R
R.configure()
import argparse,datetime,json,os,time
import common as x
import contract as C
import launch as original
p=argparse.ArgumentParser()
for name in ('package-sha256','request-result-sha256','activation-sha256','resource-plan-sha256'):p.add_argument('--'+name,required=True)
p.add_argument('--model-alias',choices=('qwen','mistral'),required=True);a=p.parse_args()
C.require(a.package_sha256==R.EXECUTIONS[R.SCOPE]['pin'],'execution_pin')
pins,controls,prior,h,protocol=x.setup(a.package_sha256);root=R.audit_root(R.SCOPE,a.model_alias)
C.require(not os.path.lexists(root),'occupied_repaired_audit_no_retry')
C.require(not h.command(['git','status','--porcelain'],cwd=x.REPO).strip() and h.command(['git','rev-parse','HEAD'],cwd=x.REPO).strip()=='de9de39a1a72db40485878452561affe2fbd24aa','clean_frozen_execution_checkout')
_,request=x.request_contract(a.request_result_sha256,pins)
for path,pin in [(x.ROOT/'activation/activation.json',a.activation_sha256),(x.ROOT/'resource_plan.json',a.resource_plan_sha256)]:C.require(h.binding(path)['sha256']==pin,'frozen_activation_resources')
parent=original.terminal(h,x.ROOT/a.model_alias,'output/result.json','gpu')
if R.SCOPE=='selection' and a.model_alias=='qwen':
 C.require(parent['job_id']=='13663' and parent['result']['sha256']=='ba716a2588a174cf0968c60e3a8aa659b47b9e806b1d94534988539fb7b25011','completed_original_Qwen')
 C.require(h.binding(x.ROOT/'audit_qwen/child.stderr.txt')['sha256']=='f23819738ab590c58396c8a27a1d9a1d4d330e3419d234ca889a8114af17a9b3','preserved_failed_audit')
qpath=x.REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py';h.verify_file(qpath,h.QUOTA_HELPER_SHA);q=x.module('audit_repair_quota',qpath)
started=time.monotonic();mount=q.mount_identity(x.W);raw=h.command(['quota','-v','-w'],timeout=30);quota=q.parse_login_quota(raw)
C.require(mount=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'} and quota['available_soft_bytes']>=8*1024**3+64*1024**2,'fresh_storage_gate')
queue=h.command(['squeue','-h','-u','poonam','-o','%i|%j']);C.require(all(not row.split('|')[1].startswith('third-') for row in queue.splitlines()),'active_project_job')
R.verify_package(R.HERE,R.PACKAGE_SHA);C.require(time.monotonic()-started<=120,'fresh_quota')
cmd=['sbatch','--parsable','--partition=cpu-short','--nodelist=csis.cn1','--cpus-per-task=2','--mem=8G','--time=60',
 '--job-name=third-v4-audit-repair-'+a.model_alias,'--output',str(root/'slurm-%j.out'),'--error',str(root/'slurm-%j.err'),
 str(R.HERE/'audit.sbatch'),str(R.HERE),'--scope',R.SCOPE,'--repair-package-sha256',R.PACKAGE_SHA,
 '--package-sha256',a.package_sha256,'--request-result-sha256',a.request_result_sha256,'--activation-sha256',a.activation_sha256,
 '--resource-plan-sha256',a.resource_plan_sha256,'--model-alias',a.model_alias]
intent={'schema':'third-byte-preserving-audit-launch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,
 'root':str(root),'original_model':parent,'repair_package_sha256':R.PACKAGE_SHA,'original_execution_package_sha256':a.package_sha256,
 'quota':quota,'quota_raw':raw,'mount':mount,'fresh_queue':queue,'reserve_bytes':8*1024**3,'stage_allowance_bytes':64*1024**2,
 'automatic_retry':False,'model_inference_calls':0,'human_annotation_calls':0}
os.umask(0o077);root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(mode=0o700,exist_ok=False)
print(json.dumps(prior.submit_once(h,root,cmd,intent),sort_keys=True))
