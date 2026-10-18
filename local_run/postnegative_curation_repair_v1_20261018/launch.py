"""Single launch; inputs stay on cluster. Never retry unknown submissions."""
import datetime,hashlib,importlib.util,json,os,pathlib,subprocess,sys
HERE=pathlib.Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace/tools/postnegative-curation-repair-v1-20261018')
W=pathlib.Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace');REPO=W.parent/'repo'
ROOT=W/'qualification/postnegative-curation-repair-v1-20261018-a01'
def check(x,m):
 if not x:raise RuntimeError(m)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(x):return subprocess.check_output(x,text=True,timeout=15)
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 pin=sys.argv[1];check(h(HERE/'package_manifest.json')==pin,'package_manifest_pin')
 manifest=json.loads((HERE/'package_manifest.json').read_bytes())
 for name,b in manifest['files'].items():check(h(HERE/name)==b['sha256'] and (HERE/name).stat().st_size==b['bytes'],'package_member:'+name)
 check(sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())<128*1024,'small_package_cap')
 check(not os.path.lexists(ROOT),'occupied_run_slot')
 check(command(['git','-C',str(REPO),'rev-parse','HEAD']).strip()=='de9de39a1a72db40485878452561affe2fbd24aa','frozen_head')
 check(not command(['git','-C',str(REPO),'status','--porcelain']).strip(),'frozen_checkout_clean')
 queue=command(['squeue','-h','-u','poonam','-o','%i|%j|%T']);check(not any(line.split('|')[1].startswith('third-') for line in queue.splitlines()),'third_job_active_reconcile_first')
 qp=REPO/'local_run/mistral_asset_acquisition_v2_20261011/quota_grants.py'
 check(h(qp)=='dc7b31d9ed6a4b5dc7a8a5a8e66ffeaad31e057747832641694234c0ed4093bc','quota_parser_pin')
 spec=importlib.util.spec_from_file_location('quota',qp);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 raw=command(['quota','-v','-w']);quota=q.parse_login_quota(raw)
 check(q.mount_identity(W)=={'source':'/dev/sdb1','target':'/nfs_home','fstype':'ext4'},'nfs_mount_pin')
 check(quota['available_soft_bytes']>=1024**3+2*1024**2,'small_aggregate_stage_margin')
 os.umask(0o077);ROOT.mkdir(mode=0o700,exist_ok=False)
 cmd=['sbatch','--parsable','--job-name=third-postnegative-curation-repair-v1','--output=/dev/null','--error=/dev/null',str(HERE/'curation_repair.sbatch'),str(HERE),pin,str(ROOT)]
 intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_manifest_sha256':pin,'command':cmd,'quota':quota,'fresh_queue':queue,'private_inputs_remain_on_cluster':True,'new_study':True,'original_v4_gate_changed':False,'automatic_retry':False}
 save(ROOT/'launch_intent.json',intent)
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
  receipt={**intent,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
  if r.returncode==0:
   job=r.stdout.strip().split(';')[0];check(job.isdigit(),'ambiguous_submission');receipt['job_id']=job
 except BaseException as e:
  save(ROOT/'launch_receipt.json',{**intent,'submission_outcome':'unknown_reconcile_without_retry','error_type':type(e).__name__});raise
 save(ROOT/'launch_receipt.json',receipt);print(json.dumps(receipt));sys.exit(r.returncode)
if __name__=='__main__':main()
