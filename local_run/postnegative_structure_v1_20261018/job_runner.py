"""Run one new protocol in node scratch; persist aggregate-only bounded evidence."""
import datetime,hashlib,json,os,pathlib,resource,shutil,socket,subprocess,sys,tempfile,time,zipfile
HERE=pathlib.Path(__file__).resolve().parent
W=pathlib.Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace'); REPO=W.parent/'repo'
NFS_BUDGET=2*1024**2
LOG_CAP=65536
RECEIPT_RESERVE=32768
RUN_NAME='postnegative-structure-v1-20261018-a01'
class SlotOccupiedError(RuntimeError):pass
def check(x,m):
 if not x:raise RuntimeError(m)
def h(p):
 digest=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
 return digest.hexdigest()
def save(p,x):
 raw=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
 check(len(raw)<=LOG_CAP,'receipt_cap')
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def tree_bytes(folder):
 total=0
 for p in folder.rglob('*'):
  check(not p.is_symlink(),'artifact_symlink')
  if p.is_file():total+=p.stat().st_size
 return total
def run_bounded(command,scratch,name,env,timeout):
 # Child output stays in scratch. The inherited file-size rlimit bounds each
 # log even if a bug produces repeated messages; no capture_output buffer.
 stdout=scratch/(name+'.stdout.private.txt');stderr=scratch/(name+'.stderr.private.txt')
 timed_out=False
 with stdout.open('xb') as out,stderr.open('xb') as err:
  try:result=subprocess.run(command,stdout=out,stderr=err,env=env,timeout=timeout)
  except subprocess.TimeoutExpired:result=None;timed_out=True
 logs={p.name:{'bytes':p.stat().st_size,'sha256':h(p)} for p in (stdout,stderr)}
 over_cap=any(v['bytes']>LOG_CAP for v in logs.values())
 receipt={'returncode':124 if timed_out else result.returncode,'timed_out':timed_out,
          'private_log_bindings':logs,'log_cap_exceeded':over_cap,'raw_logs_exported':False}
 save(scratch/(name+'.json'),receipt)
 return 125 if over_cap else receipt['returncode']
def main():
 pin,runroot=sys.argv[1:];runroot=pathlib.Path(runroot)
 check(runroot==W/'qualification'/RUN_NAME and runroot.is_dir() and not runroot.is_symlink(),'fixed_run_root')
 check(HERE==W/'tools/postnegative-structure-v1-20261018','fixed_package_root')
 check(h(HERE/'package_manifest.json')==pin,'package_manifest_hash')
 manifest=json.loads((HERE/'package_manifest.json').read_bytes())
 for name,b in manifest['files'].items():
  check(pathlib.Path(name).name==name and not (HERE/name).is_symlink(),'safe_package_member')
  check(h(HERE/name)==b['sha256'] and (HERE/name).stat().st_size==b['bytes'],'package_member:'+name)
 protocol=json.loads((HERE/'PLAN.json').read_bytes())
 check(protocol['status']=='frozen_for_stage1' and 'scipy' in protocol['runtime'],'frozen_runtime_plan')
 check(protocol['execution_policy']['nfs_small_artifact_budget_bytes']==NFS_BUDGET,'nfs_budget_pin')
 check(socket.gethostname() in ('cn1','cn3','cn4','csis.cn1','csis.cn3','csis.cn4'),'approved_compute_node')
 job=os.environ.get('SLURM_JOB_ID','');check(job.isascii() and job.isdigit() and int(job)>0,'slurm_only')
 check(os.environ.get('SLURM_CPUS_PER_TASK')=='2' and os.environ.get('SLURM_MEM_PER_NODE')=='8192','slurm_allocation')
 check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','cpu_only')
 check(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()=='de9de39a1a72db40485878452561affe2fbd24aa','frozen_remote_head')
 # This exclusive worker intent precedes all software tests and natural scoring.
 # A second scheduler submission cannot run the same study slot.
 if any((runroot/n).exists() for n in ('execution_intent.json','terminal.json','aggregate_results.zip','wrapper_failure.json')):raise SlotOccupiedError('run_slot_occupied')
 try:
  save(runroot/'execution_intent.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'job_id':job,'package_manifest_sha256':pin,'retry':False})
 except FileExistsError:raise SlotOccupiedError('run_slot_occupied')
 scratch=pathlib.Path(tempfile.mkdtemp(prefix='third-structure-'+job+'-',dir='/tmp'))
 mount=subprocess.check_output(['findmnt','-n','-o','SOURCE,FSTYPE,TARGET','-T',str(scratch)],text=True).strip()
 check('nfs' not in mount.lower(),'scratch_is_not_NFS')
 check(shutil.disk_usage(scratch).free>=8*1024**3+256*1024**2,'scratch_reserve_and_allowance')
 resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
 env=dict(os.environ,THIRD_LOCAL_REPO=str(REPO)); started=time.monotonic()
 code=run_bounded([sys.executable,'-B',str(HERE/'test_structure_stage1.py')],scratch,'software_tests',env,60)
 if code==0:
  intake=W/'qualification/llm-development-intake-v1-20261016-a01/run'
  args=[sys.executable,'-B',str(HERE/'structure_stage1.py'),'--repo',str(REPO),'--protocol',str(HERE/'PLAN.json'),'--protocol-sha256',manifest['files']['PLAN.json']['sha256'],'--allocation',str(intake/'allocation.private.json'),'--prefixes',str(intake/'development_prefixes.private.json'),'--vectors',str(W/'qualification/e5-full-sharded-v2-20261013-a01/assembly-run/cache/vectors.npy'),'--out',str(scratch/'run')]
  code=run_bounded(args,scratch,'child_process',env,1260)
 save(scratch/'process.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':job,'node':socket.gethostname(),'returncode':code,'elapsed_seconds':time.monotonic()-started,'scratch_mount':mount,'scratch_free_bytes':shutil.disk_usage(scratch).free,'scratch_path':str(scratch),'package_manifest_sha256':pin,'child_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'private_feature_or_label_output':False})
 allowed={'software_tests.json','child_process.json','process.json','run/intake.json','run/subset_oracle.json','run/completion.json','run/failure.json'}|{'run/threshold_%02d.json'%i for i in range(51)}
 files=sorted(scratch.rglob('*.json'))
 check(all(str(p.relative_to(scratch)) in allowed and p.is_file() and not p.is_symlink() for p in files),'aggregate_inventory')
 # Failure text is diagnostic-private; do not return arbitrary exception strings.
 failure=scratch/'run/failure.json'
 if failure.exists():
  old=json.loads(failure.read_bytes());sanitized={'status':'failed','exception_type':old.get('exception_type','unknown'),'reason':'stage_reported_failure_see_private_scratch','completed_threshold_indices':old.get('completed_threshold_indices',[]),'partial_results_not_complete':True,'retry_automatic':False,'elapsed_seconds':old.get('elapsed_seconds')}
  failure.rename(scratch/'stage_failure.private.txt');save(failure,sanitized)
 if code==0:
  check({str(p.relative_to(scratch)) for p in files}==allowed-{'run/failure.json'},'complete_success_inventory')
  done=json.loads((scratch/'run/completion.json').read_bytes())
  check(done['status']=='completed' and done['all_thresholds_completed'] is True,'complete_child_receipt')
 check(sum(p.stat().st_size for p in files)<2*1024**2,'aggregate_cap')
 check(tree_bytes(scratch)<256*1024**2 and shutil.disk_usage(scratch).free>=8*1024**3,'scratch_final_budget')
 archive=scratch/'aggregate_results.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(scratch)))
 check(archive.stat().st_size<1536*1024,'archive_cap')
 check(tree_bytes(HERE)+tree_bytes(runroot)+archive.stat().st_size+RECEIPT_RESERVE<=NFS_BUDGET,'combined_NFS_budget_before_copy')
 with (runroot/'aggregate_results.zip').open('xb') as target,archive.open('rb') as source:shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
 save(runroot/'terminal.json',{'status':'completed' if code==0 else 'failed','job_id':job,'returncode':code,'archive_bytes':archive.stat().st_size,'archive_sha256':h(archive),'package_manifest_sha256':pin,'automatic_retry':False,'scratch_path':str(scratch)})
 check(tree_bytes(HERE)+tree_bytes(runroot)<=NFS_BUDGET,'combined_NFS_budget_after_copy')
 print(json.dumps({'status':'completed' if code==0 else 'failed','job_id':job}));return code
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException as e:
  if isinstance(e,SystemExit):raise
  if isinstance(e,SlotOccupiedError):
   print(json.dumps({'status':'duplicate_attempt_refused'}),flush=True);sys.exit(1)
  # Keep unexpected tracebacks/raw errors in the node process, not NFS logs.
  try:
   p=pathlib.Path(sys.argv[2])/'wrapper_failure.json'
   if not p.exists():save(p,{'error_type':type(e).__name__,'reason':str(e)[:256] if type(e) is RuntimeError else 'unexpected_wrapper_failure','automatic_retry':False})
  except BaseException:pass
  print(json.dumps({'status':'wrapper_failed','error_type':type(e).__name__}),flush=True)
  sys.exit(1)
