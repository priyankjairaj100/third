"""Capture one corrected CPU audit in a fresh repair root."""
import repair_common as R
R.configure()
import argparse,contextlib,json,os,resource,sys,time,unittest
import common as x
import contract as C
import worker as parent
p=argparse.ArgumentParser()
for name in ('package-sha256','request-result-sha256','activation-sha256','resource-plan-sha256'):p.add_argument('--'+name,required=True)
p.add_argument('--model-alias',choices=('qwen','mistral'),required=True);a=p.parse_args()
C.require(a.package_sha256==R.EXECUTIONS[R.SCOPE]['pin'],'execution_pin');os.umask(0o077)
root=R.audit_root(R.SCOPE,a.model_alias);C.require(root.is_dir() and os.environ.get('SLURM_JOB_ID','').isdigit(),'launched_slurm_audit')
with (root/'software_tests.txt').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
 suite=unittest.defaultTestLoader.discover(str(R.HERE),pattern='test_*.py');result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
C.require(result.wasSuccessful() and result.testsRun==8 and not result.skipped,'repair_software_tests')
cmd=[sys.executable,'-B',str(R.HERE/'audit.py'),'--scope',R.SCOPE,'--repair-package-sha256',R.PACKAGE_SHA,*sys.argv[1:]];start=time.perf_counter()
with (root/'child.stdout.txt').open('xb') as out,(root/'child.stderr.txt').open('xb') as err:
 rc,size,exceeded=parent.capture(cmd,out,err,16*1024**2)
 for f in (out,err):f.flush();os.fsync(f.fileno())
x.publish(root/'process_completion.json',{'command':cmd,'returncode':rc,'console_bytes':size,'console_cap_exceeded':exceeded,
 'child_wall_seconds_through_exit':time.perf_counter()-start,'children_maxrss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
 'slurm_job_id':os.environ['SLURM_JOB_ID'],'package_sha256':a.package_sha256,'repair_package_sha256':R.PACKAGE_SHA,'model_inference_calls':0})
C.require(not exceeded,'console_cap_exceeded');raise SystemExit(rc)
