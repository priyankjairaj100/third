"""Retain software checks and bounded child stdout/stderr through exit."""
import contextlib,importlib.util,os,resource,sys,time,unittest
from pathlib import Path
import control as c

def main():
    args=c.parser().parse_args();c.validate_args(args);h=c.helpers()
    h.verify_package(Path(__file__).resolve().parent,args.control_package_sha256)
    h.verify_package(c.SAMPLER,args.sampler_package_sha256)
    h.verify_package(c.LOCK_HELPER,args.lock_package_sha256)
    c.require(os.environ.get('SLURM_JOB_ID'),'Slurm_required')
    root=c.root_for(args.mode)/'run';root.mkdir(mode=0o700,exist_ok=False)
    # These helpers are covered by the verified prior package; importing its
    # module does not run its old frame CLI.
    sys.path.insert(0,str(c.PRIOR))
    spec=importlib.util.spec_from_file_location('sampler_bounded_capture',c.PRIOR/'runner.py')
    capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    sys.path.insert(0,str(c.SAMPLER))
    os.environ['THIRD_SAMPLER_REPO']=str(c.REPO)
    os.environ['THIRD_SAMPLER_PROTOCOL']=str(c.PROTOCOL)
    os.environ['THIRD_PROTOCOL_EVIDENCE']=str(c.EVIDENCE)
    started=time.perf_counter()
    with (root/'tests.txt').open('x') as rawlog:
        log=capture.CappedText(rawlog,1024**2)
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            suite=unittest.defaultTestLoader.discover(str(c.SAMPLER),pattern='test_*.py')
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    validation={'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
      'skipped':len(result.skipped),'elapsed_seconds':time.perf_counter()-started,
      'passed':result.wasSuccessful() and result.testsRun==args.expected_tests and not result.skipped}
    h.write_exclusive(root/'software_validation.json',validation)
    c.require(validation['passed'],'cluster_software_validation_failed')
    command=c.child_command(args)
    with (root/'stdout.txt').open('xb') as out,(root/'stderr.txt').open('xb') as err:
        started=time.perf_counter()
        rc,size,exceeded=capture.child_capture(command,out,err,14*1024**2)
        elapsed=time.perf_counter()-started
        for stream in (out,err):stream.flush();os.fsync(stream.fileno())
    h.write_exclusive(root/'process_completion.json',{'command':command,'returncode':rc,
      'child_wall_seconds_through_exit':elapsed,'console_bytes':size,'console_cap_exceeded':exceeded,
      'runner_children_maxrss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
      'sampler_package_sha256':args.sampler_package_sha256,'slurm_job_id':os.environ['SLURM_JOB_ID']})
    c.require(not exceeded,'child_console_cap_exceeded')
    raise SystemExit(rc)

if __name__=='__main__':main()
