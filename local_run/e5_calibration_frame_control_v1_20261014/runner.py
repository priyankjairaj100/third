"""Run frame software checks then one child, retaining terminal receipts."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import selectors
import subprocess
import sys
import time
import unittest
import launch as c


class CappedText:
    def __init__(self, stream, limit): self.stream,self.limit,self.used=stream,limit,0
    def write(self, value):
        size=len(value.encode('utf-8'))
        c.require(self.used+size<=self.limit,'test_console_limit_exceeded')
        self.used+=size
        return self.stream.write(value)
    def flush(self): self.stream.flush()


def child_capture(command, out, err, limit):
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    selector=selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ,out)
    selector.register(process.stderr,selectors.EVENT_READ,err)
    size=0; exceeded=False
    try:
        while selector.get_map():
            for key,_ in selector.select(timeout=1):
                block=os.read(key.fileobj.fileno(),65536)
                if not block:
                    selector.unregister(key.fileobj);key.fileobj.close();continue
                if size+len(block)>limit:
                    exceeded=True;process.kill()
                    block=block[:max(0,limit-size)]
                key.data.write(block);size+=len(block)
        rc=process.wait()
    finally:
        selector.close()
        if process.poll() is None:process.kill();process.wait()
        for stream in (process.stdout,process.stderr):
            if not stream.closed:stream.close()
    return rc,size,exceeded


def main():
    p=argparse.ArgumentParser()
    for key in ('root','control-package-sha256','frame-package-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--expected-tests',required=True,type=int);a=p.parse_args()
    h=c.verified_prior()
    h.verify_package(Path(__file__).resolve().parent,a.control_package_sha256)
    h.verify_package(c.PACKAGE,a.frame_package_sha256)
    c.require(Path(a.root)==c.ROOT and os.environ.get('SLURM_JOB_ID'),'fixed_slurm_root_required')
    root=c.ROOT/'run';root.mkdir(mode=0o700,exist_ok=False)
    os.environ['THIRD_FRAME_REPO']=str(c.REPO)
    sys.path.insert(0,str(c.PACKAGE))
    started=time.perf_counter()
    with (root/'tests.txt').open('x') as rawlog:
        log=CappedText(rawlog,1024**2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite=unittest.defaultTestLoader.discover(str(c.PACKAGE),pattern='test_*.py')
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    validation={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
       'skipped':len(result.skipped),'elapsed_seconds':time.perf_counter()-started,
       'passed':result.wasSuccessful() and result.testsRun==a.expected_tests and not result.skipped}
    h.write_exclusive(root/'software_validation.json',validation)
    c.require(validation['passed'],'frame_software_validation_failed')
    command=c.frame_command(a.frame_package_sha256,root/'frame')
    with (root/'stdout.txt').open('xb') as out,(root/'stderr.txt').open('xb') as err:
        started=time.perf_counter()
        rc,size,exceeded=child_capture(command,out,err,14*1024**2)
        elapsed=time.perf_counter()-started
        for stream in (out,err):stream.flush();os.fsync(stream.fileno())
    h.write_exclusive(root/'process_completion.json',{'command':command,'returncode':rc,
        'console_bytes':size,'console_cap_exceeded':exceeded,
        'child_wall_seconds_through_exit':elapsed,'runner_children_maxrss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'frame_package_sha256':a.frame_package_sha256,'slurm_job_id':os.environ['SLURM_JOB_ID']})
    c.require(not exceeded,'frame_console_limit_exceeded')
    raise SystemExit(rc)


if __name__=='__main__':main()
