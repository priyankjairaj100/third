"""Run software preflight and one native-task calibration child, bounded logs."""
import contextlib
import json
import os
import resource
import selectors
import subprocess
import sys
import time
import unittest
import task_lock as t


def main():
    pin=sys.argv[1];t.verify_package(pin)
    root=t.ROOT;t.require(root.is_dir() and os.environ.get('SLURM_JOB_ID'),'launched_root')
    os.environ['THIRD_LOCAL_REPO']=str(t.REPO)
    with (root/'software_tests.txt').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        suite=unittest.defaultTestLoader.discover(str(t.HERE),pattern='test_task_lock.py')
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    t.publish(root/'software_validation.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped)})
    t.require(result.wasSuccessful() and result.testsRun==8 and not result.skipped,'software_preflight')
    command=[sys.executable,'-B',str(t.HERE/'task_lock.py'),'--package-sha256',pin]
    started=time.perf_counter();size=0;exceeded=False
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with (root/'child.stdout.txt').open('xb') as out,(root/'child.stderr.txt').open('xb') as err:
        selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ,out);selector.register(child.stderr,selectors.EVENT_READ,err)
        try:
            while selector.get_map():
                for key,_ in selector.select(timeout=1):
                    data=os.read(key.fileobj.fileno(),65536)
                    if not data:selector.unregister(key.fileobj);key.fileobj.close();continue
                    if size+len(data)>16*1024**2:
                        child.kill();exceeded=True;data=data[:max(0,16*1024**2-size)]
                    key.data.write(data);size+=len(data)
            code=child.wait()
        finally:
            selector.close()
            if child.poll() is None:child.kill();child.wait()
        for stream in (out,err):stream.flush();os.fsync(stream.fileno())
    t.publish(root/'process_completion.json',{'job_id':os.environ['SLURM_JOB_ID'],'command':command,'returncode':code,'console_bytes':size,'console_cap_exceeded':exceeded,'wall_seconds_through_child_exit':time.perf_counter()-started,'children_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'package_sha256':pin})
    t.require(not exceeded,'console_budget');raise SystemExit(code)


if __name__=='__main__':main()
