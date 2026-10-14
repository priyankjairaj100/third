"""Test, execute once, cap console output and record child lifecycle through exit."""
import argparse
import contextlib
import json
import os
import resource
import selectors
import signal
import subprocess
import sys
import time
import unittest
import common as x
import contract as C
from launch import MODES,root_for

def command(args):
    mode=args.mode
    script='qualify.py' if mode=='qualification' else ('run_model.py' if mode in ('qwen','mistral') else
        'compute_statistics.py' if mode=='statistics' else 'audit.py')
    result=[sys.executable,'-B',str(x.HERE/script),'--package-sha256',args.package_sha256,
            '--request-result-sha256',args.request_result_sha256]
    if mode!='qualification':result+=['--activation-sha256',args.activation_sha256,
                                      '--resource-plan-sha256',args.resource_plan_sha256]
    if mode in ('qwen','mistral','audit_qwen','audit_mistral'):
        result+=['--model-alias',mode.removeprefix('audit_')]
    return result

def capture(command,out,err,cap):
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                           start_new_session=True)
    selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ,out)
    selector.register(child.stderr,selectors.EVENT_READ,err);total=0;exceeded=False
    try:
        while selector.get_map():
            for key,_ in selector.select(timeout=1):
                chunk=os.read(key.fileobj.fileno(),65536)
                if not chunk:selector.unregister(key.fileobj);key.fileobj.close();continue
                if total+len(chunk)>cap and not exceeded:
                    exceeded=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                remaining=max(0,cap-total)
                key.data.write(chunk[:remaining]);total+=len(chunk[:remaining])
        return child.wait(),total,exceeded
    finally:
        selector.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=MODES,required=True)
    for name in ('package-sha256','request-result-sha256'):parser.add_argument('--'+name,required=True)
    for name in ('activation-sha256','resource-plan-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args();x.bootstrap(x.HERE,args.package_sha256)
    C.require(os.environ.get('SLURM_JOB_ID','').isdigit(),'Slurm_required')
    os.umask(0o077);root=root_for(args.mode);C.require(root.is_dir(),'launcher_stage_required')
    old=x.W/'tools/llm-annotation-inference-v1-20261014/kernel.py'
    derivation=C.strict_json((x.HERE/'DERIVATION.json').read_bytes());C.checked_file(old,derivation['source_bindings']['kernel.py'])
    os.environ['THIRD_V4_PROTOCOL']=str(x.PROTOCOL);os.environ['THIRD_V3_KERNEL']=str(old)
    with (root/'software_tests.txt').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        suite=unittest.defaultTestLoader.discover(str(x.HERE),pattern='test_*.py')
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    tested={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'passed':result.wasSuccessful() and result.testsRun==11 and not result.skipped}
    x.publish(root/'software_validation.json',tested);C.require(tested['passed'],'software_tests_failed')
    cmd=command(args);started=time.perf_counter()
    with (root/'child.stdout.txt').open('xb') as out,(root/'child.stderr.txt').open('xb') as err:
        rc,size,exceeded=capture(cmd,out,err,16*1024**2)
        for f in (out,err):f.flush();os.fsync(f.fileno())
    x.publish(root/'process_completion.json',{'command':cmd,'returncode':rc,'console_bytes':size,
        'console_cap_exceeded':exceeded,'child_wall_seconds_through_exit':time.perf_counter()-started,
        'children_maxrss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'package_sha256':args.package_sha256})
    C.require(not exceeded,'console_cap_exceeded');raise SystemExit(rc)

if __name__=='__main__':main()
