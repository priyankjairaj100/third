"""Compute-side tests, capped read-only audit child, and durable terminal receipt."""
import contextlib, json, os, platform, resource, socket, sys, time, unittest
from pathlib import Path
import control as c

EXPECTED_TESTS=12
def accepted_result(returncode,exceeded,eligible,report):
    return returncode==0 and not exceeded and eligible and report is not None and report.get('passed') is True
def main():
    a=c.parser().parse_args();c.check_args(a);ec,prior,h=c.helpers()
    c.require(platform.system()=='Linux' and socket.gethostname() in ('csis.cn1','cn1') and os.environ.get('SLURM_JOB_ID','').isdigit(),'CPU_Slurm_cn1_required')
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and Path(sys.executable).absolute()==ec.PYTHON,'CPU_only_qualified_interpreter')
    launchpath=c.root_for(a.mode)/'launch_receipt.json';deadline=time.monotonic()+10
    while not launchpath.exists() and time.monotonic()<deadline:time.sleep(.1)
    launch=h.load(launchpath);intent=h.load(c.root_for(a.mode)/'launch_intent.json')
    c.require(all(launch.get(k)==v for k,v in intent.items()),'launch_intent_changed')
    c.require(launch['job_id']==os.environ['SLURM_JOB_ID'] and launch['mode']==a.mode and launch['root']==str(c.root_for(a.mode)),'postprocess_job_identity')
    r,bindings=c.source_check(h,ec,a,launch['bindings']);c.require(bindings==launch['bindings'],'launch_inputs_changed')
    command=c.child_command(ec,a);c.require(command==launch['child_command'],'launched_command_changed')
    root=c.root_for(a.mode)/'run';root.mkdir(mode=0o700,exist_ok=False)
    # Authenticated prior capture implementation; no subprocess submission/retry here.
    sys.path.insert(0,str(prior.PRIOR));capture=c.module('postprocess_capped_capture',prior.PRIOR/'runner.py')
    started=time.perf_counter()
    with (root/'tests.txt').open('x') as raw:
        log=capture.CappedText(raw,1024**2)
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            suite=unittest.defaultTestLoader.discover(str(c.HERE),pattern='test_postprocess.py');tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        raw.flush();os.fsync(raw.fileno())
    software={'tests':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'skipped':len(tests.skipped),'passed':tests.wasSuccessful() and tests.testsRun==EXPECTED_TESTS and not tests.skipped,'seconds':time.perf_counter()-started}
    h.write_exclusive(root/'software_validation.json',software);c.require(software['passed'],'postprocess_tests_failed')
    # The unchanged child writes one report only. A partial report is limited to
    # 16 MiB; accepted reports to 8 MiB. Logs14 + tests1 + metadata leave32 MiB.
    old=resource.getrlimit(resource.RLIMIT_FSIZE);hard=old[1]
    resource.setrlimit(resource.RLIMIT_FSIZE,(min(16*1024**2,hard) if hard!=resource.RLIM_INFINITY else 16*1024**2,hard))
    with (root/'stdout.txt').open('xb') as out,(root/'stderr.txt').open('xb') as err:
        start=time.perf_counter();rc,size,exceeded=capture.child_capture(command,out,err,14*1024**2);elapsed=time.perf_counter()-start
        for stream in (out,err):stream.flush();os.fsync(stream.fileno())
    h.write_exclusive(root/'process_completion.json',{'schema':'third-llm-postprocess-process-v1','mode':a.mode,'command':command,'returncode':rc,'console_bytes':size,'console_cap_exceeded':exceeded,'child_wall_seconds_through_exit':elapsed,'children_maxrss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'slurm_job_id':os.environ['SLURM_JOB_ID'],'package':bindings['package'],'model_inference_calls':0,'retry_attempts':0})
    error=None;report=None;rb=None
    try:
        c.require(c.source_check(h,ec,a,bindings)==(r,bindings),'final_inputs_changed')
        path=root/'report.json'
        if path.exists():
            c.require(not path.is_symlink() and path.stat().st_size<=8*1024**2,'accepted_report_limit');rb=h.binding(path);report=h.load(path);c.check_seal(report)
        c.require({x.name for x in root.iterdir()}=={'tests.txt','software_validation.json','stdout.txt','stderr.txt','process_completion.json'}|({'report.json'} if path.exists() else set()),'postprocess_output_inventory')
        c.require(sum(x.stat().st_size for x in root.iterdir())<=31*1024**2,'postprocess_total_limit')
    except Exception as e:error={'type':type(e).__name__,'message':str(e)}
    eligible=all(x['eligible_terminal_process'] for x in bindings['families'].values())
    passed=error is None and accepted_result(rc,exceeded,eligible,report)
    result={'schema':'third-llm-postprocess-result-v1','passed':passed,'mode':a.mode,'package':bindings['package'],'bindings':bindings,'report':rb,'execution_eligible':eligible,'child_returncode':rc,'console_cap_exceeded':exceeded,'integrity_error':error,'slurm_job_id':os.environ['SLURM_JOB_ID'],'source_and_result_bytes_initial_final_equal':error is None,'report_is_accepted_only_when_passed':True,'model_inference_calls':0,'retry_attempts':0,'human_annotation_calls':0,'validation_performed':False,'semantic_validity_established':False}
    h.write_exclusive(root/'postprocess_result.json',result);print(json.dumps({'passed':passed,'mode':a.mode,'result':h.binding(root/'postprocess_result.json')}));raise SystemExit(0 if passed else 2)
if __name__=='__main__':main()
