"""Slurm-side capped logs and through-exit receipt; no retries or fabricated labels."""
import importlib.util,os,platform,resource,socket,sys,time
from pathlib import Path
import control as c

def main():
    a=c.parser().parse_args();c.check_args(a);prior,h=c.helpers()
    c.require(platform.system()=='Linux' and os.environ.get('SLURM_JOB_ID') and socket.gethostname() not in ('csis.mn1','csis.mn1.local'),'compute_Slurm_required')
    c.require(Path(sys.executable).absolute()==c.PYTHON,'qualified_interpreter_path')
    r,b=c.source_check(h,a,terminal=False);root=c.root_for(a.mode)/'run';root.mkdir(mode=0o700,exist_ok=False)
    receipt=c.root_for(a.mode)/'launch_receipt.json'
    deadline=time.monotonic()+10
    while not receipt.exists() and time.monotonic()<deadline:time.sleep(0.1)
    launch=h.load(receipt)
    c.require(str(launch['job_id'])==os.environ['SLURM_JOB_ID'] and launch['mode']==a.mode,'job_intent_identity')
    command=c.child_command(a,r);c.require(command==launch['child_command'],'launched_command_changed')
    limit=resource.getrlimit(resource.RLIMIT_AS)[0]
    minimum=(6 if a.mode=='qualification' else (40 if a.mode=='qwen' else 56))*1024**3
    c.require(limit==resource.RLIM_INFINITY or limit>=minimum,'inherited_address_space_limit')
    if a.mode=='qualification':c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','qualification_GPU_disabled')
    # Exact inherited bounded capture code is authenticated by the prior helper chain.
    sys.path.insert(0,str(prior.PRIOR));spec=importlib.util.spec_from_file_location('llm_control_capture',prior.PRIOR/'runner.py');capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
    cap=14*1024**2 if a.mode=='qualification' else r['families'][a.mode]['console_limit_bytes']
    with (root/'stdout.txt').open('xb') as out,(root/'stderr.txt').open('xb') as err:
        started=time.perf_counter();rc,size,exceeded=capture.child_capture(command,out,err,cap);elapsed=time.perf_counter()-started
        for stream in (out,err):stream.flush();os.fsync(stream.fileno())
    h.write_exclusive(root/'process_completion.json',{'schema':'third-llm-execution-process-v1','command':command,'returncode':rc,'child_wall_seconds_through_exit':elapsed,'console_bytes':size,'console_cap_exceeded':exceeded,'children_maxrss_kib_linux':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'mode':a.mode,'slurm_job_id':os.environ['SLURM_JOB_ID'],'bindings':b,'human_annotation_calls':0,'retry_attempts':0})
    c.require(not exceeded,'child_console_limit_exceeded');raise SystemExit(rc)
if __name__=='__main__':main()
