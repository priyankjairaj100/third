"""One v4 model process, with unchanged generation kernel and complete ledger."""
import argparse
from datetime import datetime,timezone
import json
import os
import resource
import socket
import sys
import time
import common as x
import contract as C
import kernel as K
import ledger as L
import node_policy as N

def runtime_files(store,folder):
    expected={'generation_config.json','token_eligibility.json','effective_constraint_config.json','runtime_manifest.json'}
    C.require({p.name for p in folder.iterdir()}==expected,'loader_runtime_inventory')
    files={str((folder/name).relative_to(store.root)):C.binding(folder/name) for name in expected}
    size=sum(b['bytes'] for b in files.values())
    C.require(size<=8<<20 and store.used+size<=store.limit,'runtime_output_allowance')
    store.files.update(files);store.used+=size
    return files

def expected_contract(request_sha,pins,resource_binding):
    c,result=x.request_contract(request_sha,pins)
    c.update(kernel_package=C.binding(x.HERE/'package_manifest.json'),resource_plan=resource_binding,
        models={k:{'model_id':v['model_id'],'revision':v['revision']} for k,v in pins['models'].items()},
        stage='validation',human_annotation_calls=0,human_completed=False,
        output_limit_bytes=256*1024**2,scope='model-assessed development; no human validity or publisher authenticity')
    return c,result

def main():
    total_started=time.perf_counter()
    parser=argparse.ArgumentParser()
    for name in ('package-sha256','activation-sha256','request-result-sha256','resource-plan-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--model-alias',choices=('qwen','mistral'),required=True)
    args=parser.parse_args();pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    rbind=C.binding(x.ROOT/'resource_plan.json');C.require(rbind['sha256']==args.resource_plan_sha256,'resource_pin')
    resources=C.strict_json((x.ROOT/'resource_plan.json').read_bytes());C.verify_seal(resources)
    C.require(resources['package_sha256']==args.package_sha256,'resource_package')
    expected,requests=expected_contract(args.request_result_sha256,pins,rbind)
    abind=C.binding(x.ROOT/'activation/activation.json');C.require(abind['sha256']==args.activation_sha256,'activation_pin')
    activation=C.validate_activation(x.ROOT/'activation',abind,expected)
    qualification=C.strict_json(C.checked_file(x.QUAL/'run/result.json',resources['qualification']).read_bytes())
    C.verify_seal(qualification)
    C.require(qualification['passed'] is True and qualification['package_sha256']==args.package_sha256 and
        qualification['request_contract']['request_manifest']==expected['request_manifest'],'actual_producer_consumer_qualification')
    family=resources['families'][args.model_alias]
    C.require(os.environ.get('SLURM_JOB_ID','').isdigit() and socket.gethostname() in N.ALLOWED and family['allowed_nodes']==list(N.ALLOWED),'Slurm_worker')
    C.require(os.environ.get('SLURM_CPUS_PER_TASK')=='4' and
        os.environ.get('SLURM_MEM_PER_NODE')==str(family['host_memory_bytes']//1024**2),'allocated_resources')
    C.require(sys.executable==str(x.W/pins['runtime']['interpreter_relative']),'qualified_interpreter')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        C.require(os.environ.get(name)=='1','single_numeric_thread')
    bundle=C.read_prepared_bundle(x.REQUESTS,activation,args.model_alias,protocol)
    calls=C.resolve_calls(bundle['tokens'],expected['request_manifest'],activation,protocol)
    C.require(len(calls)==resources['per_model_requests'],'complete_planned_family')
    order=C.schedule(bundle['tokens']);out=x.ROOT/args.model_alias/'output'
    C.require(str(out)==family['output_dir'] and not out.exists(),'fresh_activated_output_slot')
    watch=K.Watch()
    for path,b in [(x.ROOT/'resource_plan.json',rbind),(x.ROOT/'activation/activation.json',abind),
                   (x.QUAL/'run/result.json',resources['qualification'])]:watch.add(path,b)
    for name,b in bundle['bound_files'].items():watch.add(x.REQUESTS/name,b)
    kernel=K.FamilyKernel(x.REPO,args.model_alias,x.PROTOCOL)
    os.umask(0o077)
    ledger=L.Ledger(out,family['output_limit_bytes'],calls,lambda j:kernel.pilot.validate_judgment(j,kernel.schema))
    store=ledger.store;reserve=4<<20;store.limit-=reserve
    fatal=None;loaded=False;final_equal=False;registered_runtime={}
    store.json('run_intent.json',{'schema':'third-v4-model-run-intent','created_utc':datetime.now(timezone.utc).isoformat(),
        'activation':abind,'resource_plan':rbind,'model_alias':args.model_alias,'attempt':0,'retry':False,
        'request_indices_in_execution_order':[calls[i]['request_index'] for i in order]})
    try:
        K.runtime_preflight(kernel.pins)
        store.json('allocated_gpu_capability.json',N.verify_allocated_gpu())
        runtime=out/'runtime';runtime.mkdir(mode=0o700)
        kernel.load(activation,getattr(controls,args.model_alias.upper()),
            getattr(controls,args.model_alias.upper()+'_ASSETS'),runtime,
            'same pinned offline environment as completed v3; v4 call identities and request roles')
        registered_runtime=runtime_files(store,runtime);loaded=True
        for index in order:
            call=calls[index];writer=ledger.begin(call)
            response=K.execute_call(kernel,call,bundle['tokens'][index],bundle['rows'][index],writer)
            ledger.record(writer,response)
            if response['status']=='runtime_error':
                fatal='Recorded runtime_error; remaining calls not attempted';break
            if len(ledger.responses)%10==0:
                print(json.dumps({'model':args.model_alias,'completed_calls':len(ledger.responses),
                                  'planned_calls':len(calls),'elapsed_seconds':time.perf_counter()-total_started}),flush=True)
    except Exception as error:
        fatal=type(error).__name__+': '+str(error)
    finally:
        if (out/'runtime').exists() and not registered_runtime:
            for path in (out/'runtime').iterdir():
                if path.is_file() and not path.is_symlink():
                    name=str(path.relative_to(out));b=C.binding(path)
                    if name not in store.files:store.files[name]=b;store.used+=b['bytes']
        store.limit+=reserve
    try:
        watch.recheck();kernel.recheck();final_equal=True
    except Exception as error:
        fatal=fatal or 'Final input recheck: '+type(error).__name__+': '+str(error)
    if not loaded:fatal=fatal or 'Model load did not complete'
    runtime_hash=kernel.state['runtime_manifest_sha256'] if kernel.state is not None else None
    responses,summary=ledger.finish(fatal,runtime_hash)
    store.json('final_bindings.json',{'passed':final_equal,'files':watch.files,'model_and_runtime_files':kernel.watch.files})
    report=x.sealed({'schema':'third-v4-model-execution-v1','passed':loaded and final_equal and fatal is None,
        'created_utc':datetime.now(timezone.utc).isoformat(),'model_alias':args.model_alias,
        'package_sha256':args.package_sha256,'activation':abind,'resource_plan':rbind,
        'request_manifest':expected['request_manifest'],'planned_calls':len(calls),'ledger':summary,
        'files':dict(store.files),'ordinary_fatal':fatal,'input_recheck_passed':final_equal,
        'human_annotation_calls':0,'retry_calls':0,'semantic_validity_established':False,
        'elapsed_seconds_before_final_publication':time.perf_counter()-total_started,
        'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'hostname':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID']})
    store.json('result.json',report);store.recheck()
    print(json.dumps({'passed':report['passed'],'model':args.model_alias,'ledger':summary}),flush=True)
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
