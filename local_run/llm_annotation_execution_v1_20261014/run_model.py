#!/usr/bin/env python3
"""One newly activated family/selection run. No entropy, retries or submission."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time

ROOT=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def binding(p):
    h=hashlib.sha256();n=0
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b)
    return {'bytes':n,'sha256':h.hexdigest()}
def strict(raw):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'duplicate_json_key');d[k]=v
        return d
    def bad(s):raise ValueError('nonfinite_json')
    def number(s):
        import math
        v=float(s);require(math.isfinite(v),'nonfinite_json');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)
def read(p):
    p=Path(p);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=4*1024**2,'metadata_file')
    return strict(p.read_bytes())
def package(folder,expected):
    folder=Path(folder);require(folder.is_dir() and not folder.is_symlink(),'package_directory')
    require(binding(folder/'package_manifest.json')==expected,'package_manifest_binding')
    files=read(folder/'package_manifest.json')['files'];require(files and len(files)<=32,'package_files')
    require({p.name for p in folder.iterdir() if p.name!='__pycache__'}==set(files)|{'package_manifest.json'},'package_inventory')
    out={str((folder/'package_manifest.json').absolute()):expected}
    for name,b in files.items():
        require(Path(name).name==name and name not in ('.','..'),'package_member_name')
        p=folder/name;require(p.is_file() and not p.is_symlink() and binding(p)==b,'package_member_binding:'+name);out[str(p.absolute())]=b
    return out

def exact_import(name,path):
    require(name not in sys.modules,'module_already_loaded:'+name)
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    try:spec.loader.exec_module(m)
    except BaseException:sys.modules.pop(name,None);raise
    require(Path(m.__file__).resolve()==Path(path).resolve(),'import_origin');return m

def load_components(folder):
    # The sealed modules use these literal intra-package import names.
    return {name:exact_import(name,Path(folder)/(name+'.py')) for name in ('contract','ledger','kernel')}

def runtime_preflight(family):
    require(platform.system()=='Linux','linux_required')
    job=os.environ.get('SLURM_JOB_ID','');require(re.fullmatch('[1-9][0-9]*',job or '') is not None,'numeric_slurm_job')
    require(os.environ.get('SLURM_CPUS_PER_TASK')==str(family['cpus']),'slurm_cpu_allocation')
    require(os.environ.get('SLURM_MEM_PER_NODE')==str(family['host_memory_bytes']//1024**2),'slurm_host_memory_allocation')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(name)=='1','numeric_thread_environment:'+name)
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline_environment')
    require(family['gpu_count']==1 and family['gpu_model']=='A10080GB','resource_gpu_contract')
    return {'pid':os.getpid(),'hostname':platform.node(),'slurm_job_id':job,'python':sys.version,'executable':sys.executable,
        'slurm_cpus_per_task':os.environ['SLURM_CPUS_PER_TASK'],'slurm_mem_per_node_mib':os.environ['SLURM_MEM_PER_NODE']}

def safe_output(workspace,out,fixed):
    workspace=Path(workspace).resolve();out=Path(out).absolute()
    require(out==Path(fixed).absolute() and out.is_relative_to(workspace/'qualification'),'fixed_activated_output_path')
    require(not out.exists() and not out.is_symlink(),'output_already_exists')
    require(out.parent.is_dir(),'controller_must_create_output_parent')
    for p in (out.parent,*out.parent.parents):require(not p.is_symlink(),'output_parent_symlink')
    require(out.resolve()==out,'output_canonical_path');return out

def track_runtime(store,runtime_dir,reserve):
    """Register only the four bounded files written by the unchanged family loader."""
    names={'generation_config.json','token_eligibility.json','effective_constraint_config.json','runtime_manifest.json'}
    require({p.name for p in runtime_dir.iterdir()}==names,'runtime_exact_inventory')
    files={}
    for name in sorted(names):
        p=runtime_dir/name;require(p.is_file() and not p.is_symlink(),'runtime_regular_file');p.chmod(0o600);files[str(p.relative_to(store.root))]=binding(p)
    size=sum(b['bytes'] for b in files.values());require(size<=reserve and store.used+size<=store.limit,'runtime_output_reserve')
    require(not (set(files)&set(store.files)),'runtime_registration_duplicate');store.files.update(files);store.used+=size
    return files

def run_schedule(ledger,kernel,bundle,calls,execution_order,components):
    """Fixed policy; scheduling positions refer to original prepared request rows."""
    require(sorted(execution_order)==list(range(len(calls))),'complete_schedule_permutation')
    ordinary_fatal=None;integrity_error=None
    for pos in execution_order:
        call=calls[pos]
        try:
            writer=ledger.begin(call)
            response=components['kernel'].execute_call(kernel,call,bundle['tokens'][pos],bundle['rows'][pos],writer)
            ledger.record(writer,response)
        except Exception as exc:
            # Source/token/identity/artifact faults are integrity failures, not a
            # fabricated model answer. The intent remains available to the audit.
            ordinary_fatal=type(exc).__name__+': '+str(exc);integrity_error=ordinary_fatal;break
        if response['status']=='runtime_error':
            ordinary_fatal='Recorded runtime_error: '+response['error']['type']+': '+response['error']['message'];break
    return ordinary_fatal,integrity_error

def execute(args,components,admission,package_bindings,own_binding):
    """All authority comes from independently verified activation + fixed output slot."""
    C,L,K=components['contract'],components['ledger'],components['kernel']
    activation=admission['activation'];resources=admission['resource_plan'];family=resources['families'][args.model_alias]
    admitted_paths=admission['admission']['paths']
    for arg,key in [('requests_dir','request_dir'),('component_package','component_package'),('integration_package','integration_package'),('protocol_package','protocol_package')]:
        require(Path(getattr(args,arg)).resolve()==Path(admitted_paths[key]).resolve(),'admitted_path:'+arg)
    require(resources['packages']['runner']==own_binding,'activated_runner_package')
    require(resources['packages']['component']==binding(Path(args.component_package)/'package_manifest.json'),'activated_component_package')
    require(resources['packages']['integration']==binding(Path(args.integration_package)/'package_manifest.json'),'activated_integration_package')
    require(resources['request_manifest']['sha256']==args.request_result_sha256,'activated_request_result')
    rt=runtime_preflight(family);out=safe_output(args.workspace,args.output_dir,family['output_dir'])
    kernel=K.FamilyKernel(args.repo,args.model_alias,args.protocol_package)
    require(Path(sys.executable).absolute()==Path(args.workspace).absolute()/kernel.pins['runtime']['interpreter_relative'],'qualified_interpreter_path')
    bundle=C.read_prepared_bundle(args.requests_dir,activation,args.model_alias,kernel.protocol)
    require(len(bundle['tokens'])==resources['planned']['per_model'],'planned_family_denominator')
    require(bundle['result']['selected_pairs']==resources['planned']['canonical_pairs'] and
            bundle['result']['diagnostic_swap_pairs']==resources['planned']['swap_pairs'],'planned_phase_counts')
    calls=C.resolve_calls(bundle['tokens'],activation.contract['request_manifest'],activation,kernel.protocol)
    order=C.schedule(bundle['tokens'])
    require([calls[i]['request_index'] for i in order]==bundle['result']['model_censuses'][args.model_alias]['scheduled_request_indices'],'sealed_request_schedule')
    watch=K.Watch()
    for path,b in {**package_bindings,**admission['watch_files']}.items():watch.add(path,b)
    for name,b in bundle['bound_files'].items():watch.add(Path(args.requests_dir)/name,b)
    ledger=L.Ledger(out,family['output_limit_bytes'],calls,lambda j:kernel.pilot.validate_judgment(j,kernel.schema))
    store=ledger.store;started=time.perf_counter();fatal=integrity_error=None;loaded=False;final_equal=False
    model_bindings_initial=None;runtime_files={};model_load_attempted=False
    # Completion metadata has a reserved budget even after a bounded call budget
    # refusal. The writer itself still refuses all over-limit artifact writes.
    reserve=4*1024**2;require(store.limit>reserve+8*1024**2,'insufficient_output_metadata_reserve');store.limit-=reserve
    try:
        store.json('run_intent.json',{'schema':'third-llm-annotation-run-intent-v1','model_alias':args.model_alias,
            'activation':activation.raw_binding,'request_manifest':activation.contract['request_manifest'],
            'runner_package':own_binding,'execution_request_indices':[calls[i]['request_index'] for i in order],
            'no_retries':True,'stop_on_runtime_error':True,'runtime':rt,'argv':sys.argv,'created_utc':datetime.now(timezone.utc).isoformat()})
        store.json('input_bindings.json',{'schema':'third-llm-inference-input-bindings-v1','files':watch.files,
            'activation':activation.raw_binding,'prepared_request_manifest':activation.contract['request_manifest'],
            'resource_plan':activation.contract['resource_plan'],'runner_package':own_binding,
            'protocol_evidence_role':'Bound historical evidence; distinct from staged protocol and executing frozen repository'})
        runtime_dir=out/'runtime';runtime_dir.mkdir(mode=0o700);L.fsync_dir(out)
        require(store.limit-store.used>=8*1024**2,'runtime_reserve_before_load')
        model_load_attempted=True
        kernel.load(activation,args.model_dir,args.weights_manifest,runtime_dir,resources.get('runtime_dependency_source'))
        runtime_files=track_runtime(store,runtime_dir,8*1024**2);loaded=True
        model_bindings_initial={k:dict(v) for k,v in kernel.watch.files.items()}
        fatal,integrity_error=run_schedule(ledger,kernel,bundle,calls,order,components)
    except Exception as exc:
        fatal=type(exc).__name__+': '+str(exc);integrity_error=fatal
    finally:
        # Partial inherited runtime files are retained on failure and inventoried,
        # but cannot make a failed loader pass its exact runtime-output gate.
        if (out/'runtime').exists() and not runtime_files:
            for path in (out/'runtime').iterdir():
                if path.is_file() and not path.is_symlink():
                    name=str(path.relative_to(out));b=binding(path)
                    if name not in store.files:store.files[name]=b;store.used+=b['bytes'];path.chmod(0o600)
        store.limit+=reserve
    try:
        watch.recheck();admission['watch'].recheck();kernel.recheck();final_equal=True
    except Exception as exc:
        integrity_error='FinalRecheck: '+type(exc).__name__+': '+str(exc);fatal=fatal or integrity_error
    if not loaded:fatal=fatal or 'Model did not load'
    runtime_hash=kernel.state['runtime_manifest_sha256'] if kernel.state is not None else None
    # These durable writes may fail closed on disk exhaustion. A killed or
    # incomplete run has no accepted result; external audit uses planned_calls.
    responses,summary=ledger.finish(fatal,runtime_hash)
    store.json('final_bindings.json',{'schema':'third-llm-inference-final-bindings-v1','passed':final_equal and loaded,
        'inputs_initial_final_equal':final_equal,'model_loaded':loaded,'model_code_assets_initial':model_bindings_initial,
        'model_code_assets_final':kernel.watch.files,'input_files_final':watch.files,
        'full_model_weight_bytes_rehashed':loaded and final_equal,'output_inventory_before_final_binding':dict(store.files)})
    if fatal is not None:store.json('failure.json',{'schema':'third-llm-annotation-execution-failure-v1','ordinary_fatal':fatal,
        'artifact_integrity_error':integrity_error,'all_planned_calls_retained':True,'retry_allowed':False})
    store.recheck()
    result={'schema':'third-llm-annotation-execution-v1','passed':fatal is None and loaded and final_equal,
        'completed':fatal is None,'passed_execution_integrity':integrity_error is None and loaded and final_equal,
        'stage':'complete','alias':args.model_alias,'stop_on_runtime_error':True,
        'final_input_bindings_unchanged':final_equal,'artifact_integrity_independently_verified':False,'model_alias':args.model_alias,
        'model_id':kernel.model['model_id'],'model_revision':kernel.model['revision'],'activation':activation.raw_binding,
        'request_manifest':activation.contract['request_manifest'],'runner_package':own_binding,
        'component_package':resources['packages']['component'],'integration_package':resources['packages']['integration'],
        'resource_plan':activation.contract['resource_plan'],'planned_calls':len(calls),'recorded_calls':len(responses),
        'canonical_calls':sum(c['orientation']=='canonical' for c in calls),'diagnostic_calls':sum(c['orientation']=='swapped_diagnostic' for c in calls),
        'started_call_intents':len(ledger.started),'status_counts':summary['status_counts'],
        'model_load_attempted':model_load_attempted,'model_loaded':loaded,'initial_final_bindings_equal':final_equal,
        'ordinary_fatal':fatal,'artifact_integrity_error':integrity_error,'files':dict(store.files),
        'output_limit_bytes':family['output_limit_bytes'],'output_bytes_before_result':store.used,
        'elapsed_seconds_before_result':time.perf_counter()-started,'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'retry_calls':0,'human_annotation_calls':0,'human_completed':False,'semantic_accuracy_established':False,
        'logit_optimality_independently_replayed':False,'source_role':resources['source_role']}
    result['sha256']=sha(canonical(result));store.json('result.json',result);store.recheck();return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','workspace','protocol-package','component-package','integration-package','requests-dir','admission','activation-dir','model-dir','weights-manifest','output-dir'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('package-sha256','admission-sha256','activation-sha256','request-result-sha256'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--model-alias',choices=('qwen','mistral'),required=True);a=p.parse_args(argv)
    os.umask(0o077);pins=read(ROOT/'expectations.json')
    for name in ('package_sha256','admission_sha256','activation_sha256','request_result_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',getattr(a,name)) is not None,'raw_sha_required')
    own=binding(ROOT/'package_manifest.json');require(own['sha256']==a.package_sha256,'external_runner_manifest_sha')
    watched=package(ROOT,own)
    for folder,b in ((a.component_package,pins['component_package']),(a.integration_package,pins['integration_package']),(a.protocol_package,pins['protocol_package'])):
        watched.update(package(folder,b))
    components=load_components(a.component_package)
    exact_import('common',a.integration_package/'common.py')
    integration=exact_import('third_inference_admission',a.integration_package/'activation.py')
    ab=binding(a.admission);require(ab['sha256']==a.admission_sha256,'admission_raw_sha')
    activation_raw=binding(a.activation_dir/'activation.json');require(activation_raw['sha256']==a.activation_sha256,'activation_raw_sha')
    active,admission=integration.read_activation(a.admission,ab,a.activation_dir,activation_raw,component_contract=components['contract'])
    admission['activation']=active
    result=execute(a,components,admission,watched,own)
    print(json.dumps({k:result[k] for k in ('schema','passed','planned_calls','recorded_calls','status_counts','ordinary_fatal')}))
    return 0 if result['passed'] else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'schema':'third-llm-annotation-execution-preflight-failure-v1','passed':False,
            'error_type':type(exc).__name__,'error':str(exc),'retry_allowed':False}),file=sys.stderr);raise SystemExit(2)
