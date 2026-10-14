"""V4 source/token preparation using unchanged, pinned v3 tokenizer primitives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
REPO=W.parent/'repo'
HERE=Path(__file__).resolve().parent
ROOT=W/'qualification/llm-v4-validation-requests-v1-20261014-a01'
V4=W/'tools/model-annotation-v4-20261014'
V4_SHA='959d32cfeb07598d7eb453f1564f2f4f1566227e62792bd9fab5e38a69dedcff'
OLD=W/'tools/llm-annotation-requests-v2-20261014'
OLD_SHA='bf70b015f39ad0421d8bf66f6b357123f24aceaf9f99b5158fb9b51f490e92d7'
OLD_CONTROL=W/'tools/llm-annotation-request-control-v2-20261014'
OLD_CONTROL_SHA='d20c325301a2a0564f65f78c25fa3f9347bb560ec0e0ef5dd952bd5221a79704'
SELECTION=W/'qualification/llm-only-v4-validation-sampling-a01'
SEED_ROOT=W/'qualification/llm-only-v4-selection-a01'
SAMPLE_PACKAGE=W/'tools/llm-v4-validation-sampler-v1-20261014'
SAMPLE_SHA='110744b78528532f543701df77c4397484b188003aca3a3a47c39177c111d538'

def require(ok,why):
    if not ok:raise ValueError(why)

def bootstrap(folder,pin):
    raw=(folder/'package_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==pin,'package_pin')
    for name,b in json.loads(raw)['files'].items():
        require(Path(name).name==name and name!='package_manifest.json','safe_member')
        f=folder/name;require(f.is_file() and not f.is_symlink(),'regular_member')
        data=f.read_bytes();require({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==b,'package_member')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def admit_selection(result,plan_sha,frame):
    require(result['schema']=='third-llm-v4-validation-sample-v1' and result['passed'] is True and
            result['stage']=='complete','v4_complete_validation_sampling')
    require(result['package_sha256']==SAMPLE_SHA and result['plan_sha256']==plan_sha and
            result['frame_result']==frame and result['pair_count']==220405510,'v4_validation_identity')
    for key in ('bounded_independent_parity','all_selected_scores_pairwise_replayed','full_above_threshold_frame','overlap_requires_fresh_calls'):
        require(result[key] is True,'validation_replay:'+key)
    require(result['model_inference_calls']==0 and result['human_annotation_calls']==0 and result['new_entropy_draws']==0 and
            result['label_filter'] is False and result['inference_activated'] is False and
            result['confirmatory_study_ready'] is False,'sampling_only')
    require(type(result['population']) is int and result['population']>0 and
            result['selected_pairs']==min(200,result['population']) and
            result['swap_pairs']==(result['selected_pairs']+9)//10,'full_validation_denominator')


def main():
    total_started=time.perf_counter()
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-sha256',required=True)
    parser.add_argument('--selection-result-sha256',required=True)
    args=parser.parse_args()
    bootstrap(HERE,args.package_sha256);bootstrap(V4,V4_SHA);bootstrap(OLD,OLD_SHA)
    bootstrap(OLD_CONTROL,OLD_CONTROL_SHA);bootstrap(SAMPLE_PACKAGE,SAMPLE_SHA)
    sys.path.insert(0,str(V4));import protocol as protocol
    validation=load('v4_validation_request_primitives',SAMPLE_PACKAGE/'primitives.py')
    r=load('v4_original_request_primitives',OLD/'prepare_requests.py')
    controls=load('v4_original_request_paths',OLD_CONTROL/'control.py')
    prior,h=controls.helpers()
    s=load('v4_request_frame',r.bootstrap(prior.SAMPLER,
        'c1da01e546c4ee0b5e84503b8513ec66f6ee5ae1ded16f46b2ce6b7801e18ce2')/'sampler.py')
    watch=s.Watch()
    for folder,pin in [(HERE,args.package_sha256),(V4,V4_SHA),(OLD,OLD_SHA),(OLD_CONTROL,OLD_CONTROL_SHA),(SAMPLE_PACKAGE,SAMPLE_SHA)]:
        s.verify_package(folder,pin,watch)
    pins=r.read(OLD/'expectations.json')
    for folder,pin in [(prior.SAMPLER,pins['sampler_package']['sha256']),
        (prior.FRAME_PACKAGE,pins['frame_package']['sha256']),
        (prior.PROTOCOL,pins['protocol_package']['sha256'])]:s.verify_package(folder,pin,watch)
    a=SimpleNamespace(workspace=W,repo=REPO,frame_dir=prior.FRAME,frame_result_sha256=prior.FRAME_SHA,
                      protocol_evidence=prior.EVIDENCE,mode='diagnostic')
    r.check_runtime(a,pins)
    plan=r.read(V4/'PLAN.json');protocol.validate(plan,V4);plan_sha=s.binding(V4/'PLAN.json')['sha256']
    parent=r.read(V4/'parent_PLAN.json')
    watch.add(prior.EVIDENCE/'snapshot_manifest.json')
    require(s.binding(prior.EVIDENCE/'snapshot_manifest.json')['sha256']==prior.EVIDENCE_SHA,'snapshot_pin')
    require({str(x.relative_to(prior.EVIDENCE)) for x in prior.EVIDENCE.rglob('*') if x.is_file()
             and x.name!='snapshot_manifest.json'}==set(parent['bound_files']),'snapshot_inventory')
    for name,b in parent['bound_files'].items():watch.add(prior.EVIDENCE/name,b)
    for name,b in pins['frozen_code'].items():watch.add(REPO/name,b)
    legacy_protocol=load('v4_request_parent_protocol',prior.PROTOCOL/'protocol.py')
    legacy_protocol.validate(parent,prior.EVIDENCE)
    sys.path.insert(0,str(REPO/'empirical_execution'))
    import numpy as np
    from phase3 import calibration as cal,panels
    require(Path(cal.__file__).resolve()==REPO/'empirical_execution/phase3/calibration.py' and
            Path(panels.__file__).resolve()==REPO/'empirical_execution/phase3/panels.py','scorer_module_identity')
    frame,frame_rows,vectors=s.load_frame(a,pins['sampler_expectations'],watch,np,panels)
    rb=watch.add(SELECTION/'run/result.json');require(rb['sha256']==args.selection_result_sha256,'selection_pin')
    result=r.read(SELECTION/'run/result.json');s.verify_seal(result,True)
    admit_selection(result,plan_sha,pins['sampler_expectations']['frame_result'])
    for name,b in result['files'].items():
        require(Path(name).name==name,'safe_selection_member');watch.add(SELECTION/'run'/name,b)
    require({x.name for x in (SELECTION/'run').iterdir()}==set(result['files'])|{'result.json','process_completion.json'},'selection_inventory')
    watch.add(SELECTION/'run/process_completion.json')
    require(r.read(SELECTION/'run/process_completion.json')['passed'] is True,'sampler_process_complete')
    lockb=watch.add(SEED_ROOT/'sampling_lock.json',result['sampling_lock'])
    lock=r.read(SEED_ROOT/'sampling_lock.json');s.verify_seal(lock,True)
    require(lock['plan_sha256']==plan_sha and lock['package_sha256']==V4_SHA,'lock_lineage')
    sample=list(r.jsonl(SELECTION/'run/validation.private.jsonl'));swaps=list(r.jsonl(SELECTION/'run/validation_swap.private.jsonl'))
    raw=[];last=(-1,-1)
    tau=Fraction(result['locked_tau']['numerator'],result['locked_tau']['denominator'])
    watch.add(SELECTION/'validation_lock.json',result['validation_lock'])
    for row in sample:
        i,j=row['frame_row_indices'];score=float.fromhex(row['score_hex'])
        require(0<=i<j<len(frame_rows) and (i,j)>last,'sample_order');last=(i,j)
        require(score>float(tau),'strict_locked_threshold')
        exact=float(panels.reference_cosines(panels.graph_normalize(vectors[i:i+1]),panels.graph_normalize(vectors[j:j+1]))[0,0])
        require(exact.hex()==row['score_hex'],'selected_actual_score');raw.append((i,j,score))
    reconstructed=validation.decorate(raw,result['population'],frame_rows,protocol,plan_sha,
                                      prior.FRAME_SHA,plan['inherits_unchanged']['models'])
    require(r.exact(sample,reconstructed),'independent_identity_weights_reconstruction')
    rng=validation.generator(protocol,plan_sha,prior.FRAME_SHA,lock['seeds']['validation_swap'],'validation_swap')
    require(r.exact(swaps,s.swaps(sample,cal,rng)),'validation_swap_replay')
    require(sum((Fraction(x['inclusion_weight']['numerator'],x['inclusion_weight']['denominator']) for x in sample),
                Fraction())==result['population'],'complete_weighted_validation_population')
    wanted={i for row in sample for i in row['frame_row_indices']}
    source=W/pins['sampler_expectations']['source_relative']/'records.jsonl'
    texts=r.source_texts(source,frame_rows,wanted,140250)
    shared=r.requests(sample,swaps,texts)
    for request in shared:
        request['stage']='validation_swap' if request['diagnostic_only'] else 'validation'
    os.umask(0o077);out=ROOT/'run';out.mkdir(exist_ok=False)
    output=r.Output(out,128<<20,1<<20);started=time.perf_counter();links={}
    try:
        output.jsonl('texts.private.jsonl',(texts[k] for k in sorted(texts)))
        output.jsonl('sample.private.jsonl',sample);output.jsonl('requests.private.jsonl',shared)
        torch,transformers,runtime=r.runtime(a,pins,s,watch)
        system=(prior.EVIDENCE/parent['prompt']['system_path']).read_text()
        task=(prior.EVIDENCE/parent['prompt']['task_path']).read_text();summaries={}
        for alias,pin in pins['models'].items():
            modeldir=getattr(controls,alias.upper());assetfile=getattr(controls,alias.upper()+'_ASSETS')
            r.verify_assets(modeldir,assetfile,pin,s,watch,links)
            pilot=r.load('v4_request_'+alias+'_pilot',REPO/pin['pilot_code_relative'])
            tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_fast=True)
            identity={'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__name__,
                'tokenizer_length':len(tokenizer),'tokenizer_vocab_size':tokenizer.vocab_size,
                'tokenizer_special_ids':tokenizer.all_special_ids,'tokenizer_backend_json_sha256':r.sha(tokenizer.backend_tokenizer.to_str().encode()),
                'chat_template_sha256':r.template_sha256(tokenizer,pilot)}
            require(identity==pin['tokenizer_identity'],'qualified_tokenizer_identity')
            for cls in type(tokenizer).__mro__:
                if cls.__module__.startswith('transformers'):
                    path=Path(inspect.getfile(cls)).resolve();require(path.is_relative_to(W/'envs'),'tokenizer_source_path');watch.add(path)
            compact=[]
            def token_rows():
                for request in shared:
                    token=r.tokenize_request(request,texts,pin,pilot,tokenizer,system,task)
                    compact.append({k:token[k] for k in ('request_index','selection_index','orientation','diagnostic_only',
                                                       'input_tokens','context_fit_with_output_reserve')})
                    yield token
            output.jsonl(alias+'_tokens.private.jsonl',token_rows())
            summaries[alias]={**r.summarize_requests(compact),'tokenizer_identity':identity}
            require(not torch.cuda.is_initialized(),'no_GPU_context')
        output.metadata('input_bindings.json',{'files':watch.files,'tokenizer_asset_symlink_targets':links})
        watch.recheck()
        s.closed_inventories(a,pins['sampler_expectations'])
        for logical,target in links.items():require(str(Path(logical).resolve(strict=True))==target,'tokenizer_symlink_changed')
        for alias,pin in pins['models'].items():
            modeldir=getattr(controls,alias.upper())
            require({str(f.relative_to(modeldir)) for f in modeldir.rglob('*') if f.is_file()}==set(pin['asset_files']),
                    'final_model_asset_inventory')
        for name,b in output.files.items():require(s.binding(out/name)==b,'output_changed')
        report=s.seal({'schema':'third-llm-v4-validation-requests-v1','passed':True,'stage':'complete',
            'created_utc':datetime.now(timezone.utc).isoformat(),'package_sha256':args.package_sha256,
            'plan_sha256':plan_sha,'selection_result':rb,'sampling_lock':lockb,'frame_result':pins['sampler_expectations']['frame_result'],
            'files':dict(output.files),'runtime':runtime,'model_censuses':summaries,
            'selected_pairs':len(sample),'swap_pairs':len(swaps),'planned_requests':2*len(shared),
            'validation_population':result['population'],'locked_tau':result['locked_tau'],
            'selection_statistics':result['selection_statistics'],
            'all_calibration_text_mappings_verified':True,'all_selected_scores_weights_identities_and_swaps_replayed':True,
            'all_planned_requests_retained':True,'full_frame_context_feasibility_checked':False,
            'model_inference_calls':0,'human_annotation_calls':0,'inference_activation':False,
            'semantic_validity_established':False,'confirmatory_study_ready':False,
            'elapsed_total_seconds':time.perf_counter()-total_started,
            'elapsed_after_source_preflight_seconds':time.perf_counter()-started})
        output.metadata('result.json',report)
        print(json.dumps({'passed':True,'selected_pairs':len(sample),'planned_requests':2*len(shared)}),flush=True)
    except BaseException as error:
        output.metadata('failure.json',{'passed':False,'type':type(error).__name__,'error':str(error)})
        raise

if __name__=='__main__':main()
