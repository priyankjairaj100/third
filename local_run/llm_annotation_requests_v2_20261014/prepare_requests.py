#!/usr/bin/env python3
"""Source-bound selection requests and CPU tokenizer census; no activation or model load."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import socket
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FALSE_FLAGS = ('inference_activation', 'threshold_selected', 'model_calibration_complete',
 'semantic_validity_established', 'original_human_gate_satisfied', 'confirmatory_study_ready',
 'source_authenticity_verified', 'primary_input_acceptance', 'original_route_acceptance',
 'full_population_cache_acceptance', 'requires_human_annotations', 'weight_bytes_rehashed',
 'model_weights_loaded', 'model_forwards_executed', 'full_pair_census_reexecuted',
 'full_frame_context_feasibility_checked', 'new_sampling_performed', 'production_entropy_drawn')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def strict(raw):
    def unique(items):
        result = {}
        for k,v in items:
            require(k not in result, 'duplicate_json_key'); result[k] = v
        return result
    def bad(_): raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def template_sha256(tokenizer,pilot):
    """Qualified pilot convention hashes a JSON string, not raw template UTF-8."""
    return pilot.digest_bytes(pilot.canonical(tokenizer.chat_template))


def exact(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def read(path, cap=4 << 20):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= cap, 'metadata_cap_or_missing_file')
    with path.open('rb') as f: raw = f.read(cap + 1)
    require(len(raw) <= cap, 'metadata_cap')
    return strict(raw)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(folder, expected_sha):
    """Authenticate the imported helper with stdlib only, before running its code."""
    folder = Path(folder).absolute()
    require(not any(p.is_symlink() for p in (folder,*folder.parents)), 'package_symlink')
    manifest = folder/'package_manifest.json'
    require(sha(manifest.read_bytes()) == expected_sha, 'package_pin')
    files = read(manifest)['files']; require(files and len(files) <= 32, 'package_inventory')
    for name,b in files.items():
        require(Path(name).name == name and name != 'package_manifest.json', 'package_member_path')
        path = folder/name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == b['bytes']
                and sha(path.read_bytes()) == b['sha256'], 'package_member_binding')
    return folder


class Output:
    """Exclusive bounded private writes; leave incomplete artifacts on failure."""
    def __init__(self, folder, limit=64 << 20, reserve=1 << 20):
        self.folder=Path(folder); self.limit=limit; self.reserve=reserve; self.used=0; self.files={}
    def jsonl(self, name, rows):
        require(Path(name).name == name and name not in self.files, 'output_name')
        path=self.folder/name; h=hashlib.sha256(); size=0
        with path.open('xb') as f:
            for row in rows:
                raw=canonical(row)+b'\n'
                require(self.used+len(raw) <= self.limit-self.reserve, 'whole_preparation_output_cap')
                f.write(raw); h.update(raw); size+=len(raw); self.used+=len(raw)
            f.flush(); os.fsync(f.fileno())
        self.files[name]={'bytes':size,'sha256':h.hexdigest()}
    def metadata(self, name, value):
        raw=canonical(value)+b'\n'
        require(len(raw)<=self.reserve and self.used+len(raw)<=self.limit,'metadata_output_cap')
        with (self.folder/name).open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.used+=len(raw); b={'bytes':len(raw),'sha256':sha(raw)}; self.files[name]=b
        return b


def jsonl(path, max_line=16 << 20):
    with Path(path).open('rb') as f:
        while True:
            line=f.readline(max_line+1)
            if not line: break
            require(len(line)<=max_line and line.endswith(b'\n') and line.strip(), 'physical_LF_or_line_cap')
            yield strict(line)


def validate_selection(result, review, rb, sampler_binding, lock_binding, pins):
    require(result.get('schema')=='third-llm-selection-sample-v1' and result.get('passed') is True
            and result.get('stage')=='complete' and result.get('mode')=='selection','selection_not_complete')
    s=pins['sampler_expectations']
    for key,wanted in {'source_frame_result':s['frame_result'],'sampler_package':sampler_binding,
      'protocol_plan_sha256':s['plan_sha256'],'frame_package_sha256':s['frame_package_sha256'],
      'selected_frame_rows':s['rows'],'unordered_pair_count':s['pairs'],'scoring_block_size':256,
      'rng_stages':['selection','selection_swap'],'score_stream_encoding':'ASCII i TAB j TAB float.hex(score) LF'}.items():
        require(type(result.get(key)) is type(wanted) and result.get(key)==wanted,'selection_'+key)
    require(result.get('sampling_performed') is True and result.get('production_sampling_performed') is True
      and result.get('engineering_subset_sampled') is False and result.get('diagnostic_seeds') is None
      and result.get('parity') is None,'selection_role')
    for k in ('same_source_pairs_excluded','availability_filter_applied','requires_human_annotations',
      'threshold_selected','model_calibration_complete','semantic_validity_established','original_human_gate_satisfied',
      'inference_activation','confirmatory_study_ready','source_authenticity_verified','primary_input_acceptance',
      'original_route_acceptance','full_population_cache_acceptance'):
        require(result.get(k) is False,'selection_false_scope:'+k)
    require(all(type(result.get(k)) is int and result[k]==0 for k in ('model_inference_calls','human_annotation_calls')),
      'selection_has_labels')
    populations=result.get('bin_populations'); require(isinstance(populations,list) and len(populations)==20
      and all(type(n) is int and n>=0 for n in populations) and sum(populations)==s['pairs'],'bin_population')
    counts=[min(30,n) for n in populations]
    require(result.get('bin_sample_counts')==counts and type(result.get('selected_pairs')) is int
      and result['selected_pairs']==sum(counts) and type(result.get('diagnostic_swap_pairs')) is int
      and result['diagnostic_swap_pairs']==(sum(counts)+9)//10,'sample_counts')
    require(set(result.get('files',{}))=={'input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl'},
      'selection_file_inventory')
    require(re.fullmatch('[0-9a-f]{64}',result.get('score_stream_sha256','')) is not None,'score_stream_digest')
    require(result['lock_evidence']['sampling_lock']==lock_binding,'sampling_lock_binding')
    require(review.get('schema')=='third-llm-selection-metadata-identity-review-v1'
      and review.get('passed') is True and review.get('state')=='COMPLETED','independent_selection_review_required')
    for key,wanted in {'selection_result':rb,'selection_result_internal_seal':result['sha256'],
      'sampler_package':sampler_binding,'sampling_lock':lock_binding,'source_frame_result':s['frame_result'],
      'lock_helper_package':pins['lock_package']}.items():
        require(review.get(key)==wanted,'review_binding:'+key)
    checks=review.get('checked',{})
    for key,wanted in {'selected_pairs':result['selected_pairs'],'swap_pairs':result['diagnostic_swap_pairs'],
      'exact_weighted_denominator':s['pairs'],'all_sample_identity_orientation_weight_checks_passed':True,
      'independent_swap_reservoir_replay_passed':True}.items():
        require(type(checks.get(key)) is type(wanted) and checks.get(key)==wanted,'review_check:'+key)
    b=review.get('next_inference_builder_bindings',{})
    for key,wanted in {'selection_files':result['files'],'frame_files':s['frame_files'],
      'frame_raw_result_sha256':s['frame_result']['sha256'],'plan_raw_sha256':s['plan_sha256'],
      'sampling_lock':lock_binding,'source_records':s['frame_inherited']['prepared_source_records_file'],
      'source_audit':s['frame_inherited']['prepared_source_audit_file'],
      'inference_activation_still_required':True,'fresh_validation_calls_required':True}.items():
        require(b.get(key)==wanted,'review_builder_binding:'+key)
    require(all(review.get(k) is False for k in ('complete_census_independently_replayed',
      'selection_reservoir_independently_replayed','bin_population_counts_independently_recomputed',
      'score_stream_independently_recomputed')),'review_scope_mismatch')


def validate_sample(sample, swaps, result, frame_rows, vectors, protocol, plan, cal, panels, sampler, lock):
    require(len(sample)==result['selected_pairs'] and len(swaps)==result['diagnostic_swap_pairs'],'sample_file_count')
    raw=[]; previous=(-1,-1); counts=Counter()
    for row in sample:
        ij=row.get('frame_row_indices'); require(isinstance(ij,list) and len(ij)==2
          and all(type(i) is int for i in ij) and 0<=ij[0]<ij[1]<len(frame_rows) and tuple(ij)>previous,'pair_order_or_duplicate')
        previous=tuple(ij); score=float.fromhex(row['score_hex']); require(math.isfinite(score),'finite_score')
        require(type(row.get('bin')) is int and row['bin']==cal._bin(score),'score_bin')
        # Exact pairwise coordinate-order reference, not a full population replay.
        pair=panels.graph_normalize(vectors[ij]); exact_score=float(panels.reference_cosines(pair[:1],pair[1:])[0,0])
        require(exact_score.hex()==row['score_hex'],'retained_score_differs_from_actual_vectors')
        raw.append((*ij,score,row['bin'])); counts[row['bin']]+=1
    require([counts[i] for i in range(20)]==result['bin_sample_counts'],'per_bin_sample_count')
    wanted=sampler.decorate(raw,result['bin_populations'],frame_rows,protocol,plan,result['source_frame_result']['sha256'])
    require(exact(sample,wanted),'full_sample_identity_weight_or_extra_field')
    rng=sampler.make_rng(protocol,plan['sha256'],result['source_frame_result']['sha256'],
                         'selection_swap',lock['seed_hex']['selection_swap'],None)
    require(exact(swaps,sampler.swaps(sample,cal,rng)),'swap_reservoir_or_identity')
    require(sum((Fraction(x['inclusion_weight']['numerator'],x['inclusion_weight']['denominator']) for x in sample),
                Fraction(0))==result['unordered_pair_count'],'full_weighted_denominator')
    return raw


def source_texts(path, frame_rows, wanted_indices, expected_full_rows, line_cap=16 << 20):
    """Verify EVERY calibration record's exact physical position and text, retaining selected texts only."""
    by_original={r['original_full_row_index']:r for r in frame_rows}; seen=set(); texts={}; calibration=0; total=0
    require(len(by_original)==len(frame_rows),'duplicate_full_row_mapping')
    for index,row in enumerate(jsonl(path,line_cap)):
        total+=1; rid=row.get('record_id')
        require(isinstance(rid,str) and rid and rid not in seen,'duplicate_or_missing_source_id');seen.add(rid)
        require(isinstance(row.get('text'),str) and row['text'],'missing_source_text')
        if row.get('partition')=='calibration':
            require(index in by_original,'extra_calibration_record'); expected=by_original[index]
            require(expected['calibration_row_index']==calibration,'source_calibration_order'); calibration+=1
            for k in ('record_id','source_unit_id','source_kind','partition'):
                require(row.get(k)==expected[k],'source_frame_identity:'+k)
            require(sha(row['text'].encode('utf-8'))==expected['text_sha256'],'source_frame_text_sha')
            if expected['calibration_row_index'] in wanted_indices:
                texts[expected['calibration_row_index']]={'calibration_row_index':expected['calibration_row_index'],
                  'original_full_row_index':index,'record_id':rid,'text_sha256':expected['text_sha256'],'text':row['text']}
        else: require(index not in by_original,'calibration_partition_changed')
    require(total==expected_full_rows and calibration==len(frame_rows) and set(texts)==set(wanted_indices),'incomplete_source_coverage')
    return texts


def requests(sample, swaps, texts):
    entries=[(i,'canonical',r['canonical_frame_row_indices']) for i,r in enumerate(sample)]
    entries += [(r['selection_index'],'swapped_diagnostic',r['swapped_frame_row_indices']) for r in swaps]
    out=[]
    for index,(si,orientation,indices) in enumerate(entries):
        row=sample[si]
        out.append({'request_index':index,'selection_index':si,'stage':'selection' if orientation=='canonical' else 'selection_swap',
          'orientation':orientation,'diagnostic_only':orientation!='canonical','opaque_pair_id':row['opaque_pair_id'],
          'natural_pair_id':row['natural_pair_id'],'shown_calibration_row_indices':indices,
          'shown_record_ids':[texts[i]['record_id'] for i in indices],
          'shown_text_sha256':[texts[i]['text_sha256'] for i in indices], 'model_seeds':row['model_seeds']})
    return out


def tokenize_request(request, texts, model, pilot, tokenizer, system, task):
    a,b=(texts[i]['text'] for i in request['shown_calibration_row_indices'])
    row={'pair_id':request['opaque_pair_id'],'orientation':request['orientation'],'text_a':a,'text_b':b}
    prepared=pilot.tokenize_call({'system_prompt':system,'task_definition':task,'tokenizer':tokenizer},row)
    encoded=prepared['encoded']; require(set(encoded)=={'input_ids','attention_mask'},'unexpected_tokenizer_fields')
    require(all(x.device.type=='cpu' and x.ndim==2 and x.shape[0]==1 for x in encoded.values()),'CPU_single_request_tensors')
    ids=encoded['input_ids'][0].tolist(); mask=encoded['attention_mask'][0].tolist()
    require(len(ids)==prepared['input_tokens'] and len(mask)==len(ids) and all(x==1 for x in mask)
      and all(type(x) is int and x>=0 for x in ids),'tokenizer_ids_or_mask')
    fit=len(ids)+512<=32768
    return {**request,'model_id':model['model_id'],'model_revision':model['revision'],
      'model_seed':request['model_seeds'][model['model_id']], 'input_row_sha256':pilot.digest_bytes(pilot.canonical(row)),
      'input_row_hash_serialization':'qualified pilot canonical JSON: ensure_ascii=True, sorted keys, compact separators',
      'rendered_prompt_sha256':sha(prepared['rendered'].encode('utf-8')),
      'input_token_ids':ids,'input_tokens':len(ids),'input_token_ids_sha256':sha(canonical(ids)),
      'attention_mask_policy':'one for every unpadded token','max_new_tokens':512,'total_token_budget':32768,
      'context_fit_with_output_reserve':fit,'preparation_status':'ready_for_future_activation' if fit else 'context_overflow',
      'full_text_preserved':True,'truncation_applied':False,'activation_lock_sha256':None,'call_id':None}


def summarize_requests(records):
    schedule=sorted(records,key=lambda r:(-r['input_tokens'],r['selection_index'],r['orientation']!='canonical'))
    return {'requests':len(records),'canonical_requests':sum(not r['diagnostic_only'] for r in records),
      'diagnostic_requests':sum(r['diagnostic_only'] for r in records),
      'total_input_tokens':sum(r['input_tokens'] for r in records),
      'max_input_tokens':max((r['input_tokens'] for r in records),default=0),
      'overflow_requests':sum(not r['context_fit_with_output_reserve'] for r in records),
      'canonical_overflows':sum(not r['diagnostic_only'] and not r['context_fit_with_output_reserve'] for r in records),
      'scheduled_request_indices':[r['request_index'] for r in schedule],
      'schedule_rule':'input_tokens descending, selection_index ascending, canonical before diagnostic tie',
      'all_planned_requests_retained':True,'overflow_retains_full_sampling_denominator':True}


def verify_assets(model_dir, manifest_path, pin, sampler, watch, links):
    model_dir=Path(model_dir).absolute(); require(model_dir.name==pin['revision'],'model_directory_revision')
    watch.add(manifest_path,pin['asset_manifest']); manifest=read(manifest_path)
    require(manifest['model_id']==pin['model_id'] and manifest['revision']==pin['revision'],'asset_model_identity')
    declared={r['relative_path']:{'bytes':r['size_bytes'],'sha256':r['sha256']} for r in manifest['files']}
    require(len(declared)==len(manifest['files']) and declared==pin['asset_files'],'asset_manifest_inventory')
    require({str(p.relative_to(model_dir)) for p in model_dir.rglob('*') if p.is_file()}==set(declared),'asset_disk_inventory')
    for name,wanted in pin['tokenizer_assets'].items():
        require(not name.endswith('.safetensors') and Path(name).name==name,'tokenizer_asset_path')
        logical=model_dir/name; resolved=logical.resolve(strict=True)
        watch.add(resolved,wanted); links[str(logical)]=str(resolved)
    for name in ('config.json','tokenizer_config.json'):
        require(not read(model_dir/name).get('auto_map'),'custom_model_or_tokenizer_code_prohibited')
    return manifest


def check_runtime(args, pins):
    require(platform.system()=='Linux' and re.fullmatch('[1-9][0-9]*',os.environ.get('SLURM_JOB_ID','')),
      'Linux_Slurm_required')
    require(socket.gethostname()=='csis.cn1','qualified_CPU_node_required')
    require(platform.python_version()==pins['runtime']['python'] and Path(sys.executable).absolute()==
      Path(args.workspace).absolute()/pins['runtime']['interpreter_relative'],'pinned_judge_interpreter_required')
    for k,v in {'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
      'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
      'PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0'}.items():
        require(os.environ.get(k)==v,'required_environment:'+k)
    versions={name:importlib.metadata.version(name) for name in pins['runtime']['package_versions']}
    require(versions==pins['runtime']['package_versions'],'runtime_distribution_versions')
    return versions


def runtime(args, pins, sampler, watch):
    versions=check_runtime(args,pins)
    import torch
    import numpy as np
    import transformers
    import tokenizers
    import jinja2
    import transformers.utils.chat_template_utils as chat_utils
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(),'CUDA_must_remain_uninitialized')
    for module,name in ((torch,'torch'),(np,'numpy'),(transformers,'transformers'),(tokenizers,'tokenizers')):
        require(str(module.__version__)==versions[name],'imported_runtime_version:'+name)
    for module in (torch,np,transformers,tokenizers,jinja2,chat_utils):
        path=Path(module.__file__).resolve(); require(path.is_relative_to(Path(args.workspace)/'envs'),'external_runtime_module')
        watch.add(path)
    return torch,transformers,{'python':platform.python_version(),'interpreter':sys.executable,
      'hostname':socket.gethostname(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'packages':versions,
      'jinja2_version':importlib.metadata.version('Jinja2'),'device':'cpu','torch_imported_for_CPU_token_tensors':True,
      'model_weights_loaded':False,'CUDA_initialized':False}


def execute(args):
    start=time.perf_counter(); os.umask(0o077)
    bootstrap(HERE,args.package_sha256); pins=read(HERE/'expectations.json')
    sampler=load('third_request_sampler',bootstrap(args.sampler_package,pins['sampler_package']['sha256'])/'sampler.py')
    watch=sampler.Watch(); own=sampler.verify_package(HERE,args.package_sha256,watch)
    sampler_binding=sampler.verify_package(args.sampler_package,pins['sampler_package']['sha256'],watch)
    require(sampler_binding==pins['sampler_package'],'sampler_binding')
    out=sampler.safe(args.out); workspace=sampler.safe(args.workspace)
    require(out.is_relative_to(workspace/'qualification') and not out.exists(),'fresh_private_output_required')
    out.mkdir(parents=True,exist_ok=False); output=Output(out,pins['limits']['total_output_bytes'],pins['limits']['metadata_reserve_bytes'])
    stage='binding_preflight'; links={}
    try:
        check_runtime(args,pins)  # before source/vector hashes or tokenizer imports
        helper_binding=sampler.verify_package(args.lock_package,pins['lock_package']['sha256'],watch)
        helper=load('third_request_lock',Path(args.lock_package)/'sampling_lock.py')
        sampler.verify_package(args.protocol_package,pins['protocol_package']['sha256'],watch)
        sampler.verify_package(args.frame_package,pins['frame_package']['sha256'],watch)
        require(read(Path(args.sampler_package)/'expectations.json')==pins['sampler_expectations'],'sampler_expectations_changed')
        plan=read(Path(args.protocol_package)/'PLAN.json'); require(sampler.binding(Path(args.protocol_package)/'PLAN.json')['sha256']==pins['plan_sha256'],'plan_pin')
        historical={}; snapshot=helper.verify_historical_snapshot(args.protocol_evidence,pins['protocol_evidence_sha256'],plan,historical)
        for p,b in historical.items():watch.add(p,b)
        protocol=load('third_request_protocol',Path(args.protocol_package)/'protocol.py'); protocol.validate(plan,args.protocol_evidence)
        plan['sha256']=pins['plan_sha256']
        for name,b in pins['frozen_code'].items():watch.add(sampler.safe(args.repo)/name,b)
        sys.path.insert(0,str(sampler.safe(args.repo)/'empirical_execution'))
        import numpy as np
        from phase3 import calibration as cal,panels
        require(Path(cal.__file__).resolve()==sampler.safe(args.repo)/'empirical_execution/phase3/calibration.py'
          and Path(panels.__file__).resolve()==sampler.safe(args.repo)/'empirical_execution/phase3/panels.py','scorer_import_shadowing')
        args.frame_result_sha256=pins['sampler_expectations']['frame_result']['sha256']
        frame,frame_rows,vectors=sampler.load_frame(args,pins['sampler_expectations'],watch,np,panels)
        result_path=sampler.safe(args.selection_result)
        require(result_path==workspace/'qualification/llm-selection-sampler-v1-20261014-selection-a01/run/output/result.json','selection_path')
        rb=watch.add(result_path); require(rb['sha256']==args.selection_result_sha256,'selection_result_raw_pin')
        result=read(result_path);sampler.verify_seal(result)
        reviewb=watch.add(args.selection_review);require(reviewb['sha256']==args.selection_review_sha256,'selection_review_raw_pin')
        review=read(args.selection_review)
        lockpath=sampler.safe(args.sampling_lock)
        require(lockpath==workspace/'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json','sampling_lock_slot')
        lockb=watch.add(lockpath,pins['sampling_lock']);require(args.sampling_lock_sha256==lockb['sha256'],'sampling_lock_pin')
        validate_selection(result,review,rb,sampler_binding,lockb,pins)
        require({p.name for p in result_path.parent.iterdir()}=={'result.json',*result['files']},'selection_output_inventory')
        for name,b in result['files'].items():watch.add(result_path.parent/name,b)
        evidence=result['lock_evidence']
        expected=helper.contract(frame,sampler_binding,evidence['qualification_result'],evidence['qualification_review'],snapshot)
        lock,_=helper.read_completed_lock(lockpath,expected,helper_binding); watch.add(lockpath.parent/'intent.json')
        for name,key in [('run/output/result.json','qualification_result'),('independent_review.json','qualification_review')]:
            path=workspace/'qualification/llm-selection-sampler-v1-20261014-qualify-a01'/name
            watch.add(path,evidence[key])
            if key=='qualification_result':qualification=read(path);helper.check_qualification(qualification,sampler_binding)
            else:helper.check_review(read(path),evidence['qualification_result'],sampler_binding)
        inputs=read(result_path.parent/'input_bindings.json')
        require(inputs['sampler_package']==sampler_binding and inputs['source_frame_result']==pins['sampler_expectations']['frame_result']
          and inputs['lock_evidence']==evidence and inputs['frame_payloads_rehashed'] is True
          and inputs['original_source_bytes_rehashed'] is True,'selection_input_bindings')
        sample=list(jsonl(result_path.parent/'selection.private.jsonl'));swaps=list(jsonl(result_path.parent/'selection_swap.private.jsonl'))
        stage='retained_sample_and_full_source_checks'
        validate_sample(sample,swaps,result,frame_rows,vectors,protocol,plan,cal,panels,sampler,lock)
        wanted={i for r in sample for i in r['frame_row_indices']}
        source=workspace/pins['sampler_expectations']['source_relative']/'records.jsonl'
        texts=source_texts(source,frame_rows,wanted,frame['full_population_rows'],pins['limits']['jsonl_line_bytes'])
        shared=requests(sample,swaps,texts)
        output.jsonl('texts.private.jsonl',(texts[i] for i in sorted(texts)))
        output.jsonl('requests.private.jsonl',shared)
        output.jsonl('sample.private.jsonl',sample)
        preflight_seconds=time.perf_counter()-start
        stage='CPU_tokenizer_census'
        torch,transformers,rt=runtime(args,pins,sampler,watch)
        system=(Path(args.protocol_evidence)/plan['prompt']['system_path']).read_text()
        task=(Path(args.protocol_evidence)/plan['prompt']['task_path']).read_text()
        summaries={}; assets={}
        for alias,pin in pins['models'].items():
            started=time.perf_counter(); modeldir=getattr(args,alias+'_model'); assetfile=getattr(args,alias+'_assets_manifest')
            verify_assets(modeldir,assetfile,pin,sampler,watch,links)
            pilot=load('third_request_'+alias+'_pilot',Path(args.repo)/pin['pilot_code_relative'])
            tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_fast=True)
            actual={'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__name__,
              'tokenizer_length':len(tokenizer),'tokenizer_vocab_size':tokenizer.vocab_size,
              'tokenizer_special_ids':tokenizer.all_special_ids,'tokenizer_backend_json_sha256':sha(tokenizer.backend_tokenizer.to_str().encode()),
              'chat_template_sha256':template_sha256(tokenizer,pilot)}
            require(actual==pin['tokenizer_identity'],'qualified_tokenizer_identity_changed')
            for cls in type(tokenizer).__mro__:
                if cls.__module__.startswith('transformers'):
                    modulepath=Path(inspect.getfile(cls)).resolve()
                    require(modulepath.is_relative_to(workspace/'envs'),'external_tokenizer_class_module')
                    watch.add(modulepath)
            rows=[]
            def token_rows():
                for request in shared:
                    record=tokenize_request(request,texts,pin,pilot,tokenizer,system,task)
                    # Retain only compact census metadata, never all token tensors/IDs in RAM.
                    rows.append({k:record[k] for k in ('request_index','selection_index','orientation',
                      'diagnostic_only','input_tokens','context_fit_with_output_reserve')})
                    yield record
            output.jsonl(alias+'_tokens.private.jsonl',token_rows())
            summaries[alias]={**summarize_requests(rows),'tokenizer_seconds':time.perf_counter()-started,'tokenizer_identity':actual}
            assets[alias]={'manifest':pin['asset_manifest'],'tokenizer_asset_bytes_rehashed':True,
              'weight_bytes_rehashed':False,'tokenizer_assets':pin['tokenizer_assets']}
            del rows,tokenizer
            require(not torch.cuda.is_initialized(),'CUDA_initialized_during_tokenization')
        output.metadata('census.json',{'schema':'third-llm-request-token-census-v1','stage':'selection',
          'models':summaries,'unique_source_texts':len(texts),'selected_pairs':len(sample),'swap_pairs':len(swaps),
          'model_requests':2*len(shared),'all_sampled_pairs_retained':True,'full_frame_context_feasibility_checked':False})
        stage='finalization'; final_start=time.perf_counter()
        output.metadata('input_bindings.json',{'schema':'third-llm-request-input-bindings-v1','files':watch.files,
          'tokenizer_asset_symlink_targets':links,'historical_evidence_role':'39 historical PLAN-bound files, not the executing checkout',
          'selection_result':rb,'selection_review':reviewb,'sampling_lock':lockb,'source_frame_result':pins['sampler_expectations']['frame_result'],
          'package':own,'model_assets':assets})
        watch.recheck()
        for logical,target in links.items():require(str(Path(logical).resolve(strict=True))==target,'asset_symlink_target_changed')
        for alias,pin in pins['models'].items():
            modeldir=Path(getattr(args,alias+'_model'))
            require({str(p.relative_to(modeldir)) for p in modeldir.rglob('*') if p.is_file()}==set(pin['asset_files']),
              'final_tokenizer_asset_inventory')
        evidence_root=Path(args.protocol_evidence)
        require({str(p.relative_to(evidence_root)) for p in evidence_root.rglob('*') if p.is_file()
          and p!=evidence_root/'snapshot_manifest.json'}==set(plan['bound_files']),'final_historical_snapshot_inventory')
        require({p.name for p in Path(args.frame_dir).iterdir()}=={'result.json',*pins['sampler_expectations']['frame_files']},
          'final_frame_inventory')
        require({p.name for p in result_path.parent.iterdir()}=={'result.json',*result['files']},'final_selection_inventory')
        require({p.name for p in lockpath.parent.iterdir()}=={'intent.json','sampling_lock.json'},'final_lock_inventory')
        for name,b in output.files.items():require(sampler.binding(out/name)==b,'output_changed')
        require({p.name for p in out.iterdir()}==set(output.files),'unexpected_output_inventory')
        result={'schema':'third-llm-annotation-requests-v1','passed':True,'stage':'complete','request_stage':'selection',
          'package':own,'protocol_plan_sha256':pins['plan_sha256'],'selection_result':rb,'selection_review':reviewb,
          'sampling_lock':lockb,'source_frame_result':pins['sampler_expectations']['frame_result'],
          'files':dict(output.files),'runtime':rt,'model_censuses':summaries,'selected_pairs':len(sample),
          'diagnostic_swap_pairs':len(swaps),'unique_source_texts':len(texts),'planned_requests':2*len(shared),
          'complete_sample_identity_weights_and_retained_scores_verified':True,'all_calibration_text_mappings_verified':True,
          'full_source_frame_payload_and_tokenizer_assets_initial_final_equal':True,'all_planned_requests_retained':True,
          'output_payload_bytes_before_result':output.used,'output_cap_bytes':output.limit,
          'preflight_seconds':preflight_seconds,'final_recheck_seconds':time.perf_counter()-final_start,
          'elapsed_seconds_before_result_publication':time.perf_counter()-start,
          'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'model_inference_calls':0,'human_annotation_calls':0,'planned_human_reviews':0,
          'source_scope':'assessed development; current bytes/IDs/texts, no publisher authenticity or legacy scientific admission',
          'next_gate':'Independent request/token artifact review and a new v3 inference activation lock remain required',
          **dict.fromkeys(FALSE_FLAGS,False),'created_utc':datetime.now(timezone.utc).isoformat()}
        result['sha256']=sha(canonical(result));output.metadata('result.json',result)
        fd=os.open(out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return result
    except BaseException as exc:
        failure={'schema':'third-llm-annotation-requests-failure-v1','passed':False,'stage':stage,
          'error_type':type(exc).__name__,'error':str(exc)[:4096],'partial_outputs_usable':False,
          'model_inference_calls':0,'human_annotation_calls':0,**dict.fromkeys(FALSE_FLAGS,False)}
        try:output.metadata('failure.json',failure)
        except Exception:pass
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','workspace','sampler-package','lock-package','protocol-package','protocol-evidence',
                'frame-package','frame-dir','selection-result','selection-review','sampling-lock',
                'qwen-model','qwen-assets-manifest','mistral-model','mistral-assets-manifest','out'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('package-sha256','selection-result-sha256','selection-review-sha256','sampling-lock-sha256'):
        p.add_argument('--'+key,required=True)
    args=p.parse_args()
    try:
        r=execute(args); print(json.dumps({'passed':True,'result_sha256':r['sha256'],'planned_requests':r['planned_requests'],
          'model_inference_calls':0},sort_keys=True));return 0
    except BaseException as e:
        print(json.dumps({'passed':False,'error_type':type(e).__name__,'error':str(e)[:4096]},sort_keys=True),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
