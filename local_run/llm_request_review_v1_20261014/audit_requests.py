#!/usr/bin/env python3
"""Local request terminal audit with existing offline tokenizer replay; no model weights."""
import argparse
import base64
from fractions import Fraction
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
W=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace')
ROOT=W/'qualification/llm-annotation-requests-v2-20261014-selection-a01'
PREP_SHA='bf70b015f39ad0421d8bf66f6b357123f24aceaf9f99b5158fb9b51f490e92d7'
SELECTION_AUDIT_SHA='1980929cd3b9c677b8fe0295580a8956a193cb00513d0fce7111d4fe77047659'
NAMES={'texts.private.jsonl','requests.private.jsonl','sample.private.jsonl','qwen_tokens.private.jsonl',
       'mistral_tokens.private.jsonl','census.json','input_bindings.json','result.json'}
FALSE=('inference_activation','threshold_selected','model_calibration_complete','semantic_validity_established',
 'original_human_gate_satisfied','confirmatory_study_ready','source_authenticity_verified','primary_input_acceptance',
 'original_route_acceptance','full_population_cache_acceptance','requires_human_annotations','weight_bytes_rehashed',
 'model_weights_loaded','model_forwards_executed','full_pair_census_reexecuted','full_frame_context_feasibility_checked',
 'new_sampling_performed','production_entropy_drawn')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value,ascii_only=False):
    return json.dumps(value,ensure_ascii=ascii_only,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def binding(raw):return {'bytes':len(raw),'sha256':digest(raw)}


def strict(raw):
    def pairs(items):
        value={}
        for k,v in items:require(k not in value,'duplicate_json_key');value[k]=v
        return value
    def bad(_):raise ValueError('nonfinite_json')
    def finite(text):
        x=float(text);require(math.isfinite(x),'nonfinite_json_float');return x
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=finite)


def exact(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def read(path,cap=128<<20):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'local_file_cap_or_type')
    before=p.stat();raw=p.read_bytes();after=p.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'changed_local_read')
    return raw


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def lines(raw):
    require(not raw or raw.endswith(b'\n'),'physical_LF_required')
    return [strict(x) for x in raw.split(b'\n')[:-1]]


def seal(value):
    require(value.get('sha256')==digest(canonical({k:v for k,v in value.items() if k!='sha256'})),'internal_seal')


def read_export(path):
    raw=read(path);e=strict(raw)
    require(e.get('root')==str(ROOT) and re.fullmatch('[1-9][0-9]*',str(e.get('job_id','')))
      and not e['queue'].strip(),'export_identity_or_active_queue')
    job=str(e['job_id']);rows=[x.split('|') for x in e['accounting'].splitlines() if x.strip()]
    require(rows and all(len(r)==6 and (r[0]==job or r[0].startswith(job+'.')) and r[1:3]==['COMPLETED','0:0']
      and r[3].isdigit() and r[5]=='csis.cn1' for r in rows),'terminal_accounting')
    require(sum(r[0]==job for r in rows)==1,'exact_terminal_parent')
    files={};total=0
    for name,entry in e['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'export_path')
        content=base64.b64decode(entry['base64'],validate=True)
        require(binding(content)=={k:entry[k] for k in ('bytes','sha256')},'transport_binding')
        total+=len(content);require(total<=80<<20,'decoded_export_cap');files[name]=content
    require(total==e['total_bytes'],'transport_total')
    required={'launch_receipt.json','launch_intent.json','run/process_completion.json','run/software_validation.json',
      'run/tests.txt','run/stdout.txt','run/stderr.txt','slurm-'+job+'.out','slurm-'+job+'.err',
      *{'run/output/'+name for name in NAMES}}
    require(set(files)==required,'exact17file_terminal_inventory')
    require(sum(len(raw) for name,raw in files.items() if name.startswith('run/output/'))<=64<<20,'request_output_cap')
    require(sum(len(raw) for name,raw in files.items() if not name.startswith('run/output/'))<=16<<20,'console_metadata_cap')
    return e,files,binding(raw)


def verify_terminal(e,files,control,controlb,prepb,selectionb,reviewb,pins):
    launch=strict(files['launch_receipt.json']);intent=strict(files['launch_intent.json'])
    proc=strict(files['run/process_completion.json']);tests=strict(files['run/software_validation.json'])
    require(launch['schema']=='third-llm-request-preparation-launch-v1' and launch['root']==str(ROOT)
      and str(launch['job_id'])==str(e['job_id']) and launch['returncode']==0 and launch['control_package']==controlb,'launch_binding')
    require({k:v for k,v in launch.items() if k not in ('returncode','stdout','stderr','job_id')}==intent,'intent_receipt')
    require(launch['tokenizer_only'] is True and all(launch[k] is False for k in
      ('model_weight_reads_authorized','inference_activation','automatic_retry'))
      and launch['model_inference_calls']==launch['human_annotation_calls']==0,'launch_scope')
    expected_bindings={'preparation':prepb,'selection_result':selectionb,'selection_review':reviewb,
      'sampling_lock':pins['sampling_lock'],'qwen_assets':pins['models']['qwen']['asset_manifest'],
      'mistral_assets':pins['models']['mistral']['asset_manifest']}
    for k,v in expected_bindings.items():require(launch['bindings'][k]==v,'launch_input_binding:'+k)
    arg=SimpleNamespace(preparation_package_sha256=PREP_SHA,selection_result_sha256=selectionb['sha256'],
      selection_review_sha256=reviewb['sha256'])
    require(proc['command']==launch['child_command']==control.child_command(arg),'complete_child_command')
    require(proc['returncode']==0 and proc['console_cap_exceeded'] is False and str(proc['slurm_job_id'])==str(e['job_id'])
      and proc['preparation_package_sha256']==PREP_SHA and math.isfinite(proc['child_wall_seconds_through_exit'])
      and proc['child_wall_seconds_through_exit']>0,'process_completion')
    require(proc['console_bytes']==len(files['run/stdout.txt'])+len(files['run/stderr.txt']),'console_bytes')
    require(tests['passed'] is True and tests['tests']==19 and tests['errors']==tests['failures']==tests['skipped']==0,'cluster_tests')
    testlog=files['run/tests.txt'].decode();require('Ran 19 tests in ' in testlog and testlog.rstrip().endswith('OK')
      and testlog.count(' ... ok\n')==19,'raw_test_log')
    remote_control=W/'tools/llm-annotation-request-control-v2-20261014'
    prefix=['sbatch','--parsable','--nodelist=csis.cn1','--job-name=third-llm-requests-selection',
      '--output',str(ROOT/'slurm-%j.out'),'--error',str(ROOT/'slurm-%j.err'),str(remote_control/'requests.sbatch'),str(remote_control)]
    command=launch['command'];require(command[:len(prefix)]==prefix,'submission_prefix')
    tail=command[len(prefix):];require(len(tail)%2==0 and len(set(tail[::2]))==len(tail[::2]),'submission_args_shape')
    require(dict(zip(tail[::2],tail[1::2]))=={'--control-package-sha256':controlb['sha256'],
      '--preparation-package-sha256':PREP_SHA,'--selection-result-sha256':selectionb['sha256'],
      '--selection-review-sha256':reviewb['sha256'],'--expected-tests':'19'},'complete_submission_arguments')
    require(launch['reserve_bytes']==8<<30 and launch['output_allowance_bytes']==128<<20,'capacity_policy')
    return {'job_id':str(e['job_id']),'state':'COMPLETED','exit_code':'0:0','node':'csis.cn1',
      'child_wall_seconds_through_exit':proc['child_wall_seconds_through_exit'],'tests':19,'skips':0}


def texts_and_requests(textrows,requests,sample,swaps):
    mapping={}
    for row in sample:
        for k,i in enumerate(row['frame_row_indices']):
            item=(row['original_full_row_indices'][k],row['record_ids'][k])
            require(i not in mapping or mapping[i]==item,'inconsistent_sample_mapping');mapping[i]=item
    texts={};previous=-1
    for row in textrows:
        require(set(row)=={'calibration_row_index','original_full_row_index','record_id','text_sha256','text'},'text_schema')
        i=row['calibration_row_index'];require(type(i) is int and i>previous and i in mapping,'text_order_or_membership');previous=i
        require((row['original_full_row_index'],row['record_id'])==mapping[i] and type(row['original_full_row_index']) is int,'text_sample_join')
        require(isinstance(row['text'],str) and row['text'] and digest(row['text'].encode())==row['text_sha256'],'exported_text_sha')
        texts[i]=row
    require(set(texts)==set(mapping),'complete_selected_text_coverage')
    work=[(i,'canonical',r['canonical_frame_row_indices']) for i,r in enumerate(sample)]
    work.extend((r['selection_index'],'swapped_diagnostic',r['swapped_frame_row_indices']) for r in swaps)
    expected=[]
    for index,(i,orientation,shown) in enumerate(work):
        row=sample[i];expected.append({'request_index':index,'selection_index':i,
          'stage':'selection' if orientation=='canonical' else 'selection_swap','orientation':orientation,
          'diagnostic_only':orientation!='canonical','opaque_pair_id':row['opaque_pair_id'],
          'natural_pair_id':row['natural_pair_id'],'shown_calibration_row_indices':shown,
          'shown_record_ids':[texts[x]['record_id'] for x in shown],
          'shown_text_sha256':[texts[x]['text_sha256'] for x in shown],'model_seeds':row['model_seeds']})
    require(exact(requests,expected),'complete_request_identity_orientation_schema')
    return texts


def local_tokenizer(path,pin,watched):
    from transformers import AutoTokenizer
    folder=Path(path).resolve();files={p.name:p for p in folder.iterdir() if p.is_file()}
    required={'config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}
    if 'special_tokens_map.json' in pin['tokenizer_assets']:required.add('special_tokens_map.json')
    require(required<=set(files) and set(files)<=set(pin['tokenizer_assets'])|{'verified_download_receipt.json'},'local_tokenizer_inventory')
    for name,path in files.items():
        if name=='verified_download_receipt.json':continue
        raw=read(path,16<<20);require(binding(raw)==pin['tokenizer_assets'][name],'local_tokenizer_asset_binding')
        watched[str(path)]=binding(raw)
    require(not strict(read(files['config.json'])).get('auto_map') and not strict(read(files['tokenizer_config.json'])).get('auto_map'),
      'custom_tokenizer_code')
    tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False,use_fast=True)
    identity={'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__name__,'tokenizer_length':len(tokenizer),
      'tokenizer_vocab_size':tokenizer.vocab_size,'tokenizer_special_ids':tokenizer.all_special_ids,
      'tokenizer_backend_json_sha256':digest(tokenizer.backend_tokenizer.to_str().encode()),
      'chat_template_sha256':digest(canonical(tokenizer.chat_template,True))}
    require(identity==pin['tokenizer_identity'],'qualified_local_tokenizer_identity')
    return tokenizer


def verify_model(tokens,requests,texts,pin,tokenizer,system,task):
    require(len(tokens)==len(requests),'model_request_count');numbers=[];vocabulary_length=len(tokenizer)
    for request,row in zip(requests,tokens):
        ids=row.get('input_token_ids');require(isinstance(ids,list) and ids and all(type(i) is int and 0<=i<vocabulary_length for i in ids),'token_ID_schema_range')
        a,b=(texts[i]['text'] for i in request['shown_calibration_row_indices'])
        inputrow={'pair_id':request['opaque_pair_id'],'orientation':request['orientation'],'text_a':a,'text_b':b}
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(
          {'task_definition':task,'documents':[{'label':'A','text':a},{'label':'B','text':b}]},ensure_ascii=False)}]
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        # Independent list encoding: no torch import, CPU tensors, pilot helper or preparation code.
        encoded=tokenizer(rendered,add_special_tokens=False,truncation=False)
        require(set(encoded)=={'input_ids','attention_mask'} and encoded['input_ids']==ids
          and encoded['attention_mask']==[1]*len(ids),'independent_full_tokenizer_replay')
        fit=len(ids)+512<=32768
        expected={**request,'model_id':pin['model_id'],'model_revision':pin['revision'],
          'model_seed':request['model_seeds'][pin['model_id']], 'input_row_sha256':digest(canonical(inputrow,True)),
          'input_row_hash_serialization':'qualified pilot canonical JSON: ensure_ascii=True, sorted keys, compact separators',
          'rendered_prompt_sha256':digest(rendered.encode()),'input_token_ids':ids,'input_tokens':len(ids),
          'input_token_ids_sha256':digest(canonical(ids)),'attention_mask_policy':'one for every unpadded token',
          'max_new_tokens':512,'total_token_budget':32768,'context_fit_with_output_reserve':fit,
          'preparation_status':'ready_for_future_activation' if fit else 'context_overflow',
          'full_text_preserved':True,'truncation_applied':False,'activation_lock_sha256':None,'call_id':None}
        require(exact(row,expected),'request_token_envelope_or_prompt_hash')
        numbers.append((request,len(ids),fit))
    schedule=sorted(numbers,key=lambda x:(-x[1],x[0]['selection_index'],x[0]['orientation']!='canonical'))
    return {'requests':len(tokens),'canonical_requests':sum(not r['diagnostic_only'] for r,_,_ in numbers),
      'diagnostic_requests':sum(r['diagnostic_only'] for r,_,_ in numbers),'total_input_tokens':sum(n for _,n,_ in numbers),
      'max_input_tokens':max((n for _,n,_ in numbers),default=0),'overflow_requests':sum(not fit for _,_,fit in numbers),
      'canonical_overflows':sum(not r['diagnostic_only'] and not fit for r,_,fit in numbers),
      'scheduled_request_indices':[r['request_index'] for r,_,_ in schedule],
      'schedule_rule':'input_tokens descending, selection_index ascending, canonical before diagnostic tie',
      'all_planned_requests_retained':True,'overflow_retains_full_sampling_denominator':True,
      'tokenizer_identity':pin['tokenizer_identity']}


def audit(args):
    started=time.perf_counter();repo=Path(args.repo).resolve()
    localinputs={str(Path(x).resolve()):binding(read(x)) for x in
      (args.terminal_export,args.selection_export,args.selection_review,args.sampling_lock,__file__)}
    helperpath=HERE/'audit_full_selection.py';require(digest(read(helperpath))==SELECTION_AUDIT_SHA,'independent_selection_auditor_pin')
    helper=load('request_review_independent_selection',helperpath)
    folder=repo/'local_run/llm_annotation_requests_v2_20261014';prepb=helper.package(folder,PREP_SHA)
    controlfolder=repo/'local_run/llm_annotation_request_control_v2_20261014';controlb=helper.package(controlfolder,args.control_package_sha256)
    control=load('request_review_control',controlfolder/'control.py');pins=strict(read(folder/'expectations.json'))
    s=pins['sampler_expectations'];planpath=repo/'local_run/model_annotation_v3_20261014/PLAN.json'
    require(digest(read(planpath))==pins['plan_sha256'],'protocol_plan');plan=strict(read(planpath));localinputs[str(planpath)]=binding(read(planpath))
    for packagefolder in (folder,controlfolder):
        pm=packagefolder/'package_manifest.json';localinputs[str(pm)]=binding(read(pm))
        for name in strict(read(pm))['files']:
            member=packagefolder/name;localinputs[str(member)]=binding(read(member))
    localinputs[str(helperpath)]=binding(read(helperpath))
    sx,sfiles,sxb=helper.read_export(args.selection_export)
    sr=sfiles['run/output/result.json'];selection=strict(sr);seal(selection);sb=binding(sr)
    reviewraw=read(args.selection_review);review=strict(reviewraw);reviewb=binding(reviewraw)
    require(review['passed'] is True and review['schema']=='third-llm-selection-metadata-identity-review-v1'
      and review['selection_result']==sb and review['terminal_export']==sxb,'upstream_selection_review')
    lockraw=read(args.sampling_lock);lock=strict(lockraw);seal(lock)
    require(binding(lockraw)==pins['sampling_lock'],'production_sampling_lock')
    sample=lines(sfiles['run/output/selection.private.jsonl']);swaps=lines(sfiles['run/output/selection_swap.private.jsonl'])
    checked=helper.verify_sample(selection,sample,swaps,lock,plan)
    e,files,eb=read_export(args.terminal_export)
    terminal=verify_terminal(e,files,control,controlb,prepb,sb,reviewb,pins)
    result_raw=files['run/output/result.json'];result=strict(result_raw);seal(result)
    require(result['schema']=='third-llm-annotation-requests-v1' and result['passed'] is True
      and result['stage']=='complete' and result['request_stage']=='selection','request_completion')
    for key,wanted in {'package':prepb,'protocol_plan_sha256':pins['plan_sha256'],'selection_result':sb,
      'selection_review':reviewb,'sampling_lock':pins['sampling_lock'],'source_frame_result':s['frame_result'],
      'selected_pairs':len(sample),'diagnostic_swap_pairs':len(swaps),'planned_requests':2*(len(sample)+len(swaps)),
      'output_cap_bytes':64<<20,'model_inference_calls':0,'human_annotation_calls':0,'planned_human_reviews':0}.items():
        require(exact(result.get(key),wanted),'result_binding_or_count:'+key)
    for key in FALSE:require(result.get(key) is False,'request_false_claim:'+key)
    for key in ('complete_sample_identity_weights_and_retained_scores_verified','all_calibration_text_mappings_verified',
      'full_source_frame_payload_and_tokenizer_assets_initial_final_equal','all_planned_requests_retained'):
        require(result.get(key) is True,'missing_execution_attestation:'+key)
    require(set(result['files'])==NAMES-{'result.json'},'result_artifact_inventory')
    for name,wanted in result['files'].items():require(binding(files['run/output/'+name])==wanted,'output_artifact_hash')
    require(exact(lines(files['run/output/sample.private.jsonl']),sample),'unchanged_sample_rows')
    require(result['output_payload_bytes_before_result']==sum(x['bytes'] for x in result['files'].values()),'result_output_bytes')
    requests=lines(files['run/output/requests.private.jsonl']);textrows=lines(files['run/output/texts.private.jsonl'])
    texts=texts_and_requests(textrows,requests,sample,swaps)
    require(result['unique_source_texts']==len(texts),'unique_text_count')
    rt=result['runtime'];require(rt['python']==pins['runtime']['python'] and rt['interpreter']==str(W/pins['runtime']['interpreter_relative'])
      and rt['hostname']=='csis.cn1' and str(rt['slurm_job_id'])==str(e['job_id']) and rt['packages']==pins['runtime']['package_versions']
      and rt['device']=='cpu' and rt['torch_imported_for_CPU_token_tensors'] is True
      and rt['model_weights_loaded'] is False and rt['CUDA_initialized'] is False,'execution_runtime_identity')
    inputs=strict(files['run/output/input_bindings.json']);require(inputs['package']==prepb and inputs['selection_result']==sb
      and inputs['selection_review']==reviewb and inputs['sampling_lock']==pins['sampling_lock']
      and inputs['source_frame_result']==s['frame_result'],'input_manifest_binding')
    watched=inputs['files'];require(isinstance(watched,dict) and watched,'execution_watched_files')
    required={str(W/s['source_relative']/'records.jsonl'):s['frame_inherited']['prepared_source_records_file'],
      str(W/s['source_relative']/'audit.json'):s['frame_inherited']['prepared_source_audit_file'],
      str(W/s['frame_relative']/'result.json'):s['frame_result'],str(W/'tools/llm-annotation-requests-v2-20261014/package_manifest.json'):prepb,
      str(W/'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json'):pins['sampling_lock'],
      str(W/'qualification/llm-selection-sampler-v1-20261014-selection-a01/run/output/result.json'):sb,
      str(W/'qualification/llm-selection-sampler-v1-20261014-selection-a01/independent_review.json'):reviewb}
    required.update({str(W/s['frame_relative']/name):v for name,v in s['frame_files'].items()})
    for name,value in required.items():require(watched.get(name)==value,'execution_input_path_binding:'+name)
    for name,value in pins['frozen_code'].items():require(watched.get(str(control.REPO/name))==value,'executing_frozen_code_pin')
    for alias,pin in pins['models'].items():
        asset=inputs['model_assets'][alias]
        require(asset=={'manifest':pin['asset_manifest'],'tokenizer_asset_bytes_rehashed':True,'weight_bytes_rehashed':False,
          'tokenizer_assets':pin['tokenizer_assets']},'execution_asset_scope')
        base=control.QWEN if alias=='qwen' else control.MISTRAL
        for name,value in pin['tokenizer_assets'].items():
            resolved=inputs['tokenizer_asset_symlink_targets'].get(str(base/name))
            require(resolved and watched.get(resolved)==value,'resolved_tokenizer_asset_binding')
    # Use already-local tokenizer-only assets; no download, model loading or torch import.
    for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
      'USE_TORCH':'0','USE_TF':'0','USE_FLAX':'0','CUDA_VISIBLE_DEVICES':''}.items():os.environ[k]=v
    require('torch' not in sys.modules,'torch_free_local_replay_required')
    if args.local_renderer_site:
        renderer_site=Path(args.local_renderer_site).resolve()
        require(renderer_site.is_dir(),'local_renderer_site_absent')
        sys.path.append(str(renderer_site))
    import transformers,tokenizers,jinja2,markupsafe
    require(transformers.__version__=='4.57.6' and tokenizers.__version__=='0.22.2','local_tokenizer_runtime')
    localwatch={};summaries={};renderwatch={}
    for module in (jinja2,markupsafe):
        for path in Path(module.__file__).parent.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                require(not path.is_symlink(),'renderer_runtime_symlink')
                renderwatch[str(path)]=binding(read(path,16<<20))
    system=read(repo/plan['prompt']['system_path']).decode();task=read(repo/plan['prompt']['task_path']).decode()
    for key in ('system_path','task_path'):
        path=repo/plan['prompt'][key];require(binding(read(path))==plan['bound_files'][plan['prompt'][key]],'fixed_prompt_binding')
        localinputs[str(path)]=binding(read(path))
    for alias,pin in pins['models'].items():
        tokenizer=local_tokenizer(getattr(args,alias+'_tokenizer'),pin,localwatch)
        rows=lines(files['run/output/'+alias+'_tokens.private.jsonl'])
        summaries[alias]=verify_model(rows,requests,texts,pin,tokenizer,system,task)
        observed=result['model_censuses'][alias]
        require(set(observed)==set(summaries[alias])|{'tokenizer_seconds'},'census_schema')
        require(all(exact(observed[k],v) for k,v in summaries[alias].items()) and math.isfinite(observed['tokenizer_seconds'])
          and observed['tokenizer_seconds']>=0,'recomputed_census_and_schedule')
    census=strict(files['run/output/census.json'])
    require(exact(census,{'schema':'third-llm-request-token-census-v1','stage':'selection','models':result['model_censuses'],
      'unique_source_texts':len(texts),'selected_pairs':len(sample),'swap_pairs':len(swaps),
      'model_requests':2*len(requests),'all_sampled_pairs_retained':True,'full_frame_context_feasibility_checked':False}),'census_artifact')
    for name,wanted in localwatch.items():require(binding(read(name,16<<20))==wanted,'local_tokenizer_initial_final_change')
    for name,wanted in renderwatch.items():require(binding(read(name,16<<20))==wanted,'renderer_runtime_changed')
    for name,wanted in localinputs.items():require(binding(read(name))==wanted,'local_audit_input_initial_final_change')
    for alias,pin in pins['models'].items():
        local_tokenizer(getattr(args,alias+'_tokenizer'),pin,{})
    require('torch' not in sys.modules,'unexpected_torch_import')
    return {'schema':'third-llm-request-independent-token-review-v1','passed':True,'request_stage':'selection',
      'request_result':binding(result_raw),'request_result_internal_seal':result['sha256'],'request_package':prepb,
      'request_control_package':controlb,'terminal_export':eb,'terminal':terminal,'selection_result':sb,'selection_review':reviewb,
      'sampling_lock':pins['sampling_lock'],'source_frame_result':s['frame_result'],'checked_sample':checked,
      'prepared_files':result['files'],'model_censuses':summaries,'all_request_text_joins_and_hashes_checked':True,
      'all_input_token_IDs_and_rendered_prompt_hashes_independently_replayed':True,'complete_canonical_and_swap_coverage':True,
      'overflow_retained_in_full_denominator':True,'source_full_bytes_independently_rehashed':False,
      'complete_source_frame_mapping_independently_replayed':False,'retained_vector_scores_independently_replayed_here':False,
      'full_selection_census_independently_replayed':False,'remote_initial_final_hashes_independently_reexecuted':False,
      'execution_attestations_required_and_bound':['full source/frame hashes','all calibration row/text mapping',
        'retained pair vector scores','initial/final code/source/tokenizer equality','final remote inventories'],
      'local_replay_runtime':{'python':platform.python_version(),'transformers':transformers.__version__,
        'tokenizers':tokenizers.__version__,'jinja2':jinja2.__version__,
        'markupsafe':importlib.metadata.version('markupsafe'),'renderer_site':str(args.local_renderer_site) if args.local_renderer_site else None,
        'torch_imported':False,'model_weights_loaded':False},
      'local_renderer_files':renderwatch,'local_tokenizer_files':localwatch,'local_audit_input_bindings':localinputs,
      'local_input_and_tokenizer_initial_final_equal':True,'elapsed_seconds':time.perf_counter()-started,
      'audit_source':binding(read(__file__)),'independent_selection_auditor':binding(read(helperpath)),
      'inference_activation':False,'new_activation_still_required':True,'model_inference_calls':0,'human_annotation_calls':0,
      'semantic_validity_established':False,'source_authenticity_verified':False,'confirmatory_study_ready':False,
      'scope':'Local exported-artifact plus complete offline prompt/token replay. Original full source bytes, frame vector checks and remote initial/final equality remain bound execution evidence; no source-authenticity or semantic/human claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','terminal-export','selection-export','selection-review','sampling-lock','qwen-tokenizer','mistral-tokenizer','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--local-renderer-site',type=Path)
    p.add_argument('--control-package-sha256',required=True);args=p.parse_args()
    require(not args.out.exists() and not args.out.is_symlink(),'fresh_audit_output_required')
    try:report=audit(args)
    except Exception as exc:report={'schema':'third-llm-request-independent-token-review-v1','passed':False,
      'error_type':type(exc).__name__,'error':str(exc),'inference_activation':False,'model_inference_calls':0,'human_annotation_calls':0}
    with args.out.open('x') as f:f.write(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':report['passed'],'review':str(args.out)}));return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
