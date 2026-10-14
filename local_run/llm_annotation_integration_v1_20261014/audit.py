"""Independent selection-output audit. No model loading, output edits or logit claims."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import time
from common import (Watch,binding,canonical,jsonl,load,new_json,package,pins,read,require,seal,sealed,sha,strict)
from activation import read_activation

FLAGS={'flash':True,'math':False,'mem_efficient':False,'cudnn':False}
CALL_KEYS={'stage','opaque_pair_id','model_id','model_revision','orientation','attempt','call_id','model_seed','request_index','selection_index','request_manifest','request_row_sha256','input_token_ids_sha256','activation_lock_sha256'}
RESPONSE_EXTRA={'schema','annotator_type','human_completed','status','judgment','raw_response_sha256','runtime_manifest_sha256','artifacts','error'}
STATUSES={'valid','invalid_json','context_overflow','runtime_error','not_attempted'}
def domain(name,values):return sha(canonical(['third-llm-only-v3/'+name,*values]))
def planned_calls(tokens,request_binding,activation_sha,model,cp):
    calls=[];seen=set()
    for i,t in enumerate(tokens):
        require(type(t['request_index']) is int and t['request_index']==i,'request_index')
        require((t['stage'],t['orientation']) in {('selection','canonical'),('selection_swap','swapped_diagnostic')},'phase_orientation')
        require(t['model_id']==model['model_id'] and t['model_revision']==model['revision'],'request_model')
        seed=int(domain('model-seed',[cp['protocol_plan_sha256'],cp['frame_result_sha256'],t['opaque_pair_id'],model['model_id'],model['revision']]),16)%(2**63)
        require(type(t['model_seed']) is int and t['model_seed']==seed,'request_seed')
        key=[t['stage'],t['opaque_pair_id'],t['model_id'],t['model_revision'],t['orientation'],0]
        cid=domain('call-id',[activation_sha,*key]);require(cid not in seen,'duplicate_call');seen.add(cid)
        calls.append({k:t[k] for k in ('stage','opaque_pair_id','model_id','model_revision','orientation','model_seed','request_index','selection_index','input_token_ids_sha256')}|
          {'attempt':0,'call_id':cid,'request_manifest':request_binding,'request_row_sha256':sha(canonical(t)),'activation_lock_sha256':activation_sha})
    return calls

def check_responses(responses,calls,order):
    require(type(responses) is list and len(responses)==len(calls),'response_denominator')
    for r,c in zip(responses,calls):
        require(set(r)==CALL_KEYS|RESPONSE_EXTRA,'response_fields')
        require(all(type(r[k]) is type(c[k]) and r[k]==c[k] for k in CALL_KEYS),'response_identity')
        require(r['schema']=='third-llm-model-response-v3' and r['annotator_type']=='model' and r['human_completed'] is False,'model_only_response')
        require(r['status'] in STATUSES,'unaccepted_or_incomplete_status')
        require(r['judgment'] is None if r['status']!='valid' else type(r['judgment']) is dict,'status_judgment')
        require(r['error'] is None or (type(r['error']) is dict and set(r['error'])=={'type','message'} and all(type(v) is str and v for v in r['error'].values())),'error_shape')
        if r['status'] in ('invalid_json','runtime_error','not_attempted'):require(r['error'] is not None,'missing_failure_evidence')
        if r['status']=='valid':require(r['error'] is None,'valid_error')
    stop=False
    for i in order:
        status=responses[i]['status']
        require(status=='not_attempted' if stop else status!='not_attempted','stop_on_first_runtime_error')
        if status=='runtime_error':stop=True
    return Counter(r['status'] for r in responses)

def check_guard_steps(steps,metrics,status,ids=None):
    require(type(steps) is list and len(steps)<=513,'guard_step_count')
    accepted=[]
    for i,s in enumerate(steps):
        require(set(s)=={'step','accepted_token_id','candidates_considered','finite_native_candidates','rejected_candidates','prefix_mask_verified','selection_seconds'},'guard_step_fields')
        failed=s['accepted_token_id'] is None
        require(not failed or (status=='runtime_error' and i==len(steps)-1),'terminal_guard_failure')
        require(type(s['step']) is int and s['step']==i and s['prefix_mask_verified'] is True,'guard_step_order')
        for k in ('candidates_considered','finite_native_candidates'):require(type(s[k]) is int and s[k]>=0,'guard_count')
        require((0 if failed else 1)<=s['candidates_considered']<=s['finite_native_candidates'],'guard_candidates')
        if failed:require(s['candidates_considered']==s['finite_native_candidates'],'exhausted_guard_count')
        else:require(type(s['accepted_token_id']) is int and s['accepted_token_id']>=0,'guard_token');accepted.append(s['accepted_token_id'])
        require(type(s['rejected_candidates']) is dict and all(type(v) is int and v>0 for v in s['rejected_candidates'].values()),'guard_rejections')
        require(sum(s['rejected_candidates'].values())==s['candidates_considered']-(0 if failed else 1),'guard_rejected_denominator')
        require(type(s['selection_seconds']) in (int,float) and math.isfinite(s['selection_seconds']) and s['selection_seconds']>=0,'guard_seconds')
    require(metrics['guard_steps']==len(steps) and metrics['guard_candidates_considered']==sum(s['candidates_considered'] for s in steps) and metrics['guard_selection_seconds']==sum(s['selection_seconds'] for s in steps),'guard_aggregates')
    if ids is not None:require(accepted==ids and all(s['accepted_token_id'] is not None for s in steps),'guard_selected_tokens')
    return accepted

def tokenizer_data(tokenizer):
    """Reconstruct pinned LMFE0.11.3 token table without importing torch integration."""
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    n=len(tokenizer);special=set(tokenizer.all_special_ids);zero=tokenizer.encode('0')[-1];regular=[]
    for i in range(n):
        if i in special:continue
        after=tokenizer.decode([zero,i])[1:];alone=tokenizer.decode([i]);regular.append((i,after,len(after)>len(alone)))
    return TokenEnforcerTokenizerData(regular,lambda ids:tokenizer.decode(ids).rstrip('\ufffd'),tokenizer.eos_token_id,False,n)

def load_tokenizer(alias,path,repo,runtime,inventory,effective,w):
    for name,version in {'transformers':'4.57.6','tokenizers':'0.22.2','lm-format-enforcer':'0.11.3'}.items():require(importlib.metadata.version(name)==version,'audit_dependency:'+name)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    p=pins();model=p['component_expectations']['models'][alias];assetpins=model['asset_files'];links={}
    for name,want in assetpins.items():
        if name.endswith('.safetensors'):continue
        logical=Path(path)/name
        if not logical.exists():continue
        resolved=logical.resolve(strict=True);w.file(resolved,want,32*1024**2);links[str(logical)]=str(resolved)
    require(all((Path(path)/name).is_file() for name in ('config.json','tokenizer.json','tokenizer_config.json')),'audit_tokenizer_assets')
    t=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=True)
    actual={'tokenizer_class':type(t).__module__+'.'+type(t).__name__,'tokenizer_length':len(t),'tokenizer_vocab_size':t.vocab_size,
      'tokenizer_special_ids':t.all_special_ids,'tokenizer_backend_json_sha256':sha(t.backend_tokenizer.to_str().encode()),'chat_template_sha256':sha(canonical(t.chat_template,ascii=True))}
    require(actual==model['tokenizer_identity'] and all(runtime[k]==v for k,v in actual.items()),'audit_tokenizer_identity')
    helper_pin=p['prior_audit_helpers'][alias];w.file(Path(repo)/helper_pin['relative_path'],helper_pin['binding']);helper=load('third_output_audit_helper_'+alias,Path(repo)/helper_pin['relative_path'])
    guard_rel=str(Path(model['pilot_code_relative']).parent/'bounded_reason.py');w.file(Path(repo)/guard_rel,p['component_expectations']['source_files'][guard_rel]);guard=load('third_output_guard_'+alias,Path(repo)/guard_rel)
    schema_rel='local_run/model_annotation_v2_20261011/model_judgment.schema.json';w.file(Path(repo)/schema_rel,p['component_expectations']['source_files'][schema_rel]);schema=read(Path(repo)/schema_rel)
    alphabet=helper.verify_effective_constraint(effective,runtime['constraint'],schema,p['component_expectations']['source_files'][guard_rel]['sha256'])
    data=tokenizer_data(t);require(data.tokenizer_alphabet==alphabet,'reconstructed_alphabet')
    rawdecode=lambda ids:t.decode(list(ids),skip_special_tokens=False,clean_up_tokenization_spaces=False)
    eos=data.eos_token_id if type(data.eos_token_id) is list else [data.eos_token_id]
    eligible,eos=helper.verify_token_eligibility(inventory,data.tokenizer_tree.tokens_to_strs,eos,rawdecode)
    policy=guard.StandaloneTokenPolicy(eligible,eos,rawdecode)
    return {'tokenizer':t,'data':data,'guard':guard,'helper':helper,'schema':schema,'policy':policy,'alphabet':alphabet,'rawdecode':rawdecode,'eligible':eligible,'eos':eos,'asset_links':links}

def parse_and_replay(raw,ids,input_ids,ctx):
    from jsonschema import Draft202012Validator
    errors={};judgment=None
    try:
        judgment=strict(raw);Draft202012Validator(ctx['schema']).validate(judgment)
        require(len(judgment['reason'].split())<=80,'reason_word_limit')
    except Exception as exc:errors['strict_json_schema_words']=type(exc).__name__;judgment=None
    require(ctx['tokenizer'].decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==raw,'raw_token_decode_identity')
    try:
        emitted=ctx['helper'].replay_raw_token_prefixes(ids,ctx['data'],ctx['eligible'],ctx['eos'],ctx['rawdecode']);require(emitted==raw,'full_decode_prefix')
        state=ctx['guard'].replay_guard_tokens(input_ids,ids,ctx['schema'],ctx['data'],ctx['policy']);require(state.emitted_text==raw,'guard_emitted_text')
    except (ValueError,TypeError) as exc:errors['token_prefix_guard']=type(exc).__name__
    try:ctx['guard'].replay(raw,ctx['schema'],ctx['alphabet'])
    except (ValueError,TypeError) as exc:errors['character_grammar']=type(exc).__name__
    return judgment,errors

def audit_run(root,repo,admission_path,admission_binding,activation_dir,activation_binding,alias,tokenizer_path):
    started=time.perf_counter();activation,state=read_activation(admission_path,admission_binding,activation_dir,activation_binding)
    w=state['watch'];rdir=Path(root);resources=state['resource_plan'];ap=state['admission']['paths'];cp=pins()['component_expectations'];model=cp['models'][alias]
    result=read(rdir/'result.json');rb=w.file(rdir/'result.json');seal(result)
    require(result['schema']=='third-llm-annotation-execution-v1' and result['passed_execution_integrity'] is True and result['stage']=='complete','execution_integrity_required')
    require(result['model_alias']==alias and result['model_id']==model['model_id'] and result['model_revision']==model['revision'],'execution_model')
    for key,want in {'activation':activation_binding,'request_manifest':activation.contract['request_manifest'],'runner_package':resources['packages']['runner'],'component_package':resources['packages']['component'],'integration_package':resources['packages']['integration'],'resource_plan':activation.contract['resource_plan']}.items():require(result[key]==want,'execution_binding:'+key)
    require(result['artifact_integrity_error'] is None and result['model_loaded'] is True and result['initial_final_bindings_equal'] is True and result['retry_calls']==0 and result['human_annotation_calls']==0 and result['human_completed'] is False,'execution_gates')
    basefiles={'planned_calls.json','run_intent.json','input_bindings.json','responses.json','ledger.json','final_bindings.json'}
    runtimefiles={'runtime/'+x for x in ('generation_config.json','token_eligibility.json','effective_constraint_config.json','runtime_manifest.json')}
    allowed=basefiles|runtimefiles|({'failure.json'} if result['ordinary_fatal'] is not None else set())
    require(allowed<=set(result['files']) and all(n in allowed or n.startswith('calls/') for n in result['files']),'exact_root_artifact_roles')
    for name,b in result['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and name!='result.json','output_member');w.file(rdir/name,b)
    w.inventory(rdir,set(result['files'])|{'result.json'})
    require(sum(b['bytes'] for b in result['files'].values())==result['output_bytes_before_result'],'output_bytes')
    require(result['output_bytes_before_result']+rb['bytes']<=resources['families'][alias]['output_limit_bytes'],'output_cap')
    requests=Path(ap['request_dir']);tokens=jsonl(requests/(alias+'_tokens.private.jsonl'));shared=jsonl(requests/'requests.private.jsonl');texts={x['calibration_row_index']:x for x in jsonl(requests/'texts.private.jsonl')}
    require(len(tokens)==len(shared)==resources['planned']['per_model'],'all_requests')
    calls=planned_calls(tokens,activation.contract['request_manifest'],activation_binding['sha256'],model,cp)
    require(read(rdir/'planned_calls.json')==calls,'exact_planned_calls')
    order=sorted(range(len(tokens)),key=lambda i:(-tokens[i]['input_tokens'],tokens[i]['selection_index'],tokens[i]['orientation']!='canonical'))
    intent=read(rdir/'run_intent.json');require(intent['execution_request_indices']==order and intent['activation']==activation_binding and intent['runner_package']==resources['packages']['runner'] and intent['no_retries'] is True and intent['stop_on_runtime_error'] is True,'run_intent')
    responses=read(rdir/'responses.json');counts=check_responses(responses,calls,order);ledger=read(rdir/'ledger.json')
    require(ledger['planned_calls']==ledger['recorded_calls']==len(calls) and ledger['status_counts']==dict(counts)==result['status_counts'] and ledger['complete_planned_denominator'] is True and ledger['retry_calls']==0,'ledger_denominator')
    stopped=bool(counts['runtime_error']);require(result['passed'] is (not stopped) and result['completed'] is (not stopped) and (result['ordinary_fatal'] is not None) is stopped and ledger['ordinary_fatal']==result['ordinary_fatal'],'ordinary_stop_accounting')
    if stopped:
        failure=read(rdir/'failure.json');require(failure['ordinary_fatal']==result['ordinary_fatal'] and failure['artifact_integrity_error'] is None and failure['all_planned_calls_retained'] is True and failure['retry_allowed'] is False,'failure_receipt')
    require(result['planned_calls']==result['recorded_calls']==len(calls) and result['canonical_calls']==resources['planned']['canonical_pairs'] and result['diagnostic_calls']==resources['planned']['swap_pairs'],'result_denominator')
    ib=read(rdir/'input_bindings.json');fb=read(rdir/'final_bindings.json')
    require(fb['passed'] is True and fb['inputs_initial_final_equal'] is True and fb['model_loaded'] is True and fb['full_model_weight_bytes_rehashed'] is True,'final_binding_gates')
    require(fb['input_files_final']==ib['files'] and fb['model_code_assets_initial']==fb['model_code_assets_final'],'final_binding_equality')
    require(fb['output_inventory_before_final_binding']=={k:v for k,v in result['files'].items() if k not in ('final_bindings.json','failure.json')},'final_output_inventory_receipt')
    # Verify every weight is among the equality-bound loader attestations, without claiming an independent second model-weight read.
    weights=fb['model_code_assets_final']
    for name,b in model['asset_files'].items():
        candidates=[v for k,v in weights.items() if k.endswith('/'+model['revision']+'/'+name)]
        require(len(candidates)==1 and candidates[0]['binding']==b,'complete_weight_identity_attestation:'+name)
    runtime=read(rdir/'runtime/runtime_manifest.json');rtb=result['files']['runtime/runtime_manifest.json']
    require(runtime['sdpa_flags_inside_generation_context']==FLAGS and runtime['dtype']=='bfloat16' and runtime['visible_gpus']==1 and runtime['deterministic_algorithms'] is True and runtime['allow_tf32'] is False,'runtime_numerical_policy')
    # Qwen's unchanged loader omits huggingface-hub; the component preflight
    # checks its installed version separately. Never invent an old receipt field.
    require(runtime['packages']==pins()['family_runtime_packages'][alias],'runtime_packages')
    generation=read(rdir/'runtime/generation_config.json')
    for key,want in {'do_sample':False,'num_beams':1,'max_new_tokens':512,'use_cache':True,'repetition_penalty':1.0,'output_attentions':False,'forced_bos_token_id':None,'forced_eos_token_id':None,'remove_invalid_values':False,'renormalize_logits':False,'num_return_sequences':1}.items():require(generation.get(key)==want,'generation_config:'+key)
    for name,key in [('generation_config.json','generation_config_sha256'),('token_eligibility.json','token_eligibility_sha256'),('effective_constraint_config.json','effective_constraint_config_sha256')]:require(result['files']['runtime/'+name]['sha256']==runtime[key],'runtime_artifact_binding')
    ctx=load_tokenizer(alias,tokenizer_path,repo,runtime,read(rdir/'runtime/token_eligibility.json'),read(rdir/'runtime/effective_constraint_config.json'),w)
    system=(Path(repo)/'local_run/model_annotation_v2_20261011/judge_system_prompt.txt').read_text();task=(Path(repo)/'local_run/model_annotation_v2_20261011/judge_task_definition.txt').read_text()
    observed=[];diagnostics=[];started_ids=set();replays=[]
    for i,(response,call,token,request) in enumerate(zip(responses,calls,tokens,shared)):
        require(all(token[k]==v for k,v in request.items()),'token_shared_row');inds=request['shown_calibration_row_indices'];a,b=[texts[j] for j in inds]
        require([x['record_id'] for x in (a,b)]==request['shown_record_ids'] and [sha(x['text'].encode()) for x in (a,b)]==request['shown_text_sha256'],'prompt_text_joins')
        row={'pair_id':request['opaque_pair_id'],'orientation':request['orientation'],'text_a':a['text'],'text_b':b['text']}
        require(sha(canonical(row,ascii=True))==token['input_row_sha256'],'input_row_hash')
        messages=[{'role':'system','content':system},{'role':'user','content':__import__('json').dumps({'task_definition':task,'documents':[{'label':'A','text':a['text']},{'label':'B','text':b['text']}]},ensure_ascii=False)}]
        rendered=ctx['tokenizer'].apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=ctx['tokenizer'](rendered,add_special_tokens=False,truncation=False)['input_ids']
        require(ids==token['input_token_ids'] and len(ids)==token['input_tokens'] and sha(canonical(ids))==token['input_token_ids_sha256'] and sha(rendered.encode())==token['rendered_prompt_sha256'],'independent_prompt_tokens')
        callpath=rdir/'calls'/call['call_id'];status=response['status']
        if status=='not_attempted':require(not callpath.exists() and response['artifacts']=={} and response['raw_response_sha256'] is None,'unstarted_artifacts')
        else:
            started_ids.add(call['call_id']);require(read(callpath/'intent.json')=={'schema':'third-llm-call-intent-v3','planned_call':call,'attempt':0,'retry_allowed':False},'call_intent')
            require(read(callpath/'completion.json')==response,'response_completion')
            require({x.name for x in callpath.iterdir()}==set(response['artifacts'])|{'intent.json','completion.json'},'exact_call_inventory')
            for name,bb in response['artifacts'].items():require(result['files']['calls/'+call['call_id']+'/'+name]==bb,'call_artifact_map')
            metric=read(callpath/'metrics.json');require(metric['call_id']==call['call_id'] and metric['request_index']==i and metric['attempt']==0 and metric['status']==status and metric['input_tokens']==len(ids) and metric['model_seed']==call['model_seed'],'call_metrics')
            require(metric['full_input_tokenization'] is True and metric['truncation_applied'] is False and metric['format_only_retry_limit']==0 and metric['rendered_prompt_sha256']==token['rendered_prompt_sha256'],'call_controls')
            require(metric['effective_constraint_config_sha256']==runtime['effective_constraint_config_sha256'] and metric['token_eligibility_sha256']==runtime['token_eligibility_sha256'],'call_runtime_bindings')
            require(response['runtime_manifest_sha256']==rtb['sha256'],'response_runtime')
            if 'sdpa_flags_during_generation' in metric:require(metric['sdpa_flags_during_generation']==FLAGS,'call_flash')
            raw_present=response['raw_response_sha256'] is not None
            if status=='context_overflow':
                require(len(ids)+512>32768 and not raw_present and response['artifacts'].keys()=={'metrics.json'} and metric['output_tokens']==0 and not {'generation_seconds','sdpa_flags_during_generation','peak_gpu_allocated_bytes'}&metric.keys(),'overflow_generation')
            else:require(len(ids)+512<=32768,'generation_after_overflow')
            if status in ('valid','invalid_json'):require(raw_present and {'raw.private.txt','output_tokens.private.json','guard_steps.private.json','metrics.json'}==set(response['artifacts']),'generated_output_inventory')
            output_ids=None
            if raw_present:
                raw=(callpath/'raw.private.txt').read_bytes().decode('utf8');output_ids=read(callpath/'output_tokens.private.json')
                require(type(output_ids) is list and all(type(t) is int and t>=0 for t in output_ids) and len(output_ids)<=512,'output_ids')
                require(response['raw_response_sha256']==sha(raw.encode()) and metric['output_tokens']==len(output_ids),'raw_output_binding')
                judgment,errors=parse_and_replay(raw,output_ids,ids,ctx)
                if status=='valid':
                    require(not errors and judgment==response['judgment'],'valid_output_not_reproduced')
                    require(all(metric.get(k) is True for k in ('full_output_prefix_decode_consistency_passed','generated_token_filter_replay_passed','decoded_character_grammar_replay_passed')),'valid_replay_evidence')
                if status=='invalid_json':require(bool(errors),'invalid_status_not_reproduced')
                replays.append({'call_id':call['call_id'],'recorded_status':status,'independent_failure_checks':errors})
            if 'guard_steps.private.json' in response['artifacts']:
                steps=read(callpath/'guard_steps.private.json');accepted=check_guard_steps(steps,metric,status,output_ids)
                if output_ids is None:ctx['guard'].replay_guard_tokens(ids,accepted,ctx['schema'],ctx['data'],ctx['policy'])
            if status=='runtime_error':require(response['error'] is not None,'runtime_failure_evidence')
        obs={'opaque_pair_id':call['opaque_pair_id'],'model_id':call['model_id'],'status':status,'category':response['judgment']['category'] if status=='valid' else None}
        (observed if call['orientation']=='canonical' else diagnostics).append(obs)
    require({p.name for p in (rdir/'calls').iterdir()}==started_ids and result['started_call_intents']==len(started_ids),'started_calls_exact')
    for logical,resolved in ctx['asset_links'].items():require(str(Path(logical).resolve())==resolved,'tokenizer_symlink_changed')
    w.recheck()
    return sealed({'schema':'third-llm-output-independent-audit-v1','passed':True,'alias':alias,'execution_result':rb,'activation':activation_binding,'request_manifest':activation.contract['request_manifest'],'integration_package':resources['packages']['integration'],'runner_package':resources['packages']['runner'],'resource_plan':activation.contract['resource_plan'],'planned_calls':len(calls),'canonical_calls':len(observed),'diagnostic_calls':len(diagnostics),'status_counts':dict(counts),'canonical_observations':observed,'diagnostic_observations':diagnostics,'raw_replay_results':replays,'all_call_identities_denominators_and_tokens_verified':True,'complete_model_weight_bytes_independently_rehashed':False,'model_weight_initial_final_verification':'Bound runner attestation with exact published weight hashes; this audit does not reread weight payloads.','logit_optimality_independently_replayed':False,'semantic_validity_established':False,'human_annotation_calls':0,'source_role':resources['source_role'],'elapsed_seconds':time.perf_counter()-started})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','repo','admission','activation-dir','tokenizer-path','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('admission-sha256','activation-sha256'):p.add_argument('--'+k,required=True)
    p.add_argument('--alias',choices=('qwen','mistral'),required=True);a=p.parse_args();ab=binding(a.admission);act=binding(a.activation_dir/'activation.json')
    require(ab['sha256']==a.admission_sha256 and act['sha256']==a.activation_sha256,'external_audit_pins')
    result=audit_run(a.root,a.repo,a.admission,ab,a.activation_dir,act,a.alias,a.tokenizer_path);new_json(a.output,result);print(__import__('json').dumps({'passed':True,'audit':binding(a.output),'status_counts':result['status_counts']}))
if __name__=='__main__':main()
