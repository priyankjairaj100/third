"""Versioned independent audit primitives retained from the completed v3 auditor."""
from collections import Counter
import importlib.metadata
import math
import os
from pathlib import Path
import common as x
import contract as C
import kernel as K
require=C.require
binding=C.binding
canonical=C.canonical
sha=C.sha
strict=C.strict_json
load=x.module

def read(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=4*1024**2,'bounded_metadata')
    return strict(p.read_bytes())

def pins():
    return read(Path(__file__).with_name('prior_audit_expectations.json'))

class Watch(K.Watch):
    def file(self,path,wanted=None,cap=4*1024**2):
        require(Path(path).stat().st_size<=cap,'file_cap')
        return self.add(path,wanted)

FLAGS={'flash':True,'math':False,'mem_efficient':False,'cudnn':False}
CALL_KEYS={'stage','opaque_pair_id','model_id','model_revision','orientation','attempt','call_id','model_seed','request_index','selection_index','request_manifest','request_row_sha256','input_token_ids_sha256','activation_lock_sha256'}
RESPONSE_EXTRA={'schema','annotator_type','human_completed','status','judgment','raw_response_sha256','runtime_manifest_sha256','artifacts','error'}
STATUSES={'valid','invalid_json','context_overflow','runtime_error','not_attempted'}
def domain(name,values):return sha(canonical(['third-llm-only-v4/'+name,*values]))
def planned_calls(tokens,request_binding,activation_sha,model,cp):
    calls=[];seen=set()
    for i,t in enumerate(tokens):
        require(type(t['request_index']) is int and t['request_index']==i,'request_index')
        require((t['stage'],t['orientation']) in {('validation','canonical'),('validation_swap','swapped_diagnostic')},'phase_orientation')
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
        require(r['schema']=='third-llm-model-response-v4' and r['annotator_type']=='model' and r['human_completed'] is False,'model_only_response')
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

