"""Prospective v3 request/call/envelope components; no activation writer or launcher."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent
STATUSES=frozenset(('valid','refusal','invalid_json','context_overflow','runtime_error','missing_output','not_attempted'))
INPUT_ROW_SERIALIZATION='qualified pilot canonical JSON: ensure_ascii=True, sorted keys, compact separators'
CALL_KEYS=('stage','opaque_pair_id','model_id','model_revision','orientation','attempt','call_id','model_seed',
           'request_index','selection_index','request_manifest','request_row_sha256',
           'input_token_ids_sha256','activation_lock_sha256')

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(value,ascii=False):
    return json.dumps(value,ensure_ascii=ascii,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def strict_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate_json_key');d[k]=v
        return d
    def bad(v):raise ValueError('nonfinite_json')
    def number(v):
        x=float(v);require(math.isfinite(x),'nonfinite_json');return x
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=number)

def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'sha256_required');return value

def integer(value,minimum=0):
    require(type(value) is int and value>=minimum,'integer_required');return value

def binding(path):
    p=Path(path);h=hashlib.sha256();size=0
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
    return {'bytes':size,'sha256':h.hexdigest()}

def check_binding(b):
    require(type(b) is dict and set(b)=={'bytes','sha256'},'binding_keys')
    integer(b['bytes']);digest(b['sha256']);return b

def checked_file(path,expected,max_bytes=None):
    p=Path(path);check_binding(expected)
    require(p.is_file() and not p.is_symlink(),'regular_metadata_file_required')
    require(max_bytes is None or p.stat().st_size<=max_bytes,'metadata_size_limit')
    require(binding(p)==expected,'file_binding_mismatch:'+str(p));return p

def load_pins():return strict_json((ROOT/'expectations.json').read_bytes())

def import_exact(name,path,expected):
    p=checked_file(path,expected,1<<20)
    require(name not in sys.modules,'module_name_already_loaded:'+name)
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    require(Path(module.__file__).resolve()==p.resolve() and binding(p)==expected,'import_source_changed')
    return module

def verify_seal(obj):
    require(type(obj) is dict and 'sha256' in obj,'missing_seal')
    require(obj['sha256']==sha(canonical({k:v for k,v in obj.items() if k!='sha256'})),'canonical_seal')

@dataclass(frozen=True)
class Activation:
    """Validated prospective metadata only; not evidence the activation was authorized."""
    raw_binding:dict
    contract:dict
    intent_binding:dict

# These are raw reviewed receipts/packages, never booleans claiming they exist.
ACTIVATION_BINDINGS=('request_manifest','request_preparation_package','selection_result',
 'sampling_lock','kernel_package','resource_plan')

def validate_activation(folder,expected_raw_binding,expected_contract):
    """Reads an externally reviewed, exclusive two-file activation. Creates nothing.

    Contract format is prospective and must be adopted/tested by the future writer.
    Neither this validator nor a valid hash certifies the upstream reviews' substance.
    """
    p=Path(folder);require(p.is_dir() and not p.is_symlink(),'activation_directory')
    require({x.name for x in p.iterdir()}=={'intent.json','activation.json'},'incomplete_activation_inventory')
    obj=strict_json(checked_file(p/'activation.json',expected_raw_binding,1<<20).read_bytes());verify_seal(obj)
    require(set(obj)=={'schema','status','contract','intent','sha256'},'activation_keys')
    require(obj['schema']=='third-llm-only-annotation-activation-v4' and obj['status']=='activated','activation_status')
    intent=strict_json(checked_file(p/'intent.json',obj['intent'],1<<20).read_bytes())
    require(set(intent)=={'schema','contract_sha256','entropy_draws','attempt'},'activation_intent_keys')
    require(intent['schema']=='third-llm-only-annotation-activation-intent-v4' and
            type(intent['entropy_draws']) is int and intent['entropy_draws']==0 and
            type(intent['attempt']) is int and intent['attempt']==0,'activation_intent')
    require(intent['contract_sha256']==sha(canonical(expected_contract)),'activation_intent_contract')
    require(obj['contract']==expected_contract,'activation_contract_mismatch')
    c=obj['contract'];pins=load_pins()
    require(set(c)==set(ACTIVATION_BINDINGS)|{'protocol_plan_sha256','frame_result_sha256','protocol_evidence_sha256',
        'models','stage','human_annotation_calls','human_completed','output_limit_bytes','scope'},'activation_contract_keys')
    for k in ACTIVATION_BINDINGS:check_binding(c[k])
    require(c['request_preparation_package']==pins['request_preparation_package'],'qualified_request_preparation_package')
    for k in ('protocol_plan_sha256','frame_result_sha256','protocol_evidence_sha256'):
        require(c[k]==pins[k],'activation_identity:'+k)
    require(c['models']=={k:{'model_id':v['model_id'],'revision':v['revision']} for k,v in pins['models'].items()},'activation_models')
    require(c['stage']=='selection','only_initial_selection_stage_implemented')
    require(type(c['human_annotation_calls']) is int and c['human_annotation_calls']==0 and c['human_completed'] is False,'zero_humans')
    require(integer(c['output_limit_bytes'],1)<=512*1024**2,'output_limit')
    require(c['scope']=='model-assessed development; no human validity or publisher authenticity','source_scope')
    return Activation(dict(expected_raw_binding),c,dict(obj['intent']))

def read_jsonl(path,expected):
    p=checked_file(path,expected,128*1024**2);rows=[]
    with p.open('rb') as f:
        for raw in f:
            require(raw.endswith(b'\n') and len(raw)<=16*1024**2,'jsonl_physical_lf_required')
            require(raw.strip(),'empty_jsonl_line');rows.append(strict_json(raw))
    return rows

def reconstruct_row(request,texts):
    require(request['orientation'] in ('canonical','swapped_diagnostic'),'orientation')
    inds=request['shown_calibration_row_indices'];require(type(inds) is list and len(inds)==2 and inds[0]!=inds[1],'two_distinct_rows')
    for i in inds:integer(i)
    selected=[texts[i] for i in inds]
    for row in selected:
        require(type(row['text']) is str and sha(row['text'].encode('utf8'))==row['text_sha256'],'text_hash')
    require([r['record_id'] for r in selected]==request['shown_record_ids'],'shown_record_ids')
    require([r['text_sha256'] for r in selected]==request['shown_text_sha256'],'shown_text_sha256')
    return {'pair_id':request['opaque_pair_id'],'orientation':request['orientation'],
            'text_a':selected[0]['text'],'text_b':selected[1]['text']}

def validate_token_row(token,shared,texts,model,protocol,pins):
    shared_keys={'request_index','selection_index','stage','orientation','diagnostic_only','opaque_pair_id','natural_pair_id',
        'shown_calibration_row_indices','shown_record_ids','shown_text_sha256','model_seeds'}
    extra_keys={'model_id','model_revision','model_seed','input_row_sha256','input_row_hash_serialization',
        'rendered_prompt_sha256','input_token_ids','input_tokens','input_token_ids_sha256','attention_mask_policy',
        'max_new_tokens','total_token_budget','context_fit_with_output_reserve','preparation_status',
        'full_text_preserved','truncation_applied','activation_lock_sha256','call_id'}
    require(set(shared)==shared_keys and set(token)==shared_keys|extra_keys,'request_token_exact_keys')
    for k,v in shared.items():require(type(token.get(k)) is type(v) and token.get(k)==v,'token_shared_metadata:'+k)
    require(set(shared['model_seeds'])=={m['model_id'] for m in pins['models'].values()},'exact_two_seed_models')
    for m in pins['models'].values():
        expected=protocol.model_seed(pins['protocol_plan_sha256'],pins['frame_result_sha256'],shared['opaque_pair_id'],m['model_id'],m['revision'])
        require(type(shared['model_seeds'][m['model_id']]) is int and shared['model_seeds'][m['model_id']]==expected,'all_v3_model_seeds')
    require(token['model_id']==model['model_id'] and token['model_revision']==model['revision'],'token_model')
    seed=protocol.model_seed(pins['protocol_plan_sha256'],pins['frame_result_sha256'],shared['opaque_pair_id'],model['model_id'],model['revision'])
    require(type(token['model_seed']) is int and token['model_seed']==seed and
            shared['model_seeds'][model['model_id']]==seed,'v3_model_seed')
    require(token['activation_lock_sha256'] is None and token['call_id'] is None,'preactivation_ids_must_be_null')
    row=reconstruct_row(shared,texts)
    require(token['input_row_hash_serialization']==INPUT_ROW_SERIALIZATION and
            token['input_row_sha256']==sha(canonical(row,ascii=True)),'input_row_binding')
    ids=token['input_token_ids'];require(type(ids) is list and ids and all(type(i) is int and i>=0 for i in ids),'input_token_ids')
    require(type(token['input_tokens']) is int and token['input_tokens']==len(ids) and token['input_token_ids_sha256']==sha(canonical(ids)),'input_token_count_hash')
    digest(token['rendered_prompt_sha256'])
    require(token['attention_mask_policy']=='one for every unpadded token' and
            type(token['max_new_tokens']) is int and token['max_new_tokens']==512 and
            type(token['total_token_budget']) is int and token['total_token_budget']==32768,'tokenization_controls')
    fit=len(ids)+512<=32768
    require(token['context_fit_with_output_reserve'] is fit and
            token['preparation_status']==('ready_for_future_activation' if fit else 'context_overflow'),'context_accounting')
    require(token['full_text_preserved'] is True and token['truncation_applied'] is False,'no_text_filter_or_truncation')
    return row

def validate_planned_call(call,protocol,pins=None):
    pins=pins or load_pins();require(set(call)==set(CALL_KEYS),'planned_call_keys')
    for k in ('opaque_pair_id','call_id','request_row_sha256','input_token_ids_sha256','activation_lock_sha256'):digest(call[k])
    for k in ('request_index','selection_index','model_seed'):integer(call[k])
    require(call['model_seed']<2**63 and type(call['attempt']) is int and call['attempt']==0,'single_attempt_seed_range')
    check_binding(call['request_manifest'])
    require((call['stage'],call['orientation']) in {('selection','canonical'),('selection_swap','swapped_diagnostic')},'call_stage_orientation')
    models={(m['model_id'],m['revision']) for m in pins['models'].values()}
    require((call['model_id'],call['model_revision']) in models,'qualified_call_model')
    require(call['model_seed']==protocol.model_seed(pins['protocol_plan_sha256'],pins['frame_result_sha256'],
        call['opaque_pair_id'],call['model_id'],call['model_revision']),'call_v3_seed')
    require(call['call_id']==protocol.domain_digest('third-llm-only-v4/call-id',[call['activation_lock_sha256'],
        call['stage'],call['opaque_pair_id'],call['model_id'],call['model_revision'],call['orientation'],0]),'call_v3_id')
    return call

def resolve_calls(tokens,request_manifest,activation,protocol):
    check_binding(request_manifest);require(isinstance(activation,Activation),'validated_activation_required')
    require(request_manifest==activation.contract['request_manifest'],'activation_request_binding')
    result=[];seen=set()
    for token in tokens:
        stage=token['stage'];orientation=token['orientation']
        require((stage,orientation) in (('selection','canonical'),('selection_swap','swapped_diagnostic')),'stage_orientation')
        require(token['activation_lock_sha256'] is None and token['call_id'] is None,'already_resolved_request')
        key=(stage,token['opaque_pair_id'],token['model_id'],token['model_revision'],orientation,0)
        require(key not in seen,'duplicate_planned_call');seen.add(key)
        call={k:token[k] for k in ('stage','opaque_pair_id','model_id','model_revision','orientation','model_seed','request_index','selection_index','input_token_ids_sha256')}
        for k in ('request_index','selection_index','model_seed'):integer(call[k])
        call.update(attempt=0,activation_lock_sha256=activation.raw_binding['sha256'],request_manifest=request_manifest,
                    request_row_sha256=sha(canonical(token)),call_id=protocol.domain_digest('third-llm-only-v4/call-id',
                    [activation.raw_binding['sha256'],*key]))
        validate_planned_call(call,protocol)
        result.append(call)
    return result

def schedule(tokens):
    return sorted(range(len(tokens)),key=lambda i:(-tokens[i]['input_tokens'],tokens[i]['selection_index'],tokens[i]['orientation']!='canonical'))

def envelope(call,status,judgment=None,raw_hash=None,runtime_hash=None,artifacts=None,error=None):
    require(set(call)==set(CALL_KEYS),'planned_call_keys')
    return {'schema':'third-llm-model-response-v4',**call,'annotator_type':'model','human_completed':False,
            'status':status,'judgment':judgment,'raw_response_sha256':raw_hash,
            'runtime_manifest_sha256':runtime_hash,'artifacts':artifacts or {},'error':error}

def validate_envelope(response,call,validate_judgment):
    require(set(response)==set(CALL_KEYS)|{'schema','annotator_type','human_completed','status','judgment',
        'raw_response_sha256','runtime_manifest_sha256','artifacts','error'},'response_keys')
    require(response['schema']=='third-llm-model-response-v4' and response['annotator_type']=='model' and
            response['human_completed'] is False,'response_model_only')
    for k in CALL_KEYS:require(type(response[k]) is type(call[k]) and response[k]==call[k],'response_identity:'+k)
    status=response['status'];require(status in STATUSES,'response_status')
    require(type(response['artifacts']) is dict,'response_artifacts')
    for name,b in response['artifacts'].items():
        require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'artifact_name');check_binding(b)
    for k in ('raw_response_sha256','runtime_manifest_sha256'):
        if response[k] is not None:digest(response[k])
    if status=='valid':
        validate_judgment(response['judgment']);digest(response['raw_response_sha256']);digest(response['runtime_manifest_sha256'])
        require(response['error'] is None,'valid_cannot_have_error')
        require({'raw.private.txt','output_tokens.private.json','guard_steps.private.json','metrics.json'}<=set(response['artifacts']),'valid_missing_artifacts')
        require(response['artifacts']['raw.private.txt']['sha256']==response['raw_response_sha256'],'raw_artifact_hash')
    else:require(response['judgment'] is None,'unavailable_has_no_judgment')
    if status in ('context_overflow','not_attempted','missing_output'):
        require(response['raw_response_sha256'] is None,'ungenerated_raw_output')
    if status=='not_attempted':require(response['artifacts']=={},'not_attempted_artifacts')
    if status in ('invalid_json','refusal'):
        digest(response['raw_response_sha256']);digest(response['runtime_manifest_sha256'])
        require({'raw.private.txt','output_tokens.private.json','guard_steps.private.json','metrics.json'}<=set(response['artifacts']),'generated_failure_missing_artifacts')
        require(response['artifacts']['raw.private.txt']['sha256']==response['raw_response_sha256'],'generated_failure_raw_hash')
    if status=='invalid_json':
        require(response['error'] is not None,'invalid_format_evidence')
    if status in ('runtime_error','missing_output','not_attempted'):require(response['error'] is not None,'unavailable_error_required')
    require(response['error'] is None or (type(response['error']) is dict and set(response['error'])=={'type','message'} and
        all(type(x) is str and x for x in response['error'].values())),'error_shape')
    return response

def validate_prepared_rows(shared,tokens,text_rows,alias,protocol,pins=None):
    """Checks in-memory prepared content; raw bundle/independent review admission is separate."""
    pins=pins or load_pins();model=pins['models'][alias]
    require(type(shared) is list and shared and len(tokens)==len(shared),'request_coverage')
    texts={}
    for r in text_rows:
        require(set(r)=={'calibration_row_index','original_full_row_index','record_id','text_sha256','text'},'text_row_keys')
        i=integer(r['calibration_row_index']);integer(r['original_full_row_index'])
        require(i not in texts and type(r['record_id']) is str and r['record_id'],'unique_text_row');texts[i]=r
    canonical_rows={};swaps=set();pilot_rows=[];used=set();swap_phase=False
    for i,(shared_row,token) in enumerate(zip(shared,tokens)):
        require(type(shared_row['request_index']) is int and shared_row['request_index']==i,'stable_request_index')
        si=integer(shared_row['selection_index']);op=shared_row['opaque_pair_id'];digest(op);digest(shared_row['natural_pair_id'])
        orientation=shared_row['orientation'];diagnostic=orientation=='swapped_diagnostic'
        require(shared_row['diagnostic_only'] is diagnostic and shared_row['stage']==('selection_swap' if diagnostic else 'selection'),'request_stage')
        row=validate_token_row(token,shared_row,texts,model,protocol,pins)
        identity=protocol.identities(pins['protocol_plan_sha256'],pins['frame_result_sha256'],shared_row['shown_record_ids'])
        require(identity['opaque_pair_id']==op and identity['natural_pair_id']==shared_row['natural_pair_id'],'pair_identity')
        expected_ids=identity['canonical_record_ids'][::-1] if diagnostic else identity['canonical_record_ids']
        require(shared_row['shown_record_ids']==expected_ids,'canonical_orientation')
        if diagnostic:
            swap_phase=True;require(si in canonical_rows and si not in swaps,'unique_swap_parent')
            old=canonical_rows[si];require(old['opaque_pair_id']==op and old['shown_calibration_row_indices'][::-1]==shared_row['shown_calibration_row_indices'],'swap_reverse')
            swaps.add(si)
        else:
            require(not swap_phase and si==len(canonical_rows) and op not in {x['opaque_pair_id'] for x in canonical_rows.values()},'canonical_order_unique')
            canonical_rows[si]=shared_row
        used.update(shared_row['shown_calibration_row_indices']);pilot_rows.append(row)
    require(len(swaps)==math.ceil(len(canonical_rows)/10),'exact_swap_denominator')
    require(set(texts)==used,'exact_referenced_text_inventory')
    return pilot_rows

def read_request_bundle(folder,contract,alias,protocol):
    """Read the versioned full-text v4 bundle; no legacy sample is accepted."""
    p=Path(folder);c=contract;rb=c['request_manifest']
    result=strict_json(checked_file(p/'result.json',rb,4*1024**2).read_bytes());verify_seal(result)
    require(result['schema']=='third-llm-v4-requests-v1' and result['passed'] is True and
            result['stage']=='complete','v4_request_result')
    require(result['package_sha256']==c['request_preparation_package']['sha256'] and
            result['plan_sha256']==c['protocol_plan_sha256'] and
            result['frame_result']['sha256']==c['frame_result_sha256'],'request_source_identity')
    for key in ('selection_result','sampling_lock'):
        require(result[key]==c[key],'request_lineage:'+key)
    names={'texts.private.jsonl','sample.private.jsonl','requests.private.jsonl',
           'qwen_tokens.private.jsonl','mistral_tokens.private.jsonl','input_bindings.json'}
    require(set(result['files'])==names and {f.name for f in p.iterdir()}==names|{'result.json'},'request_inventory')
    bound={'result.json':rb}
    for name,b in result['files'].items():checked_file(p/name,b,128*1024**2);bound[name]=b
    require(sum(b['bytes'] for b in bound.values())<=128*1024**2,'total_request_bytes')
    for key in ('all_calibration_text_mappings_verified','all_selected_scores_weights_identities_and_swaps_replayed',
                'all_planned_requests_retained'):
        require(result[key] is True,'request_source_replay:'+key)
    for key in ('model_inference_calls','human_annotation_calls'):
        require(type(result[key]) is int and result[key]==0,'no_prior_request_inference')
    require(result['inference_activation'] is False,'request_is_pre_activation')
    shared=read_jsonl(p/'requests.private.jsonl',bound['requests.private.jsonl'])
    token_name=alias+'_tokens.private.jsonl';tokens=read_jsonl(p/token_name,bound[token_name])
    texts=read_jsonl(p/'texts.private.jsonl',bound['texts.private.jsonl'])
    rows=validate_prepared_rows(shared,tokens,texts,alias,protocol)
    require(result['planned_requests']==2*len(shared) and
            result['selected_pairs']==sum(not x['diagnostic_only'] for x in shared) and
            result['swap_pairs']==sum(x['diagnostic_only'] for x in shared),'complete_request_denominator')
    summary=result['model_censuses'][alias]
    require(summary['scheduled_request_indices']==[tokens[i]['request_index'] for i in schedule(tokens)],'request_schedule')
    require(summary['overflow_requests']==sum(not t['context_fit_with_output_reserve'] for t in tokens),
            'overflow_denominator')
    for name,b in bound.items():checked_file(p/name,b,128*1024**2)
    return {'result':result,'shared':shared,'tokens':tokens,'rows':rows,'bound_files':bound}

def read_prepared_bundle(folder,activation,alias,protocol):
    require(isinstance(activation,Activation),'validated_activation_required')
    return read_request_bundle(folder,activation.contract,alias,protocol)
