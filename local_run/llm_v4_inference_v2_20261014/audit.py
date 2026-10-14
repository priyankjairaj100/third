"""Independent CPU replay of v4 identities, full prompts and generated grammar."""
import argparse
from datetime import datetime,timezone
import json
import os
import socket
import time
import audit_helpers as A
import common as x
import contract as C
import node_policy as N
from run_model import expected_contract

def main():
    started=time.perf_counter();p=argparse.ArgumentParser()
    for name in ('package-sha256','request-result-sha256','activation-sha256','resource-plan-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--model-alias',choices=('qwen','mistral'),required=True);args=p.parse_args()
    pins,controls,prior,h,protocol=x.setup(args.package_sha256)
    C.require(socket.gethostname()=='csis.cn1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_only')
    rb=C.binding(x.ROOT/'resource_plan.json');C.require(rb['sha256']==args.resource_plan_sha256,'resource_pin')
    expected,_=expected_contract(args.request_result_sha256,pins,rb)
    ab=C.binding(x.ROOT/'activation/activation.json');C.require(ab['sha256']==args.activation_sha256,'activation_pin')
    activation=C.validate_activation(x.ROOT/'activation',ab,expected)
    bundle=C.read_prepared_bundle(x.REQUESTS,activation,args.model_alias,protocol)
    folder=x.ROOT/args.model_alias/'output';result=A.read(folder/'result.json');C.verify_seal(result)
    C.require(result['passed'] is True and result['package_sha256']==args.package_sha256 and
              result['activation']==ab and result['request_manifest']==expected['request_manifest'],'execution_result_identity')
    watch=A.Watch();watch.file(folder/'result.json')
    declared=result['files'];actual={str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file()}
    C.require(actual==set(declared)|{'result.json'},'complete_execution_inventory')
    for name,b in declared.items():
        C.require(not name.startswith('/') and '..' not in __import__('pathlib').Path(name).parts,'safe_output_path')
        watch.file(folder/name,b,256*1024**2)
    for name,b in bundle['bound_files'].items():watch.file(x.REQUESTS/name,b,128*1024**2)
    model=pins['models'][args.model_alias]
    calls=A.planned_calls(bundle['tokens'],expected['request_manifest'],ab['sha256'],model,pins)
    C.require(A.read(folder/'planned_calls.json')==calls,'independent_call_reconstruction')
    responses=A.read(folder/'responses.json');order=C.schedule(bundle['tokens'])
    counts=A.check_responses(responses,calls,order)
    C.require(A.read(folder/'run_intent.json')['request_indices_in_execution_order']==[calls[i]['request_index'] for i in order],
              'execution_schedule')
    final=A.read(folder/'final_bindings.json');C.require(final['passed'] is True,'final_binding_attestation')
    for name,b in model['asset_files'].items():
        matches=[v for k,v in final['model_and_runtime_files'].items() if k.endswith('/'+model['revision']+'/'+name)]
        C.require(len(matches)==1 and matches[0]['binding']==b,'all_pinned_weight_and_asset_attestations')
    capability=A.read(folder/'allocated_gpu_capability.json')
    C.require(capability['passed'] is True and capability['host'] in N.ALLOWED and
        capability['host']==result['hostname'] and capability['job_id']==result['slurm_job_id'] and
        'A100' in capability['gpu'] and capability['gpu_total_bytes']>=75*1024**3 and
        capability['cuda_capability']==[8,0] and capability['model_inference_calls']==0 and
        capability['flash_sdpa_flags']=={'flash':True,'math':False,'mem_efficient':False,'cudnn':False},'allocated_GPU_admission')
    runtime=A.read(folder/'runtime/runtime_manifest.json')
    C.require(runtime['dtype']=='bfloat16' and runtime['visible_gpus']==1 and runtime['deterministic_algorithms'] is True
              and runtime['allow_tf32'] is False and runtime['sdpa_flags_inside_generation_context']==A.FLAGS,'runtime_policy')
    C.require(runtime['packages']==A.pins()['family_runtime_packages'][args.model_alias],'runtime_packages')
    generation=A.read(folder/'runtime/generation_config.json')
    for key,value in {'do_sample':False,'num_beams':1,'max_new_tokens':512,'use_cache':True,'repetition_penalty':1.0,
        'output_attentions':False,'forced_bos_token_id':None,'forced_eos_token_id':None,'remove_invalid_values':False,
        'renormalize_logits':False,'num_return_sequences':1}.items():C.require(generation.get(key)==value,'generation:'+key)
    for name,key in [('generation_config.json','generation_config_sha256'),('token_eligibility.json','token_eligibility_sha256'),
        ('effective_constraint_config.json','effective_constraint_config_sha256')]:
        C.require(declared['runtime/'+name]['sha256']==runtime[key],'runtime_binding')
    ctx=A.load_tokenizer(args.model_alias,getattr(controls,args.model_alias.upper()),x.REPO,runtime,
        A.read(folder/'runtime/token_eligibility.json'),A.read(folder/'runtime/effective_constraint_config.json'),watch)
    system=(x.REPO/'local_run/model_annotation_v2_20261011/judge_system_prompt.txt').read_text()
    task=(x.REPO/'local_run/model_annotation_v2_20261011/judge_task_definition.txt').read_text()
    observed=[];diagnostics=[];replays=[];started_ids=set()
    from jsonschema import Draft202012Validator
    judgment_validator=Draft202012Validator(ctx['schema'])
    for response,call,token,row in zip(responses,calls,bundle['tokens'],bundle['rows']):
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps({'task_definition':task,
            'documents':[{'label':'A','text':row['text_a']},{'label':'B','text':row['text_b']}]},ensure_ascii=False)}]
        rendered=ctx['tokenizer'].apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=ctx['tokenizer'](rendered,add_special_tokens=False,truncation=False)['input_ids']
        C.require(ids==token['input_token_ids'] and C.sha(rendered.encode())==token['rendered_prompt_sha256'],'full_prompt_replay')
        status=response['status'];callpath=folder/'calls'/call['call_id']
        C.validate_envelope(response,call,judgment_validator.validate)
        C.require(status in ('valid','invalid_json','context_overflow'),'successful_run_has_all_calls_attempted')
        started_ids.add(call['call_id'])
        C.require(A.read(callpath/'intent.json')=={'schema':'third-llm-call-intent-v4','planned_call':call,'attempt':0,
                    'retry_allowed':False} and A.read(callpath/'completion.json')==response,'call_intent_completion')
        C.require({f.name for f in callpath.iterdir()}==set(response['artifacts'])|{'intent.json','completion.json'},'call_inventory')
        for name,b in response['artifacts'].items():C.require(declared['calls/'+call['call_id']+'/'+name]==b,'call_artifact_binding')
        metric=A.read(callpath/'metrics.json')
        C.require(metric['call_id']==call['call_id'] and metric['request_index']==call['request_index'] and
            metric['status']==status and metric['attempt']==0 and metric['model_seed']==call['model_seed'] and
            metric['input_tokens']==len(ids),'call_metrics')
        C.require(metric['full_input_tokenization'] is True and metric['truncation_applied'] is False and
            metric['format_only_retry_limit']==0 and metric['rendered_prompt_sha256']==token['rendered_prompt_sha256'],'no_truncation_retry')
        C.require(metric['effective_constraint_config_sha256']==runtime['effective_constraint_config_sha256'] and
            metric['token_eligibility_sha256']==runtime['token_eligibility_sha256'] and
            response['runtime_manifest_sha256']==declared['runtime/runtime_manifest.json']['sha256'],'runtime_response_binding')
        if status=='context_overflow':
            C.require(len(ids)+512>32768 and set(response['artifacts'])=={'metrics.json'} and metric['output_tokens']==0,'overflow_not_generated')
        else:
            C.require(len(ids)+512<=32768 and metric['sdpa_flags_during_generation']==A.FLAGS,'flash_generation_policy')
            raw=(callpath/'raw.private.txt').read_text();output_ids=A.read(callpath/'output_tokens.private.json')
            C.require(C.sha(raw.encode())==response['raw_response_sha256'] and metric['output_tokens']==len(output_ids)<=512,'raw_token_binding')
            judgment,errors=A.parse_and_replay(raw,output_ids,ids,ctx)
            C.require((not errors and judgment==response['judgment']) if status=='valid' else bool(errors),'recorded_format_status_replayed')
            A.check_guard_steps(A.read(callpath/'guard_steps.private.json'),metric,status,output_ids)
            replays.append({'call_id':call['call_id'],'status':status,'independent_failure_checks':errors})
        observation={'opaque_pair_id':call['opaque_pair_id'],'model_id':call['model_id'],'status':status,
                     'category':response['judgment']['category'] if status=='valid' else None}
        (observed if call['orientation']=='canonical' else diagnostics).append(observation)
    C.require({p.name for p in (folder/'calls').iterdir()}==started_ids,'all_started_calls_accounted')
    for logical,resolved in ctx['asset_links'].items():C.require(str(__import__('pathlib').Path(logical).resolve())==resolved,'tokenizer_link_unchanged')
    watch.recheck();out=x.ROOT/('audit_'+args.model_alias)/'run';out.mkdir(mode=0o700,exist_ok=False)
    x.publish(out/'result.json',x.sealed({'schema':'third-v4-independent-output-audit-v1','passed':True,
        'created_utc':datetime.now(timezone.utc).isoformat(),'model_alias':args.model_alias,'package_sha256':args.package_sha256,
        'execution_result':C.binding(folder/'result.json'),'activation':ab,'request_manifest':expected['request_manifest'],
        'canonical_observations':observed,'diagnostic_observations':diagnostics,'status_counts':dict(counts),
        'raw_replay_results':replays,'all_call_identities_denominators_and_tokens_verified':True,
        'weight_hashes_bound_from_execution':True,'weight_payloads_independently_rehashed':False,
        'semantic_validity_established':False,'human_annotation_calls':0,'elapsed_seconds':time.perf_counter()-started}))
    print(json.dumps({'passed':True,'alias':args.model_alias,'counts':dict(counts)}),flush=True)

if __name__=='__main__':main()
