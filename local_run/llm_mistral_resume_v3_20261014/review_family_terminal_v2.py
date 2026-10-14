"""Independent local terminal metadata/byte review; CPU grammar audit still required."""
import argparse,hashlib,json,math
from collections import Counter
from pathlib import Path

REPO=Path('/Users/priyankjairaj/Downloads/ACL/Three')
WORK=Path(__file__).parent
REPORTS=REPO/'local_workspace/reports'
def require(ok,why):
    if not ok: raise ValueError(why)
def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(b): return hashlib.sha256(b).hexdigest()
def binding(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':sha(raw)}
def read(path): return json.loads(path.read_bytes())
def seal(value):
    body=dict(value);digest=body.pop('sha256');require(sha(canonical(body))==digest,'internal_seal')
def domain(label,values): return sha(canonical(['third-llm-only-v3/'+label,*values]))

def review(alias, launch_receipt=None):
    root=REPORTS/('llm-only-v3-selection-'+alias+'-a01');out=root/'run/output'
    exported=read(root/'terminal_export.json');job=exported['job_id']
    require(not exported['queue'].strip(),'active_queue')
    rows=[line.split('|') for line in exported['accounting'].splitlines() if line]
    parents=[row for row in rows if row[0]==job]
    require(len(parents)==1 and len(parents[0])==6,'parent_accounting')
    parent=parents[0]
    require((parent[1],parent[2]) in {('COMPLETED','0:0'),('FAILED','2:0')},'eligible_terminal_status')
    require(all(row[0]==job or row[0].startswith(job+'.') for row in rows),'unrelated_job')
    require(parent[5] and 'mn1' not in parent[5],'compute_node')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual==set(exported['files'])|{'terminal_export.json'},'export_inventory')
    for name,b in exported['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'export_member_name')
        p=root/name;require(not p.is_symlink() and binding(p)=={k:b[k] for k in ('bytes','sha256')},'export_member_binding')
    require(sum(x['bytes'] for x in exported['files'].values())==exported['total_bytes'],'transport_total')
    launch=read(root/'launch_receipt.json');intent=read(root/'launch_intent.json');process=read(root/'run/process_completion.json')
    require(launch==read(launch_receipt or WORK/('inference_'+alias+'_launch_receipt.json')),'local_submission_receipt')
    require(all(launch.get(k)==v for k,v in intent.items()),'launch_intent')
    require(launch['job_id']==job and launch['returncode']==0 and launch['mode']==alias,'job_identity')
    require(process['schema']=='third-llm-execution-process-v1' and process['command']==launch['child_command'],'exact_child_command')
    require(process['mode']==alias and str(process['slurm_job_id'])==job and process['console_cap_exceeded'] is False,'child_identity')
    require(process['returncode']==int(parent[2].split(':')[0]),'child_terminal_code')
    require(process['retry_attempts']==process['human_annotation_calls']==0,'no_retries_or_humans')
    require(all(launch['bindings'].get(k)==v for k,v in process['bindings'].items()),'child_input_bindings')
    require(process['console_bytes']==sum((root/'run'/n).stat().st_size for n in ('stdout.txt','stderr.txt')),'console_bytes')
    resources=read(WORK/'inference_resource_plan.json');seal(resources)
    active=read(REPORTS/'llm-only-v3-annotation-activation-a01/activation.json');seal(active)
    ab=binding(REPORTS/'llm-only-v3-annotation-activation-a01/activation.json')
    result=read(out/'result.json');seal(result)
    require(result['schema']=='third-llm-annotation-execution-v1' and result['alias']==result['model_alias']==alias,'result_schema_family')
    for key,want in {'activation':ab,'request_manifest':resources['request_manifest'],'resource_plan':binding(WORK/'inference_resource_plan.json'),'runner_package':resources['packages']['runner'],'component_package':resources['packages']['component'],'integration_package':resources['packages']['integration']}.items():
        require(result[key]==want,'result_binding:'+key)
    require(result['human_annotation_calls']==result['retry_calls']==0 and result['human_completed'] is False,'result_scope')
    require(result['passed_execution_integrity'] is True and result['artifact_integrity_error'] is None,'execution_integrity')
    require(result['model_loaded'] is True and result['initial_final_bindings_equal'] is True,'model_asset_attestations')
    inventory={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    require(inventory==set(result['files'])|{'result.json'},'exact_output_inventory')
    for name,b in result['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'result_member_name')
        require(binding(out/name)==b,'result_artifact_binding')
    require(sum(b['bytes'] for b in result['files'].values())==result['output_bytes_before_result'],'result_output_bytes')
    require(result['output_bytes_before_result']+binding(out/'result.json')['bytes']<=resources['families'][alias]['output_limit_bytes'],'output_cap')
    request_root=REPORTS/'llm-annotation-requests-v2-20261014/selection-a01/run/output'
    tokens=[json.loads(line) for line in (request_root/(alias+'_tokens.private.jsonl')).read_text().splitlines()]
    calls=read(out/'planned_calls.json');responses=read(out/'responses.json');ledger=read(out/'ledger.json')
    require(len(calls)==len(responses)==len(tokens)==resources['planned']['per_model']==99,'full_denominator')
    for i,(call,response,t) in enumerate(zip(calls,responses,tokens)):
        require(call['request_index']==response['request_index']==t['request_index']==i,'request_index')
        require(all(response.get(k)==v for k,v in call.items()),'response_call_identity')
        require(call['request_row_sha256']==sha(canonical(t)) and call['request_manifest']==resources['request_manifest'],'request_identity')
        require(all(call[k]==t[k] for k in ('stage','opaque_pair_id','model_id','model_revision','orientation','model_seed','selection_index','input_token_ids_sha256')),'prepared_call_fields')
        cid=domain('call-id',[ab['sha256'],*[t[k] for k in ('stage','opaque_pair_id','model_id','model_revision','orientation')],0])
        require(call['call_id']==cid and call['attempt']==0 and call['activation_lock_sha256']==ab['sha256'],'activated_unique_call')
        require(response['annotator_type']=='model' and response['human_completed'] is False,'model_response')
        require(response['status'] in ('valid','invalid_json','context_overflow','runtime_error','not_attempted'),'response_status')
        cp=out/'calls'/cid
        if response['status']=='not_attempted':require(not cp.exists(),'unstarted_path')
        else:require(read(cp/'completion.json')==response,'durable_completion')
    counts=dict(Counter(r['status'] for r in responses));require(result['status_counts']==ledger['status_counts']==counts,'complete_status_counts')
    require(result['planned_calls']==result['recorded_calls']==ledger['planned_calls']==ledger['recorded_calls']==99,'ledger_counts')
    require(result['canonical_calls']==90 and result['diagnostic_calls']==9 and ledger['complete_planned_denominator'] is True,'canonical_swap_denominators')
    run=read(out/'run_intent.json');order=sorted(range(99),key=lambda i:(-tokens[i]['input_tokens'],tokens[i]['selection_index'],tokens[i]['orientation']!='canonical'))
    require(run['execution_request_indices']==order and run['activation']==ab,'sealed_schedule')
    fb=read(out/'final_bindings.json');ib=read(out/'input_bindings.json')
    require(fb['input_files_final']==ib['files'] and fb['model_code_assets_initial']==fb['model_code_assets_final'],'initial_final_asset_identity')
    seconds=process['child_wall_seconds_through_exit'];require(type(seconds) in (float,int) and math.isfinite(seconds) and seconds>0,'process_timing')
    return {'schema':'third-llm-family-local-terminal-review-v1','passed':True,'alias':alias,'job_id':job,'scheduler':{'state':parent[1],'exit_code':parent[2],'elapsed_seconds':int(parent[3]),'node':parent[5]},'execution_passed':result['passed'],'execution_result':binding(out/'result.json'),'transport':binding(root/'terminal_export.json'),'transport_files':len(exported['files']),'transport_bytes':exported['total_bytes'],'launch_receipt':binding(root/'launch_receipt.json'),'process_receipt':binding(root/'run/process_completion.json'),'review_source':binding(Path(__file__)),'activation':ab,'request_manifest':resources['request_manifest'],'planned_calls':99,'canonical_calls':90,'diagnostic_calls':9,'status_counts':counts,'child_wall_seconds_through_exit':seconds,'children_maxrss_kib_linux':process['children_maxrss_kib_linux'],'all_exported_and_declared_output_bytes_locally_verified':True,'all_prepared_call_identities_and_denominators_verified':True,'independent_CPU_prompt_token_grammar_audit_still_required':True,'full_model_weight_bytes_independently_rehashed':False,'new_model_calls':0,'human_annotation_calls':0,'semantic_validity_established':False,'scope':'Local terminal metadata and exact exported artifact bytes; scheduler/model execution and weight checks are bound execution attestations.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('alias',choices=('qwen','mistral'));p.add_argument('--output',type=Path,required=True);p.add_argument('--launch-receipt',type=Path);a=p.parse_args()
    report=review(a.alias,a.launch_receipt)
    if a.launch_receipt is not None:report['explicit_launch_receipt']={'path':str(a.launch_receipt),'binding':binding(a.launch_receipt)}
    with a.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','alias','job_id','scheduler','execution_passed','status_counts')}))
