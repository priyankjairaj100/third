"""Prospective admission reader and exclusive activation writer. No entropy or jobs."""
import argparse
import json
from pathlib import Path
from common import (HERE,Watch,binding,canonical,fsync_dir,load,new_json,package,pins,read,require,sealed,sha,seal,valid_binding)

PACKAGE_KEYS={'component','runner','integration','statistics','protocol','preparation','control'}
POLICY={'one_model_per_process':True,'no_retries':True,'stop_on_runtime_error':True,
        'remaining_after_runtime_error':'not_attempted','invalid_json_and_overflow_retain_denominator':True}
def validate_resource_plan(r):
    seal(r)
    require(set(r)=={'schema','stage','packages','request_manifest','request_review','sampling_lock','planned','families','policy','quota','source_role','sha256'},'resource_keys')
    require(r['schema']=='third-llm-selection-resource-plan-v1' and r['stage']=='selection','resource_stage')
    require(set(r['packages'])==PACKAGE_KEYS,'resource_packages')
    for b in list(r['packages'].values())+[r[k] for k in ('request_manifest','request_review','sampling_lock')]:valid_binding(b)
    require(r['policy']==POLICY and all(r['policy'][k] is True for k in POLICY if type(POLICY[k]) is bool),'resource_policy')
    n=r['planned'];require(set(n)=={'canonical_pairs','swap_pairs','per_model','total'} and all(type(v) is int and v>0 for v in n.values()),'planned_counts')
    require(n['canonical_pairs']<=600 and n['swap_pairs']==(n['canonical_pairs']+9)//10 and n['per_model']==n['canonical_pairs']+n['swap_pairs'] and n['total']==2*n['per_model'],'planned_denominator')
    require(set(r['families'])=={'qwen','mistral'},'two_families')
    for alias,f in r['families'].items():
        require(set(f)=={'cpus','host_memory_bytes','gpu_count','gpu_model','slurm_time_seconds','output_limit_bytes','console_limit_bytes','output_dir'},'family_resource_keys')
        require(all(type(f[k]) is int and f[k]>0 for k in f if k not in ('gpu_model','output_dir')),'resource_integer')
        output=Path(f['output_dir']);base=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace/qualification')
        require(output.is_absolute() and output.is_relative_to(base) and '..' not in output.parts and output!=base,'fixed_private_family_output')
        require(f['cpus']==4 and f['host_memory_bytes']==(48 if alias=='qwen' else 64)*1024**3 and f['gpu_count']==1 and f['gpu_model']=='A10080GB','qualified_family_allocation')
        require(f['slurm_time_seconds']<=28800 and f['output_limit_bytes']<=64*1024**2 and f['console_limit_bytes']<=14*1024**2,'bounded_family_resource')
    require(set(r['quota'])=={'reserve_bytes','total_output_commitment_bytes'} and all(type(v) is int for v in r['quota'].values()),'quota_fields')
    require(len({f['output_dir'] for f in r['families'].values()})==2,'distinct_family_outputs')
    require(r['quota']['reserve_bytes']==8*1024**3 and r['quota']['total_output_commitment_bytes']>=sum(f['output_limit_bytes']+f['console_limit_bytes']+1024**2 for f in r['families'].values()),'quota_reserve_commitment')
    require(r['source_role']==pins()['source_role'],'source_role')
    return r

def inspect_admission(path,expected):
    """Deep read-only checks. Caller must separately review the pinned resource plan."""
    w=Watch();w.file(path,expected,1<<20);a=read(path)
    require(set(a)=={'schema','paths','bindings'} and a['schema']=='third-llm-selection-admission-v1','admission_schema')
    paths=a['paths'];bs=a['bindings']
    names={'request_dir','request_review','request_audit_source','selection_result','selection_review','sampling_lock','resource_plan','producer_consumer_report'}|{k+'_package' for k in PACKAGE_KEYS}
    require(set(paths)==names and set(bs)==names-{'request_dir'},'admission_fields')
    p=pins()
    for key in ('request_review','request_audit_source'):require(bs[key]==p[key],'reviewed_actual_evidence:'+key)
    for name in names-{'request_dir'}:
        valid_binding(bs[name])
        if name.endswith('_package'):package(paths[name],bs[name],w)
        else:w.file(paths[name],bs[name],4*1024**2)
    require(Path(paths['integration_package']).resolve()==HERE,'executed_integration_package')
    for name,want in p['packages'].items():require(bs[name+'_package']==want,'pinned_package:'+name)
    resource=validate_resource_plan(read(paths['resource_plan']))
    require(resource['request_manifest']==p['request_result'],'reviewed_actual_request_result')
    require(resource['packages']=={k:bs[k+'_package'] for k in PACKAGE_KEYS},'resource_package_bindings')
    review=read(paths['request_review']);request_dir=Path(paths['request_dir']);request=read(request_dir/'result.json')
    w.file(request_dir/'result.json',resource['request_manifest']);seal(request)
    require(resource['request_review']==bs['request_review'] and resource['sampling_lock']==bs['sampling_lock'],'resource_input_bindings')
    require(review['schema']==p['request_review_schema'] and review['passed'] is True and review['request_stage']=='selection','request_review_status')
    require(review['request_result']==resource['request_manifest'] and review['audit_source']==bs['request_audit_source'],'request_review_identity')
    for key in ('all_input_token_IDs_and_rendered_prompt_hashes_independently_replayed','complete_canonical_and_swap_coverage','overflow_retained_in_full_denominator'):
        require(review.get(key) is True,'request_review_gate:'+key)
    require(request['schema']=='third-llm-annotation-requests-v1' and request['passed'] is True and request['stage']=='complete','request_result_status')
    require(request['package']==p['packages']['preparation'],'request_package')
    for key in ('selection_result','selection_review','sampling_lock'):
        require(request[key]==bs[key] and review[key]==bs[key],'request_lineage:'+key)
    require(review['prepared_files']==request['files'],'review_prepared_files')
    for name,b in request['files'].items():require(Path(name).name==name,'request_member');w.file(request_dir/name,b)
    w.inventory(request_dir,set(request['files'])|{'result.json'})
    require(sum(w.files[str((request_dir/n).absolute())]['bytes'] for n in set(request['files'])|{'result.json'})<=64*1024**2,'prepared_output_cap')
    counts={'canonical_pairs':request['selected_pairs'],'swap_pairs':request['diagnostic_swap_pairs'],'per_model':request['planned_requests']//2,'total':request['planned_requests']}
    require(resource['planned']==counts,'actual_request_counts')
    for key in ('model_inference_calls','human_annotation_calls','planned_human_reviews'):require(type(request[key]) is int and request[key]==0,'preparation_is_preactivation')
    cp=p['component_expectations']
    require(request['protocol_plan_sha256']==cp['protocol_plan_sha256'] and request['source_frame_result']['sha256']==cp['frame_result_sha256'],'request_frame_plan')
    tests=read(paths['producer_consumer_report'])
    require(tests.get('schema')=='third-llm-inference-producer-consumer-validation-v1' and tests.get('passed') is True,'producer_consumer_report')
    require(tests.get('packages')==resource['packages'] and tests.get('request_manifest')==resource['request_manifest'],'producer_consumer_bindings')
    require(type(tests.get('tests')) is int and tests['tests']>0 and tests.get('failures')==0 and tests.get('errors')==0 and tests.get('skipped')==0 and tests.get('model_calls')==0,'producer_consumer_tests')
    contract={'request_manifest':resource['request_manifest'],'request_preparation_package':p['packages']['preparation'],'request_preparation_review':bs['request_review'],
      'selection_result':bs['selection_result'],'selection_review':bs['selection_review'],'sampling_lock':bs['sampling_lock'],
      'kernel_package':p['packages']['component'],'inference_audit_package':bs['integration_package'],'statistics_package':p['packages']['statistics'],
      'producer_consumer_test_report':bs['producer_consumer_report'],'resource_plan':bs['resource_plan'],
      **{k:cp[k] for k in ('protocol_plan_sha256','frame_result_sha256','protocol_evidence_sha256')},
      'models':{k:{'model_id':v['model_id'],'revision':v['revision']} for k,v in cp['models'].items()},
      'stage':'selection','human_annotation_calls':0,'human_completed':False,
      'output_limit_bytes':max(x['output_limit_bytes'] for x in resource['families'].values()),'scope':p['source_role']}
    w.recheck()
    return {'admission':a,'activation_contract':contract,'resource_plan':resource,'watch':w,'request_result':request,'request_review':review}

def read_activation(admission_path,admission_binding,folder,activation_binding,component_contract=None):
    state=inspect_admission(admission_path,admission_binding);p=state['admission']['paths']
    c=component_contract or load('third_integration_bound_component_contract',Path(p['component_package'])/'contract.py')
    require(Path(c.__file__).resolve()==(Path(p['component_package'])/'contract.py').resolve(),'component_contract_import_path')
    activation=c.validate_activation(folder,activation_binding,state['activation_contract'])
    state['watch'].file(Path(folder)/'activation.json',activation_binding,1<<20)
    state['watch'].file(Path(folder)/'intent.json',activation.intent_binding,1<<20)
    state['watch'].inventory(folder,{'intent.json','activation.json'})
    state['watch_files']=dict(state['watch'].files)
    return activation,state

def create_activation(admission_path,admission_binding,out):
    state=inspect_admission(admission_path,admission_binding);p=Path(out)
    require(all(not Path(f['output_dir']).exists() for f in state['resource_plan']['families'].values()),'prior_family_output_refused')
    p.mkdir(mode=0o700,parents=False,exist_ok=False);p.chmod(0o700);fsync_dir(p.parent)
    contract=state['activation_contract'];intent={'schema':'third-llm-only-annotation-activation-intent-v3','contract_sha256':sha(canonical(contract)),'entropy_draws':0,'attempt':0}
    ib=new_json(p/'intent.json',intent);state['watch'].recheck()
    result=sealed({'schema':'third-llm-only-annotation-activation-v3','status':'activated','contract':contract,'intent':ib})
    rb=new_json(p/'activation.json',result);read_activation(admission_path,admission_binding,p,rb)
    return rb

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--admission-sha256',required=True);parser.add_argument('--out',type=Path);parser.add_argument('--create',action='store_true');a=parser.parse_args()
    b=binding(a.admission);require(b['sha256']==a.admission_sha256,'external_admission_sha')
    if a.create:require(a.out is not None,'new_output_required');print(json.dumps(create_activation(a.admission,b,a.out)))
    else:inspect_admission(a.admission,b);print(json.dumps({'admission_valid':True,'activation_created':False,'entropy_draws':0,'model_calls':0}))
if __name__=='__main__':main()
