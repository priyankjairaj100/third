#!/usr/bin/env python3
"""CPU producer/consumer and exact prepared-token qualification. Never loads weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import run_model as R


def test_suite(folder,env,timeout=180,command=None):
    started=time.perf_counter()
    if command is None:command=[sys.executable,'-B','-m','unittest','discover','-s',str(folder),'-p','test_*.py','-v']
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=timeout)
    elapsed=time.perf_counter()-started
    match=re.search(r'^Ran ([1-9][0-9]*) tests? in [0-9.]+s$',result.stderr,re.MULTILINE)
    R.require(result.returncode==0 and match is not None and result.stderr.rstrip().endswith('\nOK'),'unit_suite_failure_or_skips')
    return {'command':command,'tests':int(match[1]),'failures':0,'errors':0,'skipped':0,'wall_seconds':elapsed},result.stdout.encode(),result.stderr.encode()


def verify_tokenizer(alias,model_dir,manifest_path,family,watch):
    pin=family.model;model_dir=Path(model_dir)
    R.require(model_dir.name==pin['revision'],'tokenizer_model_revision')
    watch.add(manifest_path,pin['asset_manifest']);manifest=R.read(manifest_path)
    declared={x['relative_path']:{'bytes':x['size_bytes'],'sha256':x['sha256']} for x in manifest['files']}
    R.require(len(declared)==len(manifest['files']) and declared==pin['asset_files'],'asset_inventory')
    watch.snapshot(model_dir,declared)
    for name,b in declared.items():R.require((model_dir/name).is_file() and (model_dir/name).stat().st_size==b['bytes'],'asset_file_stat')
    # ONLY tokenizer/config bytes are read; safetensors content is not opened.
    for name,b in pin['tokenizer_assets'].items():watch.add(model_dir/name,b,allow_asset_symlink=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,use_fast=True)
    identity={'chat_template_sha256':family.pilot.digest_bytes(family.pilot.canonical(tokenizer.chat_template)),
      'tokenizer_backend_json_sha256':R.sha(tokenizer.backend_tokenizer.to_str().encode()),
      'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__name__,
      'tokenizer_length':len(tokenizer),'tokenizer_vocab_size':tokenizer.vocab_size,'tokenizer_special_ids':tokenizer.all_special_ids}
    R.require(identity==pin['tokenizer_identity'],'qualified_tokenizer_identity')
    return tokenizer,identity


def engineering_files(manifest):
    R.require(set(manifest)=={'schema','selection_rule','source_role','families'} and
        manifest['schema']=='third-historical-output-engineering-replay-v1' and
        manifest['selection_rule']=='Fixed call-0000 from each already exposed pilot; no new generation or empirical labels' and
        manifest['source_role']=='historical engineering only' and set(manifest['families'])=={'qwen','mistral'},'engineering_replay_manifest')
    names={'runtime':'runtime_manifest.json','token_eligibility':'token_eligibility.json',
        'effective_constraint':'effective_constraint_config.json','response':'calls/call-0000/response.json'}
    names.update({key:'calls/call-0000/attempt-0.'+suffix for key,suffix in {
        'input_tokens':'input_tokens.private.json','output_tokens':'output_tokens.private.json',
        'raw_output':'raw.private.txt','guard_steps':'guard_steps.private.json','metrics':'metrics.json'}.items()})
    output={}
    for alias,item in manifest['families'].items():
        R.require(set(item)=={'root','call','files'} and item['call']=='call-0000' and
            set(item['files'])==set(names.values()),'engineering_replay_file_set')
        root=Path(item['root']);R.require(root.is_absolute() and str(root)==str(root.absolute()) and '..' not in root.parts,'engineering_replay_root')
        output[alias]={key:{'path':str(root/name),'binding':item['files'][name]} for key,name in names.items()}
    return output


def engineering_replay(manifest_path,expected_sha,args,audit,components,watch):
    b=R.binding(manifest_path);R.require(b['sha256']==expected_sha,'engineering_replay_manifest_pin');watch.add(manifest_path,b)
    manifest=R.read(manifest_path)
    reports={};integration_common=sys.modules['common'];audit_watch=integration_common.Watch()
    for alias,files in engineering_files(manifest).items():
        values={}
        for key,item in files.items():
            R.require(set(item)=={'path','binding'},'engineering_replay_file_binding')
            path=Path(item['path']);watch.add(path,item['binding'])
            R.require(path.stat().st_size<=8*1024**2,'engineering_replay_file_cap')
            values[key]=path.read_bytes().decode('utf8') if key=='raw_output' else R.strict(path.read_bytes())
        cp=components['contract'].load_pins();model=cp['models'][alias]
        R.require(files['runtime']['binding']==model['pilot_runtime_receipt'],'known_qualified_runtime_receipt')
        rt=values['runtime'];response=values['response']
        R.require(response['status']=='valid' and response['human_completed'] is False and
            response['model_id']==model['model_id'] and response['model_revision']==model['revision'],'historical_valid_model_response')
        R.require(response['raw_output_sha256']==files['raw_output']['binding']['sha256'] and
            response['runtime_manifest_sha256']==files['runtime']['binding']['sha256'],'historical_response_bindings')
        R.require(rt['token_eligibility_sha256']==files['token_eligibility']['binding']['sha256'] and
            rt['effective_constraint_config_sha256']==files['effective_constraint']['binding']['sha256'],'historical_constraint_bindings')
        context=audit.load_tokenizer(alias,str(getattr(args,alias+'_model_dir')),args.repo,rt,
            values['token_eligibility'],values['effective_constraint'],audit_watch)
        judgment,errors=audit.parse_and_replay(values['raw_output'],values['output_tokens'],values['input_tokens'],context)
        R.require(not errors and judgment==response['judgment'],'historical_output_replay_failed')
        audit.check_guard_steps(values['guard_steps'],values['metrics'],'valid',values['output_tokens'])
        for logical,resolved in context['asset_links'].items():R.require(str(Path(logical).resolve())==resolved,'historical_tokenizer_target_changed')
        reports[alias]={'passed':True,'historical_call_id':response['call_id'],'input_tokens':len(values['input_tokens']),
            'output_tokens':len(values['output_tokens']),'artifacts':files,'new_model_calls':0,'new_study_labels':0,
            'raw_token_prefix_grammar_and_strict_judgment_replayed':True,'guard_steps_and_metrics_replayed':True}
    audit_watch.recheck()
    for path,b in audit_watch.files.items():watch.add(path,b)
    return {'manifest':R.binding(manifest_path),'families':reports,'scope':manifest['source_role'],'selection_rule':manifest['selection_rule']}

def qualify(args):
    started=time.perf_counter();os.umask(0o077);pins=R.read(R.ROOT/'expectations.json')
    own=R.binding(R.ROOT/'package_manifest.json');R.require(own['sha256']==args.package_sha256,'external_runner_pin')
    watched=R.package(R.ROOT,own)
    for folder,pin in ((args.component_package,pins['component_package']),(args.integration_package,pins['integration_package']),(args.protocol_package,pins['protocol_package'])):
        watched.update(R.package(folder,pin))
    components=R.load_components(args.component_package);C,L,K=components['contract'],components['ledger'],components['kernel']
    K.runtime_preflight(C.load_pins())
    R.require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'),'qualification_gpu_visibility_must_be_empty')
    R.require(Path(sys.executable).absolute()==Path(args.workspace).absolute()/C.load_pins()['runtime']['interpreter_relative'],'qualified_interpreter')
    R.exact_import('common',args.integration_package/'common.py')
    admission=R.exact_import('activation',args.integration_package/'activation.py')
    independent_audit=R.exact_import('third_cpu_qualification_audit',args.integration_package/'audit.py')
    resource_binding=R.binding(args.resource_plan);R.require(resource_binding['sha256']==args.resource_plan_sha256,'resource_plan_pin')
    resources=admission.validate_resource_plan(R.read(args.resource_plan))
    for role,folder in [('control',args.control_package),('statistics',args.statistics_package)]:
        watched.update(R.package(folder,resources['packages'][role]))
    R.require(resources['packages']['runner']==own and resources['packages']['component']==pins['component_package'] and
              resources['packages']['integration']==pins['integration_package'],'qualification_packages')
    watch=K.Watch()
    for path,b in watched.items():watch.add(path,b)
    watch.add(args.resource_plan,resource_binding)
    prepared=R.read(args.requests_dir/'result.json');rb=R.binding(args.requests_dir/'result.json')
    R.require(rb['sha256']==args.request_result_sha256 and rb==resources['request_manifest'],'prepared_result_pin')
    C.verify_seal(prepared)
    R.require(prepared['schema']=='third-llm-annotation-requests-v1' and prepared['passed'] is True and
              prepared['stage']=='complete' and prepared['request_stage']=='selection','prepared_status')
    R.require(prepared['package']==resources['packages']['preparation'] and prepared['sampling_lock']==resources['sampling_lock'],'prepared_package_lock')
    names={'texts.private.jsonl','requests.private.jsonl','sample.private.jsonl','qwen_tokens.private.jsonl',
           'mistral_tokens.private.jsonl','census.json','input_bindings.json'}
    R.require(set(prepared['files'])==names,'prepared_file_set');watch.snapshot(args.requests_dir,names|{'result.json'})
    watch.add(args.requests_dir/'result.json',rb)
    for name,b in prepared['files'].items():watch.add(args.requests_dir/name,b)
    shared=C.read_jsonl(args.requests_dir/'requests.private.jsonl',prepared['files']['requests.private.jsonl'])
    texts=C.read_jsonl(args.requests_dir/'texts.private.jsonl',prepared['files']['texts.private.jsonl'])
    R.require(len(shared)==resources['planned']['per_model'] and prepared['planned_requests']==resources['planned']['total'],'request_denominator')
    output=L.PrivateOutput(args.output_dir,16*1024**2)
    try:
        env=dict(os.environ,THIRD_INFERENCE_REPO=str(args.repo),THIRD_INFERENCE_PROTOCOL=str(args.protocol_package),
                 THIRD_INFERENCE_COMPONENTS=str(args.component_package),
                 THIRD_INTEGRATION_REPO=str(args.repo),THIRD_INTEGRATION_COMPONENTS=str(args.component_package),THIRD_INTEGRATION_PROTOCOL=str(args.protocol_package),
                 THIRD_INTEGRATION_STATISTICS=str(args.statistics_package),THIRD_INTEGRATION_RUNNER=str(R.ROOT),
                 PYTHONDONTWRITEBYTECODE='1')
        suites={};expected_tests={'components':30,'integration':16,'execution':14,'control':8,'statistics':14}
        for name,folder in [('components',args.component_package),('integration',args.integration_package),('execution',R.ROOT),('control',args.control_package),('statistics',args.statistics_package)]:
            command=None
            if name=='statistics':
                command=[sys.executable,'-B',str(R.ROOT/'run_statistics_tests.py'),'--statistics-package',str(folder),
                    '--statistics-package-sha256',resources['packages']['statistics']['sha256'],'--repo',str(args.repo)]
            receipt,stdout,stderr=test_suite(folder,env,command=command)
            R.require(receipt['tests']==expected_tests[name],'source_bound_suite_count:'+name)
            output.write(output.root/(name+'.tests.stdout.private.txt'),stdout)
            output.write(output.root/(name+'.tests.stderr.private.txt'),stderr);suites[name]=receipt
        output.json('software_suites.json',suites)
        censuses={};full_count=0;families=[]
        for alias in ('qwen','mistral'):
            family=K.FamilyKernel(args.repo,alias,args.protocol_package);families.append(family)
            token_name=alias+'_tokens.private.jsonl';tokens=C.read_jsonl(args.requests_dir/token_name,prepared['files'][token_name])
            rows=C.validate_prepared_rows(shared,tokens,texts,alias,family.protocol)
            tokenizer,identity=verify_tokenizer(alias,getattr(args,alias+'_model_dir'),getattr(args,alias+'_weights_manifest'),family,watch)
            prompt=args.repo/'local_run/model_annotation_v2_20261011'
            # Only the tokenizer fields of state are populated. No load(), model,
            # activation object, final call ID, GPU context or forward exists.
            family.state={'tokenizer':tokenizer,'system_prompt':(prompt/'judge_system_prompt.txt').read_text(),
                          'task_definition':(prompt/'judge_task_definition.txt').read_text()}
            before=time.perf_counter()
            for row,token in zip(rows,tokens):family.verify_tokenization(row,token);full_count+=1
            order=C.schedule(tokens);actual=[tokens[i]['request_index'] for i in order]
            R.require(actual==prepared['model_censuses'][alias]['scheduled_request_indices'],'actual_schedule')
            censuses[alias]={'requests':len(tokens),'canonical':sum(r['orientation']=='canonical' for r in tokens),
                'swaps':sum(r['orientation']=='swapped_diagnostic' for r in tokens),'overflow':sum(not r['context_fit_with_output_reserve'] for r in tokens),
                'all_full_input_ids_masks_rendered_and_input_row_hashes_equal':True,'scheduled_request_indices':actual,
                'tokenizer_identity':identity,'retokenization_seconds':time.perf_counter()-before,
                'model_weights_read':False,'model_weights_loaded':False,'model_calls':0}
            family.recheck();family.state=None
        R.require(full_count==resources['planned']['total'],'complete_two_family_token_replay')
        replay=engineering_replay(args.engineering_replay_manifest,args.engineering_replay_sha256,args,independent_audit,components,watch)
        output.json('historical_engineering_replay.json',replay)
        watch.recheck()
        for family in families:family.recheck()
        output.json('token_replay.json',censuses);output.json('input_bindings.json',watch.files)
        result={'schema':'third-llm-inference-producer-consumer-validation-v1','passed':True,'packages':resources['packages'],
            'request_manifest':rb,'resource_plan':resource_binding,'tests':sum(x['tests'] for x in suites.values()),
            'failures':0,'errors':0,'skipped':0,'actual_request_rows_replayed':full_count,'model_censuses':censuses,
            'historical_engineering_output_replays_passed':True,'historical_engineering_replay_manifest':replay['manifest'],
            'model_calls':0,'model_weights_loaded':False,'weight_bytes_rehashed':False,'activation_created':False,
            'production_call_ids_created':False,'entropy_draws':0,'human_annotation_calls':0,
            'synthetic_software_fixture_outcomes_are_not_source_labels':True,'new_gpu_orchestration_tested':False,
            'first_real_generation_must_be_recorded_planned_attempt_after_activation':True,
            'files':dict(output.files),'elapsed_seconds':time.perf_counter()-started}
        result['sha256']=R.sha(R.canonical(result));output.json('qualification.json',result);output.recheck();return result
    except Exception as exc:
        output.json('failure.json',{'schema':'third-llm-inference-cpu-qualification-failure-v1','passed':False,
            'error_type':type(exc).__name__,'error':str(exc),'model_calls':0,'activation_created':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','workspace','protocol-package','component-package','integration-package','requests-dir','resource-plan','output-dir',
              'qwen-model-dir','qwen-weights-manifest','mistral-model-dir','mistral-weights-manifest','control-package','statistics-package','engineering-replay-manifest'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('package-sha256','request-result-sha256','resource-plan-sha256','engineering-replay-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=qualify(a);print(json.dumps({k:r[k] for k in ('schema','passed','tests','actual_request_rows_replayed','model_calls')}))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'passed':False,'error_type':type(exc).__name__,'error':str(exc),'model_calls':0}),file=sys.stderr);raise SystemExit(2)
