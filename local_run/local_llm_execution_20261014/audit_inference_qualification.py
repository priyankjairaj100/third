#!/usr/bin/env python3
"""Local, fixed13573 terminal-evidence review. No remote/model calls or artifact edits."""
import base64,json,math,re,sys
from pathlib import Path
from types import SimpleNamespace
import audit_requests as A
R=Path('/Users/priyankjairaj/Downloads/ACL/Three');W=A.W;H=Path(__file__).resolve().parent
QROOT=W/'qualification/llm-only-v3-execution-qualification-a01'
RAW=H/'inference_qualification_terminal_export.json';OUT=H/'inference_qualification_independent_review.json'
req=A.require
bind=lambda p:A.binding(A.read(p))
roles={'component':'llm_annotation_inference_v1_20261014','runner':'llm_annotation_execution_v1_20261014',
 'integration':'llm_annotation_integration_v1_20261014','statistics':'llm_annotation_statistics_v1_20261014',
 'control':'llm_annotation_control_v1_20261014','preparation':'llm_annotation_requests_v2_20261014','protocol':'model_annotation_v3_20261014'}
remote=lambda role:W/'tools'/roles[role].replace('_','-')

def run():
 req(bind(H/'audit_requests.py')['sha256']=='eaef54bb18a9e29ea6183fc1372bd0932b7b9698213bc9dc331faa3ae9c7616a','review_helper_pin')
 initial=bind(RAW);e=A.strict(A.read(RAW));req(e['schema']=='third-llm-execution-terminal-export-v1' and e['job_id']=='13573' and e['root']==str(QROOT) and e['queue']=='','terminal_identity')
 expected_accounting=['13573|COMPLETED|0:0|22||csis.cn1','13573.batch|COMPLETED|0:0|22|1060K|csis.cn1']
 req(e['accounting'].splitlines()==expected_accounting,'terminal_all_steps')
 files={}
 for n,v in e['files'].items():
  req(not Path(n).is_absolute() and '..' not in Path(n).parts,'transport_path');raw=base64.b64decode(v['base64'],validate=True)
  req(A.binding(raw)=={k:v[k] for k in ('bytes','sha256')},'transport_hash');files[n]=raw
 req(len(files)==22 and sum(map(len,files.values()))==e['total_bytes']==90821,'transport22files_bytes')
 extracted=R/'local_workspace/reports/llm-only-v3-execution-qualification-a01'
 for n,raw in files.items():req(A.read(extracted/n)==raw,'extracted_exact:'+n)
 load=lambda n:A.strict(files[n]);q=load('run/output/qualification.json');A.seal(q)
 req(q['schema']=='third-llm-inference-producer-consumer-validation-v1' and q['passed'] is True,'qualified_status')
 req(q['tests']==82 and q['skipped']==q['errors']==q['failures']==0 and q['actual_request_rows_replayed']==198,'qualified_counts')
 for k in ('activation_created','model_weights_loaded','new_gpu_orchestration_tested','production_call_ids_created','weight_bytes_rehashed'):req(q[k] is False,'scope_false:'+k)
 for k in ('model_calls','human_annotation_calls','entropy_draws'):req(type(q[k]) is int and q[k]==0,'scope_zero:'+k)
 req(q['first_real_generation_must_be_recorded_planned_attempt_after_activation'] is True and q['synthetic_software_fixture_outcomes_are_not_source_labels'] is True,'scope_limits')
 expected_outputs={x+'.tests.'+y+'.private.txt' for x in ('components','integration','execution','control','statistics') for y in ('stderr','stdout')}|{'software_suites.json','token_replay.json','historical_engineering_replay.json','input_bindings.json'}
 req(set(q['files'])==expected_outputs,'output14files')
 required={'launch_intent.json','launch_receipt.json','run/process_completion.json','run/stdout.txt','run/stderr.txt','slurm-13573.out','slurm-13573.err'}|{'run/output/'+x for x in expected_outputs|{'qualification.json'}}
 req(set(files)==required,'terminal_exact_inventory')
 for n,v in q['files'].items():req(A.binding(files['run/output/'+n])==v,'qualification_file_binding')
 req(sum(len(v) for n,v in files.items() if n.startswith('run/output/'))<=16<<20,'output16MiB_cap')
 plan=A.strict(A.read(H/'inference_resource_plan.json'));A.seal(plan);planb=bind(H/'inference_resource_plan.json')
 req(q['resource_plan']==planb and q['packages']==plan['packages'] and set(q['packages'])==set(roles),'resource_packages')
 expected_inputs={}
 def expect(path,value):
  path=str(path);req(path not in expected_inputs or expected_inputs[path]==value,'conflicting_expected_input');expected_inputs[path]=value
 for role,foldername in roles.items():
  folder=R/'local_run'/foldername;manifest=folder/'package_manifest.json';req(bind(manifest)==q['packages'][role],'package_manifest:'+role)
  members=A.strict(A.read(manifest))['files']
  for n,v in members.items():req(bind(folder/n)==v,'package_member:'+role+'/'+n)
  if role!='preparation':
   expect(remote(role)/'package_manifest.json',q['packages'][role])
   for n,v in members.items():expect(remote(role)/n,v)
 req(q['packages']['runner']['sha256']=='221774c18ac2b1ab99cc37880cec45c2e36c4d547a882ef81a7d329d86472387','sealed_runner')
 control=A.load('terminal_qual_control',R/'local_run'/roles['control']/'control.py')
 launch=load('launch_receipt.json');intent=load('launch_intent.json');process=load('run/process_completion.json')
 req({k:v for k,v in launch.items() if k not in ('returncode','stdout','stderr','job_id')}==intent,'launch_intent')
 req(launch['schema']=='third-llm-execution-launch-v1' and launch['returncode']==0 and launch['job_id']=='13573' and launch['mode']=='qualification' and launch['root']==str(QROOT),'launch_status')
 args=SimpleNamespace(mode='qualification',resource_plan_sha256=planb['sha256'])
 req(process['command']==launch['child_command']==control.child_command(args,plan),'exact_child_command')
 expectedcmd=['sbatch','--parsable','--partition=cpu-short','--nodelist=csis.cn1','--cpus-per-task=4','--mem=8G','--time=20','--job-name=third-llm-infer-qualification','--output',str(QROOT/'slurm-%j.out'),'--error',str(QROOT/'slurm-%j.err'),str(remote('control')/'execute.sbatch'),str(remote('control')),'--mode','qualification','--control-package-sha256',q['packages']['control']['sha256'],'--resource-plan-sha256',planb['sha256']]
 req(launch['command']==expectedcmd,'exact_sbatch_command')
 req(process['returncode']==0 and process['mode']=='qualification' and process['slurm_job_id']=='13573' and process['console_cap_exceeded'] is False and process['retry_attempts']==process['human_annotation_calls']==0,'process_complete')
 req(process['console_bytes']==len(files['run/stdout.txt'])+len(files['run/stderr.txt'])<=14<<20,'console_cap')
 req(0<q['elapsed_seconds']<process['child_wall_seconds_through_exit']<=22,'timing_scope')
 req(process['bindings']=={k:v for k,v in launch['bindings'].items() if k!='request_terminal'},'launch_compute_binding_equality')
 req(launch['reserve_bytes']==8<<30 and launch['admitted_remaining_output_commitment_bytes']==224<<20 and launch['quota']['available_soft_bytes']>= (8<<30)+(224<<20),'quota_policy_at_launch')
 req(launch['automatic_retry'] is False and launch['sampling_entropy_draws']==launch['human_annotation_calls']==0,'control_no_retry_entropy')
 review=A.strict(A.read(H/'request_independent_review.json'));req(bind(H/'request_independent_review.json')==plan['request_review'],'request_review_pin')
 req(review['passed'] is True and review['request_result']==q['request_manifest']==plan['request_manifest'],'request_identity')
 req(review['all_input_token_IDs_and_rendered_prompt_hashes_independently_replayed'] is True,'prior_actual_local_token_replay')
 te=A.strict(A.read(H/'requests_terminal_export.json'));req(A.binding(A.read(H/'requests_terminal_export.json'))==review['terminal_export'],'request_export_pin')
 request_files={n:base64.b64decode(v['base64'],validate=True) for n,v in te['files'].items() if n.startswith('run/output/')}
 for n,raw in request_files.items():expect(control.REQUEST_ROOT/n,A.binding(raw))
 req(A.binding(request_files['run/output/result.json'])==q['request_manifest'],'request_raw_binding')
 suites=load('run/output/software_suites.json');req(set(suites)=={'components','integration','execution','control','statistics'},'five_suites')
 suite_roles={'components':'component','integration':'integration','execution':'runner','control':'control','statistics':'statistics'}
 for name,n in {'components':30,'integration':16,'execution':14,'control':8,'statistics':14}.items():
  row=suites[name];req(row['tests']==n and row['errors']==row['failures']==row['skipped']==0,'suite_counts')
  log=files['run/output/'+name+'.tests.stderr.private.txt'].decode();req(log.count(' ... ok\n')==n and re.search(r'^Ran '+str(n)+r' tests in [0-9.]+s$',log,re.M) and log.rstrip().endswith('\nOK'),'actual_test_log')
  cmd=[str(control.PYTHON),'-B','-m','unittest','discover','-s',str(remote(suite_roles[name])),'-p','test_*.py','-v']
  if name=='statistics':cmd=[str(control.PYTHON),'-B',str(remote('runner')/'run_statistics_tests.py'),'--statistics-package',str(remote('statistics')),'--statistics-package-sha256',q['packages']['statistics']['sha256'],'--repo',str(control.REPO)]
  req(row['command']==cmd and math.isfinite(row['wall_seconds']) and row['wall_seconds']>0,'test_command')
 tokens=load('run/output/token_replay.json');req(tokens==q['model_censuses'] and set(tokens)=={'qwen','mistral'},'two_token_censuses')
 cp=A.strict(A.read(R/'local_run'/roles['component']/'expectations.json'))
 for alias,t in tokens.items():
  prior=review['model_censuses'][alias];req(t['requests']==99 and t['canonical']==90 and t['swaps']==9 and t['overflow']==0,'token_complete_counts')
  req(t['scheduled_request_indices']==prior['scheduled_request_indices'] and t['tokenizer_identity']==prior['tokenizer_identity']==cp['models'][alias]['tokenizer_identity'],'token_schedule_identity')
  req(t['all_full_input_ids_masks_rendered_and_input_row_hashes_equal'] is True and t['model_calls']==0 and t['model_weights_read'] is False and t['model_weights_loaded'] is False,'token_execution_scope')
 history=load('run/output/historical_engineering_replay.json');hm=A.strict(A.read(H/'engineering_replay_manifest.json'));hmb=bind(H/'engineering_replay_manifest.json')
 req(q['historical_engineering_output_replays_passed'] is True and q['historical_engineering_replay_manifest']==history['manifest']==hmb,'historical_manifest_binding')
 expect(control.ADMISSION/'engineering_replay_manifest.json',hmb);expect(control.ADMISSION/'resource_plan.json',planb)
 inputs=load('run/output/input_bindings.json');req(len(inputs)==117 and all(set(v)=={'binding','resolved_path'} for v in inputs.values()),'input117_schema')
 for alias,dirname in [('qwen','annotation-pilot-v5-20261011'),('mistral','mistral-pilot-v1-20261011')]:
  recorded=history['families'][alias];m=hm['families'][alias];base=R/'local_workspace/reports'/dirname/'a01/inference'
  actual={str(Path(m['root'])/n):v for n,v in m['files'].items()};req({v['path']:v['binding'] for v in recorded['artifacts'].values()}==actual,'exact_historical9_inputs')
  for n,v in m['files'].items():req(bind(base/n)==v,'local_historical_byte_identity');expect(Path(m['root'])/n,v)
  response=A.strict(A.read(base/'calls/call-0000/response.json'));req(recorded['historical_call_id']==response['call_id'] and recorded['passed'] is True and recorded['raw_token_prefix_grammar_and_strict_judgment_replayed'] is True and recorded['guard_steps_and_metrics_replayed'] is True,'two_historical_passes')
  req(recorded['new_model_calls']==recorded['new_study_labels']==0,'historical_not_new_labels')
  for key in ('input_tokens','output_tokens'):req(recorded[key]==len(A.strict(A.read(base/('calls/call-0000/attempt-0.'+key+'.private.json')))),'historical_token_counts')
  model=cp['models'][alias];modeldir=control.MODELS[alias];expect(control.WEIGHTS[alias],model['asset_manifest'])
  for name,v in model['asset_files'].items():
   if name.endswith('.safetensors'):continue
   logical=str(modeldir/name);req(logical in inputs and inputs[logical]['binding']==v,'tokenizer_metadata_pin');expect(logical,v)
   resolved=inputs[logical]['resolved_path'];expect(resolved,v);req(inputs[resolved]['resolved_path']==resolved,'resolved_asset_binding')
 for alias in ('qwen','mistral'):
  ip=A.strict(A.read(R/'local_run'/roles['integration']/'expectations.json'))
  helper=ip['prior_audit_helpers'][alias];req(bind(R/helper['relative_path'])==helper['binding'],'audit_helper_local');expect(control.REPO/helper['relative_path'],helper['binding'])
  guard=str(Path(cp['models'][alias]['pilot_code_relative']).parent/'bounded_reason.py');req(bind(R/guard)==cp['source_files'][guard],'guard_local');expect(control.REPO/guard,cp['source_files'][guard])
 schema='local_run/model_annotation_v2_20261011/model_judgment.schema.json';expect(control.REPO/schema,cp['source_files'][schema])
 req(set(expected_inputs)==set(inputs),'exact_all117_input_paths')
 for path,value in expected_inputs.items():req(inputs[path]['binding']==value,'all117_input_binding_equality')
 req(not any(p.endswith('.safetensors') for p in inputs),'no_weight_content_hashes')
 req(bind(RAW)==initial,'unchanged_export')
 return {'schema':'third-llm-inference-qualification-independent-review-v1','passed':True,'qualification':A.binding(files['run/output/qualification.json']),
  'qualification_internal_seal':q['sha256'],'terminal_export':initial,'job_id':'13573','state':'COMPLETED','exit_code':'0:0','elapsed_seconds':22,'node':'csis.cn1',
  'transport_files':22,'transport_bytes':90821,'all_transport_and_extracted_bytes_verified':True,'all117_input_bindings_verified':True,
  'packages':q['packages'],'request_manifest':q['request_manifest'],'request_independent_review':plan['request_review'],'resource_plan':planb,
  'tests':82,'suite_counts':{k:v['tests'] for k,v in suites.items()},'failures':0,'errors':0,'skipped':0,
  'actual_token_replay_rows':198,'canonical_pairs':90,'diagnostic_swaps':9,'per_model_requests':99,'overflow_requests':0,
  'matches_prior_independent_full_local_token_replay':True,'historical_engineering_replay_manifest':hmb,'historical_outputs_replayed':2,
  'historical_artifacts_independently_matched':18,'model_calls':0,'human_annotation_calls':0,'model_weights_loaded':False,'weight_bytes_rehashed':False,
  'activation_created':False,'production_call_ids_created':False,'entropy_draws':0,'new_gpu_orchestration_tested':False,
  'whole_child_seconds_through_exit':process['child_wall_seconds_through_exit'],'children_maxrss_kib_linux':process['children_maxrss_kib_linux'],
  'maxrss_interpretation':'Child resource.getrusage maximum1122016KiB; Slurm1060K is wrapper-only and not full process memory.',
  'checks':['exact qualified package members and sealed source dependencies','full launch/intent/child/suite command bindings',
   'actual raw82-test logs with zero failures/errors/skips','198 tokenizer execution receipts and exact schedules/identities agree with prior independent local full-ID replay',
   'two fixed already-exposed historical calls with exact18artifact bytes; reported token/grammar/guard success','all117 input identities plus source-bound absence of weight/model/activation calls'],
  'limitations':['Remote tokenizer and historical guard execution is attested by the sealed CPU program and terminal artifacts; this review does not rerun that job.',
   'Prior request review independently replayed every full input ID/render hash locally; complete source/frame mapping remains separately bound earlier execution evidence.',
   'No new GPU inference, labels, semantic validity, publisher authenticity, held-out validation or original human-gate completion established.'],
  'source_authenticity_verified':False,'semantic_validity_established':False,'confirmatory_study_ready':False,
  'review_source':bind(Path(__file__))}

if __name__=='__main__':
 req(not OUT.exists(),'exclusive_review_output')
 result=run()
 with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'passed':result['passed'],'review':bind(OUT),'qualification':result['qualification']}))
