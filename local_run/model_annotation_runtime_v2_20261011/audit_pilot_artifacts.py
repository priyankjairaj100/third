"""Independent post-run artifact checks. Requires jsonschema; never changes model labels."""
import argparse, collections, hashlib, json, pathlib
from jsonschema import Draft202012Validator
p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('--repo',type=pathlib.Path,required=True);a=p.parse_args();root=a.root
repo=a.repo
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
read=lambda p:json.loads(p.read_text())
lines=lambda p:[json.loads(x) for x in p.read_text().split('\n') if x]
pre=root/'preparation';out=root/'inference';m=read(pre/'manifest.json');summary=read(out/'summary.json')
assert summary['completed'] is True and summary['fatal_error'] is None
assert sha(pre/'inputs.jsonl')==m['inputs_sha256']
assert sha(pre/'weights_manifest.json')==m['weights_manifest_sha256']
assert sha(pre/'preparation.json')==m['preparation_sha256']
pr=read(pre/'preparation.json');assert pr['source_pack_seal_verified'] is True
assert sha(pre/'private_source_mapping.json')==pr['source_mapping_sha256']
assert sha(repo/'local_run/model_annotation_runtime_v2_20261011/run_pilot.py')==m['driver_sha256']
plan=repo/'local_run/model_annotation_v2_20261011/PLAN.json';assert sha(plan)==m['plan_sha256']
inputs=lines(pre/'inputs.jsonl');responses=lines(out/'responses.jsonl');attempts=lines(out/'attempts.jsonl')
assert len(inputs)==len(responses)==44
assert len({r['pair_id'] for r in inputs})==40
assert sum(r['orientation']=='canonical' for r in inputs)==40
schema=read(plan.parent/'model_response.schema.json');validator=Draft202012Validator(schema)
runtime=read(out/'runtime_manifest.json')
assert runtime['gpu'].find('A100')>=0 and runtime['visible_gpus']==1
assert runtime['dtype']=='bfloat16' and runtime['attention_implementation']=='eager'
assert runtime['deterministic_algorithms'] is True
by_pair={};swaps=[]
for i,(item,response) in enumerate(zip(inputs,responses)):
 validator.validate(response)
 assert response['human_completed'] is False and response['annotator_type']=='model'
 assert response['pair_id']==item['pair_id'] and response['orientation']==item['orientation']
 assert response['input_sha256']==hashlib.sha256(canon(item)).hexdigest()
 assert response['runtime_manifest_sha256']==sha(out/'runtime_manifest.json')
 assert response['prompt_sha256']==sha(out/'prompt_manifest.json')
 assert response['decoding_config_sha256']==sha(out/'decoding_config.json')
 c=out/'calls'/f'call-{i:04d}'
 assert read(c/'response.json')==response
 first=read(c/'attempt-0.messages.private.json')
 shown=json.loads(first[1]['content']); assert shown['documents']==[{'label':'A','text':item['text_a']},{'label':'B','text':item['text_b']}]
 these=[x for x in attempts if x['input_index']==i];assert 1<=len(these)<=2
 assert [x['attempt'] for x in these]==list(range(len(these)))
 for j,t in enumerate(these):
  assert t['full_input_tokenization'] is True and t['truncation_applied'] is False
  if 'output_tokens' in t:
   assert t['input_tokens']+384<=16384 and t['output_tokens']<=384
   assert sha(c/f'attempt-{j}.raw.private.txt')==t['raw_output_sha256']
   assert len(read(c/f'attempt-{j}.output_tokens.private.json'))==t['output_tokens']
  if j==1: assert t['format_only_retry'] is True
 if response['raw_output_sha256']:
  hashes=[sha(x) for x in c.glob('attempt-*.raw.private.txt')];assert response['raw_output_sha256'] in hashes
 if item['orientation']=='canonical': by_pair[item['pair_id']]=(item,response)
 else: swaps.append((item,response))
changed=0;binary_changed=0;comparable=0;positive={'exact_copy','substantially_same_meaning'}
for item,response in swaps:
 original,base=by_pair[item['pair_id']]
 assert item['text_a']==original['text_b'] and item['text_b']==original['text_a']
 if base['status']==response['status']=='valid':
  comparable+=1;x=base['judgment']['category'];y=response['judgment']['category'];changed+=x!=y;binary_changed+=(x in positive)!=(y in positive)
counts=dict(collections.Counter(r['status'] for r in responses));assert counts==summary['status_counts_all_calls']
assert changed==summary['swapped_order_diagnostics'].get('category_changed',0)
assert binary_changed==summary['swapped_order_diagnostics'].get('binary_positive_changed',0)
assert not summary['v1_human_gate_satisfied'] and not summary['confirmatory_study_ready'] and summary['human_responses_created']==0
report={'schema':'third-pilot-independent-artifact-audit-v1','passed':True,'scope':'Actual pilot response provenance, schema, raw output hashes, complete input coverage, swapped order integrity, runtime and zero-human gate checks; not semantic correctness','pairs':40,'scheduled_calls':44,'attempts':len(attempts),'status_counts':counts,'swapped_comparable':comparable,'swapped_category_changed':changed,'swapped_binary_changed':binary_changed,'max_input_tokens':max(x.get('input_tokens',0) for x in attempts),'total_input_tokens':sum(x.get('input_tokens',0) for x in attempts),'total_generated_tokens':sum(x.get('output_tokens',0) for x in attempts),'generation_seconds':sum(x.get('generation_seconds',0) for x in attempts),'peak_gpu_allocated_bytes':max(x.get('peak_gpu_allocated_bytes',0) for x in attempts),'peak_gpu_reserved_bytes':max(x.get('peak_gpu_reserved_bytes',0) for x in attempts),'process_peak_rss_kib':summary['process_peak_rss_kib'],'source_bindings':{str(p.relative_to(root)):sha(p) for p in [pre/'manifest.json',pre/'preparation.json',pre/'inputs.jsonl',pre/'weights_manifest.json',out/'runtime_manifest.json',out/'responses.jsonl',out/'attempts.jsonl',out/'summary.json']}}
(root/'independent_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
