"""Synthetic software fixtures only; no production activation, entropy or inference."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import common as c
import activation as a
import audit as d
import ingest as g
REPO=Path(os.environ.get('THIRD_INTEGRATION_REPO',os.environ.get('THIRD_INFERENCE_REPO',HERE.parents[1])))
COMP=Path(os.environ.get('THIRD_INTEGRATION_COMPONENTS',os.environ.get('THIRD_INTEGRATION_COMPONENT',REPO/'local_run/llm_annotation_inference_v1_20261014')))
PROTO=Path(os.environ.get('THIRD_INTEGRATION_PROTOCOL',REPO/'local_run/model_annotation_v3_20261014'))
STATS=Path(os.environ.get('THIRD_INTEGRATION_STATISTICS',REPO/'local_run/llm_annotation_statistics_v1_20261014'))
C=c.load('contract',COMP/'contract.py');L=c.load('ledger',COMP/'ledger.py')
P=c.load('integration_test_protocol',PROTO/'protocol.py');S=c.load('integration_test_stats',STATS/'statistics_core.py')
CP=c.pins()['component_expectations'];MODELS=CP['models'];B={'bytes':1,'sha256':'a'*64}
SCHEMA=c.read(REPO/'local_run/model_annotation_v2_20261011/model_judgment.schema.json')
GOOD={'category':'exact_copy','distinction_types':['none'],'reason':'The same content.'}

def resource():
    f=lambda alias:{'cpus':4,'host_memory_bytes':(48 if alias=='qwen' else 64)*1024**3,'gpu_count':1,'gpu_model':'A10080GB',
      'slurm_time_seconds':14400,'output_limit_bytes':64*1024**2,'console_limit_bytes':14*1024**2,
      'output_dir':'/nfs_home/users/poonam/ccu-acl2027/local_workspace/qualification/SOFTWARE_FIXTURE_ONLY/'+alias}
    return c.sealed({'schema':'third-llm-selection-resource-plan-v1','stage':'selection','packages':{k:B for k in a.PACKAGE_KEYS},
      'request_manifest':B,'request_review':B,'sampling_lock':B,'planned':{'canonical_pairs':1,'swap_pairs':1,'per_model':2,'total':4},
      'families':{k:f(k) for k in MODELS},'policy':dict(a.POLICY),'quota':{'reserve_bytes':8*1024**3,'total_output_commitment_bytes':192*1024**2},'source_role':c.pins()['source_role']})

def fixture_tokens(alias):
    m=MODELS[alias];pair='c'*64;seed=P.model_seed(CP['protocol_plan_sha256'],CP['frame_result_sha256'],pair,m['model_id'],m['revision']);rows=[]
    for i,(stage,orientation) in enumerate([('selection','canonical'),('selection_swap','swapped_diagnostic')]):
        rows.append({'request_index':i,'selection_index':0,'stage':stage,'orientation':orientation,'opaque_pair_id':pair,
          'model_id':m['model_id'],'model_revision':m['revision'],'model_seed':seed,'input_token_ids':[100],
          'input_token_ids_sha256':c.sha(c.canonical([100])),'input_tokens':1,'activation_lock_sha256':None,'call_id':None})
    return rows

def calls(alias='qwen'):
    return d.planned_calls(fixture_tokens(alias),B,B['sha256'],MODELS[alias],CP)

def validate(value):
    from jsonschema import Draft202012Validator
    Draft202012Validator(SCHEMA).validate(value);c.require(len(value['reason'].split())<=80,'words')

def tiny_context(raw):
    """Real pinned LMFE/guard replay with a one-token software-only vocabulary."""
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    guard=c.load('integration_test_actual_guard',REPO/'local_run/model_annotation_runtime_v5_20261011/bounded_reason.py')
    helper=c.load('integration_test_actual_prior_auditor',REPO/c.pins()['prior_audit_helpers']['qwen']['relative_path'])
    table={0:raw,**{i+2:chr(i+32) for i in range(95)}}
    decode=lambda ids,**kw:''.join(table.get(i,'') for i in ids)
    data=TokenEnforcerTokenizerData([(i,v,False) for i,v in table.items()],decode,1,False,97)
    return {'schema':SCHEMA,'tokenizer':NS(decode=decode),'helper':helper,'data':data,'guard':guard,
      'policy':guard.StandaloneTokenPolicy(frozenset(table),frozenset({1}),decode),'alphabet':data.tokenizer_alphabet,
      'rawdecode':decode,'eligible':frozenset(table),'eos':frozenset({1})}

def report(alias,responses):
    return {'schema':'third-llm-output-independent-audit-v1','passed':True,'alias':alias,
      'all_call_identities_denominators_and_tokens_verified':True,'request_manifest':B,'activation':B,
      'planned_calls':2,'canonical_calls':1,'diagnostic_calls':1,
      'canonical_observations':g.observed_from_responses(responses),'diagnostic_observations':g.observed_from_responses(responses,True)}

class ActivationTests(unittest.TestCase):
    def test_resource_requires_all_seven_packages_and_exact_family_allocations(self):
        r=resource();a.validate_resource_plan(r)
        for change in ('control','memory','counts','retry','reserve','output'):
            b=copy.deepcopy(r);b.pop('sha256')
            if change=='control':b['packages'].pop('control')
            if change=='memory':b['families']['qwen']['host_memory_bytes']=1
            if change=='counts':b['planned']['total']=3
            if change=='retry':b['policy']['no_retries']=False
            if change=='reserve':b['quota']['reserve_bytes']=1
            if change=='output':b['families']['qwen']['output_dir']='/tmp/unbound'
            with self.assertRaises(ValueError):a.validate_resource_plan(c.sealed(b))
    def test_resource_rejects_bool_integer_and_unsealed_mutation(self):
        r=resource();r['planned']['canonical_pairs']=True
        with self.assertRaises(ValueError):a.validate_resource_plan(r)
        r.pop('sha256')
        with self.assertRaises(ValueError):a.validate_resource_plan(c.sealed(r))
    def test_byte_identity_and_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.json';b=c.new_json(p,{'fixture_only':True})
            with self.assertRaises(FileExistsError):c.new_json(p,{})
            w=c.Watch();w.file(p,b);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):w.recheck()
    def test_strict_json_rejects_duplicate_nonfinite(self):
        for raw in ('{"a":1,"a":2}','1e309','NaN'):
            with self.assertRaises(ValueError):c.strict(raw)
    def test_sealed_sources_and_actual_request_review_are_bound(self):
        p=c.pins()
        for alias,root in [('component',COMP),('protocol',PROTO),('statistics',STATS)]:c.package(root,p['packages'][alias])
        self.assertEqual(p['request_result']['sha256'],'ca81b6bf7f60d6805245a4c9d74e1e42e2db4794caa04d60e74f7052d97fc6a2')

class OutputTests(unittest.TestCase):
    def test_independent_call_identity_matches_component(self):
        for alias in MODELS:
            token=fixture_tokens(alias);activation=C.Activation(B,{'request_manifest':B},B)
            self.assertEqual(calls(alias),C.resolve_calls(token,B,activation,P))
            self.assertEqual(calls(alias)[0]['model_seed'],calls(alias)[1]['model_seed'])
            self.assertNotEqual(calls(alias)[0]['call_id'],calls(alias)[1]['call_id'])
    def test_independent_call_rejects_seed_revision_stage_mutation(self):
        for field,value in [('model_seed',1),('model_revision','wrong'),('stage','validation')]:
            t=fixture_tokens('qwen');t[0][field]=value
            with self.assertRaises(ValueError):d.planned_calls(t,B,B['sha256'],MODELS['qwen'],CP)
    def test_stop_on_runtime_error_preserves_reordered_denominator(self):
        cc=calls();rr=[C.envelope(cc[0],'not_attempted',error={'type':'Stop','message':'previous error'}),C.envelope(cc[1],'runtime_error',error={'type':'Runtime','message':'fixture'})]
        self.assertEqual(d.check_responses(rr,cc,[1,0]),{'not_attempted':1,'runtime_error':1})
        rr[0]=C.envelope(cc[0],'context_overflow')
        with self.assertRaises(ValueError):d.check_responses(rr,cc,[1,0])
    def test_incomplete_missing_or_wrong_identity_is_not_accepted(self):
        cc=calls();rr=[C.envelope(x,'context_overflow') for x in cc]
        for changed in [rr[:1],[dict(rr[0],call_id='f'*64),rr[1]],[dict(rr[0],status='missing_output'),rr[1]]]:
            with self.assertRaises(ValueError):d.check_responses(changed,cc,[0,1])
    def test_actual_guard_and_strict_json_replay(self):
        raw=json.dumps(GOOD);j,errors=d.parse_and_replay(raw,[0,1],[100],tiny_context(raw))
        self.assertEqual(j,GOOD);self.assertEqual(errors,{})
        raw=json.dumps(dict(GOOD,reason='word '*81));j,errors=d.parse_and_replay(raw,[0],[100],tiny_context(raw))
        self.assertTrue(errors);self.assertIn('strict_json_schema_words',errors)
    def test_wrong_raw_decode_identity_is_integrity_failure(self):
        raw=json.dumps(GOOD);ctx=tiny_context(raw)
        with self.assertRaisesRegex(ValueError,'raw_token_decode_identity'):d.parse_and_replay(raw+' ',[0],[100],ctx)
    def test_guard_counts_failure_and_token_sequence(self):
        step={'step':0,'accepted_token_id':5,'candidates_considered':2,'finite_native_candidates':3,'rejected_candidates':{'blocked':1},'prefix_mask_verified':True,'selection_seconds':0.1}
        metrics={'guard_steps':1,'guard_candidates_considered':2,'guard_selection_seconds':0.1}
        d.check_guard_steps([step],metrics,'valid',[5])
        with self.assertRaises(ValueError):d.check_guard_steps([step],metrics,'valid',[6])
        bad=dict(step,accepted_token_id=None,candidates_considered=3,rejected_candidates={'blocked':3});metrics['guard_candidates_considered']=3
        d.check_guard_steps([bad],metrics,'runtime_error')
        with self.assertRaises(ValueError):d.check_guard_steps([bad],metrics,'valid')
    def test_qwen_runtime_uses_actual_qualified_receipt_fields(self):
        maps=c.pins()['family_runtime_packages'];self.assertNotIn('huggingface-hub',maps['qwen']);self.assertIn('huggingface-hub',maps['mistral'])

class IngestionTests(unittest.TestCase):
    def completed_fixture(self,alias):
        raw=json.dumps(GOOD).encode();cc=calls(alias)
        with tempfile.TemporaryDirectory() as td:
            ledger=L.Ledger(Path(td)/'synthetic-only',100000,cc,validate)
            for call in cc:
                wr=ledger.begin(call);wr.raw('raw.private.txt',raw);wr.json('output_tokens.private.json',[0,1]);wr.json('guard_steps.private.json',[]);wr.json('metrics.json',{'synthetic_only':True})
                rr=C.envelope(call,'valid',GOOD,c.sha(raw),'b'*64,dict(wr.files));ledger.record(wr,rr)
            responses,summary=ledger.finish();d.check_responses(responses,cc,[0,1])
            for response in responses:
                folder=ledger.store.root/'calls'/response['call_id'];self.assertEqual(c.read(folder/'completion.json'),response)
                j,errors=d.parse_and_replay((folder/'raw.private.txt').read_bytes().decode(),c.read(folder/'output_tokens.private.json'),[100],tiny_context(raw.decode()));self.assertEqual(j,response['judgment']);self.assertFalse(errors)
            return responses
    def test_synthetic_producer_completion_independent_replay_ingest_statistics(self):
        reports={a:report(a,self.completed_fixture(a)) for a in MODELS}
        row={'opaque_pair_id':'c'*64,'score_hex':float(.95).hex(),'bin':19,'bin_population':1,'bin_sample_size':1,'inclusion_weight':{'numerator':1,'denominator':1},'inclusion_probability':{'numerator':1,'denominator':1}}
        counts=resource()['planned'];stats,swaps=g.selection_statistics([row],[0]*19+[1],reports,MODELS,B,B,counts,S)
        self.assertTrue(stats['selection_passed']);self.assertEqual(stats['selected_tau'],{'numerator':1,'denominator':2});self.assertEqual(stats['diagnostics']['pairs'],1)
        reports['mistral']['canonical_observations'][0].update(status='not_attempted',category=None)
        stats,_=g.selection_statistics([row],[0]*19+[1],reports,MODELS,B,B,counts,S)
        self.assertFalse(stats['selection_passed']);self.assertEqual(stats['grid_ledger'][0]['weighted_denominator'],{'numerator':1,'denominator':1})
    def test_wrong_model_pair_order_activation_report_refused(self):
        reports={a:report(a,[C.envelope(x,'context_overflow') for x in calls(a)]) for a in MODELS};counts=resource()['planned']
        g.validate_panel_reports(reports,MODELS,B,B,counts)
        for key,value in [('activation',{'bytes':2,'sha256':'b'*64}),('planned_calls',1)]:
            bad=copy.deepcopy(reports);bad['qwen'][key]=value
            with self.assertRaises(ValueError):g.validate_panel_reports(bad,MODELS,B,B,counts)
        bad=copy.deepcopy(reports);bad['qwen']['canonical_observations'][0]['model_id']=MODELS['mistral']['model_id']
        with self.assertRaises(ValueError):g.validate_panel_reports(bad,MODELS,B,B,counts)
    def test_diagnostic_cannot_enter_canonical_statistics(self):
        cc=calls();rr=[C.envelope(x,'context_overflow') for x in cc]
        self.assertEqual(len(g.observed_from_responses(rr)),1);self.assertEqual(len(g.observed_from_responses(rr,True)),1)
        rr[1]['orientation']='canonical'
        with self.assertRaises(ValueError):g.observed_from_responses(rr)

if __name__=='__main__':unittest.main()
