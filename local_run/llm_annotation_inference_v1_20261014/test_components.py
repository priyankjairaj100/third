"""Small synthetic fixtures only. No model loads, entropy draws, labels or remote calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import contract as c
import kernel as k
from ledger import ArtifactIntegrityError,Ledger,PrivateOutput
REPO=Path(os.environ.get('THIRD_INFERENCE_REPO',HERE.parents[1]))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
PROTOCOL=Path(os.environ.get('THIRD_INFERENCE_PROTOCOL',REPO/'local_run/model_annotation_v3_20261014'))
P=module('test_v3_protocol',PROTOCOL/'protocol.py')
PILOT=module('test_v3_qualified_pilot',REPO/'local_run/model_annotation_runtime_v5_20261011/run_pilot.py')
SCHEMA=json.loads((REPO/'local_run/model_annotation_v2_20261011/model_judgment.schema.json').read_text())
B={'bytes':12,'sha256':'a'*64};PINS=c.load_pins()
GOOD={'category':'exact_copy','distinction_types':['none'],'reason':'Same content.'}

def fixture():
    text={}
    for i,(rid,s) in enumerate([('a','First\u2028line.\nUnicode café.'),('b','Second\tline. 😀')]):
        text[i]={'calibration_row_index':i,'original_full_row_index':10+i,'record_id':rid,'text':s,'text_sha256':c.sha(s.encode())}
    ident=P.identities(PINS['protocol_plan_sha256'],PINS['frame_result_sha256'],['a','b'])
    indices=[next(i for i,r in text.items() if r['record_id']==rid) for rid in ident['canonical_record_ids']]
    model=PINS['models']['qwen'];seed=P.model_seed(PINS['protocol_plan_sha256'],PINS['frame_result_sha256'],ident['opaque_pair_id'],model['model_id'],model['revision'])
    shared=[];tokens=[]
    for i,inds in enumerate([indices,indices[::-1]]):
        r={'request_index':i,'selection_index':0,'stage':'selection_swap' if i else 'selection',
           'orientation':'swapped_diagnostic' if i else 'canonical','diagnostic_only':bool(i),
           'opaque_pair_id':ident['opaque_pair_id'],'natural_pair_id':ident['natural_pair_id'],
           'shown_calibration_row_indices':inds,'shown_record_ids':[text[j]['record_id'] for j in inds],
           'shown_text_sha256':[text[j]['text_sha256'] for j in inds],'model_seeds':{m['model_id']:P.model_seed(PINS['protocol_plan_sha256'],PINS['frame_result_sha256'],ident['opaque_pair_id'],m['model_id'],m['revision']) for m in PINS['models'].values()}}
        row=c.reconstruct_row(r,text)
        t={**r,'model_id':model['model_id'],'model_revision':model['revision'],'model_seed':seed,
           'input_row_sha256':c.sha(c.canonical(row,ascii=True)),'input_row_hash_serialization':c.INPUT_ROW_SERIALIZATION,
           'rendered_prompt_sha256':'b'*64,'input_token_ids':[1,4,7],'input_tokens':3,'input_token_ids_sha256':c.sha(c.canonical([1,4,7])),
           'attention_mask_policy':'one for every unpadded token','max_new_tokens':512,'total_token_budget':32768,
           'context_fit_with_output_reserve':True,'preparation_status':'ready_for_future_activation','full_text_preserved':True,
           'truncation_applied':False,'activation_lock_sha256':None,'call_id':None}
        shared.append(r);tokens.append(t)
    activation=c.Activation(B,{'request_manifest':B},B)
    calls=c.resolve_calls(tokens,B,activation,P)
    return shared,tokens,list(text.values()),calls,activation

def validate(j):return PILOT.validate_judgment(j,SCHEMA)
def raw_artifacts(raw=b'{}'):
    b={'bytes':len(raw),'sha256':c.sha(raw)}
    return {x:b for x in ('raw.private.txt','output_tokens.private.json','guard_steps.private.json','metrics.json')}

def active_contract():
    return {**{x:B for x in c.ACTIVATION_BINDINGS},'request_preparation_package':PINS['request_preparation_package'],
        **{x:PINS[x] for x in ('protocol_plan_sha256','frame_result_sha256','protocol_evidence_sha256')},
        'models':{a:{'model_id':m['model_id'],'revision':m['revision']} for a,m in PINS['models'].items()},
        'stage':'selection','human_annotation_calls':0,'human_completed':False,'output_limit_bytes':10_000_000,
        'scope':'model-assessed development; no human validity or publisher authenticity'}

def write_activation(path,contract):
    path.mkdir();intent={'schema':'third-llm-only-annotation-activation-intent-v3','contract_sha256':c.sha(c.canonical(contract)),'entropy_draws':0,'attempt':0}
    (path/'intent.json').write_bytes(c.canonical(intent))
    obj={'schema':'third-llm-only-annotation-activation-v3','status':'activated','contract':contract,'intent':c.binding(path/'intent.json')}
    obj['sha256']=c.sha(c.canonical(obj));(path/'activation.json').write_bytes(c.canonical(obj));return c.binding(path/'activation.json')

class ContractTests(unittest.TestCase):
    def test_strict_json(self):
        for raw in ('{"x":1,"x":2}','NaN','1e309','{} prose'):
            with self.assertRaises(ValueError):c.strict_json(raw)
    def test_actual_frozen_source_pins(self):
        for rel,pin in PINS['source_files'].items():self.assertEqual(c.binding(k.source_path(REPO,PROTOCOL,rel)),pin)
    def test_unicode_and_exact_swap_rows(self):
        shared,tokens,texts,_,_=fixture();rows=c.validate_prepared_rows(shared,tokens,texts,'qwen',P)
        self.assertEqual(rows[0]['text_a'],rows[1]['text_b']);self.assertIn('\u2028',''.join(x['text'] for x in texts))
    def test_wrong_orientation_or_text_refused(self):
        shared,tokens,texts,_,_=fixture();texts[0]['text']+='x'
        with self.assertRaisesRegex(ValueError,'text_hash'):c.validate_prepared_rows(shared,tokens,texts,'qwen',P)
        shared,tokens,texts,_,_=fixture();shared[0]['shown_calibration_row_indices'].reverse()
        with self.assertRaises(ValueError):c.validate_prepared_rows(shared,tokens,texts,'qwen',P)
    def test_extra_request_fields_and_unqualified_seed_refused(self):
        shared,tokens,texts,_,_=fixture();tokens[0]['score']=0.9
        with self.assertRaisesRegex(ValueError,'exact_keys'):c.validate_prepared_rows(shared,tokens,texts,'qwen',P)
        shared,tokens,texts,_,_=fixture();tokens[0]['model_seed']+=1
        with self.assertRaisesRegex(ValueError,'model_seed'):c.validate_prepared_rows(shared,tokens,texts,'qwen',P)
    def test_exact_ceil_swap_for_one_pair(self):
        s,t,x,_,_=fixture()
        with self.assertRaisesRegex(ValueError,'exact_swap_denominator'):c.validate_prepared_rows(s[:1],t[:1],x,'qwen',P)
    def test_lf_jsonl_preserves_unicode_separator(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';p.write_bytes(c.canonical({'text':'a\u2028b'})+b'\n')
            self.assertEqual(c.read_jsonl(p,c.binding(p)),[{'text':'a\u2028b'}])
    def test_seed_shared_call_id_distinct(self):
        s,t,x,calls,a=fixture();self.assertEqual(calls[0]['model_seed'],calls[1]['model_seed']);self.assertNotEqual(calls[0]['call_id'],calls[1]['call_id'])
        a2=c.Activation({'bytes':12,'sha256':'c'*64},a.contract,B)
        self.assertNotEqual(c.resolve_calls(t,B,a2,P)[0]['call_id'],calls[0]['call_id'])
        with self.assertRaisesRegex(ValueError,'duplicate_planned_call'):c.resolve_calls([t[0],t[0]],B,a,P)
    def test_planned_call_id_seed_and_revision_tamper(self):
        call=fixture()[3][0];c.validate_planned_call(call,P)
        for key,value in [('call_id','f'*64),('model_seed',call['model_seed']+1),('model_revision','wrong'),('attempt',False)]:
            bad=dict(call);bad[key]=value
            with self.assertRaises(ValueError):c.validate_planned_call(bad,P)
    def test_activation_external_hash_exact_inventory_and_no_entropy(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'activation';contract=active_contract();b=write_activation(p,contract)
            self.assertEqual(c.validate_activation(p,b,contract).raw_binding,b)
            (p/'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory'):c.validate_activation(p,b,contract)
    def test_activation_contract_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'activation';contract=active_contract();b=write_activation(p,contract);contract['human_completed']=True
            with self.assertRaises(ValueError):c.validate_activation(p,b,contract)
    def test_generated_nonvalid_needs_evidence(self):
        call=fixture()[3][0]
        for status in ('invalid_json','refusal'):
            response=c.envelope(call,status,raw_hash='a'*64,runtime_hash='b'*64,error={'type':'Format','message':'bad'})
            with self.assertRaises(ValueError):c.validate_envelope(response,call,validate)
    def test_valid_schema_and_identity_are_strict(self):
        call=fixture()[3][0];a=raw_artifacts();response=c.envelope(call,'valid',GOOD,a['raw.private.txt']['sha256'],'b'*64,a)
        c.validate_envelope(response,call,validate)
        bad=copy.deepcopy(response);bad['judgment']['reason']='word '*81
        with self.assertRaises(ValueError):c.validate_envelope(bad,call,validate)
        bad=copy.deepcopy(response);bad['attempt']=False
        with self.assertRaises(ValueError):c.validate_envelope(bad,call,validate)
    def test_longest_first_schedule_original_indices(self):
        _,tokens,_,_,_=fixture();tokens[1]['input_tokens']=100
        self.assertEqual(c.schedule(tokens),[1,0]);tokens[0]['input_tokens']=100
        self.assertEqual(c.schedule(tokens),[0,1])

class LedgerTests(unittest.TestCase):
    def calls(self):
        call=fixture()[3][0];out=[]
        for i in range(3):r=dict(call,request_index=i,call_id=hashlib.sha256(str(i).encode()).hexdigest());out.append(r)
        return out
    def test_exclusive_intents_and_reordered_fatal_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            calls=self.calls();ledger=Ledger(Path(td)/'new',100000,calls,validate)
            writer=ledger.begin(calls[2]);r=c.envelope(calls[2],'context_overflow',runtime_hash='b'*64)
            ledger.record(writer,r);ledger.begin(calls[0])
            with self.assertRaisesRegex(ValueError,'at_most_one_attempt'):ledger.begin(calls[0])
            rows,summary=ledger.finish('fatal',runtime_hash='b'*64)
            self.assertEqual([r['status'] for r in rows],['missing_output','not_attempted','context_overflow'])
            self.assertEqual(summary['recorded_calls'],3)
    def test_selection_phase_canonical_and_swap_but_no_validation(self):
        calls=fixture()[3]
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'both',100000,calls,validate)
            rows,summary=ledger.finish('synthetic no-model failure')
            self.assertEqual([r['stage'] for r in rows],['selection','selection_swap'])
            bad=copy.deepcopy(calls);bad[1]['stage']='validation_swap'
            with self.assertRaisesRegex(ValueError,'one_selection_phase'):Ledger(Path(td)/'bad',100000,bad,validate)
    def test_no_fabricated_success_if_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            l=Ledger(Path(td)/'new',100000,self.calls(),validate)
            with self.assertRaisesRegex(ValueError,'incomplete_ledger'):l.finish()
    def test_no_overwrite_and_budget_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'new';s=PrivateOutput(p,5);s.write(p/'a',b'12345')
            with self.assertRaises(ArtifactIntegrityError):s.write(p/'b',b'6')
            with self.assertRaises(FileExistsError):PrivateOutput(p,5)
    def test_output_path_traversal_refused(self):
        with tempfile.TemporaryDirectory() as td:
            s=PrivateOutput(Path(td)/'new',100)
            with self.assertRaises(ArtifactIntegrityError):s.write(s.root/'..'/'outside',b'x')
            self.assertFalse((Path(td)/'outside').exists())
    def test_output_changed_final_recheck(self):
        with tempfile.TemporaryDirectory() as td:
            s=PrivateOutput(Path(td)/'new',100);s.json('a',{})
            (s.root/'a').write_text('bad')
            with self.assertRaisesRegex(ValueError,'output_changed'):s.recheck()
    def test_watch_raw_file_and_symlink_target(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_bytes(b'one');(p/'b').write_bytes(b'one');(p/'link').symlink_to(p/'a')
            w=k.Watch();w.add(p/'link',allow_asset_symlink=True);(p/'link').unlink();(p/'link').symlink_to(p/'b')
            with self.assertRaisesRegex(ValueError,'final_binding_changed'):w.recheck()

class Tensor:
    def __init__(self,x):self.x=x;self.shape=(1,len(x))
    def __getitem__(self,index):return self if not isinstance(index,tuple) else Tensor(self.x[index[1]])
    def tolist(self):return self.x
    def to(self,*a):return self
    def detach(self):return self
    def cpu(self):return self
class Context:
    def __enter__(self):return self
    def __exit__(self,*a):pass
class KernelTests(unittest.TestCase):
    def mock_kernel(self,call,token,row,fail=False,raw=None):
        raw=raw if raw is not None else json.dumps(GOOD);seeds=[];generated=[]
        class Guard:
            def __init__(self,*a):self.steps=[];self.state=NS(emitted_text=raw)
        def generate(**kw):
            generated.append(kw)
            if fail:raise RuntimeError('synthetic CUDA failure')
            return Tensor(token['input_token_ids']+[8,9])
        cuda=NS(manual_seed_all=lambda s:seeds.append(('cuda',s)),reset_peak_memory_stats=lambda x:None,synchronize=lambda:None,
                max_memory_allocated=lambda x:12,max_memory_reserved=lambda x:16)
        torch=NS(manual_seed=lambda s:seeds.append(('cpu',s)),cuda=cuda,inference_mode=Context)
        state={'torch':torch,'model':NS(generate=generate),'tokenizer':NS(decode=lambda *a,**kw:raw),
            'tokenizer_data':object(),'token_policy':object(),'generation':object(),'constraint_alphabet':'abc',
            'flash_context':Context,'flash_flags':lambda:{'flash':True,'math':False,'mem_efficient':False,'cudnn':False},
            'effective_constraint_config_sha256':'a'*64,'token_eligibility_sha256':'b'*64,'runtime_manifest_sha256':'c'*64}
        pilot=NS(build_constraint=lambda *a:(None,NS(token_enforcer=object())),replay_tokens=lambda *a:True,
                 replay_constraint=lambda *a:True,parse_judgment=PILOT.parse_judgment,validate_judgment=PILOT.validate_judgment)
        prepared={'encoded':{'input_ids':Tensor(token['input_token_ids']),'attention_mask':Tensor([1]*token['input_tokens'])},
                  'rendered_prompt_sha256':token['rendered_prompt_sha256']}
        kernel=NS(state=state,pilot=pilot,protocol=P,guard=NS(GreedyGuardProcessor=Guard),schema=SCHEMA,
                  model=PINS['models']['qwen'],activation=fixture()[4],verify_tokenization=lambda *a:prepared)
        return kernel,seeds,generated
    def run_case(self,token=None,fail=False,raw=None):
        shared,tokens,texts,calls,_=fixture();token=token or tokens[0];call=dict(calls[0],request_row_sha256=c.sha(c.canonical(token)))
        row=c.reconstruct_row(shared[0],{r['calibration_row_index']:r for r in texts});kernel,seeds,generated=self.mock_kernel(call,token,row,fail,raw)
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'new',200000,[call],validate);writer=ledger.begin(call)
            response=k.execute_call(kernel,call,token,row,writer);ledger.record(writer,response);ledger.finish()
        return response,seeds,generated
    def test_separate_protocol_root_does_not_modify_executing_checkout(self):
        self.assertEqual(k.source_path('/old/repo','/private/tools/v3','local_run/model_annotation_v3_20261014/protocol.py'),Path('/private/tools/v3/protocol.py'))
        self.assertEqual(k.source_path('/old/repo','/private/tools/v3','local_run/model_annotation_runtime_v5_20261011/run_pilot.py'),Path('/old/repo/local_run/model_annotation_runtime_v5_20261011/run_pilot.py'))
    def test_linux_numeric_slurm_and_all_dependency_pins(self):
        with patch.object(k.platform,'system',return_value='Linux'),patch.object(k.platform,'python_version',return_value='3.13.7'), \
             patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(k.importlib.metadata,'version',side_effect=PINS['runtime']['package_versions'].__getitem__):
            self.assertTrue(k.runtime_preflight(PINS))
            with patch.dict(os.environ,{'SLURM_JOB_ID':'not_a_job'}):
                with self.assertRaisesRegex(ValueError,'positive_slurm'):k.runtime_preflight(PINS)
            with patch.object(k.importlib.metadata,'version',return_value='wrong'):
                with self.assertRaisesRegex(ValueError,'package_version'):k.runtime_preflight(PINS)
    def test_exact_tokenization_checks_full_ids_and_mask(self):
        shared,tokens,texts,calls,_=fixture();row=c.reconstruct_row(shared[0],{r['calibration_row_index']:r for r in texts})
        encoded={'input_ids':Tensor(tokens[0]['input_token_ids']),'attention_mask':Tensor([1,1,1])}
        prepared={'encoded':encoded,'input_tokens':3,'rendered_prompt_sha256':tokens[0]['rendered_prompt_sha256']}
        kernel=k.FamilyKernel.__new__(k.FamilyKernel);kernel.state={};kernel.pilot=NS(tokenize_call=lambda *a:prepared,digest_bytes=PILOT.digest_bytes,canonical=PILOT.canonical)
        self.assertEqual(kernel.verify_tokenization(row,tokens[0]),prepared)
        encoded['attention_mask']=Tensor([1,0,1])
        with self.assertRaisesRegex(ValueError,'prepared_tokens_changed'):kernel.verify_tokenization(row,tokens[0])
    def test_valid_one_greedy_flash_call_and_seed(self):
        r,seeds,calls=self.run_case();self.assertEqual(r['status'],'valid');self.assertEqual(len(calls),1)
        self.assertEqual([x[1] for x in seeds],[r['model_seed']]*2);self.assertFalse(calls[0]['output_attentions'])
        self.assertEqual(len(calls[0]['logits_processor']),1)
    def test_overflow_no_seed_or_generation(self):
        t=fixture()[1][0];t['input_token_ids']=[1]*32257;t['input_tokens']=32257
        r,seeds,calls=self.run_case(t);self.assertEqual(r['status'],'context_overflow');self.assertEqual(seeds,[]);self.assertEqual(calls,[])
    def test_boundary_and_no_format_retry(self):
        t=fixture()[1][0];t['input_token_ids']=[1]*32256;t['input_tokens']=32256
        r,seeds,calls=self.run_case(t,raw='not json');self.assertEqual(r['status'],'invalid_json');self.assertEqual(len(calls),1)
        self.assertIn('raw.private.txt',r['artifacts'])
    def test_runtime_failure_preserves_one_attempt(self):
        r,seeds,calls=self.run_case(fail=True);self.assertEqual(r['status'],'runtime_error');self.assertEqual(len(calls),1)
        self.assertIsNone(r['judgment']);self.assertIn('guard_steps.private.json',r['artifacts'])
    def test_foreign_call_binding_before_generation(self):
        shared,tokens,texts,calls,_=fixture();call=dict(calls[0],request_row_sha256='f'*64)
        kernel,seeds,generated=self.mock_kernel(call,tokens[0],{})
        with self.assertRaisesRegex(ValueError,'call_request_binding'):k.execute_call(kernel,call,tokens[0],{},None)
        self.assertFalse(generated)
    def test_flash_fallback_is_error_not_second_attempt(self):
        s,t,x,calls,_=fixture();row=c.reconstruct_row(s[0],{r['calibration_row_index']:r for r in x});kernel,_,generated=self.mock_kernel(calls[0],t[0],row)
        kernel.state['flash_flags']=lambda:{'flash':True,'math':True,'mem_efficient':False,'cudnn':False}
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'new',100000,[calls[0]],validate);writer=ledger.begin(calls[0]);r=k.execute_call(kernel,calls[0],t[0],row,writer)
            self.assertEqual(r['status'],'runtime_error');self.assertFalse(generated)

if __name__=='__main__':unittest.main()
