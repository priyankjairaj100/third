"""Synthetic software checks only: no production source, tokenizer assets or model calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('request_test_module',HERE/'prepare_requests.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
REPO=Path(os.environ.get('THIRD_REQUEST_REPO',HERE.parents[1]))
SAMPLER=Path(os.environ.get('THIRD_REQUEST_SAMPLER_PACKAGE',REPO/'local_run/llm_selection_sampler_v1_20261014'))
PROTOCOL=Path(os.environ.get('THIRD_REQUEST_PROTOCOL',REPO/'local_run/model_annotation_v3_20261014'))
sampler=r.load('request_test_sampler',SAMPLER/'sampler.py')
protocol=r.load('request_test_protocol',PROTOCOL/'protocol.py')
pilots={k:r.load('request_test_pilot_'+k,REPO/'local_run'/folder/'run_pilot.py') for k,folder in
  [('qwen','model_annotation_runtime_v5_20261011'),('mistral','model_annotation_runtime_mistral_v1_20261011')]}


def fixture_source():
    texts=['one\nline\u2028separator\u2029and\x85next','é and e\u0301','middle training record','final question']
    source=[{'record_id':str(i),'source_unit_id':'same' if i<2 else 'other','source_kind':'fixture',
      'partition':'train' if i==2 else 'calibration','text':t} for i,t in enumerate(texts)]
    frame=[]
    for i,row in enumerate(source):
        if row['partition']=='calibration':
            frame.append({k:row[k] for k in ('record_id','source_unit_id','source_kind','partition')}|
              {'calibration_row_index':len(frame),'original_full_row_index':i,'text_sha256':r.sha(row['text'].encode())})
    return source,frame


class FakeTensor:
    def __init__(self,values):self.values=values;self.shape=(1,len(values));self.ndim=2;self.device=SimpleNamespace(type='cpu')
    def __getitem__(self,index):
        if index!=0:raise IndexError(index)
        return SimpleNamespace(tolist=lambda:list(self.values))


class FakeTokenizer:
    """Integer bytes only; not a semantic/model tokenizer."""
    def __init__(self,n=None):self.n=n;self.messages=None
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs=={'tokenize':False,'add_generation_prompt':True}
        self.messages=messages;return 'FIXTURE\n'+messages[0]['content']+'\n'+messages[1]['content']
    def __call__(self,text,**kwargs):
        assert kwargs=={'add_special_tokens':False,'truncation':False,'return_tensors':'pt'}
        ids=list(text.encode()) if self.n is None else [7]*self.n
        return {'input_ids':FakeTensor(ids),'attention_mask':FakeTensor([1]*len(ids))}


def fixture_request():
    request={'request_index':0,'selection_index':0,'stage':'selection','orientation':'canonical','diagnostic_only':False,
      'opaque_pair_id':'a'*64,'natural_pair_id':'b'*64,'shown_calibration_row_indices':[0,1],
      'shown_record_ids':['a','b'],'shown_text_sha256':['c'*64,'d'*64],'model_seeds':{'fixture-model':7}}
    texts={0:{'text':'é\u2028full\nA'},1:{'text':'B e\u0301'}}
    model={'model_id':'fixture-model','revision':'fixture-revision'}
    return request,texts,model


def selection_fixture():
    pins=copy.deepcopy(r.read(HERE/'expectations.json'));s=pins['sampler_expectations'];s['rows']=4;s['pairs']=6
    b={'bytes':123,'sha256':'a'*64}; lock=pins['sampling_lock']; populations=[6]+[0]*19
    result={'schema':'third-llm-selection-sample-v1','passed':True,'stage':'complete','mode':'selection',
      'source_frame_result':s['frame_result'],'sampler_package':pins['sampler_package'],'protocol_plan_sha256':s['plan_sha256'],
      'frame_package_sha256':s['frame_package_sha256'],'selected_frame_rows':4,'unordered_pair_count':6,'scoring_block_size':256,
      'rng_stages':['selection','selection_swap'],'score_stream_encoding':'ASCII i TAB j TAB float.hex(score) LF',
      'sampling_performed':True,'production_sampling_performed':True,'engineering_subset_sampled':False,
      'diagnostic_seeds':None,'parity':None,'model_inference_calls':0,'human_annotation_calls':0,'bin_populations':populations,
      'bin_sample_counts':populations,'selected_pairs':6,'diagnostic_swap_pairs':1,
      'files':dict.fromkeys(('input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl'),b),
      'score_stream_sha256':'b'*64,'lock_evidence':{'sampling_lock':lock},'sha256':'c'*64}
    for key in ('same_source_pairs_excluded','availability_filter_applied','requires_human_annotations','threshold_selected',
      'model_calibration_complete','semantic_validity_established','original_human_gate_satisfied','inference_activation',
      'confirmatory_study_ready','source_authenticity_verified','primary_input_acceptance','original_route_acceptance',
      'full_population_cache_acceptance'):result[key]=False
    review={'schema':'third-llm-selection-metadata-identity-review-v1','passed':True,'state':'COMPLETED',
      'selection_result':b,'selection_result_internal_seal':result['sha256'],'sampler_package':pins['sampler_package'],
      'sampling_lock':lock,'source_frame_result':s['frame_result'],'lock_helper_package':pins['lock_package'],
      'checked':{'selected_pairs':6,'swap_pairs':1,'exact_weighted_denominator':6,
        'all_sample_identity_orientation_weight_checks_passed':True,'independent_swap_reservoir_replay_passed':True},
      'next_inference_builder_bindings':{'selection_files':result['files'],'frame_files':s['frame_files'],
        'frame_raw_result_sha256':s['frame_result']['sha256'],'plan_raw_sha256':s['plan_sha256'],'sampling_lock':lock,
        'source_records':s['frame_inherited']['prepared_source_records_file'],
        'source_audit':s['frame_inherited']['prepared_source_audit_file'],
        'inference_activation_still_required':True,'fresh_validation_calls_required':True}}
    for k in ('complete_census_independently_replayed','selection_reservoir_independently_replayed',
      'bin_population_counts_independently_recomputed','score_stream_independently_recomputed'):review[k]=False
    return pins,result,review,b


class RequestsTests(unittest.TestCase):
    def test_actual_Qwen_and_Mistral_templates_match_qualified_JSON_hash_convention(self):
        fixtures=r.read(HERE/'tokenizer_templates.fixture.json')['models']
        pins=r.read(HERE/'expectations.json')['models'];self.assertEqual(set(fixtures),{'qwen','mistral'})
        for alias,fixture in fixtures.items():
            self.assertEqual(fixture['source_config'],pins[alias]['tokenizer_assets']['tokenizer_config.json'])
            template=fixture['chat_template'];tokenizer=SimpleNamespace(chat_template=template)
            expected=pins[alias]['tokenizer_identity']['chat_template_sha256']
            self.assertEqual(fixture['expected_pilot_hash'],expected)
            self.assertEqual(r.sha(template.encode()),fixture['expected_raw_hash'])
            self.assertNotEqual(fixture['expected_raw_hash'],expected)
            self.assertEqual(r.template_sha256(tokenizer,pilots[alias]),expected)

    def test_source_unicode_physical_lines_and_all_frame_mapping(self):
        source,frame=fixture_source()
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.jsonl';p.write_bytes(b''.join(r.canonical(x)+b'\n' for x in source))
            actual=r.source_texts(p,frame,{0,2},4)
            self.assertEqual(actual[0]['text'],source[0]['text']);self.assertEqual(actual[2]['record_id'],'3')
            self.assertEqual(set(actual),{0,2})

    def test_source_wrong_unselected_text_refused(self):
        source,frame=fixture_source();source[1]['text']='changed unselected record'
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s';p.write_bytes(b''.join(r.canonical(x)+b'\n' for x in source))
            with self.assertRaisesRegex(ValueError,'source_frame_text_sha'):r.source_texts(p,frame,{0},4)

    def test_source_missing_duplicate_wrong_index_refused(self):
        for change in ('duplicate','missing','index'):
            source,frame=fixture_source()
            if change=='duplicate':source[2]['record_id']='0'
            if change=='missing':source.pop()
            if change=='index':frame[1]['original_full_row_index']=2
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'s';p.write_bytes(b''.join(r.canonical(x)+b'\n' for x in source))
                with self.assertRaises(ValueError):r.source_texts(p,frame,{0},4)

    def test_jsonl_requires_physical_final_LF_and_rejects_large_line(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s';p.write_bytes(b'{"text":"a"}')
            with self.assertRaisesRegex(ValueError,'physical_LF'):list(r.jsonl(p))
            p.write_bytes(b'{"text":"a"}\n')
            with self.assertRaisesRegex(ValueError,'line_cap'):list(r.jsonl(p,5))

    def test_strict_json_rejects_duplicate_nonfinite(self):
        for x in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):r.strict(x)

    def test_both_unchanged_pilot_renderers_and_unicode_hash(self):
        request,texts,model=fixture_request();results=[]
        for pilot in pilots.values():
            tokenizer=FakeTokenizer();out=r.tokenize_request(request,texts,model,pilot,tokenizer,'system','task')
            expected={'task_definition':'task','documents':[{'label':'A','text':texts[0]['text']},{'label':'B','text':texts[1]['text']}]}
            self.assertEqual(tokenizer.messages[1]['content'],json.dumps(expected,ensure_ascii=False))
            row={'pair_id':request['opaque_pair_id'],'orientation':'canonical','text_a':texts[0]['text'],'text_b':texts[1]['text']}
            self.assertEqual(out['input_row_sha256'],r.sha(pilot.canonical(row)))
            self.assertNotEqual(out['input_row_sha256'],r.sha(r.canonical(row)))
            self.assertIsNone(out['call_id']);self.assertEqual(out['model_seed'],7);results.append(out)
        self.assertEqual(results[0],results[1])

    def test_overflow_boundary_retains_complete_request(self):
        request,texts,model=fixture_request()
        for n,fit in ((32256,True),(32257,False)):
            out=r.tokenize_request(request,texts,model,pilots['qwen'],FakeTokenizer(n),'s','t')
            self.assertEqual(out['context_fit_with_output_reserve'],fit);self.assertEqual(len(out['input_token_ids']),n)
            self.assertEqual(out['opaque_pair_id'],request['opaque_pair_id']);self.assertFalse(out['truncation_applied'])

    def test_schedule_has_full_denominator_and_fixed_sample_ties(self):
        rows=[dict(request_index=i,selection_index=s,orientation=o,diagnostic_only=o!='canonical',input_tokens=n,
          context_fit_with_output_reserve=n<=32256) for i,s,o,n in
          [(0,0,'canonical',5),(1,1,'canonical',5),(2,0,'swapped_diagnostic',5),(3,2,'canonical',40000)]]
        out=r.summarize_requests(rows)
        self.assertEqual(out['scheduled_request_indices'],[3,0,2,1]);self.assertEqual(out['requests'],4)
        self.assertEqual(out['canonical_overflows'],1);self.assertTrue(out['overflow_retains_full_sampling_denominator'])

    def test_output_cap_failure_retains_partial_without_success(self):
        with tempfile.TemporaryDirectory() as t:
            o=r.Output(t,limit=100,reserve=40)
            with self.assertRaisesRegex(ValueError,'output_cap'):o.jsonl('x.jsonl',[{'x':'a'*20},{'x':'b'*70}])
            self.assertLessEqual((Path(t)/'x.jsonl').stat().st_size,60)
            self.assertFalse((Path(t)/'result.json').exists());o.metadata('failure.json',{'passed':False})
            self.assertLessEqual(sum(x.stat().st_size for x in Path(t).iterdir()),100)

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            o=r.Output(t);o.jsonl('x.jsonl',[{'x':1}])
            with self.assertRaises(ValueError):o.jsonl('x.jsonl',[{'x':2}])
            with self.assertRaises(FileExistsError):o.metadata('x.jsonl',{})

    def test_selection_requires_passed_bound_review_and_complete_counts(self):
        p,x,review,b=selection_fixture();r.validate_selection(x,review,b,p['sampler_package'],p['sampling_lock'],p)
        for field,value in [('passed',False),('selection_result',{'bytes':3,'sha256':'d'*64})]:
            bad=copy.deepcopy(review);bad[field]=value
            with self.assertRaises(ValueError):r.validate_selection(x,bad,b,p['sampler_package'],p['sampling_lock'],p)
        bad=copy.deepcopy(x);bad['bin_populations'][0]=5
        with self.assertRaises(ValueError):r.validate_selection(bad,review,b,p['sampler_package'],p['sampling_lock'],p)

    def test_selection_cannot_promote_scope_or_drop_overflow(self):
        p,x,review,b=selection_fixture()
        for key in ('availability_filter_applied','source_authenticity_verified','inference_activation'):
            bad=copy.deepcopy(x);bad[key]=True
            with self.assertRaises(ValueError):r.validate_selection(bad,review,b,p['sampler_package'],p['sampling_lock'],p)
        bad=copy.deepcopy(review);bad['complete_census_independently_replayed']=True
        with self.assertRaises(ValueError):r.validate_selection(x,bad,b,p['sampler_package'],p['sampling_lock'],p)

    def test_exact_types_do_not_accept_boolean_numerators(self):
        self.assertFalse(r.exact({'numerator':True},{'numerator':1}))
        self.assertFalse(r.exact([True],[1]));self.assertTrue(r.exact({'a':[1,False]},{'a':[1,False]}))

    def test_actual_frozen_vector_scores_identities_weights_and_swaps(self):
        import numpy as np
        sys.path.insert(0,str(REPO/'empirical_execution'))
        from phase3 import calibration as cal,panels
        vectors=np.asarray([[1,0,0],[0,1,0],[1,1,0],[-1,0,1]],dtype=np.float32)
        rows=[{'record_id':str(i),'source_unit_id':'same' if i<2 else str(i),'original_full_row_index':i}
          for i in range(4)]
        plan={'sha256':'a'*64,'models':[{'model_id':'fixture-model','revision':'fixture'}]};frame='b'*64
        raw=[];pop=[0]*20
        for i,j,score in cal.iter_pair_scores(vectors,256):
            b=cal._bin(score);pop[b]+=1;raw.append((i,j,score,b))
        raw.sort();sample=sampler.decorate(raw,pop,rows,protocol,plan,frame)
        lock={'seed_hex':{'selection_swap':'0'*64}}
        swaps=sampler.swaps(sample,cal,sampler.make_rng(protocol,plan['sha256'],frame,'selection_swap','0'*64,None))
        result={'selected_pairs':6,'diagnostic_swap_pairs':1,'bin_populations':pop,'bin_sample_counts':pop,
          'source_frame_result':{'sha256':frame},'unordered_pair_count':6}
        r.validate_sample(sample,swaps,result,rows,vectors,protocol,plan,cal,panels,sampler,lock)
        self.assertTrue(sample[0]['same_source'])
        for mutation in ('score','weight','order','swaps'):
            a,b=copy.deepcopy(sample),copy.deepcopy(swaps)
            if mutation=='score':a[0]['score_hex']=float(0.25).hex()
            if mutation=='weight':a[0]['inclusion_probability']['numerator']=True
            if mutation=='order':a[0],a[1]=a[1],a[0]
            if mutation=='swaps':b[0]['swapped_frame_row_indices'].reverse()
            with self.assertRaises(ValueError):r.validate_sample(a,b,result,rows,vectors,protocol,plan,cal,panels,sampler,lock)

    def test_ceil_swaps_are_retained_for_small_n(self):
        for n,wanted in ((1,1),(9,1),(10,1),(11,2)):
            sample=[{'canonical_frame_row_indices':[0,1],'opaque_pair_id':str(i),'natural_pair_id':str(i),
                     'model_seeds':{'fixture':i}} for i in range(n)]
            swaps=[{'selection_index':i,'swapped_frame_row_indices':[1,0]} for i in range(wanted)]
            texts={0:{'record_id':'a','text_sha256':'a'*64},1:{'record_id':'b','text_sha256':'b'*64}}
            records=r.requests(sample,swaps,texts)
            self.assertEqual(len(records),n+wanted);self.assertEqual(sum(x['diagnostic_only'] for x in records),wanted)
            self.assertEqual(records[-1]['shown_record_ids'],['b','a'])

    def test_asset_symlinks_bind_resolved_tokenizer_files_without_weight_reads(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t).resolve();model=base/'revision';model.mkdir();blob=base/'blob';blob.write_bytes(b'{}')
            for name in ('config.json','tokenizer_config.json'):(model/name).symlink_to(blob)
            (model/'model.safetensors').write_bytes(b'unread weight payload')
            tok={name:{'bytes':2,'sha256':r.sha(b'{}')} for name in ('config.json','tokenizer_config.json')}
            files={**tok,'model.safetensors':{'bytes':21,'sha256':'0'*64}}
            manifest={'model_id':'fixture','revision':'revision','files':[{'relative_path':k,'size_bytes':v['bytes'],
                       'sha256':v['sha256']} for k,v in files.items()]}
            path=base/'assets.json';path.write_bytes(r.canonical(manifest))
            pin={'model_id':'fixture','revision':'revision','asset_manifest':sampler.binding(path),
                 'asset_files':files,'tokenizer_assets':tok}
            watch=sampler.Watch();links={};r.verify_assets(model,path,pin,sampler,watch,links)
            self.assertEqual(links[str(model/'config.json')],str(blob))
            self.assertNotIn(str(model/'model.safetensors'),watch.files)
            blob.write_bytes(b'[]')
            with self.assertRaises(ValueError):watch.recheck()

    def test_watch_catches_input_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t).resolve()/'x';p.write_bytes(b'initial');w=sampler.Watch();w.add(p);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):w.recheck()

    def test_runtime_refuses_non_slurm_before_imports(self):
        args=SimpleNamespace(workspace=Path('/fixture'))
        with patch.object(r.platform,'system',return_value='Linux'),patch.dict(r.os.environ,{'SLURM_JOB_ID':''}):
            with self.assertRaisesRegex(ValueError,'Linux_Slurm_required'):r.check_runtime(args,r.read(HERE/'expectations.json'))


if __name__=='__main__':unittest.main()
