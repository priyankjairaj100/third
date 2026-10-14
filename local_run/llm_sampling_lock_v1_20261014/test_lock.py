"""Software metadata fixtures only; all entropy calls are mocked, never production."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sampling_lock as s

FOLDER=Path(__file__).parent
REPO=FOLDER.resolve().parents[1]
FRAME=REPO/'local_workspace/reports/e5-calibration-frame-v1-20261014/a01/run/frame/result.json'
# Exact reviewed metadata is copied from the public source summary's bound result
# when available; tests for pure lock behavior never need frame payloads.
B={'bytes':1,'sha256':'a'*64}
SEEDS={name:f'{i:064x}' for i,name in enumerate(s.SEED_NAMES)}

def fixture_contract():
    return {'schema':'software-only-contract','frame_sha256':'b'*64,'meaning':'synthetic test metadata, no empirical claim'}

def fixture_qualification():
    return s.seal({'schema':'third-llm-selection-qualification-v1','passed':True,'mode':'qualification','stage':'complete',
      'source_frame_result':s.pins()['frame_result'],'sampler_package':B,'selected_frame_rows':512,
      'unordered_pair_count':130816,'scoring_block_size':256,'production_sampling_performed':False,
      'parity':dict.fromkeys(s.PARITY_KEYS,True),'runtime':{'python':'3.13.7','numpy':'2.5.3','hostname':'csis.cn1','numeric_threads':1,'gpu_used':False,'rng':'Python random.Random MT19937','concurrent_node_load_not_controlled':True,'interpreter':'/software-fixture/envs/encoder-cpu/bin/python','slurm_job_id':'1','cpu_affinity':[0]},'files':{**{n:B for n in ('input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl','diagnostic_frame_rows.private.jsonl')},'diagnostic_frame_vectors.npy':{'bytes':512*768*4+128,'sha256':'c'*64}},
      'diagnostic_frame_artifacts_are_exact_first512_slice':True,
      'protocol_plan_sha256':s.pins()['plan_file']['sha256'],'frame_package_sha256':s.pins()['frame_package']['sha256'],
      'rng_stages':s.pins()['sampling_algorithm']['diagnostic_stages'],
      'diagnostic_seeds':{k:hashlib.sha256(('third-llm-only-v3/engineering-first512-v1/'+k).encode()).hexdigest() for k in ('selection','selection_swap')},
      'lock_evidence':None,'bin_populations':[130816]+[0]*19,'bin_sample_counts':[30]+[0]*19,'selected_pairs':30,'diagnostic_swap_pairs':3,
      'model_inference_calls':0,'human_annotation_calls':0,'same_source_pairs_excluded':False,'availability_filter_applied':False,
      'primary_input_acceptance':False,**dict.fromkeys(s.FALSE_FLAGS,False)})

class LockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='.software-lock-test-',dir=FOLDER)
        self.root=Path(self.temp.name);(self.root/'qualification').mkdir()
    def tearDown(self):self.temp.cleanup()
    def test_pure_builder_and_validator_draw_no_entropy(self):
        with mock.patch.object(s.os,'urandom',side_effect=AssertionError('must not draw')):
            lock=s.build_lock(fixture_contract(),SEEDS,'2026-10-14T10:00:00+00:00',B)
            self.assertTrue(s.validate_lock(lock,fixture_contract(),B)['passed'])
    def test_changed_contract_and_false_gates_refused_even_resealed(self):
        for key,value in [('inference_activation',True),('requires_human_annotations',True),('planned_human_reviews',100),('sampling_performed',True)]:
            lock=s.build_lock(fixture_contract(),SEEDS,'2026-10-14T10:00:00+00:00',B);lock.pop('sha256');lock[key]=value
            with self.assertRaises(ValueError):s.validate_lock(s.seal(lock),fixture_contract(),B)
        lock=s.build_lock(fixture_contract(),SEEDS,'2026-10-14T10:00:00+00:00',B)
        with self.assertRaises(ValueError):s.validate_lock(lock,{'frame_sha256':'c'*64},B)
    def test_seed_shape_and_helper_binding(self):
        bad=dict(SEEDS);bad['selection']='aa'
        with self.assertRaises(ValueError):s.build_lock(fixture_contract(),bad,'2026-10-14T10:00:00+00:00',B)
        lock=s.build_lock(fixture_contract(),SEEDS,'2026-10-14T10:00:00+00:00',B)
        with self.assertRaises(ValueError):s.validate_lock(lock,fixture_contract(),{'bytes':1,'sha256':'c'*64})
    def test_intent_precedes_exactly_four_mock_draws_and_existing_slot_refused(self):
        out=self.root/s.pins()['output_relative'];seen=[]
        def fake_entropy(count):
            self.assertEqual(count,32);self.assertTrue((out/'intent.json').is_file());self.assertFalse((out/'sampling_lock.json').exists())
            seen.append(count);return bytes([len(seen)])*32
        with mock.patch.object(s.os,'urandom',side_effect=fake_entropy):
            result=s.create_once(self.root,fixture_contract(),B,{})
            self.assertEqual(seen,[32]*4)
            with self.assertRaises(FileExistsError):s.create_once(self.root,fixture_contract(),B,{})
            self.assertEqual(seen,[32]*4)
        lock,binding=s.read_completed_lock(Path(result['path']),fixture_contract(),B)
        self.assertEqual(binding,result['binding']);self.assertEqual(len(lock['seed_hex']),4)
        self.assertEqual((out.stat().st_mode & 0o777),0o700)
    def test_entropy_failure_retains_intent_and_never_retries(self):
        with mock.patch.object(s.os,'urandom',side_effect=OSError('software injected failure')) as entropy:
            with self.assertRaises(OSError):s.create_once(self.root,fixture_contract(),B,{})
            self.assertEqual(entropy.call_count,1)
            with self.assertRaises(FileExistsError):s.create_once(self.root,fixture_contract(),B,{})
            self.assertEqual(entropy.call_count,1)
        out=self.root/s.pins()['output_relative'];self.assertTrue((out/'intent.json').exists());self.assertTrue((out/'failure.json').exists())
        self.assertFalse((out/'sampling_lock.json').exists())
    def test_metadata_change_after_draw_consumes_slot_without_lock(self):
        watched=self.root/'metadata.json';watched.write_text('{}');expected={watched:s.binding(watched.read_bytes())}
        def entropy(count):watched.write_text('{"changed":true}');return b'x'*count
        with mock.patch.object(s.os,'urandom',side_effect=entropy):
            with self.assertRaises(ValueError):s.create_once(self.root,fixture_contract(),B,expected)
        out=self.root/s.pins()['output_relative'];self.assertFalse((out/'sampling_lock.json').exists());self.assertTrue((out/'failure.json').exists())
    def test_failure_or_extra_inventory_not_a_completed_lock(self):
        with mock.patch.object(s.os,'urandom',return_value=b'x'*32):r=s.create_once(self.root,fixture_contract(),B,{})
        path=Path(r['path']);(path.parent/'failure.json').write_text('{}')
        with self.assertRaises(ValueError):s.read_completed_lock(path,fixture_contract(),B)
    def test_qualification_requires_full_parity_identity_and_counts(self):
        good=fixture_qualification();s.check_qualification(good,B)
        for key,value in [('selected_frame_rows',511),('unordered_pair_count',130815),('production_sampling_performed',True)]:
            bad=copy.deepcopy(good);bad.pop('sha256');bad[key]=value
            with self.assertRaises(ValueError):s.check_qualification(s.seal(bad),B)
        bad=copy.deepcopy(good);bad.pop('sha256');bad['parity']['full512_bins_and_reservoir']=False
        with self.assertRaises(ValueError):s.check_qualification(s.seal(bad),B)
    def test_copied_first512_artifacts_required(self):
        for mode in ('missing','false_flag','wrong_size'):
            bad=fixture_qualification();bad.pop('sha256')
            if mode=='missing':bad['files'].pop('diagnostic_frame_rows.private.jsonl')
            elif mode=='false_flag':bad['diagnostic_frame_artifacts_are_exact_first512_slice']=False
            else:bad['files']['diagnostic_frame_vectors.npy']['bytes']-=4
            with self.assertRaises(ValueError):s.check_qualification(s.seal(bad),B)
    def test_qualification_runtime_or_seed_drift_refused(self):
        for section,key,value in [('runtime','numpy','2.0'),('diagnostic_seeds','selection','0'*64)]:
            bad=fixture_qualification();bad.pop('sha256');bad[section][key]=value
            with self.assertRaises(ValueError):s.check_qualification(s.seal(bad),B)
    def test_independent_review_must_bind_result_sampler_frame_and_checks(self):
        review={'schema':'third-llm-selection-qualification-independent-review-v1','passed':True,'qualification_result':B,
          'sampler_package':B,'source_frame_result':s.pins()['frame_result'],'checks':dict.fromkeys(s.PARITY_KEYS,True),
          'production_sampling_performed':False,'scope':'software fixture'}
        s.check_review(review,B,B)
        review['qualification_result']={'bytes':1,'sha256':'c'*64}
        with self.assertRaises(ValueError):s.check_review(review,B,B)
    def test_duplicate_json_nonfinite_and_symlinks_refused(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):s.strict(raw)
        p=self.root/'real';p.write_text('{}');q=self.root/'link';q.symlink_to(p)
        with self.assertRaises(ValueError):s.read_small(q,100)
    def test_hist_snapshot_exact_inventory(self):
        snap=self.root/'snapshot';snap.mkdir();files={}
        for i in range(39):
            p=snap/f'{i}.json';p.write_text('{}');files[p.name]=s.binding(p.read_bytes())
        raw=s.canonical({'files':files});(snap/'snapshot_manifest.json').write_bytes(raw)
        self.assertEqual(s.verify_historical_snapshot(snap,s.binding(raw)['sha256'],{'bound_files':files},{}),s.binding(raw))
        (snap/'nested').mkdir();(snap/'nested/snapshot_manifest.json').write_text('{}')
        with self.assertRaises(ValueError):s.verify_historical_snapshot(snap,s.binding(raw)['sha256'],{'bound_files':files},{})

if __name__=='__main__':unittest.main()
