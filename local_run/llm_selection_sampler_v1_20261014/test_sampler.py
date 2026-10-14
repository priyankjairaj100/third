"""Tiny software fixtures; no natural-data sampling, production entropy, or model calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sampler as s

REPO = Path(os.environ.get('THIRD_SAMPLER_REPO',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(REPO/'empirical_execution'))
from phase3 import calibration as cal, panels
PROTOCOL = Path(os.environ.get('THIRD_SAMPLER_PROTOCOL',str(REPO/'local_run/model_annotation_v3_20261014')))
protocol=s.load_module('third_sampler_test_protocol',PROTOCOL/'protocol.py')
PLAN={'sha256':'a'*64,'models':[{'model_id':'model-one','revision':'r1'},{'model_id':'model-two','revision':'r2'}]}
FRAME_SHA='b'*64


def source_id():
    return next('account:test:'+str(i) for i in range(1000)
                if panels.digest_partition(panels.hash_integer('ccu-v1-source-split','account:test:'+str(i)))=='calibration')


def rows(n):
    return [{'calibration_row_index':i,'original_full_row_index':i*2,
             'record_id':'doc\u2028'+str(i),'source_unit_id':source_id(),
             'source_kind':'native_metadata_requires_provenance_review',
             'partition':'calibration','text_sha256':hashlib.sha256(str(i).encode()).hexdigest()}
            for i in range(n)]


def frame_for(rs):
    return {'rows':len(rs),'full_population_rows':len(rs)*2,
            'calibration_ordered_ids_sha256':s.digest([x['record_id'] for x in rs]),
            'calibration_ordered_source_ids_sha256':s.digest([x['source_unit_id'] for x in rs]),
            'original_full_row_indices_sha256':s.digest([x['original_full_row_index'] for x in rs])}


def vectors(n=41,d=5):
    # Deterministic algebraic fixture, not generated empirical documents.
    return np.array([[((i+3)*(j+5)%23)-11 for j in range(d)] for i in range(n)],dtype=np.float32)+np.float32(.125)


def rngs(stage=s.STAGES['qualification'][0]):
    return [s.make_rng(protocol,PLAN['sha256'],FRAME_SHA,stage,s.DIAGNOSTIC_SEEDS['selection'],i) for i in range(20)]


class SamplerTests(unittest.TestCase):
    def test_01_bin_edges_unclipped(self):
        for x in (-1.00000001,-1.,-.9,0.,.5,.999,1.,1.00000001):
            expected=max(0,min(19,sum(x>=float(k/10) for k in range(-10,11))-1))
            self.assertEqual(cal._bin(x),expected)
        self.assertEqual(cal._bin(1.00000001),19)

    def test_02_collect_matches_separate_reference(self):
        v=vectors(); a=s.collect(v,cal,rngs()); b=s.independent_reference(v,panels,rngs())
        for key in ('populations','raw','stream_sha256','pair_count'):self.assertEqual(a[key],b[key])
        self.assertEqual(b['pairwise_pairs'],171)

    def test_03_cross_tile_order_and_full_coverage(self):
        v=vectors(260,2); a=s.collect(v,cal,rngs()); b=s.independent_reference(v,panels,rngs())
        self.assertEqual(a['stream_sha256'],b['stream_sha256'])
        self.assertEqual(a['pair_count'],260*259//2)
        self.assertEqual(a['raw'],b['raw'])

    def test_04_small_bins_census_same_source_kept(self):
        rs=rows(8); a=s.collect(vectors(8),cal,rngs())
        sample=s.decorate(a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA)
        self.assertEqual(len(sample),28)
        self.assertTrue(all(x['same_source'] for x in sample))
        self.assertTrue(all(x['inclusion_probability']=={'numerator':1,'denominator':1} for x in sample))

    def test_05_exact_weights_id_orientation_swaps(self):
        rs=rows(41); a=s.collect(vectors(),cal,rngs()); sample=s.decorate(a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA)
        swap_rng=lambda:s.make_rng(protocol,PLAN['sha256'],FRAME_SHA,s.STAGES['qualification'][1],s.DIAGNOSTIC_SEEDS['selection_swap'],None)
        swap=s.swaps(sample,cal,swap_rng())
        s.verify_decoration(sample,swap,a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA,swap_rng())
        self.assertEqual(len(swap),(len(sample)+9)//10)
        corrupt=copy.deepcopy(sample);corrupt[0]['inclusion_weight']['numerator']+=1
        with self.assertRaises(ValueError):s.verify_decoration(corrupt,swap,a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA,swap_rng())

    def test_06_domain_separation_and_stable_identity(self):
        a=rngs()[0].getstate();b=rngs('selection')[0].getstate();self.assertNotEqual(a,b)
        self.assertEqual(rngs()[3].getstate(),rngs()[3].getstate())
        identity=protocol.identities(PLAN['sha256'],FRAME_SHA,['b','a'])
        self.assertEqual(identity,protocol.identities(PLAN['sha256'],FRAME_SHA,['a','b']))
        self.assertEqual(protocol.model_seed(PLAN['sha256'],FRAME_SHA,identity['opaque_pair_id'],'m','r'),
                         protocol.model_seed(PLAN['sha256'],FRAME_SHA,identity['opaque_pair_id'],'m','r'))

    def test_07_workload_geometry(self):
        self.assertEqual(s.workload(512),{'rows':512,'unordered_pairs':130816,'blocks':2,'upper_tiles':3,
                         'computed_tile_entries':196608,'coordinate_products':150994944})
        f=s.workload(20996)
        self.assertEqual((f['unordered_pairs'],f['upper_tiles'],f['computed_tile_entries'],f['coordinate_products']),
                         (220405510,3486,223102992,171343097856))

    def test_08_row_mapping_unicode_valid(self):
        rs=rows(19);self.assertIs(s.validate_rows(rs,frame_for(rs),panels),rs)
        self.assertEqual(len((b''.join(s.canonical(x,True)+b'\n' for x in rs)).split(b'\n'))-1,19)

    def test_09_duplicate_id_rejected_even_resealed(self):
        rs=rows(19);rs[1]['record_id']=rs[0]['record_id']
        with self.assertRaisesRegex(ValueError,'duplicate_id'):s.validate_rows(rs,frame_for(rs),panels)

    def test_10_wrong_order_noncalibration_and_extra_fields(self):
        for kind in ('order','partition','extra'):
            rs=rows(19);f=frame_for(rs)
            if kind=='order':rs[1]['original_full_row_index']=rs[0]['original_full_row_index']
            elif kind=='partition':rs[0]['partition']='test'
            else:rs[0]['text']='must not appear'
            with self.assertRaises(ValueError):s.validate_rows(rs,f,panels)

    def test_11_binding_change_symlink_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td).resolve()/'x';p.write_bytes(b'x');w=s.Watch();w.add(p);p.write_bytes(b'y')
            with self.assertRaises(ValueError):w.recheck()
            link=Path(td).resolve()/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):s.binding(link)
            with self.assertRaises(FileExistsError):s.publish(p,{'new':True})
            self.assertEqual(p.read_bytes(),b'y')

    def test_12_strict_json_and_seals(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):s.strict(raw)
        obj=s.seal({'x':'\u2028'});s.verify_seal(obj)
        obj['x']='changed'
        with self.assertRaises(ValueError):s.verify_seal(obj)

    def test_13_output_cap_and_reference_bound(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td).resolve()/'x'
            with self.assertRaises(ValueError):s.write_bytes(p,b'123',2)
            self.assertFalse(p.exists())
        with self.assertRaises(ValueError):s.independent_reference(vectors(513,2),panels,rngs())

    def test_14_partial_stream_failure_no_usable_result(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td).resolve();out=w/'qualification/out';args=SimpleNamespace(out=out,workspace=w,mode='qualification')
            with patch.object(s,'load_context',side_effect=ValueError('fixture_binding_refusal')):
                with self.assertRaises(ValueError):s.execute(args)
            failure=s.read_json(out/'FAILED.json')
            self.assertFalse(failure['partial_output_usable']);self.assertFalse((out/'result.json').exists())
            with self.assertRaises(ValueError):s.execute(args)

    def test_15_incomplete_collector_refused(self):
        fake=SimpleNamespace(iter_pair_scores=lambda *x:iter([(0,1,0.2)]),_bin=cal._bin,_reservoir_insert=cal._reservoir_insert)
        with self.assertRaisesRegex(ValueError,'incomplete_census'):s.collect(vectors(3),fake,rngs())

    def test_16_actual_frozen_pins_and_frame_metadata(self):
        pins=s.read_json(s.FOLDER/'expectations.json')
        for name,wanted in pins['frozen_files'].items():self.assertEqual(s.binding(REPO/name),wanted)
        self.assertEqual(s.binding(PROTOCOL/'PLAN.json')['sha256'],pins['plan_sha256'])
        self.assertEqual(s.binding(PROTOCOL/'package_manifest.json')['sha256'],pins['protocol_package_sha256'])


class AdditionalChecks(unittest.TestCase):
    def test_17_orientation_mutation_rejected(self):
        rs=rows(8); a=s.collect(vectors(8),cal,rngs()); sample=s.decorate(a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA)
        fresh=lambda:s.make_rng(protocol,PLAN['sha256'],FRAME_SHA,s.STAGES['qualification'][1],s.DIAGNOSTIC_SEEDS['selection_swap'],None)
        swap=s.swaps(sample,cal,fresh())
        sample[0]['canonical_frame_row_indices'].reverse()
        with self.assertRaisesRegex(ValueError,'orientation'):s.verify_decoration(sample,swap,a['raw'],a['populations'],rs,protocol,PLAN,FRAME_SHA,fresh())

    def test_18_runtime_requires_linux_slurm(self):
        with patch.object(s.platform,'system',return_value='Linux'), patch.dict(os.environ,{'SLURM_JOB_ID':''}):
            with self.assertRaisesRegex(ValueError,'Slurm'):s.validate_runtime(None,{},np,s.Watch())
        with patch.object(s.platform,'system',return_value='Darwin'), patch.dict(os.environ,{'SLURM_JOB_ID':'123'}):
            with self.assertRaisesRegex(ValueError,'Slurm'):s.validate_runtime(None,{},np,s.Watch())

    def test_19_protocol_historical_snapshot_bindings(self):
        path=Path(os.environ.get('THIRD_PROTOCOL_EVIDENCE',str(REPO/'local_workspace/reports/model-annotation-v3-evidence-v1-20261014')))
        pins=s.read_json(s.FOLDER/'expectations.json'); manifest=s.read_json(path/'snapshot_manifest.json')
        plan=s.read_json(PROTOCOL/'PLAN.json')
        self.assertEqual(s.binding(path/'snapshot_manifest.json')['sha256'],pins['protocol_evidence_sha256'])
        self.assertEqual(manifest['files'],plan['bound_files']); self.assertEqual(len(manifest['files']),39)
        for name,wanted in manifest['files'].items():self.assertEqual(s.binding(path/name),wanted)
        self.assertTrue(protocol.validate(plan,path)['passed'])

    def test_20_published_frame_seal_shape_flags(self):
        pins=s.read_json(s.FOLDER/'expectations.json')
        # Actual reviewed scalar bindings; no natural vectors or scores are read by unit tests.
        self.assertEqual(pins['frame_result']['sha256'],'d4fc45ec0d38ddc94c2a9c82ee3948020d0acd1f47fd84582b87ad5e273aedb1')
        self.assertEqual(pins['frame_files']['vectors.npy']['bytes'],20996*768*4+128)
        self.assertEqual(pins['rows']*(pins['rows']-1)//2,pins['pairs'])


class AdmissionChecks(unittest.TestCase):
    def test_21_vectors_and_changed_payload_refused(self):
        good=vectors(8);s.validate_vectors(good,[8,5],np)
        for bad in (good.astype(np.float64),np.zeros_like(good),np.full_like(good,np.nan),good[:3]):
            with self.assertRaises(ValueError):s.validate_vectors(bad,[8,5],np)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td).resolve()/'vectors.npy';np.save(path,good,allow_pickle=False)
            expected=s.binding(path);np.save(path,good+1,allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'binding_mismatch'):s.Watch().add(path,expected)

    def test_22_completed_synthetic_qualification_and_copy(self):
        with tempfile.TemporaryDirectory() as td:
            workspace=Path(td).resolve();out=workspace/'qualification/output'
            args=SimpleNamespace(out=out,workspace=workspace,mode='qualification',frame_result_sha256=FRAME_SHA)
            for k in ('sampling_lock','sampling_lock_sha256','qualification_result','qualification_review','lock_package'):setattr(args,k,None)
            pins={'qualification_rows':512,'rows':20996,'output_directory_cap_bytes':16<<20,
                  'frame_result':{'bytes':100,'sha256':FRAME_SHA},'frame_package_sha256':'c'*64}
            v=vectors(512,2);rs=rows(512);f={'frame_scope':'synthetic software fixture only'}
            context=(pins,{'bytes':100,'sha256':'d'*64},PLAN,protocol,cal,panels,{'synthetic':True},f,rs,v)
            with patch.object(s,'load_context',return_value=context),patch.object(s,'closed_inventories'):
                result=s.execute(args)
            self.assertTrue(result['passed']);self.assertFalse(result['production_sampling_performed'])
            self.assertEqual(result['unordered_pair_count'],130816)
            self.assertEqual(set(result['files']),{'input_bindings.json','selection.private.jsonl','selection_swap.private.jsonl',
                              'diagnostic_frame_vectors.npy','diagnostic_frame_rows.private.jsonl'})
            self.assertEqual(np.load(out/'diagnostic_frame_vectors.npy').tobytes(),v.tobytes())
            s.verify_seal(result)
            for name,bound in result['files'].items():self.assertEqual(s.binding(out/name),bound)


if __name__=='__main__':unittest.main()
