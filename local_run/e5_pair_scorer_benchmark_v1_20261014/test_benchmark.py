"""Software checks only; actual qualification vectors are read without models."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import benchmark as b

REPO = Path(os.environ['THIRD_BENCHMARK_REPO']).resolve()
QUALIFICATION = Path(os.environ['THIRD_BENCHMARK_QUALIFICATION']).resolve()
sys.path.insert(0, str(REPO / 'empirical_execution'))
from phase3 import calibration as cal, panels


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Algebraic software fixtures, never natural records or study evidence.
        cls.vectors = np.random.default_rng(7123).normal(size=(19, 768)).astype(np.float32)
        cls.table = b.reference_table(cls.vectors, panels, np)

    def test_pair_order_exact_coverage_and_tail(self):
        for n in (19, 256, 260, 512):
            order = list(b.pair_order(n, 256))
            self.assertEqual(len(order), n * (n - 1) // 2)
            self.assertEqual(set(order), {(i,j) for i in range(n) for j in range(i+1,n)})

    def test_full_geometry(self):
        g = b.geometry(20996)
        self.assertEqual((g['unordered_pairs'],g['upper_tiles'],g['last_block_rows']),
                         (220405510,3486,4))
        self.assertEqual(g['coordinate_products'],171343097856)

    def test_genuine_score_hex_equal_across_block_sizes(self):
        for size in (1,7,19,256):
            result = b.measure(self.vectors, list(range(19)), self.table, cal,
                               'boundary_and_bins', size, time.perf_counter()+20)
            self.assertTrue(result['completed'])
            self.assertEqual(result['pairs_checked'],171)
            self.assertEqual(sum(result['bin_counts']),171)

    def test_repeated_fixture_exact_parity_and_modes(self):
        mapping = [i % 19 for i in range(260)]
        scores = [b.measure(self.vectors[mapping], mapping, self.table, cal,
                            mode,256,time.perf_counter()+20) for mode in b.MODES]
        self.assertEqual(scores[0]['score_stream_sha256'],scores[1]['score_stream_sha256'])
        self.assertEqual(scores[1]['pairs_checked'],33670)

    def test_duplicate_missing_and_reordered_pairs_rejected(self):
        real = list(cal.iter_pair_scores(self.vectors,19))
        for changed in (real[:-1],real+[real[0]], [real[1],real[0]]+real[2:]):
            fake = SimpleNamespace(iter_pair_scores=lambda *a, data=changed: iter(data))
            with self.assertRaisesRegex(ValueError,'pair_order|incomplete_pair'):
                b.measure(self.vectors,list(range(19)),self.table,fake,'score_only',19,time.perf_counter()+20)

    def test_score_mutation_rejected(self):
        real = list(cal.iter_pair_scores(self.vectors,19))
        i,j,v=real[0]; real[0]=(i,j,np.nextafter(v,np.inf))
        fake=SimpleNamespace(iter_pair_scores=lambda *a:iter(real))
        with self.assertRaisesRegex(ValueError,'score_hex_parity'):
            b.measure(self.vectors,list(range(19)),self.table,fake,'score_only',19,time.perf_counter()+20)

    def test_expired_budget_is_incomplete(self):
        with self.assertRaises(b.BudgetExpired) as caught:
            b.measure(self.vectors,list(range(19)),self.table,cal,'score_only',19,-1)
        self.assertFalse(caught.exception.partial['completed'])
        self.assertEqual(caught.exception.partial['pairs_checked'],0)

    def test_partial_budget_preserves_counter(self):
        mapping=[i%19 for i in range(256)]
        moments=iter([0,0,2,2])
        with self.assertRaises(b.BudgetExpired) as caught:
            b.measure(self.vectors[mapping],mapping,self.table,cal,'score_only',256,1,
                      clock=lambda:next(moments))
        self.assertEqual(caught.exception.partial['pairs_checked'],4096)
        self.assertFalse(caught.exception.partial['completed'])

    def test_complete_bounded_schedule_has_28_cases(self):
        cases=[]
        report=b.run_suite(self.vectors,cal,panels,np,cases.append,seconds=30)
        self.assertEqual(report['cases'],28)
        self.assertEqual(len(cases),28)
        self.assertTrue(all(c['completed'] for c in cases))
        self.assertEqual(sum(c['kind']=='genuine_qualification_correctness' for c in cases),4)
        self.assertEqual(sum(c['kind']=='software_shape_warmup' for c in cases),6)
        self.assertEqual(sum(c['kind']=='software_shape_measurement' for c in cases),18)
        self.assertEqual(report['full_target_pair_count_executed'],0)

    def test_invalid_vectors_refused(self):
        invalid=[self.vectors.astype(np.float64),self.vectors[:18],self.vectors.copy(),self.vectors.copy()]
        invalid[2][0]=0;invalid[3][0,0]=np.nan
        for value in invalid:
            with self.assertRaises(ValueError):
                b.reference_table(value,panels,np)

    def test_invalid_mapping_refused(self):
        for mapping in (list(range(18)),[19]*19,[True]*19):
            with self.assertRaisesRegex(ValueError,'invalid_fixture_map'):
                b.measure(self.vectors,mapping,self.table,cal,'score_only',19,time.perf_counter()+20)

    def test_raw_scores_unclipped_and_frozen_bin_edges(self):
        self.assertEqual(cal._bin(-1.1),0);self.assertEqual(cal._bin(1.1),19)
        for k,edge in enumerate(cal.BIN_EDGES[1:-1],1):
            self.assertEqual(cal._bin(edge),k)
            self.assertEqual(cal._bin(np.nextafter(edge,-np.inf)),k-1)
        # Exact positive and negative axis vectors retain the raw endpoint scores.
        rows=np.zeros((19,768),np.float32); rows[:,0]=1;rows[1,0]=-1
        table=b.reference_table(rows,panels,np)
        self.assertEqual(table[0,0],struct.pack('<d',1.0))
        self.assertEqual(table[0,1],struct.pack('<d',-1.0))

    def test_binding_detects_mutation_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(b'old');wanted=b.binding(p)
            p.write_bytes(b'new')
            with self.assertRaises(ValueError):b.verify_binding(p,wanted)
            link=Path(tmp)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):b.binding(link)

    def test_package_member_and_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'code.py').write_bytes(b'pass\n')
            m=p/'package_manifest.json';m.write_text(json.dumps({'files':{'code.py':b.binding(p/'code.py')}}))
            sha=b.binding(m)['sha256'];b.verify_package(p,sha)
            (p/'code.py').write_bytes(b'fail\n')
            with self.assertRaises(ValueError):b.verify_package(p,sha)
            with self.assertRaises(ValueError):b.verify_package(p,'0'*64)

    def test_output_and_receipt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=b.new_output(Path(tmp)/'qualification'/'new',tmp)
            b.publish(p/'result.json',{'passed':False})
            with self.assertRaises((ValueError,FileExistsError)):b.new_output(p,tmp)
            with self.assertRaises(FileExistsError):b.publish(p/'result.json',{'passed':True})
            self.assertFalse(json.loads((p/'result.json').read_text())['passed'])
            with self.assertRaises(ValueError):b.new_output(Path(tmp)/'outside',tmp)

    def test_actual_qualified_vector_and_frozen_source_bindings(self):
        pins=json.loads((Path(b.__file__).parent/'expectations.json').read_text())
        b.verify_binding(QUALIFICATION/'result.json',pins['qualification_result'])
        b.verify_binding(QUALIFICATION/'assembled/vectors.npy',pins['qualification_vectors'])
        for name,expected in pins['repository_files'].items():b.verify_binding(REPO/name,expected)
        v=np.load(QUALIFICATION/'assembled/vectors.npy',mmap_mode='r',allow_pickle=False)
        golden=np.load(QUALIFICATION/'golden/vectors.npy',mmap_mode='r',allow_pickle=False)
        self.assertEqual(v.tobytes(),golden.tobytes())
        table=b.reference_table(v,panels,np)
        for size in (1,7,19,256):
            result=b.measure(v,list(range(19)),table,cal,'boundary_and_bins',size,time.perf_counter()+20)
            self.assertEqual(result['pairs_checked'],171)
        # No claim that this local numerical environment matches cluster timing.


if __name__=='__main__':unittest.main()
