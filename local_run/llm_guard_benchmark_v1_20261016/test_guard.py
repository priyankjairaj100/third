"""Synthetic exact-arithmetic/ordering fixtures, never natural study results."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,os.environ.get('THIRD_LOCAL_REPO',str(Path(__file__).resolve().parents[2])))
from empirical_execution.phase3 import panels
from guard_core import scan


class Checks(unittest.TestCase):
    def setUp(self):
        self.z=np.array([[1,0],[0,1],[-1,0],[1,0],[0.8,0.6],[0,1]],dtype=np.float32)
        self.ids=['train-a','train-b','train-c','z-held','a-held','b-held']
        self.taus=[i/100 for i in range(50,100)]+[0.995]
    def test_all_grid_matches_original_guard(self):
        got=scan(self.z,[0,1,2],[3,4,5],self.ids[3:],self.taus,panels,2)
        rows=[{'record_id':rid,'partition':'train' if i<3 else 'test'} for i,rid in enumerate(self.ids)]
        for k,tau in enumerate(self.taus):
            _,_,audit=panels.semantic_guard(rows,self.z,tau,block_size=2,diagnostic_only=True)
            actual={self.ids[i]:(self.ids[int(got['witness_indices'][i,k])],float(got['witness_scores'][i,k]))
                    for i in range(3) if got['witness_indices'][i,k]>=0}
            self.assertEqual(actual,{r['record_id']:(r['heldout_witness_id'],r['score']) for r in audit['removed']})
        self.assertEqual(got['pairs'],9)
    def test_lexical_witness_can_differ_from_max_score(self):
        got=scan(self.z,[0],[3,4],['z-held','a-held'],[0.5],panels,1)
        self.assertEqual(got['witness_indices'][0,0],4)
        self.assertGreater(got['maxima'][0],got['witness_scores'][0,0])
    def test_strict_equal_threshold_has_no_witness(self):
        got=scan(self.z,[0],[3],['same'],[1.0],panels,1)
        self.assertEqual(got['maxima'][0],1.0);self.assertEqual(got['witness_indices'][0,0],-1)
    def test_chunk_and_heldout_order_invariance(self):
        a=scan(self.z,[0,1,2],[3,4,5],self.ids[3:],self.taus,panels,1)
        b=scan(self.z,[0,1,2],[5,3,4],[self.ids[i] for i in [5,3,4]],self.taus,panels,3)
        for k in ('witness_indices','witness_scores','maxima'):np.testing.assert_array_equal(a[k],b[k])
    def test_overlap_and_duplicate_identity_rejected(self):
        for args in (([0],[0],['h']),([0,0],[3],['h']),([0],[3,4],['h','h'])):
            with self.assertRaises(ValueError):scan(self.z,*args,[0.5],panels)
    def test_wrong_dtype_zero_nan_and_bad_index_rejected(self):
        with self.assertRaises(ValueError):scan(self.z.astype(np.float64),[0],[3],['h'],[0.5],panels)
        for value in (0,np.nan):
            z=self.z.copy();z[0]=value
            with self.assertRaises(ValueError):scan(z,[0],[3],['h'],[0.5],panels)
        with self.assertRaises(ValueError):scan(self.z,[True],[3],['h'],[0.5],panels)
    def test_invalid_grid_and_block_rejected(self):
        for grid in ([],[0.7,0.5],[float('nan')],[1.01]):
            with self.assertRaises(ValueError):scan(self.z,[0],[3],['h'],grid,panels)
        with self.assertRaises(ValueError):scan(self.z,[0],[3],['h'],[0.5],panels,False)


if __name__=='__main__':unittest.main()
