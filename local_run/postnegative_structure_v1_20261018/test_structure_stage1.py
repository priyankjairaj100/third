"""Tiny algebraic software fixtures; no empirical results."""
import importlib.util
import itertools
from fractions import Fraction
from pathlib import Path
import sys
import os
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s',HERE/'structure_stage1.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
REPO=Path(os.environ.get('THIRD_LOCAL_REPO','/Users/priyankjairaj/Downloads/ACL/Three'))
sys.path.insert(0,str(REPO/'empirical_execution'))
from ccu.core import BlockerGraph
from phase3.reference_graph import build_reference_graph

class Checks(unittest.TestCase):
    def fixture(self):
        ids=('a','b','c','d','e','f'); src=('u','v','w','w','z','q')
        return BlockerGraph(ids,src,np.arange(6,dtype=np.int64),np.array([0,0,1,2,4,5,5]),np.array([0,1,0,2,0]),.9)
    def test_exact_expectations_exhaustively(self):
        g=self.fixture();h={0:2,1:3,2:1}
        for k in range(1,6):
            total=0;cases=0
            for deleted in itertools.combinations(range(6),k):
                F=set(deleted);total+=sum(i not in F and len(b)>0 and set(map(int,b))<=F for i,b in enumerate(g.blockers));cases+=1
            r=s.expected_record_admissions(h,6,k)
            self.assertEqual(Fraction(r['numerator'])/Fraction(r['denominator']),Fraction(total,cases))
    def test_singleton_mechanisms(self):
        g=self.fixture();r=s.describe(g,{'u','v','w','z','q'})
        self.assertEqual(r['initially_selected'],2)
        self.assertEqual(r['singleton_record_admission_total'],3)
        self.assertEqual(r['singleton_excluded_blocker_admission_total'],1)
        self.assertEqual(r['singleton_record_requests_with_admission'],2)
        self.assertEqual(r['genuine_single_source_admission_total'],3)
    def test_unknown_source_excluded(self):
        r=s.describe(self.fixture(),{'v','w','z','q'})
        self.assertEqual(r['genuine_single_source_admission_total'],1)
    def test_scalar_parity_fixed_grid(self):
        x=np.asarray([[1,0,0],[1,1,0],[0,1,0],[.5,.25,1],[1,-1,0],[.25,.75,.125]],np.float32)
        r=s.independent_small_oracle(x,list('abcdef'),list('uvwxyz'),build_reference_graph)
        self.assertEqual(r['thresholds'],51);self.assertTrue(r['passed'])
    def test_strict_boundary(self):
        x=np.eye(3,dtype=np.float32);g=build_reference_graph(x,list('abc'),0.0)
        self.assertEqual(g.edge_count,0)
    def test_fixed_grid(self):
        self.assertEqual(len(s.GRID),51);self.assertEqual(s.GRID[-1],Fraction(199,200))
        self.assertEqual(len(set(s.GRID)),51)
if __name__=='__main__':unittest.main()
