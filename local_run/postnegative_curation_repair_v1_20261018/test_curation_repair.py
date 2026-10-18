"""Small algebraic software fixtures only; no empirical evidence."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r',HERE/'curation_repair.py');r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
REPO=Path(os.environ.get('THIRD_LOCAL_REPO','/Users/priyankjairaj/Downloads/ACL/Three'))
sys.path.insert(0,str(REPO/'empirical_execution'))
from phase3.reference_graph import build_reference_graph
from ccu import structure
A=r.load_algorithm(REPO)

class Checks(unittest.TestCase):
    def fixture(self,tau=.6):
        x=np.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,-1,0],[1,0,0]],np.float32)
        ids=[f'fixture-{i}' for i in range(8)];sources=['a','a','b','c','c','d','unknown','e']
        return build_reference_graph(x,ids,tau,sources,seed=0,block_size=3),x
    def test_record_compositions(self):
        g,x=self.fixture();z=r.one_arm(g,x,'R',set(g.source_ids)-{'unknown'},A,build_reference_graph,structure)
        self.assertEqual(z['status'],'completed');self.assertEqual(z['singleton_checks'],8)
        self.assertEqual([q['checkpoint'] for q in z['checkpoints']],[1,8])
        self.assertEqual(z['checkpoints'][-1]['selected_records'],0)
    def test_source_compositions_and_owner_overlap(self):
        g,x=self.fixture();z=r.one_arm(g,x,'S',set(g.source_ids)-{'unknown'},A,build_reference_graph,structure)
        self.assertEqual(z['singleton_checks'],5);self.assertEqual(z['sampling_frame_size'],5)
        self.assertEqual([q['checkpoint'] for q in z['checkpoints']],[1,5])
        self.assertEqual(z['checkpoints'][-1]['deleted_records'],7)
    def test_same_requests_all_thresholds(self):
        digests=[]
        for tau in (.5,.9,.995):
            g,x=self.fixture(tau);digests.append(r.one_arm(g,x,'R',set(g.source_ids)-{'unknown'},A,build_reference_graph,structure)['request_order_digest'])
        self.assertEqual(len(set(digests)),1)
    def test_empty_source_frame_explicit(self):
        g,x=self.fixture();self.assertEqual(r.one_arm(g,x,'S',set(),A,build_reference_graph,structure)['status'],'unavailable_empty_request_frame')
    def test_exact_integer_basis_is_membership(self):
        g,x=self.fixture();ids=list(g.record_ids);ui={s:i for i,s in enumerate(ids)}
        c=r.polynomial(g,'record',ui,ui,len(ids),8)
        self.assertTrue(all(type(v)is int for row in c.values() for v in row))
        state=A.IndexedState(8,8,8,range(8),c)
        self.assertEqual(r.membership_ids(state,ids),r.independent_selected(g,'record',set()))
    def test_invalid_request_atomic_and_retry_contract(self):
        g,x=self.fixture();ids=list(g.record_ids);ui={s:i for i,s in enumerate(ids)}
        c=r.polynomial(g,'record',ui,ui,len(ids),2);state=A.IndexedState(8,2,8,range(8),c);m=state.meter();before=state.snapshot()
        with self.assertRaises(ValueError):state.delete([99],m)
        self.assertEqual(state.snapshot(),before)
        with self.assertRaises(ValueError):state.delete([0,1,2],m)
        self.assertEqual(state.snapshot(),before)
        state.delete([0],m);after=state.snapshot();state.delete([0],m);self.assertEqual(state.snapshot(),after)
if __name__=='__main__':unittest.main()
