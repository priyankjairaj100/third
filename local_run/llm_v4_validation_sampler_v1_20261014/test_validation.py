"""Analytic sampling and bound fixtures; no natural records or model judgments."""
from collections import Counter
from fractions import Fraction
import importlib.util
import itertools
from math import comb
from pathlib import Path
import random
import sys
import unittest
import primitives as v

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m

class EnumeratedChoices:
    def __init__(self,values):self.values=iter(values)
    def randrange(self,n):
        value=next(self.values);assert 0<=value<n;return value

class ValidationTests(unittest.TestCase):
    def test_exhaustive_reservoir_is_uniform(self):
        # All 3*4*5*6 random outcomes for n=6,k=2 yield each subset 24 times.
        stream=[(i,j,.99) for i in range(4) for j in range(i+1,4)]
        counts=Counter()
        for choice in itertools.product(range(3),range(4),range(5),range(6)):
            result=v.collect(iter(stream),4,.98,EnumeratedChoices(choice),capacity=2)
            counts[tuple((i,j) for i,j,_ in result['raw'])]+=1
        self.assertEqual(len(counts),comb(6,2));self.assertEqual(set(counts.values()),{24})
    def test_strict_cut_retains_every_eligible_pair_in_census(self):
        stream=[(0,1,.99),(0,2,.995),(1,2,.999)]
        r=v.collect(iter(stream),3,.995,random.Random(3))
        self.assertEqual(r['raw'],[(1,2,.999)]);self.assertEqual(r['population'],1)
        self.assertEqual(v.collect(iter(stream),3,.999,random.Random(3))['raw'],[])
    def test_incomplete_and_nonfinite_streams_rejected(self):
        for stream in ([(0,1,.99)],[(0,1,float('nan')),(0,2,.99),(1,2,.99)]):
            with self.assertRaises(ValueError):v.collect(iter(stream),3,.9,random.Random(3))
    def test_census_identity_and_model_seed_match_selection(self):
        folder=Path(__file__).resolve().parents[1]/'model_annotation_v4_20261014'
        if not folder.exists():folder=Path('/nfs_home/users/poonam/ccu-acl2027/local_workspace/tools/model-annotation-v4-20261014')
        p=load('validation_test_protocol',folder/'protocol.py')
        rows=[{'record_id':'fixture-'+str(i),'original_full_row_index':i,'source_unit_id':'source-'+str(i)} for i in range(3)]
        models=[{'model_id':'fixture-model','revision':'fixture-revision'}]
        r=v.decorate([(0,1,.999)],1,rows,p,'a'*64,'b'*64,models)[0]
        identity=p.identities('a'*64,'b'*64,['fixture-0','fixture-1'])
        self.assertEqual(r['canonical_record_ids'],identity['canonical_record_ids'])
        self.assertEqual(r['opaque_pair_id'],identity['opaque_pair_id'])
        self.assertEqual(r['inclusion_weight'],{'numerator':1,'denominator':1})
        self.assertEqual(r['model_seeds']['fixture-model'],p.model_seed('a'*64,'b'*64,r['opaque_pair_id'],'fixture-model','fixture-revision'))
    def test_empty_population_cannot_be_decorated(self):
        with self.assertRaises(ValueError):v.decorate([],0,[],None,'a'*64,'b'*64,[])

if __name__=='__main__':unittest.main()
