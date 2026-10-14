"""Small software fixtures, never empirical data."""
import copy
from collections import Counter
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import unittest
import protocol as p
import sampling as s
import select_worker as worker

FOLDER = Path(__file__).resolve().parent
PLAN = json.loads((FOLDER/'PLAN.json').read_bytes())

class FollowupTests(unittest.TestCase):
    def test_parent_and_negative_history_preserved(self):
        self.assertTrue(p.validate(PLAN,FOLDER)['passed'])
    def test_judges_quality_and_human_scope_cannot_change(self):
        for path,value in [(['inherits_unchanged','judge_runtime','format_only_retry_limit'],1),
            (['inherits_unchanged','model_labels','drop_unavailable_from_denominator'],True),
            (['selection','precision_min'],'9/10'),(['validation','validation_lower_bound_min'],.8),
            (['resources','human_annotation_calls'],1),(['resources','retries'],1)]:
            plan=copy.deepcopy(PLAN);target=plan
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            with self.assertRaises(ValueError):p.validate(plan,FOLDER)
    def test_strict_boundary_strata(self):
        cuts=PLAN['selection']['stratum_upper_cuts']
        for k,t in enumerate(cuts):
            self.assertEqual(s.stratum(t,cuts),k)
            self.assertEqual(s.stratum(math.nextafter(t,math.inf),cuts),k+1)
            self.assertEqual(s.stratum(math.nextafter(t,-math.inf),cuts),k)
    def test_exact_reservoir_uniformity_all_rng_paths(self):
        # N=4, n=2: enumerate all 3*4 equiprobable replacement draws.
        counts=Counter()
        class Rng:
            def __init__(self,values):self.values=iter(values)
            def randrange(self,n):
                v=next(self.values);assert v<n;return v
        for a,b in product(range(3),range(4)):
            sample=[];rng=Rng([a,b])
            for i in range(4):s.insert(sample,i,i+1,2,rng)
            counts[tuple(sorted(sample))]+=1
        self.assertEqual(len(counts),6)
        self.assertEqual(set(counts.values()),{2})
    def test_small_population_census_and_weights(self):
        class NoDraw:
            def randrange(self,n):raise AssertionError('census should not draw')
        stream=[(0,1,.9),(0,2,.95),(1,2,.99)]
        r=s.select(stream,3,[.9,.95],30,[NoDraw(),NoDraw(),NoDraw()])
        self.assertEqual(r['populations'],[1,1,1]);self.assertEqual(r['raw'],[(0,1,.9,0),(0,2,.95,1),(1,2,.99,2)])
        self.assertEqual(s.weights([0,2,60],30),[None,Fraction(1),Fraction(2)])
    def test_no_duplicate_missing_pair_or_nonfinite_admission(self):
        for stream in [[(0,1,.9),(0,1,.9),(1,2,.9)],[(0,1,.9)],[(0,1,math.nan)]]:
            with self.assertRaises(ValueError):s.select(stream,3,[.9],30,s.generators('a'*64,'b'*64,'c'*64,2))
    def test_fresh_version_domains_and_stable_orientation(self):
        x=p.identities('a'*64,'b'*64,['x','y'])
        self.assertEqual(x,p.identities('a'*64,'b'*64,['y','x']))
        self.assertTrue(p.PREFIX.endswith('v4/'))
    def test_disjoint_strata_match_independent_tail_counts(self):
        cuts=PLAN['selection']['stratum_upper_cuts']
        scores=[-1.,1.]+[x for t in cuts for x in (math.nextafter(t,-math.inf),t,math.nextafter(t,math.inf))]
        tail={'threshold_grid':cuts,'observed':{'pair_count':len(scores),
              'counts_strictly_above':[sum(x>t for x in scores) for t in cuts]}}
        actual=worker.expected_populations(tail,cuts)
        expected=[sum(s.stratum(x,cuts)==k for x in scores) for k in range(len(cuts)+1)]
        self.assertEqual(actual,expected)
        self.assertEqual(sum(actual),len(scores))

if __name__ == '__main__':
    unittest.main()
