"""Independent exact-coverage checks and overlap failure fixtures."""
from fractions import Fraction
from math import comb
import os
from pathlib import Path
import unittest
import statistics_core as s
from test_statistics import fixture

REPO=Path(os.environ.get('THIRD_FROZEN_REPO',str(Path(__file__).resolve().parents[2])))

class ValidationStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bound=staticmethod(s.frozen_bound(REPO))
    def test_exact_coverage_by_brute_force_small_populations(self):
        for N in range(1,9):
            for n in range(1,N+1):
                for K in range(N+1):
                    failure=0
                    for successes in range(max(0,n+K-N),min(n,K)+1):
                        lower=self.bound(N,n,successes)['lower_success_count']
                        if lower>K:failure+=comb(K,successes)*comb(N-K,n-successes)
                    self.assertLessEqual(Fraction(failure,comb(N,n)),Fraction(1,20))
    def test_empty_and_full_census_boundary(self):
        self.assertFalse(self.bound(0,0,0)['defined'])
        self.assertEqual(self.bound(38,38,35)['lower_success_count'],35)
        self.assertEqual(self.bound(38,38,34)['lower_success_count'],34)
    def test_fresh_census_passes_and_category_instability_invalidates(self):
        populations,selected,observed=fixture();selection=s.selection(selected,populations,observed)
        rows=[r for r in selected if r['bin']==12]
        fresh=[dict(r) for r in observed if r['opaque_pair_id'].startswith('12-')]
        result=s.validation(rows,30,fresh,selected,populations,observed,selection,self.bound)
        self.assertTrue(result['model_validation_passed']);self.assertEqual(result['overlap_pairs'],30)
        fresh[0]['category']='substantially_same_meaning'
        result=s.validation(rows,30,fresh,selected,populations,observed,selection,self.bound)
        self.assertTrue(result['numeric_lower_bound_passed']);self.assertFalse(result['model_validation_passed'])
        self.assertFalse(result['model_validation_available']);self.assertEqual(len(result['changed_overlap_observations']),1)
    def test_missing_output_keeps_full_denominator_and_blocks_overlap_stability(self):
        populations,selected,observed=fixture();selection=s.selection(selected,populations,observed)
        rows=[r for r in selected if r['bin']==12]
        fresh=[dict(r) for r in observed if r['opaque_pair_id'].startswith('12-')][1:]
        result=s.validation(rows,30,fresh,selected,populations,observed,selection,self.bound)
        self.assertEqual(result['finite_population_bound']['n'],30)
        self.assertEqual(result['finite_population_bound']['x'],29)
        self.assertFalse(result['model_validation_passed'])

if __name__=='__main__':unittest.main()
