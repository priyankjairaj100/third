"""Analytic model-label fixtures for right-closed stratification and strict cuts."""
from fractions import Fraction
import unittest
import statistics_core as s

def fixture():
    populations=[0]*13;populations[10]=60;populations[12]=30;rows=[];observed=[]
    for b,score in ((10,.99),(12,.999)):
        for k in range(30):
            identity=str(b)+'-'+str(k);weight=Fraction(populations[b],30)
            rows.append({'opaque_pair_id':identity,'bin':b,'bin_population':populations[b],
                'bin_sample_size':30,'score_hex':score.hex(),'inclusion_weight':s.fjson(weight),
                'inclusion_probability':s.fjson(1/weight)})
            for model in s.MODELS:observed.append({'opaque_pair_id':identity,'model_id':model,
                'status':'valid','category':'exact_copy' if b==12 else 'unrelated'})
    return populations,rows,observed

class StatisticsTests(unittest.TestCase):
    def test_strict_cut_selects_099_with_exact_denominator(self):
        populations,rows,observed=fixture();r=s.selection(rows,populations,observed)
        self.assertEqual(s.rational(r['selected_tau']),Fraction(99,100))
        cut=next(x for x in r['grid_ledger'] if s.rational(x['tau'])==Fraction(99,100))
        self.assertEqual(s.rational(cut['weighted_denominator']),30)
        self.assertEqual(s.rational(cut['weighted_positive_numerator']),30)
    def test_two_unavailable_top_pairs_make_all_thresholds_fail(self):
        populations,rows,observed=fixture()
        for r in observed:
            if r['opaque_pair_id'] in ('12-0','12-1') and r['model_id']==s.MODELS[0]:
                r.update(status='invalid_json',category=None)
        r=s.selection(rows,populations,observed)
        self.assertFalse(r['selection_passed']);self.assertIsNone(r['selected_tau'])
    def test_corrupted_stratum_or_weight_is_not_accepted(self):
        populations,rows,observed=fixture();rows[0]['inclusion_weight']=s.fjson(Fraction(1))
        with self.assertRaises(ValueError):s.selection(rows,populations,observed)
        populations,rows,observed=fixture();rows[0]['bin']=11
        with self.assertRaises(ValueError):s.selection(rows,populations,observed)

if __name__=='__main__':unittest.main()
