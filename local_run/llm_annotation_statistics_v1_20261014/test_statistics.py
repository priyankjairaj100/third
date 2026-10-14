import bisect
import copy
from fractions import Fraction
import itertools
import math
from pathlib import Path
import unittest

import statistics_core as s

REPO = Path(__file__).resolve().parents[2]


def fixture(scores, prefix='p'):
    bins = [max(0, min(19, bisect.bisect_right([float(Fraction(i, 10)) for i in range(-10, 11)], x)-1)) for x in scores]
    populations = [bins.count(i) for i in range(20)]
    assert max(populations) <= 30
    rows = [{'opaque_pair_id': prefix+str(i), 'score_hex': float(value).hex(), 'bin': b,
             'bin_population': populations[b], 'bin_sample_size': populations[b],
             'inclusion_probability': s.fjson(Fraction(1)), 'inclusion_weight': s.fjson(Fraction(1))}
            for i, (value, b) in enumerate(zip(scores, bins))]
    return rows, populations


def observed(rows, category='exact_copy', status='valid'):
    return [{'opaque_pair_id': row['opaque_pair_id'], 'model_id': model, 'status': status,
             'category': category if status == 'valid' else None} for row in rows for model in s.MODELS]


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound = staticmethod(s.frozen_bound(REPO))

    def test_all_statuses_keep_denominator(self):
        rows, pops = fixture([.96]*10)
        obs = observed(rows)
        for status in sorted(s.STATUSES - {'valid'}):
            changed = copy.deepcopy(obs)
            changed[-1].update(status=status, category=None)
            r = s.selection(rows, pops, changed)
            self.assertFalse(r['selection_passed'])
            self.assertEqual(r['grid_ledger'][0]['conservative_precision'], s.fjson(Fraction(9, 10)))
            self.assertEqual(r['grid_ledger'][0]['weighted_denominator'], s.fjson(Fraction(10)))

    def test_absent_pair_and_model_are_explicit_missing(self):
        rows, pops = fixture([.96]*20)
        r = s.selection(rows, pops, observed(rows)[:-3])
        self.assertFalse(r['selection_passed'])
        self.assertEqual(r['diagnostics']['status_counts'][s.MODELS[0]]['missing_output'], 1)
        self.assertEqual(r['diagnostics']['status_counts'][s.MODELS[1]]['missing_output'], 2)
        self.assertEqual(r['grid_ledger'][0]['conservative_precision'], s.fjson(Fraction(9, 10)))

    def test_positive_categories_can_disagree(self):
        rows, pops = fixture([.96])
        obs = observed(rows); obs[1]['category'] = 'substantially_same_meaning'
        r = s.selection(rows, pops, obs)
        self.assertTrue(r['selection_passed'])
        self.assertEqual(r['diagnostics']['valid_six_way_disagreements'], 1)
        self.assertEqual(r['selected_tau'], s.fjson(Fraction(1, 2)))

    def test_uncertain_and_substantive_negative_sensitivity(self):
        rows, pops = fixture([.96])
        obs = observed(rows, 'uncertain')
        r = s.selection(rows, pops, obs)['grid_ledger'][0]
        self.assertEqual(r['conservative_precision'], s.fjson(Fraction(0)))
        self.assertEqual(r['missing_uncertain_sensitivity_upper'], s.fjson(Fraction(1)))
        obs[0]['category'] = 'unrelated'
        r = s.selection(rows, pops, obs)['grid_ledger'][0]
        self.assertEqual(r['missing_uncertain_sensitivity_upper'], s.fjson(Fraction(0)))

    def test_strict_boundary_and_empty_support(self):
        rows, pops = fixture([.5])
        r = s.selection(rows, pops, observed(rows))
        self.assertFalse(r['selection_passed'])
        self.assertIsNone(r['grid_ledger'][0]['conservative_precision'])
        self.assertEqual(r['failed_diagnostic_tau'], s.fjson(Fraction(199, 200)))
        rows, pops = fixture([1.0000000000000002])
        r = s.selection(rows, pops, observed(rows))
        self.assertTrue(r['grid_ledger'][-1]['passes_selection_rule'])

    def test_least_threshold_and_exact_95_percent(self):
        rows, pops = fixture([.81]+[.96]*19)
        obs = observed(rows); obs[:2] = observed(rows[:1], 'unrelated')
        r = s.selection(rows, pops, obs)
        self.assertEqual(r['selected_tau'], s.fjson(Fraction(1, 2)))
        self.assertEqual(r['grid_ledger'][0]['conservative_precision'], s.fjson(Fraction(19, 20)))
        # One more substantive negative lowers the first region below threshold,
        # but score .96 support clears after the .81 negative is excluded.
        rows, pops = fixture([.81, .81]+[.96]*18)
        obs = observed(rows); obs[:4] = observed(rows[:2], 'unrelated')
        r = s.selection(rows, pops, obs)
        self.assertEqual(r['selected_tau'], s.fjson(Fraction(81, 100)))

    def test_exact_unequal_weight_denominator(self):
        rows, pops = fixture([.81]*30+[.96]*30)
        pops[18] = 300
        for row in rows[:30]:
            row['bin_population'] = 300
            row['inclusion_weight'] = s.fjson(Fraction(10))
            row['inclusion_probability'] = s.fjson(Fraction(1, 10))
        obs = observed(rows); obs[:60] = observed(rows[:30], 'unrelated')
        r = s.selection(rows, pops, obs)
        self.assertEqual(r['grid_ledger'][0]['conservative_precision'], s.fjson(Fraction(1, 11)))
        self.assertEqual(r['grid_ledger'][0]['weighted_denominator'], s.fjson(Fraction(330)))
        self.assertEqual(r['selected_tau'], s.fjson(Fraction(81, 100)))

    def test_malformed_design_and_observations_rejected(self):
        rows, pops = fixture([.96])
        for field, value in [('score_hex', 'nan'), ('bin', 18), ('bin_population', True),
                             ('inclusion_weight', {'numerator': 2, 'denominator': 2})]:
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.assertRaises(ValueError): s.selection(bad, pops, observed(rows))
        for obs in [observed(rows)*2, observed(rows)+[dict(observed(rows)[0], opaque_pair_id='unknown')],
                    [dict(observed(rows)[0], status='runtime_error')],
                    [dict(observed(rows)[0], model_id='replacement')]]:
            with self.assertRaises(ValueError): s.selection(rows, pops, obs)
        with self.assertRaises(ValueError): s.selection([], pops, [])

    def validation_case(self, validation_rows, validation_observed, N=None, selection_override=None):
        earlier, pops = fixture([.96])
        old = observed(earlier)
        selected = s.selection(earlier, pops, old)
        if selection_override is not None: selected.update(selection_override)
        return s.validation(validation_rows, len(validation_rows) if N is None else N, validation_observed,
                            earlier, pops, old, selected, self.bound)

    def test_frozen_bound_actual_import_and_census(self):
        bound = self.bound(10, 10, 9)
        self.assertEqual(bound['lower_proportion'], s.fjson(Fraction(9, 10)))
        self.assertFalse(self.bound(0, 0, 0)['defined'])

    def test_validation_census_pass_and_undefined_empty(self):
        rows, _ = fixture([.96]*10, 'v')
        obs = observed(rows); obs[-1].update(status='missing_output', category=None)
        r = self.validation_case(rows, obs)
        self.assertTrue(r['model_validation_passed'])
        self.assertTrue(r['zero_overlap_is_zero_replay_evidence'])
        r = self.validation_case([], [])
        self.assertFalse(r['model_validation_available'])
        self.assertFalse(r['model_validation_passed'])

    def test_positive_category_change_invalidates_overlap(self):
        rows, _ = fixture([.96])
        obs = observed(rows); obs[0]['category'] = 'substantially_same_meaning'
        r = self.validation_case(rows, obs)
        self.assertTrue(r['numeric_lower_bound_passed'])
        self.assertFalse(r['model_validation_available'])
        self.assertEqual(len(r['changed_overlap_observations']), 1)

    def test_status_change_invalidates_overlap(self):
        rows, _ = fixture([.96])
        obs = observed(rows); obs[0].update(status='runtime_error', category=None)
        r = self.validation_case(rows, obs)
        self.assertFalse(r['model_validation_available'])
        self.assertEqual(r['overlap_model_comparisons'], 2)

    def test_forged_threshold_stale_result_and_incomplete_validation_rejected(self):
        rows, _ = fixture([.96])
        with self.assertRaisesRegex(ValueError, 'selection_result_not_reproduced'):
            self.validation_case(rows, observed(rows), selection_override={'selected_tau': s.fjson(Fraction(9, 10))})
        with self.assertRaisesRegex(ValueError, 'validation_sample_incomplete'):
            self.validation_case(rows, observed(rows), N=201)
        earlier, pops = fixture([.96]); old = observed(earlier); selected = s.selection(earlier, pops, old)
        with self.assertRaisesRegex(ValueError, 'selection_result_not_reproduced'):
            s.validation(rows, 1, observed(rows), earlier, pops, [], selected, self.bound)

    def test_exact_bound_small_population_coverage(self):
        # Exhaustive finite-population tail coverage, independent combinatorial
        # outcome probabilities. This tests the inherited numerical convention.
        for N in range(1, 13):
            for n in range(1, N+1):
                bounds = [self.bound(N, n, x)['lower_success_count'] for x in range(n+1)]
                for K in range(N+1):
                    numerator = sum(math.comb(K, x)*math.comb(N-K, n-x)
                                    for x in range(max(0, n-(N-K)), min(n, K)+1) if bounds[x] > K)
                    self.assertLessEqual(Fraction(numerator, math.comb(N, n)), Fraction(1, 20))


if __name__ == '__main__':
    unittest.main()
