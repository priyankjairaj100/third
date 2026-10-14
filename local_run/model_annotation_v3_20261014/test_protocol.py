"""Small algebraic software fixtures only; never saved as empirical annotations."""
import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path
import protocol as p

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((Path(__file__).parent / 'PLAN.json').read_bytes())

def responses(a, b, status='valid'):
    return [{'model_id': 'software-model-a', 'status': 'valid', 'category': a},
            {'model_id': 'software-model-b', 'status': status, 'category': b}]

class ProtocolTests(unittest.TestCase):
    def test_current_bindings_and_prospective_only(self):
        result = p.validate(PLAN, ROOT)
        self.assertTrue(result['passed']); self.assertFalse(result['scientific_gates_completed'])
    def test_all_six_way_pairs_conservative_truth_table(self):
        for a in p.CATEGORIES:
            for b in p.CATEGORIES:
                self.assertEqual(p.panel_label(responses(a, b)), int(a in p.POSITIVE and b in p.POSITIVE))
    def test_unavailable_retains_complete_weight(self):
        a = responses('exact_copy', 'substantially_same_meaning')
        for status in p.STATUSES - {'valid'}:
            b = responses('exact_copy', None, status)
            self.assertEqual(p.weighted_precision([(Fraction(2), a), (Fraction(3), b)]), Fraction(2, 5))
        self.assertIsNone(p.weighted_precision([]))
    def test_no_duplicate_judge_or_failed_substantive_vote(self):
        r = responses('exact_copy', 'exact_copy'); r[1]['model_id'] = r[0]['model_id']
        with self.assertRaises(ValueError): p.panel_label(r)
        with self.assertRaises(ValueError): p.panel_label(responses('exact_copy', 'exact_copy', 'runtime_error'))
    def test_replay_change_even_two_positive_categories(self):
        a = responses('exact_copy', 'exact_copy'); b = responses('exact_copy', 'substantially_same_meaning')
        self.assertEqual(p.panel_label(a), p.panel_label(b)); self.assertTrue(p.overlap_instability(a, b))
        self.assertTrue(p.overlap_instability(a, responses('exact_copy', None, 'context_overflow')))
        self.assertFalse(p.overlap_instability(a, copy.deepcopy(a)))
    def test_identity_unordered_unicode_not_normalized(self):
        a = p.identities('a'*64, 'b'*64, ['record-é', 'record-z'])
        self.assertEqual(a, p.identities('a'*64, 'b'*64, ['record-z', 'record-é']))
        self.assertNotEqual(a, p.identities('a'*64, 'b'*64, ['record-e\u0301', 'record-z']))
        self.assertNotEqual(a, p.identities('c'*64, 'b'*64, ['record-é', 'record-z']))
    def test_domains_structured_no_collision_and_seed_binding(self):
        self.assertNotEqual(p.domain_digest(p.PREFIX+'sampling', ['a|b','c']), p.domain_digest(p.PREFIX+'sampling', ['a','b|c']))
        self.assertNotEqual(p.domain_digest(p.PREFIX+'sampling', ['x']), p.domain_digest(p.PREFIX+'orientation', ['x']))
        seed = p.model_seed('a'*64,'b'*64,'c'*64,'model','1'*40)
        self.assertEqual(seed, p.model_seed('a'*64,'b'*64,'c'*64,'model','1'*40))
        self.assertNotEqual(seed, p.model_seed('a'*64,'b'*64,'c'*64,'model','2'*40))
        self.assertTrue(0 <= seed < 2**63)
    def test_refuse_human_or_fabricated_gate(self):
        for section, key, value in [('supersession','human_reviews_planned',100),
             ('supersession','original_v1_three_human_gate_satisfied',True),
             ('activation','completed',True),('activation','human_annotation_dependency',True),
             ('not_completed','confirmatory_study_ready',True)]:
            v=copy.deepcopy(PLAN); v[section][key]=value
            with self.assertRaises(ValueError): p.validate(v, ROOT)
    def test_refuse_numeric_denominator_runtime_or_replay_relaxation(self):
        for section,key,value in [('calibration','validation_pair_max',100),('calibration','selection_weighted_precision_min',0.9),
            ('calibration','validation_lower_bound_min',0.8),('model_labels','drop_unavailable_from_denominator',True),
            ('judge_runtime','max_new_tokens',384),('judge_runtime','format_only_retry_limit',1),
            ('validation_stability','reuse_selection_or_pilot_output_as_validation',True)]:
            v=copy.deepcopy(PLAN); v[section][key]=value
            with self.assertRaises(ValueError): p.validate(v, ROOT)
    def test_refuse_changed_bound_source(self):
        v=copy.deepcopy(PLAN); next(iter(v['bound_files'].values()))['sha256']='0'*64
        with self.assertRaises(ValueError): p.validate(v, ROOT)

if __name__ == '__main__': unittest.main()
