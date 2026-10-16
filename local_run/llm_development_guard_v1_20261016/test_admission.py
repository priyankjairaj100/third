"""Synthetic gate fixtures only; no model calls or natural evidence."""
import base64
import copy
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import admission as a
import common as c
import fixture_statistics as s
from prepare_admission import prepare


def audits(observations):
    return {alias: {'canonical_observations': [r for r in observations if r['model_id'] == model]}
            for alias, model in a.MODELS.items()}


def fixture(production_shape=False):
    populations = [0]*13
    if production_shape:
        for b in [0] + list(range(2, 13)):
            populations[b] = 30
        populations[0] += 220405510-sum(populations)
    else:
        populations[10] = 60; populations[12] = 30
    rows = []; observations = []
    for b, N in enumerate(populations):
        if not N:
            continue
        value = ([.4]+a.CUTS[1:]+[.999])[b]
        for k in range(min(30, N)):
            pair = f'{b}-{k}'; n = min(30, N)
            rows.append({'opaque_pair_id': pair, 'bin': b, 'bin_population': N, 'bin_sample_size': n,
                         'score_hex': value.hex(), 'inclusion_weight': a.fjson(Fraction(N, n)),
                         'inclusion_probability': a.fjson(Fraction(n, N))})
            observations.extend({'opaque_pair_id': pair, 'model_id': model, 'status': 'valid',
                                 'category': 'exact_copy' if b == 12 else 'unrelated'} for model in s.MODELS)
    selected = s.selection(rows, populations, observations)
    return rows, populations, observations, selected


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(os.environ.get('THIRD_LOCAL_REPO', str(Path(__file__).resolve().parents[2])))
        cls.bound = staticmethod(s.frozen_bound(repo))

    def setUp(self):
        self.rows, self.pop, self.obs, self.selected = fixture()
        self.tau, self.earlier = a.selection(self.rows, self.pop, audits(self.obs), self.selected)

    def fresh(self, N=30, overlap=True):
        rows = []; obs = []
        for k in range(min(200, N)):
            pair = f'12-{k}' if overlap else f'fresh-{k}'
            rows.append({'opaque_pair_id': pair, 'score_hex': (.999).hex(),
                         'validation_population': N, 'validation_sample_size': min(200, N),
                         'inclusion_weight': a.fjson(Fraction(N, min(200, N))),
                         'inclusion_probability': a.fjson(Fraction(min(200, N), N))})
            obs.extend({'opaque_pair_id': pair, 'model_id': model, 'status': 'valid', 'category': 'exact_copy'}
                       for model in s.MODELS)
        return rows, obs

    def report(self, rows, obs, N):
        return s.validation(rows, N, obs, self.rows, self.pop, self.obs, self.selected, self.bound)

    def validate(self, rows, obs, N, report=None):
        return a.validation(rows, N, audits(obs), report or self.report(rows, obs, N), self.tau, self.earlier)

    def test_all_51_cuts_and_strict_boundary(self):
        self.assertEqual(self.tau, Fraction(99, 100))
        self.assertEqual(self.selected['grid_ledger'][-2]['sample_support'], 30)
        self.assertEqual(len(self.selected['grid_ledger']), 51)

    def test_negative_selection_cannot_admit(self):
        obs = copy.deepcopy(self.obs)
        for r in obs:
            if r['opaque_pair_id'] in ('12-0', '12-1') and r['model_id'] == s.MODELS[0]:
                r.update(status='invalid_json', category=None)
        result = s.selection(self.rows, self.pop, obs)
        self.assertFalse(result['selection_passed'])
        with self.assertRaisesRegex(ValueError, 'negative_selection'):
            a.selection(self.rows, self.pop, audits(obs), result)

    def test_grid_mutation_or_forged_threshold_rejected(self):
        for mutation in ('omitted', 'reordered', 'threshold', 'arithmetic'):
            r = copy.deepcopy(self.selected)
            if mutation == 'omitted': r['grid_ledger'].pop()
            if mutation == 'reordered': r['grid_ledger'].reverse()
            if mutation == 'threshold': r['selected_tau'] = a.fjson(Fraction(199, 200))
            if mutation == 'arithmetic': r['grid_ledger'][0]['weighted_denominator'] = a.fjson(Fraction(1))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                a.selection(self.rows, self.pop, audits(self.obs), r)

    def test_weights_denominator_model_identity_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0]['inclusion_weight'] = a.fjson(Fraction(1))
        with self.assertRaisesRegex(ValueError, 'weights'): a.selection(rows, self.pop, audits(self.obs), self.selected)
        for mutation in ('missing', 'duplicate', 'model'):
            panel = audits(copy.deepcopy(self.obs))
            if mutation == 'missing': panel['qwen']['canonical_observations'].pop()
            if mutation == 'duplicate': panel['qwen']['canonical_observations'].append(panel['qwen']['canonical_observations'][0])
            if mutation == 'model': panel['qwen']['canonical_observations'][0]['model_id'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): a.labels(self.rows, panel)

    def test_nonfinite_scores_and_noncanonical_fractions_rejected(self):
        for value in ('nan', 'inf', '0.99'):
            with self.subTest(value=value), self.assertRaises(ValueError): a.score({'score_hex': value})
        for value in ({'numerator': True, 'denominator': 1}, {'numerator': 2, 'denominator': 2}):
            with self.assertRaises(ValueError): a.fraction(value)

    def test_census_and_stable_overlap_pass(self):
        rows, obs = self.fresh()
        result = self.validate(rows, obs, 30)
        self.assertEqual(result['overlap_pairs'], 30)
        self.assertEqual(result['lower_success_count'], 30)

    def test_positive_category_change_is_unstable(self):
        rows, obs = self.fresh(); obs[0]['category'] = 'substantially_same_meaning'
        r = self.report(rows, obs, 30)
        self.assertTrue(r['numeric_lower_bound_passed']); self.assertFalse(r['model_validation_passed'])
        with self.assertRaisesRegex(ValueError, 'negative_or_unstable'): self.validate(rows, obs, 30, r)

    def test_census_pass_boundary_and_failed_numeric_gate(self):
        rows, obs = self.fresh(overlap=False)
        for negatives in (3, 4):
            changed = copy.deepcopy(obs)
            for r in changed:
                if int(r['opaque_pair_id'].split('-')[-1]) < negatives and r['model_id'] == s.MODELS[0]:
                    r.update(status='invalid_json', category=None)
            if negatives == 3:
                self.assertEqual(self.validate(rows, changed, 30)['conservative_positives'], 27)
            else:
                with self.assertRaisesRegex(ValueError, 'negative_or_unstable'): self.validate(rows, changed, 30)

    def test_exact_hypergeometric_bound_and_corruption(self):
        rows, obs = self.fresh(N=400, overlap=False); r = self.report(rows, obs, 400)
        got = self.validate(rows, obs, 400, r)
        self.assertLess(got['lower_success_count'], 400)
        bad = copy.deepcopy(r); bad['finite_population_bound']['lower_success_count'] += 1
        with self.assertRaisesRegex(ValueError, 'exact_95_percent_bound'): self.validate(rows, obs, 400, bad)

    def test_fresh_sample_filter_and_overlap_omission_rejected(self):
        rows, obs = self.fresh(); r = self.report(rows, obs, 30)
        with self.assertRaisesRegex(ValueError, 'SRS_size'): self.validate(rows[:-1], obs, 30, r)
        changed = copy.deepcopy(rows); changed[0]['score_hex'] = float(self.tau).hex()
        with self.assertRaisesRegex(ValueError, 'above_threshold'): self.validate(changed, obs, 30, r)
        obs[0]['category'] = 'substantially_same_meaning'
        with self.assertRaisesRegex(ValueError, 'stability_replay'): self.validate(rows, obs, 30, r)

    def production_bundle(self):
        self.rows, self.pop, self.obs, self.selected = fixture(production_shape=True)
        self.tau, self.earlier = a.selection(self.rows, self.pop, audits(self.obs), self.selected)
        rows, obs = self.fresh(); validation = self.report(rows, obs, 30)
        raw = {}; data = {}; bindings = {}
        def add(name, value):
            if name.endswith('.jsonl'):
                payload = b''.join(c.canonical(row)+b'\n' for row in value)
            else:
                value = copy.deepcopy(value)
                value['sha256'] = hashlib.sha256(c.canonical(value)).hexdigest()
                payload = c.canonical(value)+b'\n'
            data[name] = value; raw[name] = payload; bindings[name] = c.byte_binding(payload)
        add('selection_sample.jsonl', self.rows); add('validation_sample.jsonl', rows)
        expected = c.strict((c.HERE/'expectations.json').read_bytes())
        frame = {'passed': True, 'plan_sha256': c.PLAN_PIN, 'frame_result': expected['frame_result'],
                 'pair_count': 220405510, 'score_stream_sha256': '899114828d4f9d130d57f900660efc5d07f6b90e4a76c76d3eaedc7233323ea2'}
        add('selection_sampling.json', {**frame, 'selected_pairs': 360, 'populations': self.pop,
                                      'files': {'selection.private.jsonl': bindings['selection_sample.jsonl']}})
        for alias in a.MODELS:
            add('selection_'+alias+'.json', {**audits(self.obs)[alias], 'passed': True,
                'package_sha256': c.SELECTION_PIN, 'activation': 'synthetic-selection', 'request_manifest': 'old'})
            add('validation_'+alias+'.json', {**audits(obs)[alias], 'passed': True,
                'package_sha256': c.VALIDATION_PIN, 'activation': 'synthetic-validation', 'request_manifest': 'fresh'})
        add('selection_statistics.json', {'passed': True, 'package_sha256': c.SELECTION_PIN,
            'sampling_result': bindings['selection_sampling.json'], 'statistics': self.selected,
            'audits': {alias: bindings['selection_'+alias+'.json'] for alias in a.MODELS}})
        add('validation_sampling.json', {**frame, 'package_sha256': c.VALIDATION_SAMPLER_PIN,
            'selection_statistics': bindings['selection_statistics.json'], 'full_above_threshold_frame': True,
            'label_filter': False, 'overlap_requires_fresh_calls': True, 'locked_tau': a.fjson(self.tau),
            'population': 30, 'selected_pairs': 30, 'selection_overlap_pairs': 30,
            'files': {'validation.private.jsonl': bindings['validation_sample.jsonl']}})
        add('validation_statistics.json', {'passed': True, 'package_sha256': c.VALIDATION_PIN,
            'validation_sample': bindings['validation_sampling.json'], 'selection_statistics': bindings['selection_statistics.json'],
            'statistics': validation, 'audits': {alias: bindings['validation_'+alias+'.json'] for alias in a.MODELS}})
        def review_bind(mapping): return {k: bindings[v] for k, v in mapping.items()}
        sr = {'passed': True, 'all51_exact_cuts_independently_recomputed': True, 'selection_passed': True,
              'selected_tau': a.fjson(self.tau), 'human_annotation_calls': 0,
              'bindings': review_bind({'sample':'selection_sample.jsonl', 'qwen':'selection_qwen.json',
                                      'mistral':'selection_mistral.json', 'statistics':'selection_statistics.json'})}
        vr = {'passed': True, 'model_validation_passed': True, 'exact_bound_and_decision_independently_recomputed': True,
              'locked_tau': a.fjson(self.tau), 'human_annotation_calls': 0, 'semantic_validity_established': False,
              **self.validate(rows, obs, 30, validation), 'bindings': review_bind({
                  'sample':'validation_sample.jsonl', 'sampling':'validation_sampling.json',
                  'statistics':'validation_statistics.json', 'selection_statistics':'selection_statistics.json',
                  'qwen':'validation_qwen.json', 'mistral':'validation_mistral.json',
                  'selection_qwen':'selection_qwen.json', 'selection_mistral':'selection_mistral.json'})}
        done = {'selection_pipeline_complete': True, 'selection_passed': True, 'validation_run': True,
                'independent_validation_review': vr}
        for name, value in zip(c.LOCAL_FILES, (done, sr, vr)):
            raw[name] = c.canonical(value)+b'\n'; bindings[name] = c.byte_binding(raw[name]); data[name] = value
        return data, bindings, raw

    def test_complete_synthetic_bundle_preparation_and_tamper(self):
        data, bindings, raw = self.production_bundle()
        self.assertTrue(a.check_data(data, bindings)['passed'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); controller = root/'controller'; controller.mkdir()
            for source, target in zip(c.LOCAL_FILES, ('completion.json', 'selection_completion.json', 'independent_validation_review.json')):
                (controller/target).write_bytes(raw[source])
            export = {name: {**bindings[name], 'base64': base64.b64encode(raw[name]).decode()} for name in c.REMOTE_FILES}
            (root/'export.json').write_bytes(c.canonical(export))
            result = prepare(controller, root/'export.json', root/'admission')
            self.assertTrue(result['prepared'])
            with self.assertRaisesRegex(ValueError, 'occupied'): prepare(controller, root/'export.json', root/'admission')
            (root/'admission/validation_sample.jsonl').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'bound_admission'): a.check_bundle(root/'admission', result['admission_sha256'])

    def test_terminal_flags_frame_and_fresh_call_lineage_rejected(self):
        data, bindings, _ = self.production_bundle()
        for mutation in ('completion', 'frame', 'fresh_calls', 'human'):
            d = copy.deepcopy(data)
            if mutation == 'completion': d['controller_completion.json']['validation_run'] = False
            if mutation == 'frame':
                for name in ('selection_sampling.json', 'validation_sampling.json'):
                    d[name]['frame_result']['sha256'] = 'f'*64
            if mutation == 'fresh_calls':
                d['validation_qwen.json']['activation'] = d['selection_qwen.json']['activation']
            if mutation == 'human': d['selection_review.json']['human_annotation_calls'] = 1
            for name in c.REMOTE_FILES:
                if name.endswith('.json'):
                    d[name].pop('sha256'); d[name]['sha256'] = hashlib.sha256(c.canonical(d[name])).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): a.check_data(d, bindings)

    def test_missing_or_failed_completion_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); controller = root/'controller'; controller.mkdir()
            with self.assertRaises(FileNotFoundError): prepare(controller, root/'missing-export', root/'out')
            self.assertFalse((root/'out').exists())
            (controller/'completion.json').write_text('{"selection_pipeline_complete":true,"selection_passed":false,"validation_run":false}')
            with self.assertRaisesRegex(ValueError, 'terminal_positive'): prepare(controller, root/'missing-export', root/'out')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__': unittest.main()
