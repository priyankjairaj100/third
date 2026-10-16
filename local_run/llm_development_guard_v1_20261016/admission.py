"""Replay selection/validation arithmetic from exact completed artifacts.

No threshold argument or forced-pass flag exists. This does not run model calls.
"""
from collections import Counter
from fractions import Fraction
from math import comb, isfinite
import bisect
from pathlib import Path
import common as c

MODELS = {'qwen': 'Qwen/Qwen2.5-7B-Instruct', 'mistral': 'mistralai/Mistral-Nemo-Instruct-2407'}
CATEGORIES = ('exact_copy', 'substantially_same_meaning', 'overlapping_information',
              'merely_related', 'unrelated', 'uncertain')
STATUSES = ('valid', 'invalid_json', 'refusal', 'context_overflow', 'runtime_error',
            'missing_output', 'not_attempted')
GRID = [Fraction(i, 100) for i in range(50, 100)] + [Fraction(199, 200)]
CUTS = [.5, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99, .995]


def fraction(value):
    c.require(isinstance(value, dict) and set(value) == {'numerator', 'denominator'}, 'fraction_fields')
    a, b = value['numerator'], value['denominator']
    c.require(type(a) is int and type(b) is int and b > 0, 'fraction_integer')
    q = Fraction(a, b)
    c.require((q.numerator, q.denominator) == (a, b), 'reduced_fraction')
    return q


def score(row):
    value = float.fromhex(row['score_hex'])
    c.require(isfinite(value) and value.hex() == row['score_hex'], 'finite_canonical_score')
    return value


def fjson(value):
    return None if value is None else {'numerator': value.numerator, 'denominator': value.denominator}


def labels(rows, audits):
    ids = [r['opaque_pair_id'] for r in rows]
    c.require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), 'unique_pair_ids')
    found = {}
    for alias, model in MODELS.items():
        observations = audits[alias]['canonical_observations']; values = {}
        for row in observations:
            c.require(set(row) == {'opaque_pair_id', 'model_id', 'status', 'category'}, 'canonical_observation_fields')
            identity = row['opaque_pair_id']
            c.require(identity in ids and identity not in values and row['model_id'] == model, 'pair_model_identity')
            c.require(row['status'] in STATUSES, 'known_status')
            c.require(row['category'] in CATEGORIES if row['status'] == 'valid' else row['category'] is None,
                      'status_category')
            values[identity] = row
        c.require(set(values) == set(ids) and len(observations) == len(rows), 'complete_denominator')
        found[alias] = values
    return found


def positive(panel, identity):
    return all(panel[a][identity]['status'] == 'valid' and
               panel[a][identity]['category'] in CATEGORIES[:2] for a in MODELS)


def selection(rows, populations, audits, stats):
    c.require(len(populations) == 13 and all(type(n) is int and n >= 0 for n in populations), '13_populations')
    counts = Counter()
    for row in rows:
        b = bisect.bisect_left(CUTS, score(row)); N = populations[b]; n = min(30, N)
        c.require(n > 0 and row['bin'] == b and row['bin_population'] == N and row['bin_sample_size'] == n,
                  'stratified_sample_identity')
        c.require(fraction(row['inclusion_weight']) == Fraction(N, n) and
                  fraction(row['inclusion_probability']) == Fraction(n, N), 'selection_weights')
        counts[b] += 1
    c.require([counts[i] for i in range(13)] == [min(30, n) for n in populations], 'complete_strata')
    panel = labels(rows, audits); chosen = None
    c.require(len(stats['grid_ledger']) == len(GRID), 'original_51_grid')
    for tau, cut in zip(GRID, stats['grid_ledger']):
        c.require(fraction(cut['tau']) == tau and cut['tau_float_hex'] == float(tau).hex(), 'unchanged_grid')
        above = [r for r in rows if score(r) > float(tau)]
        den = sum((fraction(r['inclusion_weight']) for r in above), Fraction())
        num = sum((fraction(r['inclusion_weight']) * positive(panel, r['opaque_pair_id']) for r in above), Fraction())
        passed = den > 0 and num / den >= Fraction(19, 20)
        c.require(cut['weighted_denominator'] == fjson(den) and cut['weighted_positive_numerator'] == fjson(num)
                  and cut['conservative_precision'] == fjson(num/den if den else None)
                  and cut['sample_support'] == len(above) and cut['passes_selection_rule'] is passed,
                  'independent_selection_arithmetic')
        if passed and chosen is None:
            chosen = tau
    c.require(stats['selected_tau'] == fjson(chosen) and stats['selection_passed'] is (chosen is not None),
              'least_strict_selection')
    c.require(chosen is not None, 'negative_selection_no_guard')
    return chosen, panel


def validation(rows, N, audits, stats, selected_tau, earlier):
    c.require(type(N) is int and N > 0 and len(rows) == min(200, N), 'fresh_nonempty_SRS_size')
    n = len(rows)
    c.require(fraction(stats['locked_tau']) == selected_tau, 'locked_tau_unchanged')
    for row in rows:
        c.require(score(row) > float(selected_tau) and row['validation_population'] == N
                  and row['validation_sample_size'] == n, 'full_above_threshold_sample')
        c.require(fraction(row['inclusion_weight']) == Fraction(N, n) and
                  fraction(row['inclusion_probability']) == Fraction(n, N), 'validation_weights')
    current = labels(rows, audits); ids = set(current['qwen'])
    x = sum(positive(current, i) for i in ids); bound = stats['finite_population_bound']
    c.require(bound['defined'] is True and (bound['N'], bound['n'], bound['x']) == (N, n, x), 'bound_denominator')
    L = bound['lower_success_count']
    c.require(type(L) is int and 0 <= L <= N, 'integer_lower_count')
    if n == N:
        c.require(L == x, 'exact_census')
    elif x == 0:
        c.require(L == 0, 'zero_success')
    else:
        def tail(K):
            return sum(comb(K, j)*comb(N-K, n-j) for j in range(max(x, n-(N-K)), min(n, K)+1))
        denominator = comb(N, n)
        c.require(20*tail(L) > denominator and (L == 0 or 20*tail(L-1) <= denominator), 'exact_95_percent_bound')
    c.require(fraction(bound['lower_proportion']) == Fraction(L, N), 'lower_fraction')
    overlap = ids & set(earlier['qwen']) & set(earlier['mistral']); changes = []
    for identity in sorted(overlap):
        for alias in MODELS:
            a, b = earlier[alias][identity], current[alias][identity]
            if (a['status'], a['category']) != (b['status'], b['category']):
                changes.append((identity, a['model_id']))
    reported = [(r['opaque_pair_id'], r['model_id']) for r in stats['changed_overlap_observations']]
    c.require(set(changes) == set(reported) and len(changes) == len(reported)
              and stats['overlap_pairs'] == len(overlap), 'overlap_stability_replay')
    passed = 10*L >= 9*N and not changes
    c.require(stats['numeric_lower_bound_passed'] is (10*L >= 9*N)
              and stats['model_validation_available'] is (not changes)
              and stats['model_validation_passed'] is passed, 'validation_decision_replay')
    c.require(passed, 'negative_or_unstable_validation_no_guard')
    return {'population': N, 'sample_size': n, 'conservative_positives': x,
            'lower_success_count': L, 'overlap_pairs': len(overlap), 'changed_overlap_observations': 0}


def check_data(data, bindings):
    done, review, vr = (data[n] for n in c.LOCAL_FILES)
    c.require(done['selection_pipeline_complete'] is True and done['selection_passed'] is True
              and done['validation_run'] is True and done['independent_validation_review'] == vr,
              'terminal_controller_validation_required')
    c.require(review['passed'] is True and review['all51_exact_cuts_independently_recomputed'] is True
              and review['selection_passed'] is True, 'completed_selection_review')
    c.require(vr['passed'] is True and vr['model_validation_passed'] is True
              and vr['exact_bound_and_decision_independently_recomputed'] is True, 'completed_validation_review')
    c.require(review['human_annotation_calls'] == vr['human_annotation_calls'] == 0
              and vr['semantic_validity_established'] is False, 'model_only_review_scope')
    for name in c.REMOTE_FILES:
        if name.endswith('.json'):
            c.seal_check(data[name]); c.require(data[name]['passed'] is True, 'passed_actual_artifact')
    source, selected = data['selection_sampling.json'], data['selection_statistics.json']
    fresh, report = data['validation_sampling.json'], data['validation_statistics.json']
    c.require(source['plan_sha256'] == fresh['plan_sha256'] == c.PLAN_PIN, 'frozen_plan')
    expected = c.strict((c.HERE/'expectations.json').read_bytes())
    c.require(source['frame_result'] == fresh['frame_result'] == expected['frame_result'], 'same_pinned_calibration_frame')
    c.require(source['pair_count'] == fresh['pair_count'] == 220405510
              and source['score_stream_sha256'] == fresh['score_stream_sha256'] ==
              '899114828d4f9d130d57f900660efc5d07f6b90e4a76c76d3eaedc7233323ea2', 'same_full_pair_population')
    c.require(source['files']['selection.private.jsonl'] == bindings['selection_sample.jsonl']
              and selected['sampling_result'] == bindings['selection_sampling.json'], 'selection_sample_binding')
    c.require(fresh['files']['validation.private.jsonl'] == bindings['validation_sample.jsonl']
              and report['validation_sample'] == bindings['validation_sampling.json'], 'validation_sample_binding')
    c.require(fresh['selection_statistics'] == report['selection_statistics'] == bindings['selection_statistics.json'],
              'locked_selection_result')
    c.require(selected['package_sha256'] == c.SELECTION_PIN and report['package_sha256'] == c.VALIDATION_PIN
              and fresh['package_sha256'] == c.VALIDATION_SAMPLER_PIN, 'execution_package_lineage')
    c.require(fresh['full_above_threshold_frame'] is True and fresh['label_filter'] is False
              and fresh['overlap_requires_fresh_calls'] is True, 'fresh_validation_design')
    for alias in MODELS:
        old, new = data['selection_'+alias+'.json'], data['validation_'+alias+'.json']
        c.require(selected['audits'][alias] == bindings['selection_'+alias+'.json'] and
                  report['audits'][alias] == bindings['validation_'+alias+'.json'], 'actual_CPU_audit_binding')
        c.require(old['package_sha256'] == c.SELECTION_PIN and new['package_sha256'] == c.VALIDATION_PIN,
                  'audit_execution_identity')
        c.require(old['activation'] != new['activation'] and old['request_manifest'] != new['request_manifest'],
                  'fresh_calls_required')
    selection_bindings = {'sample': 'selection_sample.jsonl', 'qwen': 'selection_qwen.json',
                          'mistral': 'selection_mistral.json', 'statistics': 'selection_statistics.json'}
    validation_bindings = {'sample': 'validation_sample.jsonl', 'sampling': 'validation_sampling.json',
                           'statistics': 'validation_statistics.json', 'selection_statistics': 'selection_statistics.json',
                           'qwen': 'validation_qwen.json', 'mistral': 'validation_mistral.json',
                           'selection_qwen': 'selection_qwen.json', 'selection_mistral': 'selection_mistral.json'}
    c.require(review['bindings'] == {k: bindings[v] for k, v in selection_bindings.items()}, 'selection_review_bytes')
    c.require(vr['bindings'] == {k: bindings[v] for k, v in validation_bindings.items()}, 'validation_review_bytes')
    rows = data['selection_sample.jsonl']
    c.require(len(rows) == source['selected_pairs'] == 360 and sum(source['populations']) == 220405510,
              'actual_complete_selection_scope')
    tau, earlier = selection(rows, source['populations'],
                             {a: data['selection_'+a+'.json'] for a in MODELS}, selected['statistics'])
    c.require(fjson(tau) == review['selected_tau'] == vr['locked_tau'] == fresh['locked_tau'], 'locked_review_tau')
    c.require(fresh['selected_pairs'] == len(data['validation_sample.jsonl']), 'complete_fresh_sample')
    got = validation(data['validation_sample.jsonl'], fresh['population'],
                     {a: data['validation_'+a+'.json'] for a in MODELS}, report['statistics'], tau, earlier)
    c.require(got['overlap_pairs'] == fresh['selection_overlap_pairs'], 'sample_overlap')
    c.require(all(vr[k] == v for k, v in got.items()), 'independent_review_agrees')
    return {'passed': True, 'locked_tau': fjson(tau), 'model_validation_passed': True,
            'all51_cuts_replayed': True, 'exact_finite_population_bound_replayed': True,
            'canonical_overlap_replayed': True, 'human_annotation_calls': 0,
            'human_validity_claimed': False, **got}


def read_payloads(folder, manifest):
    wanted_names = set(c.REMOTE_FILES) | set(c.LOCAL_FILES)
    c.require(set(manifest['files']) == wanted_names, 'complete_admission_files')
    data, bindings = {}, {}
    for name in sorted(wanted_names):
        p = folder/name
        c.require(p.stat().st_size <= 2 << 20, 'admission_file_cap')
        bindings[name] = c.binding(p)
        c.require(bindings[name] == manifest['files'][name] and bindings[name]['bytes'] <= 2 << 20,
                  'bound_admission_payload')
        raw = p.read_bytes()
        c.require(c.byte_binding(raw) == bindings[name], 'admission_read_race')
        data[name] = [c.strict(line) for line in raw.splitlines()] if name.endswith('.jsonl') else c.strict(raw)
    return data, bindings


def check_bundle(folder, pin, *, remote=False):
    folder = Path(folder)
    c.require(c.binding(folder/'admission.json')['sha256'] == pin, 'admission_pin')
    manifest = c.strict((folder/'admission.json').read_bytes())
    c.require(manifest['schema'] == 'third-full-development-guard-admission-v1', 'admission_schema')
    data, bindings = read_payloads(folder, manifest)
    if remote:
        c.require(folder == c.BUNDLE, 'fixed_remote_bundle')
        for name, path in c.REMOTE_FILES.items():
            c.require(c.binding(path) == bindings[name], 'live_calibration_payload_changed')
    result = check_data(data, bindings)
    c.require(manifest['review'] == result, 'admission_review_replayed')
    return result
