"""Exact design-based category intervals for frozen human audit frames.

The unit is an original three-rater majority, conditional on fixed potential
outcomes. Blank, skipped, incomplete, and unavailable contexts stay unidentified.
The intervals do not quantify latent semantic truth or new-rater variation.
"""
from fractions import Fraction
from math import comb
from pathlib import Path
import argparse
import hashlib
import json

from phase5.human_analysis import validate_responses, _domains, DISTINCTION_TYPES, CORPORA
from phase4.requests import digest


def rational(value):
    value = value if isinstance(value, Fraction) else Fraction(str(value))
    if not 0 < value < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    return value


def fraction_json(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def hypergeom_tail(N, K, n, k, *, upper):
    """Exact rational tail. Invalid binomial coefficients contribute zero."""
    if any(type(v) is not int for v in (N, K, n, k)) or not 0 <= K <= N or not 0 <= n <= N:
        raise ValueError('valid integer population and sample counts required')
    low, high = max(0, n - (N - K)), min(n, K)
    low, high = (max(low, k), high) if upper else (low, min(high, k))
    numerator = sum(comb(K, j) * comb(N - K, n - j) for j in range(low, high + 1))
    return Fraction(numerator, comb(N, n))


def count_interval(N, n, observed_positive, missing, *, alpha=Fraction(1, 20)):
    """Union exact equal-tail intervals over every missing binary completion."""
    alpha = rational(alpha)
    if any(type(v) is not int for v in (N, n, observed_positive, missing)) or not 0 <= observed_positive <= n <= N or not 0 <= missing <= n - observed_positive:
        raise ValueError('valid integer counts required')
    if not n:
        return [0, N]
    k_low, k_high = observed_positive, observed_positive + missing
    # The upper tail increases with K. The lower tail decreases with K.
    low, high = k_low, N - n + k_low
    while low < high:
        mid = (low + high) // 2
        if hypergeom_tail(N, mid, n, k_low, upper=True) >= alpha / 2:
            high = mid
        else:
            low = mid + 1
    lower = low
    low, high = k_high, N - n + k_high
    while low < high:
        mid = (low + high + 1) // 2
        if hypergeom_tail(N, mid, n, k_high, upper=False) >= alpha / 2:
            low = mid
        else:
            high = mid - 1
    return [lower, low]


def sampling_design(manifest, data):
    """Choose one outcome-free owner for each unit from its sampling strata."""
    pair = data['is_pair']
    strata = []
    for index, st in enumerate(manifest['strata']):
        frame = st['frame_pair_ids'] if pair else st['frame_unit_ids']
        sample = st['selected_pair_ids'] if pair else st['sample_unit_ids']
        strata.append({'index': index, 'frame': set(frame), 'sample': set(sample),
                       'N': len(frame), 'n': len(sample)})
    owners = {}
    for item in data['units']:
        memberships = [st for st in strata if item in st['frame']]
        if not memberships:
            raise ValueError('every frame unit needs a sampling stratum')
        # High inclusion probability improves efficiency. The index breaks ties.
        owner = min(memberships, key=lambda st: (-Fraction(st['n'], st['N']), st['index']))
        owners[item] = owner['index']
    return strata, owners


def category_interval(domain, values, strata, owners, *, alpha):
    """Valid for overlapping independent SRS strata and arbitrary nonresponse.

    Only each owner's original SRS enters its randomization calculation.
    Known outcomes from the complete union add deterministic count constraints.
    """
    domain = set(domain)
    if not domain:
        return {'frame_units': 0, 'count_interval': None, 'prevalence_interval': None,
                'observed_units': 0, 'point_estimate_identified': False, 'strata': []}
    if any(v not in (None, 0, 1) for v in values.values()):
        raise ValueError('binary original-majority outcomes or missing values required')
    active = [st for st in strata if any(owners[i] == st['index'] for i in domain)]
    per_stratum_alpha = rational(alpha) / len(active)
    rows = []
    for st in active:
        owned = {i for i in domain if owners[i] == st['index']}
        sampled = owned & st['sample']
        positive = sum(values.get(i) == 1 for i in sampled)
        missing = sum(values.get(i) is None for i in sampled)
        interval = count_interval(st['N'], st['n'], positive, missing, alpha=per_stratum_alpha)
        # Outside owned, the transformed binary outcome is identically zero.
        known_positive = sum(values.get(i) == 1 for i in owned)
        known_negative = sum(values.get(i) == 0 for i in owned)
        lower = max(interval[0], known_positive)
        upper = min(interval[1], len(owned) - known_negative)
        if lower > upper:
            # This can occur on the interval's allowed noncoverage event.
            # Report an empty confidence set; never hide inconsistency by clipping.
            interval = None
        else:
            interval = [lower, upper]
        rows.append({'stratum_index': st['index'], 'stratum_population': st['N'],
                     'stratum_sample': st['n'], 'owned_domain_units': len(owned),
                     'owned_sample_positive': positive, 'owned_sample_missing': missing,
                     'known_union_positive': known_positive, 'known_union_negative': known_negative,
                     'alpha': fraction_json(per_stratum_alpha), 'count_interval': interval})
    empty = any(r['count_interval'] is None for r in rows)
    counts = None if empty else [sum(r['count_interval'][j] for r in rows) for j in (0, 1)]
    known = sum(values.get(i) is not None for i in domain)
    return {'frame_units': len(domain), 'observed_units': known,
            'count_interval': counts, 'prevalence_interval': [v / len(domain) for v in counts] if counts else None,
            'empty_confidence_set': empty, 'point_estimate_identified': known == len(domain),
            'census_prevalence': sum(values.get(i) == 1 for i in domain) / len(domain) if known == len(domain) else None,
            'strata': rows}


def analyze_intervals(manifest, bundle, *, alpha=Fraction(1, 20), software_fixture_only=False):
    alpha = rational(alpha)
    data = validate_responses(manifest, bundle, software_fixture_only=software_fixture_only)
    strata, owners = sampling_design(manifest, data)
    endpoints = []
    for (corpus, role), domain in _domains(data['is_pair'], data['units']).items():
        scopes = ['pair'] if data['is_pair'] else ['complete_original_context', 'displayed_context_only']
        for scope in scopes:
            for question, choices in list(data['questions'].items()) + [('distinction_type:' + t, [0, 1]) for t in DISTINCTION_TYPES]:
                categories = choices + ['no_majority'] if not question.startswith('distinction_type:') else [1]
                for category in categories:
                    values = {}
                    for item in domain:
                        if item not in data['outcomes']:
                            continue
                        unit = data['units'][item]
                        allowed = data['is_pair'] or (not unit['missing_comparison'] and
                                  (scope == 'displayed_context_only' or unit['complete_context_displayed']))
                        outcome = data['outcomes'][item][question]
                        values[item] = int(outcome == category) if allowed and outcome is not None else None
                    endpoints.append({'corpus': corpus, 'population': role, 'context_scope': scope,
                                      'question': question, 'category': category,
                                      '_domain': domain, '_values': values})
    active_count = sum(bool(e['_domain']) for e in endpoints)
    results = []
    for endpoint in endpoints:
        domain, values = endpoint.pop('_domain'), endpoint.pop('_values')
        results.append({**endpoint, **category_interval(domain, values, strata, owners,
                        alpha=alpha / max(1, active_count))})
    balanced = []
    group_keys = sorted({(r['population'], r['context_scope'], r['question'], str(r['category'])) for r in results})
    for key in group_keys:
        rows = [r for r in results if (r['population'], r['context_scope'], r['question'], str(r['category'])) == key]
        available = len(rows) == len(CORPORA) and {r['corpus'] for r in rows} == set(CORPORA) and all(r['prevalence_interval'] is not None for r in rows)
        balanced.append({'population': key[0], 'context_scope': key[1], 'question': key[2],
                         'category': rows[0]['category'],
                         'prevalence_interval': [sum(r['prevalence_interval'][j] for r in rows) / len(CORPORA) for j in (0, 1)] if available else None,
                         'definition': 'unweighted mean of the three specified corpus prevalences; no missing-corpus reweighting',
                         'inherits_same_simultaneous_event': True})
    result = {'schema': 'ccu-human-design-inference-v1', 'manifest_sha256': manifest['manifest_sha256'],
              'response_bundle_sha256': digest(bundle), 'alpha': fraction_json(alpha),
              'family': 'all reported nonempty corpus/population/context/question/category endpoints in this one manifest',
              'nonempty_endpoint_count': active_count, 'coverage': 'simultaneous finite-population design coverage at least 1-alpha',
              'estimand': 'fixed original three-rater protocol outcomes conditional on frame, blocker choices and context occurrence choices',
              'assumptions': ['each recorded stratum sample follows its declared SRSWOR design',
                              'potential original three-rater outcomes do not change with sample selection',
                              'sampling seeds and frame choices precede response inspection'],
              'nonresponse': 'arbitrary outcome-dependent missingness allowed; all binary completions retained',
              'owner_rule': 'highest stratum inclusion probability, then original manifest stratum index',
              'owner_sha256': digest(owners), 'raters_or_repeated_roles_counted_as_independent_units': False,
              'human_origin_authenticated': False, 'actual_human_response_count': data['actual_human_response_count'],
              'evidence_role': 'software_fixture_only' if software_fixture_only else manifest.get('evidence_role'),
              'semantic_truth_coverage': False, 'new_rater_population_coverage': False,
              'point_estimates': 'use unchanged Phase 5 HT/Hajek estimates; these intervals target finite-frame prevalence',
              'results': results, 'equal_corpus_balanced': balanced,
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                    for name in ('phase5/human_analysis.py', 'phase5/context_audit.py', 'phase4/admission_audit.py')}}
    result['sha256'] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', default='0.05', help='whole-report family error, decimal or exact fraction')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous inference output')
    result = analyze_intervals(json.loads(args.manifest.read_text()), json.loads(args.responses.read_text()), alpha=Fraction(args.alpha))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'actual_human_response_count': result['actual_human_response_count']}))


if __name__ == '__main__':
    main()
