"""Finite combinatorial tests and actual blank-pack acceptance. No human data."""
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path
import hashlib
import json

from phase6.human_inference import count_interval, hypergeom_tail, category_interval, analyze_intervals

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = {}
    cases = 0
    minimum = Fraction(1)
    for N in range(1, 9):
        for n in range(N + 1):
            for K in range(N + 1):
                coverage = Fraction(0)
                for k in range(max(0, n - N + K), min(n, K) + 1):
                    interval = count_interval(N, n, k, 0, alpha=Fraction(1, 10))
                    probability = Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))
                    coverage += probability * (interval[0] <= K <= interval[1])
                    # Delete arbitrary positive and negative responses from the same sample.
                    for hidden_positive in range(k + 1):
                        for hidden_negative in range(n - k + 1):
                            partial = count_interval(N, n, k - hidden_positive,
                                                     hidden_positive + hidden_negative, alpha=Fraction(1, 10))
                            assert partial[0] <= interval[0] and partial[1] >= interval[1]
                assert coverage >= Fraction(9, 10)
                minimum = min(minimum, coverage)
                cases += 1
    checks['exact_small_population_coverage'] = True
    checks['arbitrary_missingness_never_narrows_complete_interval'] = True
    checks['census_is_exact'] = count_interval(12, 12, 7, 0) == [7, 7]
    checks['blank_census_unidentified'] = count_interval(12, 12, 0, 12) == [0, 12]
    checks['empty_sample_unidentified'] = count_interval(100, 0, 0, 0) == [0, 100]
    checks['exact_tail_matches_manual'] = hypergeom_tail(5, 2, 2, 1, upper=True) == Fraction(7, 10)
    # Unequal overlapping design: these four symbols are algebraic units only.
    strata = [{'index': 0, 'frame': set('abc'), 'sample': set('ab'), 'N': 3, 'n': 2},
              {'index': 1, 'frame': set('bcd'), 'sample': set('d'), 'N': 3, 'n': 1}]
    owners = dict(a=0, b=0, c=0, d=1)
    result = category_interval(set('abcd'), dict(a=1, b=0, d=1), strata, owners, alpha=Fraction(1, 20))
    checks['overlap_deduplicated_known_union_constraints'] = result['observed_units'] == 3 and result['count_interval'] == [2, 3]
    out = ROOT / 'phase6/results/human_inference_blank'
    out.mkdir(parents=True, exist_ok=True)
    packs = {'pair': ROOT / 'phase4/results/human_admission_engineering_pack',
             'context': ROOT / 'phase5/results/context_audit_engineering/blank_pack'}
    actual = {}
    for name, path in packs.items():
        manifest = json.loads((path / 'private_sampling_manifest.json').read_text())
        bundle = json.loads((path / 'responses.template.json').read_text())
        # Split 5% across both reports, including every reported endpoint.
        output = analyze_intervals(manifest, bundle, alpha=Fraction(1, 40))
        checks[name + '_actual_zero_human_responses'] = output['actual_human_response_count'] == 0
        checks[name + '_blank_nonempty_frames_have_full_intervals'] = all(r['prevalence_interval'] == [0.0, 1.0] for r in output['results'] if r['frame_units'])
        checks[name + '_empty_frames_remain_undefined'] = all(r['prevalence_interval'] is None for r in output['results'] if not r['frame_units'])
        checks[name + '_missing_corpora_not_reweighted'] = all(r['prevalence_interval'] is None for r in output['equal_corpus_balanced'])
        (out / (name + '_intervals.json')).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        actual[name] = {'manifest_sha256': manifest['manifest_sha256'], 'nonempty_endpoints': output['nonempty_endpoint_count'],
                        'actual_human_responses': 0, 'per_report_alpha': '1/40'}
    report = {'schema': 'ccu-human-inference-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'exact_coverage_cases': cases, 'minimum_coverage_at_nominal_90_percent': str(minimum),
              'actual_blank_packs': actual, 'software_fixtures_are_empirical_data': False,
              'human_collection_performed': False,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name('human_inference.py')]}}
    (ROOT / 'phase6/results/human_inference_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
