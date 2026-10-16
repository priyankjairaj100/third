#!/usr/bin/env python3
"""Aggregate the completed negative v4 selection; no inference or retuning.

Reads private, already exported artifacts. Outputs counts and exact fractions,
never record IDs, pair IDs, model responses or source text. This separate author
replay complements the terminal CPU audits and controller arithmetic review.
"""
import argparse
import bisect
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PINS = {
    'sample': {'bytes': 291785, 'sha256': '324a0bd35384f9dc4898cc1ba91c615746ce2fb2a7c74903563b976cd031840a'},
    'qwen': {'bytes': 118666, 'sha256': 'c94f64e2a04aadbadb231740da0ce5b2679d589cc35d96fb9c075d73ab578d01'},
    'mistral': {'bytes': 127301, 'sha256': '327df2e39f6d2fdca54b38a24676063aaea21f4c8698080ef34dcc9f6c8854b6'},
    'statistics': {'bytes': 22645, 'sha256': 'bffdc4d15aa54bc88d2ffb97c875631a14146a383bfdeb4ca836be3e3fb02f2f'},
}
POSITIVE = {'exact_copy', 'substantially_same_meaning'}
GRID = [Fraction(i, 100) for i in range(50, 100)] + [Fraction(199, 200)]
EDGES = [.5, .9, .91, .92, .93, .94, .95, .96, .97, .98, .99, .995]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return Fraction(value['numerator'], value['denominator'])


def fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def binding(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def positive(row):
    return row['status'] == 'valid' and row['category'] in POSITIVE


def possible(row):
    return row['status'] != 'valid' or row['category'] in POSITIVE | {'uncertain'}


def seal_check(obj):
    raw = json.dumps({k: v for k, v in obj.items() if k != 'sha256'},
                     sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode()
    need(hashlib.sha256(raw).hexdigest() == obj['sha256'], 'canonical seal')


def analyze(private):
    decoded = {}
    for name, pin in PINS.items():
        path = private / (name + ('.private.jsonl' if name == 'sample' else '.private.json'))
        need(path.is_file() and not path.is_symlink(), 'regular input required')
        raw = path.read_bytes()
        need(binding(raw) == pin, 'fixed terminal binding: ' + name)
        decoded[name] = ([json.loads(line) for line in raw.splitlines()]
                         if name == 'sample' else json.loads(raw))
    rows = decoded['sample']
    models = ('qwen', 'mistral')
    audited = {alias: decoded[alias] for alias in models}
    stats = decoded['statistics']
    seal_check(stats)
    need(stats['passed'] and not stats['independent_validation_performed'], 'terminal state')
    ids = {r['opaque_pair_id'] for r in rows}
    need(len(rows) == len(ids) == 360, 'canonical sample identity')
    labels, swaps, counts = {}, {}, {}
    for alias, audit in audited.items():
        seal_check(audit)
        need(audit['passed'] and audit['all_call_identities_denominators_and_tokens_verified'], 'full audit')
        need(audit['human_annotation_calls'] == audit['audit_repair']['model_calls_repeated'] == 0, 'no repeat or human')
        need(stats['audits'][alias] == PINS[alias], 'statistics audit binding')
        observations = audit['canonical_observations']
        labels[alias] = {r['opaque_pair_id']: r for r in observations}
        need(len(observations) == len(labels[alias]) == 360 and set(labels[alias]) == ids, 'complete model panel')
        diag = audit['diagnostic_observations']
        need(len(diag) == len({r['opaque_pair_id'] for r in diag}) == 36, 'swap panel')
        changed = binary = 0
        for row in diag:
            canonical = labels[alias][row['opaque_pair_id']]
            changed += (row['status'], row['category']) != (canonical['status'], canonical['category'])
            binary += positive(row) != positive(canonical)
        swaps[alias] = {'pairs': 36, 'status_or_category_changes': changed,
                        'binary_positive_changes': binary, 'diagnostic_only': True}
        need(swaps[alias] == stats['swap_diagnostics'][alias], 'swap arithmetic')
        counts[alias] = {'canonical_status_counts': dict(Counter(r['status'] for r in observations)),
                         'canonical_category_counts': dict(Counter(r['category'] for r in observations if r['status'] == 'valid')),
                         'canonical_positive_count': sum(map(positive, observations)),
                         'all396_status_counts': dict(Counter(r['status'] for r in observations + diag)),
                         'model_result': audit['execution_result'],
                         'audit': PINS[alias], 'audit_job_id': audit['audit_repair']['audit_job_id']}
        need(counts[alias]['all396_status_counts'] == audit['status_counts'], 'status totals')
    joint = {key: all(positive(labels[a][key]) for a in models) for key in ids}
    sensitivity = {key: all(possible(labels[a][key]) for a in models) for key in ids}
    confusion = Counter()
    for key in ids:
        outcomes = tuple(labels[a][key]['category'] if labels[a][key]['status'] == 'valid'
                         else labels[a][key]['status'] for a in models)
        confusion[outcomes] += 1
    bin_rows = []
    for index in range(13):
        group = [r for r in rows if r['bin'] == index]
        if not group:
            continue
        population = group[0]['bin_population']
        need(len(group) == min(30, population), 'frozen bin allocation')
        for r in group:
            need(r['bin_population'] == population and r['bin_sample_size'] == len(group), 'bin metadata')
            need(rational(r['inclusion_weight']) == Fraction(population, len(group)), 'design weight')
            need(rational(r['inclusion_probability']) == Fraction(len(group), population), 'design probability')
            need(bisect.bisect_left(EDGES, float.fromhex(r['score_hex'])) == index, 'score bin')
        bin_rows.append({'bin': index, 'population_pairs': population, 'sample_pairs': len(group),
                         'joint_positives': sum(joint[r['opaque_pair_id']] for r in group),
                         'sensitivity_possible_positives': sum(sensitivity[r['opaque_pair_id']] for r in group)})
    need(sum(x['population_pairs'] for x in bin_rows) == 220405510, 'full census sum')
    ledger = []
    original = stats['statistics']
    need(len(original['grid_ledger']) == len(GRID), '51 cuts')
    for tau, old in zip(GRID, original['grid_ledger']):
        above = [r for r in rows if float.fromhex(r['score_hex']) > float(tau)]
        den = sum((rational(r['inclusion_weight']) for r in above), Fraction())
        num = sum((rational(r['inclusion_weight']) * joint[r['opaque_pair_id']] for r in above), Fraction())
        upper = sum((rational(r['inclusion_weight']) * sensitivity[r['opaque_pair_id']] for r in above), Fraction())
        need(den > 0, 'support at every observed cut')
        replay = {'tau': fraction(tau), 'tau_float_hex': float(tau).hex(), 'sample_support': len(above),
                  'weighted_denominator': fraction(den), 'weighted_positive_numerator': fraction(num),
                  'conservative_precision': fraction(num / den),
                  'missing_uncertain_sensitivity_upper': fraction(upper / den),
                  'passes_selection_rule': num / den >= Fraction(19, 20)}
        need(replay == old, 'exact full cut replay')
        ledger.append(replay)
    need(not any(x['passes_selection_rule'] for x in ledger), 'expected terminal failure')
    need(original['selection_passed'] is False and original['selected_tau'] is None, 'negative decision')
    disagreements = sum(all(labels[a][key]['status'] == 'valid' for a in models) and
                        labels['qwen'][key]['category'] != labels['mistral'][key]['category'] for key in ids)
    diagnostics = original['diagnostics']
    need(disagreements == diagnostics['valid_six_way_disagreements'], 'disagreement count')
    need(sum(joint.values()) == diagnostics['conservative_positives'], 'joint count')
    peak = max(rational(x['conservative_precision']) for x in ledger)
    upper_peak = max(rational(x['missing_uncertain_sensitivity_upper']) for x in ledger)
    return {'schema': 'third-v4-negative-selection-aggregate-v1',
            'source_bindings': PINS, 'separate_author_arithmetic_replay_passed': True,
            'all51_full_cut_records_exactly_recomputed': True,
            'selection_passed': False, 'selected_tau': None, 'fresh_validation_run': False,
            'human_annotation_calls': 0, 'model_calls_repeated': 0, 'new_model_calls': 0,
            'semantic_validity_established': False, 'model_defined_target_only': True,
            'models': counts, 'canonical_pairs': len(rows), 'joint_positive_pairs': sum(joint.values()),
            'both_valid_pairs': sum(all(labels[a][key]['status'] == 'valid' for a in models) for key in ids),
            'valid_six_way_disagreements': disagreements,
            'maximum_weighted_joint_positive_proportion': fraction(peak),
            'maximum_missing_uncertain_sensitivity': fraction(upper_peak),
            'selection_requirement': fraction(Fraction(19, 20)),
            'grid_ledger': ledger, 'nonempty_bins': bin_rows,
            'canonical_category_cross_counts': [{'qwen': q, 'mistral': m, 'pairs': n}
                                                for (q, m), n in sorted(confusion.items())],
            'swap_diagnostics': swaps,
            'interpretation': [
                'Selection failed for this fixed representation, rater pair, rubric and protocol.',
                'The 27/360 raw joint count is descriptive of the stratified sample, not a population prevalence estimate.',
                'Six-way disagreements count only the 355 pairs with two valid responses.',
                'The sensitivity calculation is favorable missing/uncertain imputation, not a confidence bound.',
                'No new threshold, rater, prompt, representation, guard, development or held-out activation is authorized.',
                'Failures do not prove absence of semantic duplicates or invalidate the algebraic theory.'],
            'publication_policy': 'Aggregate counts and bindings only; no pair/record IDs, source text or per-pair labels.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.private)
    report['analysis_source'] = binding(Path(__file__).read_bytes())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'canonical_pairs': report['canonical_pairs'],
                      'selection_passed': False, 'exact_cuts_replayed': 51}))


if __name__ == '__main__':
    main()
