"""Pure, model-only estimators. No ingestion, entropy, inference or acceptance API."""
from collections import Counter
import bisect
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

MODELS = ('Qwen/Qwen2.5-7B-Instruct', 'mistralai/Mistral-Nemo-Instruct-2407')
CATEGORIES = ('exact_copy', 'substantially_same_meaning', 'overlapping_information',
              'merely_related', 'unrelated', 'uncertain')
POSITIVE = frozenset(CATEGORIES[:2])
STATUSES = frozenset(('valid', 'invalid_json', 'refusal', 'context_overflow',
                     'runtime_error', 'missing_output', 'not_attempted'))
GRID = tuple(Fraction(i, 100) for i in range(50, 100)) + (Fraction(199, 200),)
CALIBRATION_SHA = '2a6ce3b19de3063f32b010ebfa19ffc4003ffc09acfc41a1ead4f74bc6c81b13'
PANELS_SHA = 'bd2409958ac9a9e27faf463b44b2f839260a3195c7cde32fb3bfb76aeb14a04b'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def rational(value):
    require(isinstance(value, dict) and set(value) == {'numerator', 'denominator'}, 'exact_fraction_required')
    a, b = value['numerator'], value['denominator']
    require(type(a) is int and type(b) is int and b > 0, 'invalid_fraction')
    result = Fraction(a, b)
    require((result.numerator, result.denominator) == (a, b), 'reduced_fraction_required')
    return result


def fjson(value):
    return None if value is None else {'numerator': value.numerator, 'denominator': value.denominator}


def score(row):
    require(isinstance(row.get('score_hex'), str), 'raw_score_hex_required')
    value = float.fromhex(row['score_hex'])
    require(math.isfinite(value) and value.hex() == row['score_hex'], 'canonical_finite_score_required')
    return value


def complete_panels(rows, observed):
    """Explicit absent pair/model records become missing_output, never disappear.

    `observed` is a list of already ingested canonical observations containing
    only opaque_pair_id/model_id/status/category. Transport/schema validation,
    stage/orientation/attempt bindings and raw-output provenance are prerequisites
    of the separate future ingestion layer; this core cannot establish them.
    """
    ids = [r['opaque_pair_id'] for r in rows]
    require(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids), 'duplicate_or_invalid_pair')
    panels = {x: {m: {'status': 'missing_output', 'category': None} for m in MODELS} for x in ids}
    seen = set()
    for item in observed:
        require(isinstance(item, dict) and set(item) == {'opaque_pair_id', 'model_id', 'status', 'category'}, 'canonical_observation_fields')
        pair, model = item['opaque_pair_id'], item['model_id']
        require(pair in panels and model in MODELS and (pair, model) not in seen, 'unknown_or_duplicate_observation')
        status, category = item['status'], item['category']
        require(status in STATUSES, 'unknown_status')
        require(category in CATEGORIES if status == 'valid' else category is None, 'status_category_mismatch')
        seen.add((pair, model))
        panels[pair][model] = {'status': status, 'category': category}
    return panels


def panel_summary(panel):
    label = int(all(x['status'] == 'valid' and x['category'] in POSITIVE for x in panel.values()))
    # Sensitivity only: a missing/uncertain model could be positive, while an
    # observed substantive negative keeps the two-model intersection negative.
    possible = int(all(x['status'] != 'valid' or x['category'] in POSITIVE or x['category'] == 'uncertain'
                       for x in panel.values()))
    return label, possible


def diagnostic_counts(panels):
    statuses = {m: Counter(x[m]['status'] for x in panels.values()) for m in MODELS}
    categories = {m: Counter(x[m]['category'] for x in panels.values() if x[m]['status'] == 'valid') for m in MODELS}
    disagreement = sum(all(x[m]['status'] == 'valid' for m in MODELS) and
                       x[MODELS[0]]['category'] != x[MODELS[1]]['category'] for x in panels.values())
    return {'pairs': len(panels), 'status_counts': {m: dict(v) for m, v in statuses.items()},
            'category_counts': {m: dict(v) for m, v in categories.items()},
            'valid_six_way_disagreements': disagreement,
            'conservative_positives': sum(panel_summary(x)[0] for x in panels.values())}


def validate_selection_design(rows, populations):
    require(isinstance(populations, list) and len(populations) == 13 and
            all(type(n) is int and n >= 0 for n in populations), 'thirteen_exact_tail_populations')
    counts = Counter()
    for row in rows:
        b = row['bin']
        require(type(b) is int and 0 <= b < 13, 'bin_index')
        N, n = populations[b], min(30, populations[b])
        require(n > 0 and type(row['bin_population']) is int and row['bin_population'] == N
                and type(row['bin_sample_size']) is int and row['bin_sample_size'] == n, 'bin_design_changed')
        require(rational(row['inclusion_weight']) == Fraction(N, n) and
                rational(row['inclusion_probability']) == Fraction(n, N), 'inclusion_weight_changed')
        expected_bin = bisect.bisect_left([.5,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99,.995], score(row))
        require(b == expected_bin, 'score_bin_changed')
        counts[b] += 1
    require([counts[i] for i in range(13)] == [min(30, n) for n in populations], 'selection_sample_incomplete')
    require(sum((rational(x['inclusion_weight']) for x in rows), Fraction(0)) == sum(populations), 'weighted_population_mismatch')


def selection(rows, populations, observed):
    validate_selection_design(rows, populations)
    panels = complete_panels(rows, observed)
    ledger = []
    chosen = None
    for tau in GRID:
        above = [x for x in rows if score(x) > float(tau)]
        den = sum((rational(x['inclusion_weight']) for x in above), Fraction(0))
        num = sum((rational(x['inclusion_weight']) * panel_summary(panels[x['opaque_pair_id']])[0]
                   for x in above), Fraction(0))
        possible = sum((rational(x['inclusion_weight']) * panel_summary(panels[x['opaque_pair_id']])[1]
                        for x in above), Fraction(0))
        passed = den > 0 and num / den >= Fraction(19, 20)
        ledger.append({'tau': fjson(tau), 'tau_float_hex': float(tau).hex(), 'sample_support': len(above),
                       'weighted_denominator': fjson(den), 'weighted_positive_numerator': fjson(num),
                       'conservative_precision': fjson(num / den if den else None),
                       'missing_uncertain_sensitivity_upper': fjson(possible / den if den else None),
                       'passes_selection_rule': passed})
        if passed and chosen is None:
            chosen = tau
    return {'schema': 'third-llm-only-selection-statistics-v4', 'selection_passed': chosen is not None,
            'selected_tau': fjson(chosen), 'failed_diagnostic_tau': fjson(GRID[-1]) if chosen is None else None,
            'grid_ledger': ledger, 'diagnostics': diagnostic_counts(panels),
            'model_defined_target_only': True, 'human_annotation_calls': 0,
            'semantic_validity_established': False, 'inference_activation': False}


def frozen_bound(repo):
    """Load only the pinned numerical helper, never human collector/evaluator APIs."""
    repo = Path(repo)
    for name, pin in [('calibration.py', CALIBRATION_SHA), ('panels.py', PANELS_SHA)]:
        path = repo / 'empirical_execution/phase3' / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'frozen_numeric_source_changed')
    package_name = '_third_llm_statistics_frozen_phase3'
    folder = repo / 'empirical_execution/phase3'
    module = ModuleType(package_name)
    module.__path__ = [str(folder)]
    sys.modules[package_name] = module
    cspec = importlib.util.spec_from_file_location(package_name + '.calibration', folder / 'calibration.py')
    calibration = importlib.util.module_from_spec(cspec)
    cspec.loader.exec_module(calibration)
    return calibration.finite_population_lower_bound


def validation(rows, population, observed, selection_rows, selection_populations, selection_observed, selection_result, bound):
    require(type(population) is int and population >= 0 and len(rows) == min(200, population), 'validation_sample_incomplete')
    require(selection(selection_rows, selection_populations, selection_observed) == selection_result,
            'selection_result_not_reproduced')
    require(selection_result.get('selection_passed') is True, 'failed_selection_cannot_validate')
    tau = rational(selection_result['selected_tau'])
    require(tau in GRID, 'threshold_off_grid')
    require(all(score(x) > float(tau) for x in rows), 'validation_must_be_strictly_above_locked_tau')
    panels = complete_panels(rows, observed)
    earlier = complete_panels(selection_rows, selection_observed)
    overlap = sorted(set(panels) & set(earlier))
    changed = []
    for pair in overlap:
        for model in MODELS:
            if panels[pair][model] != earlier[pair][model]:
                changed.append({'opaque_pair_id': pair, 'model_id': model,
                                'selection': earlier[pair][model], 'validation': panels[pair][model]})
    successes = sum(panel_summary(x)[0] for x in panels.values())
    lower = bound(population, len(rows), successes, alpha=Fraction(1, 20))
    numeric_pass = bool(lower['defined'] and 10 * lower['lower_success_count'] >= 9 * population)
    available = bool(lower['defined'] and not changed)
    return {'schema': 'third-llm-only-validation-statistics-v1', 'locked_tau': fjson(tau),
            'finite_population_bound': lower, 'numeric_lower_bound_passed': numeric_pass,
            'overlap_pairs': len(overlap), 'overlap_model_comparisons': 2 * len(overlap),
            'changed_overlap_observations': changed, 'observed_overlap_stable': not changed,
            'zero_overlap_is_zero_replay_evidence': not overlap,
            'model_validation_available': available, 'model_validation_passed': numeric_pass and available,
            'diagnostics': diagnostic_counts(panels), 'model_defined_target_only': True,
            'fixed_potential_labels_assumed_not_proven': True, 'human_annotation_calls': 0,
            'semantic_validity_established': False, 'confirmatory_study_ready': False}
