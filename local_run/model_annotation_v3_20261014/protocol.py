"""Planning-only checks/reference helpers. No sampling, entropy draws or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

PREFIX = 'third-llm-only-v3/'
CATEGORIES = ('exact_copy', 'substantially_same_meaning', 'overlapping_information',
              'merely_related', 'unrelated', 'uncertain')
POSITIVE = frozenset(CATEGORIES[:2])
STATUSES = frozenset(('valid', 'invalid_json', 'refusal', 'context_overflow',
                     'runtime_error', 'missing_output', 'not_attempted'))


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def domain_digest(domain, values):
    require(domain in {PREFIX + x for x in ('sampling', 'natural-pair', 'opaque-pair',
            'orientation', 'model-seed', 'call-id')}, 'Unknown v3 domain')
    require(isinstance(values, list), 'Structured list values required')
    return hashlib.sha256(canonical([domain, *values])).hexdigest()


def identities(plan_sha256, frame_sha256, record_ids):
    for value in (plan_sha256, frame_sha256):
        require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 required')
    require(isinstance(record_ids, (list, tuple)) and len(record_ids) == 2 and
            all(isinstance(x, str) and x for x in record_ids) and record_ids[0] != record_ids[1],
            'Two distinct exact record IDs required')
    ids = sorted(record_ids, key=lambda x: x.encode('utf-8'))
    natural = domain_digest(PREFIX + 'natural-pair', [frame_sha256, ids])
    opaque = domain_digest(PREFIX + 'opaque-pair', [plan_sha256, frame_sha256, natural])
    reverse = int(domain_digest(PREFIX + 'orientation', [plan_sha256, frame_sha256, natural]), 16) % 2
    return {'natural_pair_id': natural, 'opaque_pair_id': opaque,
            'canonical_record_ids': ids[::-1] if reverse else ids}


def model_seed(plan_sha256, frame_sha256, opaque_pair_id, model_id, revision):
    # No stage or orientation argument: exact replay and diagnostic swaps share seed.
    return int(domain_digest(PREFIX + 'model-seed',
        [plan_sha256, frame_sha256, opaque_pair_id, model_id, revision]), 16) % (2**63)


def checked_responses(responses):
    require(isinstance(responses, list) and len(responses) == 2, 'Exactly two model records required')
    by_model = {}
    for record in responses:
        require(isinstance(record, dict) and set(record) == {'model_id', 'status', 'category'},
                'Reference helper expects model_id/status/category only')
        model = record['model_id']
        require(isinstance(model, str) and model and model not in by_model, 'Two distinct model IDs required')
        require(record['status'] in STATUSES, 'Explicit valid or unavailable status required')
        require(record['category'] in CATEGORIES if record['status'] == 'valid'
                else record['category'] is None, 'Unavailable is not a substantive judgment')
        by_model[model] = record
    return by_model


def panel_label(responses):
    records = checked_responses(responses)
    return int(all(r['status'] == 'valid' and r['category'] in POSITIVE for r in records.values()))


def overlap_instability(selection, validation):
    a, b = checked_responses(selection), checked_responses(validation)
    require(set(a) == set(b), 'Overlap must contain the same two models')
    return any((a[m]['status'], a[m]['category']) != (b[m]['status'], b[m]['category']) for m in a)


def weighted_precision(sample):
    """Reference full-denominator ratio; inputs are exact inclusion weights + responses."""
    numerator, denominator = Fraction(0), Fraction(0)
    for weight, responses in sample:
        require(type(weight) is Fraction and weight > 0, 'Positive exact Fraction weight required')
        denominator += weight
        numerator += weight * panel_label(responses)
    return None if not denominator else numerator / denominator


def binding(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def validate(plan, repo):
    repo = Path(repo)
    require(plan['schema'] == 'third-llm-only-annotation-plan-v3-20261014' and
            plan['status'] == 'prospective_not_activated', 'Prospective v3 role required')
    human = plan['supersession']
    require(human['human_annotation_required'] is False and
            type(human['human_reviews_planned']) is int and human['human_reviews_planned'] == 0 and
            type(human['human_reviews_completed']) is int and human['human_reviews_completed'] == 0 and
            human['original_v1_three_human_gate_satisfied'] is False and
            human['v2_single_author_audit_completed'] is False, 'No active human annotation or completed historical gate')
    activation = plan['activation']
    require(activation['required'] is True and activation['completed'] is False and
            activation['human_annotation_dependency'] is False, 'Prospective model-only activation required')
    require(all(v is False or (type(v) is int and v == 0) for v in plan['not_completed'].values()),
            'Planning cannot complete evidence gates')
    labels = plan['model_labels']
    require(tuple(labels['categories']) == CATEGORIES and set(labels['positive_categories']) == POSITIVE and
            labels['retain_all_sampled_units'] is True and labels['drop_unavailable_from_denominator'] is False and
            labels['models_assumed_independent'] is False and labels['confidence_score_used'] is False,
            'Conservative full-denominator panel required')
    v2 = json.loads((repo / 'local_run/model_annotation_v2_20261011/PLAN.json').read_bytes())
    cal = plan['calibration']
    for key in ('selection_bins', 'selection_pairs_per_bin_max', 'selection_pair_max', 'threshold_grid',
                'selection_weighted_precision_min', 'validation_pair_max', 'validation_lower_bound_min',
                'validation_alpha', 'calls_max_one_cell_base', 'swap_calls_max_one_cell',
                'failed_tau_0995_is_only_diagnostic'):
        require(type(cal[key]) is type(v2['calibration'][key]) and cal[key] == v2['calibration'][key],
                'Frozen v2 numerical policy changed: ' + key)
    require(cal['post_validation_retuning_allowed'] is False and
            cal['selection_calibration_requires_context_available_filter'] is False,
            'No outcome tuning or context-based frame filtering')
    rt = plan['judge_runtime']
    controls = {'max_new_tokens': 512, 'total_token_budget': 32768, 'batch_size': 1,
                'dtype': 'bfloat16', 'do_sample': False, 'attention_implementation': 'sdpa',
                'sdpa_backend': 'FLASH_ATTENTION', 'attention_fallback_allowed': False,
                'lm_format_enforcer_version': '0.11.3'}
    require(rt['format_only_retry_limit'] == 0 and rt['semantic_retry_limit'] == 0 and
            rt['truncation_allowed'] is False and rt['summarization_allowed'] is False and
            rt['max_input_tokens_with_output_reserve'] == 32768 - 512, 'Runtime coverage/retry contract changed')
    require(len(plan['models']) == 2 and len({m['family'] for m in plan['models']}) == 2,
            'Two distinct fixed model families required')
    for model in plan['models']:
        runtime = json.loads((repo / model['runtime_plan_path']).read_bytes())
        asset = json.loads((repo / model['asset_expectations_path']).read_bytes())
        require(model['model_id'] == asset['model_id'] and model['revision'] == asset['revision'], 'Model identity mismatch')
        require(model['substantive_calibration_qualified'] is False, 'Pilot is not calibration')
        report = json.loads((repo / model['technical_result_path']).read_bytes())
        if model['family'] == 'Qwen':
            pilot_pass = report['technical_pilot_passed'] is True and report['summary']['status_counts_all_calls'] == {'valid': 44}
            runtime_sha = report['summary']['runtime_plan_sha256']
        else:
            pilot_pass = report['passed'] is True and report['outcomes']['valid_outputs'] == 44
            runtime_sha = report['bindings']['runtime_plan_sha256']
        require(model['technical_pilot_qualified'] is True and pilot_pass and
                runtime_sha == binding(repo / model['runtime_plan_path'])['sha256'], 'Executed pilot/runtime binding mismatch')
        for key, value in controls.items():
            require(type(rt[key]) is type(value) and rt[key] == value == runtime[key], 'Runtime pin mismatch: ' + key)
        require(runtime['format_retry_limit'] == 0 and model['runtime_package_versions'] == runtime['required_packages'] and
                model['effective_constraint_policy'] == runtime['effective_constraint_policy'], 'Effective runtime differs from pilot')
    stability = plan['validation_stability']
    require(stability['fresh_canonical_calls_for_all_validation_pairs'] is True and
            stability['reuse_selection_or_pilot_output_as_validation'] is False and
            stability['preserve_original_labels'] is True, 'Fresh validation and original labels required')
    budget = plan['cost_budget']
    require(budget['all_calls_max'] == budget['base_calls_max'] + budget['swapped_calls_max'] == 1760 and
            budget['generation_retries'] == 0 and budget['human_annotation_calls'] == 0, 'Complete call budget required')
    domains = plan['randomness']['domains']
    require(set(domains.values()) == {PREFIX + x for x in ('sampling', 'natural-pair', 'opaque-pair',
            'orientation', 'model-seed', 'call-id')}, 'Distinct explicit v3 domains required')
    for name, expected in plan['bound_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Repository-relative bindings required')
        require(binding(repo / name) == expected, 'Changed bound file: ' + name)
    return {'passed': True, 'scope': 'Prospective protocol and local file bindings only; no activation, sampling, inference or acceptance',
            'bound_files_verified': len(plan['bound_files']), 'human_annotation_calls': 0, 'model_inference_calls': 0,
            'scientific_gates_completed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(json.loads((Path(__file__).parent / 'PLAN.json').read_bytes()), a.repo), sort_keys=True))
