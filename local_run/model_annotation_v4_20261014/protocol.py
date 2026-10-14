"""Planning-only checks/reference helpers. No sampling, entropy draws or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

PREFIX = 'third-llm-only-v4/'
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
            'orientation', 'model-seed', 'call-id')}, 'Unknown v4 domain')
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


def validate(plan, folder):
    folder = Path(folder)
    parent = json.loads((folder/'parent_PLAN.json').read_bytes())
    require(binding(folder/'parent_PLAN.json') == plan['parent_plan'], 'parent_plan_binding')
    require(plan['parent_plan']['sha256'] == '7910272ad6d05e012b101c782ea2268b5585eb6ab0f9b49f38c1aac9b5e13419', 'parent_pin')
    require(plan['schema'] == 'third-llm-only-tail-followup-plan-v4-20261014' and
            plan['status'] == 'prospective_not_activated', 'prospective_v4')
    require(plan['inherits_unchanged'] == {k:parent[k] for k in
        ('first_cell','models','judge_runtime','prompt','model_labels','validation_stability','swapped_diagnostics')}, 'inherited_settings_changed')
    selection = plan['selection']
    require(selection['stratum_upper_cuts'] == [.5,.9,.91,.92,.93,.94,.95,.96,.97,.98,.99,.995] and
            selection['strata'] == 13 and selection['capacity_per_stratum'] == 30 and
            selection['canonical_pair_cap'] == 390, 'sampling_design')
    require(selection['threshold_grid'] == parent['calibration']['threshold_grid'] and
            selection['precision_min'] == '19/20' and not selection['empty_support_passes'] and
            not selection['reuse_prior_labels'] and not selection['availability_filter'] and
            selection['new_model_labels_required'] is True, 'selection_quality_policy')
    require(plan['validation'] == {k:v for k,v in parent['calibration'].items() if
        k.startswith('validation_') or k in ('bound_target','post_validation_retuning_allowed')}, 'validation_changed')
    require(plan['resources']['human_annotation_calls'] == 0 and plan['resources']['retries'] == 0 and
            plan['resources']['calls_max'] == 2*(390+200+39+20), 'resource_denominator')
    require(plan['randomness']['prefix'] == PREFIX and plan['randomness']['seeds'] ==
            ['selection','validation','selection_swap','validation_swap'], 'new_seed_domains')
    require(plan['prior_exposure']['failure_preserved'] is True and
            plan['prior_exposure']['followup_is_post_v3_development'] is True and
            plan['advance_rules']['semantic_truth_claim'] is False and
            plan['advance_rules']['human_agreement_claim'] is False and
            plan['advance_rules']['source_acceptance_inherited'] is False, 'scope')
    prior = json.loads((folder/'prior_v3_terminal_result.json').read_bytes())
    require(prior['selection_passed'] is False and prior['selected_tau'] is None and
            prior['conservative_positives_of90'] == 0, 'negative_v3_history')
    return {'passed':True,'inference_activated':False,'scientific_gates_completed':False}
