"""Post-guard whole-source development prefixes; no task labels accepted."""
from collections import Counter, defaultdict
import hashlib
import common as c


def digest(salt, unit):
    return int.from_bytes(hashlib.sha256((salt+'\0'+unit).encode()).digest(), 'big')


def inventory(rows):
    groups = {}; seen = set(); counts = Counter()
    for i, row in enumerate(rows):
        c.require(set(row) == {'original_full_row_index', 'record_id', 'source_unit_id', 'source_kind',
                              'partition', 'text_sha256', 'role', 'genuine_source_request_eligible'}, 'metadata_only_rows')
        c.require(row['original_full_row_index'] == i and row['record_id'] not in seen, 'aligned_unique_rows')
        seen.add(row['record_id']); unit = row['source_unit_id']; role = row['role']
        c.require(unit not in groups or groups[unit] == role, 'source_role_disjoint')
        groups[unit] = role
        original = digest('ccu-v1-source-split', unit)
        part = 'train' if original*10 < 7*2**256 else ('calibration' if original*20 < 17*2**256 else 'test')
        reserved = digest('third-llm-development-reserve-v1-20261016', unit)*10 < 2**256
        expected = 'calibration' if part == 'calibration' else ('development_' if reserved else 'evaluation_')+part
        c.require(row['partition'] == part and role == expected, 'frozen_partition_and_reserve')
        c.require(row['genuine_source_request_eligible'] is row['source_kind'].startswith('native_'), 'source_semantics')
        counts[role] += 1
    return dict(counts)


def whole_prefix(rows, target, salt):
    c.require(type(target) is int and target > 0, 'positive_target')
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_unit_id']].append(row['record_id'])
    chosen = []; units = []
    for unit in sorted(groups, key=lambda s: (digest(salt, s), s)):
        if len(chosen) >= target:
            break
        chosen.extend(sorted(groups[unit])); units.append(unit)
    return {'target': target, 'actual': len(chosen), 'shortfall': max(0, target-len(chosen)),
            'overshoot': max(0, len(chosen)-target), 'record_ids': chosen, 'source_ids': units}


def build_panels(rows, removed, plan):
    ids = [r['record_id'] for r in removed]
    c.require(len(ids) == len(set(ids)), 'duplicate_removed')
    train = {r['record_id'] for r in rows if r['role'] == 'development_train'}
    c.require(set(ids) <= train, 'only_development_train_may_be_removed')
    dead = set(ids); kept = [r for r in rows if r['role'] == 'development_train' and r['record_id'] not in dead]
    return {'retained_development_train_ids': [r['record_id'] for r in kept],
            'development_test_ids': [r['record_id'] for r in rows if r['role'] == 'development_test'],
            'prefixes': {str(n): whole_prefix(kept, n, plan['development_order_salt'])
                         for n in plan['development_targets']},
            'all_reserved_sources_still_excluded_from_evaluation': True,
            'full_calibration_test_membership_unchanged': True,
            'effects_launched': False, 'original_primary_acceptance': False}
