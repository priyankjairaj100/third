"""Charged full-state audits of separate durable method replays.

Every reconstructed replay snapshot must match its actual worker snapshot hash.
All coordinates are then compared with a fresh retained-data construction.
The original measured service supplies matching heads, inventories and final state.
These are FP64 diagnostics and exact structural checks, not bitwise-history claims.
"""
from pathlib import Path
import hashlib
import io
import json
import resource
import time
import numpy as np

from ccu.core import BlockerGraph
from phase3.panels import graph_normalize, reference_cosines
from phase6 import run_isolated
from phase6.methods import build_method, load_method, PRIMARY_METHODS
from phase6.workers import decode
from phase6.measurement import state_inventory
from phase4.requests import digest


class StateAuditBudgetError(RuntimeError):
    pass


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def direct_graph(features, ids, sources, threshold, seed, maximum_pair_coordinates):
    """Independent bounded pair tiles. The frozen scorer defines each predicate."""
    ids, sources = tuple(ids), tuple(sources)
    x = np.asarray(features)
    n = len(ids)
    if n * (n - 1) // 2 * x.shape[1] > maximum_pair_coordinates:
        raise StateAuditBudgetError('fresh pair-coordinate audit budget exceeded')
    if len(set(ids)) != n or len(sources) != n:
        raise ValueError('unique aligned record and source IDs required')
    order = np.asarray(sorted(range(n), key=lambda i: (hashlib.sha256(f'priority-v1|{seed}|{ids[i]}'.encode()).digest(), ids[i])), np.int64)
    normalized = graph_normalize(x)
    blockers = [[] for _ in ids]
    tile = 128
    for lower in range(0, n, tile):
        later = order[lower:lower + tile]
        for earlier_offset in range(0, lower + 1, tile):
            earlier = order[earlier_offset:earlier_offset + tile]
            scores = reference_cosines(normalized[later], normalized[earlier])
            hits = (scores > threshold) & (earlier_offset + np.arange(len(earlier))[None, :] < lower + np.arange(len(later))[:, None])
            for i, j in zip(*np.nonzero(hits)):
                blockers[int(later[i])].append(int(earlier[j]))
    for values in blockers:
        values.sort()
    ptr = np.asarray([0] + list(np.cumsum([len(v) for v in blockers])), np.int64)
    edges = np.asarray([i for values in blockers for i in values], np.int64)
    return BlockerGraph(ids, sources, order, ptr, edges, float(threshold))


def symbolic_state(graph, horizon, unit):
    """Independent exact integer coefficients of record-statistic symbols."""
    owners = graph.record_ids if unit == 'record' else graph.source_ids
    result = {}
    for row, blocker_indices in enumerate(graph.blockers):
        blocker_units = {owners[int(i)] for i in blocker_indices}
        if owners[row] in blocker_units or len(blocker_units) > horizon:
            continue
        tail = tuple(sorted(blocker_units))
        head = tuple(sorted(blocker_units | {owners[row]}))
        for key, sign in ((tail, 1), (head, -1)):
            if len(key) > horizon:
                continue
            record = graph.record_ids[row]
            values = result.setdefault(key, {})
            values[record] = values.get(record, 0) + sign
    return {key: {record: weight for record, weight in values.items() if weight}
            for key, values in result.items() if any(values.values())}


def substitute_symbols(state, deleted, remaining_horizon):
    result = {}
    for key, values in state.items():
        new = tuple(u for u in key if u not in deleted)
        if len(new) > remaining_horizon:
            continue
        target = result.setdefault(new, {})
        for record, weight in values.items():
            target[record] = target.get(record, 0) + weight
    return {key: {record: weight for record, weight in values.items() if weight}
            for key, values in result.items() if any(values.values())}


def stable_coefficients(state):
    return {tuple(sorted(state.unit_ids[i] for i in key)): np.asarray(value)
            for key, value in state.coefficients().items()}


def stable_groups(state):
    if getattr(state, 'method', None) != 'P-R':
        return None
    result = []
    for nodes, ground in state.groups:
        converted = [tuple(sorted(state.unit_ids[i] for i in key)) for key in nodes]
        result.append({'nodes': sorted(converted), 'grounded': bool(ground),
                       'omitted_node': None if ground else converted[0],
                       'stored_basis': sorted(converted if ground else converted[1:])})
    return sorted(result, key=lambda value: json.dumps(value, sort_keys=True))


def array_comparison(actual, expected, absolute, relative):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {'passed': False, 'shape_actual': list(a.shape), 'shape_expected': list(b.shape), 'coordinates_checked': 0}
    delta = np.abs(a.astype(np.longdouble) - b.astype(np.longdouble))
    allowed = absolute + relative * np.abs(b.astype(np.longdouble))
    bad = (~np.isfinite(delta)) | (delta > allowed)
    return {'passed': not bool(np.any(bad)), 'coordinates_checked': int(a.size),
            'failing_coordinates': int(np.count_nonzero(bad)),
            'maximum_absolute_difference': float(np.max(delta, initial=0)),
            'frobenius_difference': float(np.sqrt(np.sum(delta * delta))),
            'bitwise_equal': a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            'absolute_tolerance': absolute, 'relative_tolerance': relative,
            'rigorous_interval_claim': False}


def independent_coefficients(symbols, x, y, ids, maximum_coordinates):
    d, c = x.shape[1], y.shape[1]
    packed = d * (d + 1) // 2 + d * c + 1
    if len(symbols) * packed > maximum_coordinates:
        raise StateAuditBudgetError('full coefficient-coordinate audit budget exceeded')
    lookup = {rid: i for i, rid in enumerate(ids)}
    tri = np.triu_indices(d)
    result = {}
    for key, weights in symbols.items():
        value = np.zeros(packed, np.longdouble)
        for record, sign in sorted(weights.items()):
            index = lookup[record]
            a, b = x[index].astype(np.longdouble), y[index].astype(np.longdouble)
            value[:len(tri[0])] += sign * (a[:, None] * a[None, :])[tri]
            value[len(tri[0]):-1] += sign * (a[:, None] * b[None, :]).ravel()
            value[-1] += sign
        result[key] = value
    return result, packed


def compare_coefficients(actual, expected, packed, absolute, relative):
    rows = []
    for key in sorted(set(actual) | set(expected)):
        a = actual.get(key, np.zeros(packed))
        b = expected.get(key, np.zeros(packed, np.longdouble))
        check = array_comparison(a, b, absolute, relative)
        count_ok = bool(float(a[-1]).is_integer() and a[-1] == b[-1])
        rows.append({'key': list(key), 'actual_stored_key': key in actual,
                     'expected_designated_key': key in expected, 'count_coordinate_exact': count_ok,
                     'count_actual': int(a[-1]) if np.isfinite(a[-1]) and float(a[-1]).is_integer() else None,
                     'count_expected': int(b[-1]), **check})
    return {'passed': all(row['passed'] and row['count_coordinate_exact'] for row in rows),
            'complete_union_of_keys_compared': True,
            'coordinates_checked': sum(row['coordinates_checked'] for row in rows),
            'stored_key_sets_equal': set(actual) == set(expected),
            'zero_padding_rule': 'absent coefficients are zero; no near-zero pruning',
            'rows': rows}


def payload_comparison(state, graph, x, y, unit, horizon, initial_selected_ids, absolute, relative):
    owners = graph.record_ids if unit == 'record' else graph.source_ids
    selected = [i for i, values in enumerate(graph.blockers) if not len(values)]
    method = getattr(state, 'method', 'B-A' if getattr(state, 'retain_all', False) else 'B-E')
    if method == 'B-F':
        selected = [i for i, rid in enumerate(graph.record_ids) if rid in initial_selected_ids]
    z = x[selected].astype(np.longdouble)
    target = y[selected].astype(np.longdouble)
    moments = state.moments()
    checks = {'count_exact': moments.count == len(selected),
              'gram': array_comparison(moments.gram, z.T @ z, absolute, relative),
              'cross': array_comparison(moments.cross, z.T @ target, absolute, relative)}
    if method in ('B-E', 'B-A'):
        expected = {}
        for i, rid in enumerate(graph.record_ids):
            blockers = {owners[int(j)] for j in graph.blockers[i]}
            if method == 'B-A' or (len(blockers) <= horizon and owners[i] not in blockers):
                expected[rid] = {'owner': owners[i], 'blockers': tuple(sorted(blockers))}
        actual = {rid: {'owner': value['owner'], 'blockers': tuple(sorted(value['blockers']))}
                  for rid, value in state.logical_membership().items()}
        checks.update(eligible_membership_exact=actual == expected,
                      selected_membership_exact=set(state.selected_ids()) == {graph.record_ids[i] for i in selected},
                      expected_eligible_records=len(expected), actual_eligible_records=len(actual))
        lookup = {rid: i for i, rid in enumerate(graph.record_ids)}
        payload_equal = True
        blocker_counts_equal = True
        for row in np.flatnonzero(state.live):
            rid = state.record_ids[row]
            if rid not in lookup:
                payload_equal = False
                continue
            i = lookup[rid]
            payload_equal &= np.array_equal(state.x[row], x[i]) and np.array_equal(state.y[row], y[i])
            blocker_counts_equal &= int(state.counts[row]) == len(expected[rid]['blockers']) if rid in expected else False
        checks.update(payload_values_exact=bool(payload_equal), blocker_counts_exact=bool(blocker_counts_equal))
    elif method == 'B-F':
        checks.update(payload_values_exact=np.array_equal(state.x, x[selected]) and np.array_equal(state.y, y[selected]),
                      payload_owner_order_exact=tuple(state.owners) == tuple(owners[i] for i in selected),
                      target='surviving_original_selection; no fresh-curation substitution')
    elif method in ('O-G', 'O-T'):
        checks['selected_membership_exact'] = {state.ids[i] for i in state.selected_indices()} == {graph.record_ids[i] for i in selected}
    flags = [value for key, value in checks.items() if key.endswith('_exact')]
    checks['passed'] = all(flags) and checks['gram']['passed'] and checks['cross']['passed']
    return checks


def snapshot_parts(state):
    raw = state.snapshot_bytes()
    return {'state.npz': raw} if isinstance(raw, bytes) else raw


def audit_service_state(package, method, requests, reference_service_dir, audit_out, *, horizon,
                        unit, lambda_reg, policy, solver='cholesky', full_state=True):
    """Audit a separate actual service. Keep its timing outside measured repair."""
    started = time.perf_counter()
    package, reference, out = map(Path, (package, reference_service_dir, audit_out))
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'ccu-complete-state-audit-v1', 'method': method, 'full_state_requested': full_state,
              'passed': False, 'scope': 'separate durable audit service; original measured intermediate states remain private',
              'canonical_symbolic_equality_is_not_FP64_bitwise_history_independence': True,
              'FP64_moment_checks_are_diagnostics': True, 'checkpoints': []}
    try:
        if method not in PRIMARY_METHODS:
            raise ValueError('this audit requires one of the eight primary FP64 methods')
        absolute = float(policy.get('state_audit_absolute_tolerance', 1e-11))
        relative = float(policy.get('state_audit_relative_tolerance', 1e-10))
        if not np.isfinite(absolute) or not np.isfinite(relative) or min(absolute, relative) < 0:
            raise ValueError('finite nonnegative prospective state audit tolerances required')
        maximum_pairs = policy.get('state_audit_max_pair_coordinates', 100_000_000)
        maximum_coefficients = policy.get('state_audit_max_coefficient_coordinates', 4_000_000)
        if any(type(v) is not int or v < 0 for v in (maximum_pairs, maximum_coefficients)):
            raise ValueError('prospective audit work caps require nonnegative integers')
        result['prospective_audit_policy'] = {'absolute_tolerance': absolute, 'relative_tolerance': relative,
                                             'max_pair_coordinates': maximum_pairs,
                                             'max_coefficient_coordinates': maximum_coefficients}
        frozen = json.loads((reference / 'prospective_configuration.json').read_text())
        if any(frozen[key] != value for key, value in [('method', method), ('unit', unit), ('horizon', horizon),
                                                    ('lambda_reg', lambda_reg), ('solver', solver), ('requests', requests)]):
            raise ValueError('audit request differs from measured service configuration')
        if frozen['policy'] != policy:
            raise ValueError('audit resource policy differs from measured service')
        reference_report = json.loads((reference / 'service_report.json').read_text())
        if not reference_report.get('success'):
            raise ValueError('measured service did not complete successfully')
        for name, expected_hash in frozen['input_hashes'].items():
            if hash_file(package / name) != expected_hash:
                raise ValueError('audit input differs from measured service input')
        reference_repair = json.loads((reference / 'repair/repair_report.json').read_text())
        if len(reference_repair['releases']) != len(requests):
            raise ValueError('measured release count differs from requested checkpoints')
        for name, expected_hash in reference_repair['state_hashes'].items():
            if hash_file(reference / 'repair' / name) != expected_hash:
                raise ValueError('measured final snapshot differs from its actual worker hash')
        for index, release in enumerate(reference_repair['releases']):
            if hash_file(reference / f'repair/head_{index:04d}.npy') != release['head_sha256']:
                raise ValueError('measured head differs from its actual worker hash')
        manifest = json.loads((package / 'bundle.json').read_text())
        for name, entry in manifest['files'].items():
            if hash_file(package / name) != entry['sha256']:
                raise ValueError('audit input bundle changed')
        meta = json.loads((package / 'metadata.json').read_text())
        cx, x, y = [np.load(package / name, allow_pickle=False) for name in ('curator.npy', 'learner.npy', 'targets.npy')]
        ids, sources = meta['record_ids'], meta['source_ids']
        original_graph = direct_graph(cx, ids, sources, meta['threshold'], meta['priority_seed'], maximum_pairs)
        initial_selected = {ids[i] for i, blockers in enumerate(original_graph.blockers) if not len(blockers)}
        owners = ids if unit == 'record' else sources
        original_units = set(owners)
        symbols = symbolic_state(original_graph, horizon, unit)
        run = out / 'audit_service'
        report = run_isolated.run_service(package, method, requests, run, horizon=horizon, unit=unit,
                    lambda_reg=lambda_reg, policy=policy, solver=solver, persistence='every_release',
                    compaction_fraction=frozen['compaction_fraction'], batch_rows=frozen['batch_rows'], panel=frozen['panel'])
        result['audit_service_success'] = report['success']
        if not report['success']:
            raise RuntimeError('separate state audit service failed; no state gate waiver')
        actual_report = json.loads((run / 'repair/repair_report.json').read_text())
        measured_report = json.loads((reference / 'repair/repair_report.json').read_text())
        state = load_method(method, run / 'construction/state.npz')
        result['initial_snapshot_matches_measured'] = hash_file(run / 'construction/state.npz') == hash_file(reference / 'construction/state.npz')
        cumulative = set()
        for index, batch in enumerate(requests):
            mark = time.perf_counter()
            if any(identifier not in original_units for identifier in batch):
                raise ValueError('unknown requested unit')
            fresh = set(batch) - cumulative
            cumulative.update(fresh)
            remaining = horizon - len(cumulative)
            state.delete(batch)
            head, _ = decode(state, lambda_reg, solver)
            parts = snapshot_parts(state)
            expected_hashes = {f['file']: f['sha256'] for f in actual_report['releases'][index]['serialized_inventory']['files']}
            snapshot_bound = {name: hashlib.sha256(raw).hexdigest() for name, raw in parts.items()} == expected_hashes
            actual_head = np.load(run / f'repair/head_{index:04d}.npy', allow_pickle=False)
            measured_head = np.load(reference / f'repair/head_{index:04d}.npy', allow_pickle=False)
            head_bound = (head.weights.dtype == actual_head.dtype == measured_head.dtype and
                          head.weights.shape == actual_head.shape == measured_head.shape and
                          head.weights.tobytes() == actual_head.tobytes() == measured_head.tobytes())
            keep = np.asarray([i for i, owner in enumerate(owners) if owner not in cumulative], dtype=int)
            retained_ids, retained_sources = [ids[i] for i in keep], [sources[i] for i in keep]
            graph = direct_graph(cx[keep], retained_ids, retained_sources, meta['threshold'], meta['priority_seed'], maximum_pairs)
            fresh_symbols = symbolic_state(graph, remaining, unit)
            symbols = substitute_symbols(symbols, fresh, remaining)
            symbolic_equal = symbols == fresh_symbols
            if method in ('P-I', 'P-S', 'P-R'):
                alive = state.state.alive if method == 'P-I' else state.alive
                actual_alive = {state.unit_ids[i] for i in alive}
                actual_horizon = state.remaining_horizon
            elif method in ('B-E', 'B-A'):
                actual_alive = {state.unit_ids[i] for i in np.flatnonzero(state.unit_alive)}
                actual_horizon = state.horizon
            else:
                actual_alive, actual_horizon = set(state.alive), state.horizon
            row = {'index': index, 'deleted_unique_units': len(cumulative), 'remaining_horizon': remaining,
                   'actual_worker_snapshot_hashes': expected_hashes, 'reconstructed_actual_snapshot_bound': snapshot_bound,
                   'replay_and_measured_heads_bitwise_equal': bool(head_bound),
                   'observed_inventory_matches_measured': actual_report['releases'][index]['logical_state_inventory'] == measured_report['releases'][index]['logical_state_inventory'],
                   'inventory_scope': 'descriptive wire inventory can include process-specific alias/set order; its hash is not canonical abstract state',
                   'remaining_horizon_exact': actual_horizon == remaining,
                   'alive_units_exact': actual_alive == original_units - cumulative,
                   'canonical_record_symbol_state_exact': symbolic_equal,
                   'canonical_symbol_scope': 'exact integer linear forms in original record statistics; independent substitution and fresh incidence construction',
                   'fresh_retained_pair_scores_recomputed': True, 'retained_records': len(keep)}
            if method in ('P-I', 'P-S', 'P-R') and full_state:
                extended, packed = independent_coefficients(fresh_symbols, x[keep], y[keep], retained_ids, maximum_coefficients)
                coefficients = stable_coefficients(state)
                row['coefficients'] = compare_coefficients(coefficients, extended, packed, absolute, relative)
                built = build_method(method, graph, x[keep], y[keep], remaining, unit=unit)
                row['fresh_constructor_coefficients'] = compare_coefficients(coefficients, stable_coefficients(built), packed, absolute, relative)
                row['structural_rank_groups_exact'] = stable_groups(state) == stable_groups(built)
                row['rank_groups_actual'] = stable_groups(state)
                row['rank_groups_fresh'] = stable_groups(built)
            row['moments_and_membership'] = payload_comparison(state, graph, x[keep], y[keep], unit, remaining, initial_selected, absolute, relative)
            required = [row[key] for key in ('reconstructed_actual_snapshot_bound', 'replay_and_measured_heads_bitwise_equal',
                        'remaining_horizon_exact', 'alive_units_exact', 'canonical_record_symbol_state_exact')]
            required.append(row['moments_and_membership']['passed'])
            if 'coefficients' in row:
                required.extend([row['coefficients']['passed'], row['fresh_constructor_coefficients']['passed'], row['structural_rank_groups_exact']])
            row['passed'] = all(required)
            row['audit_seconds'] = time.perf_counter() - mark
            result['checkpoints'].append(row)
            write(out / 'audit_progress.json', result)
        result['final_snapshot_matches_measured'] = all(hash_file(run / 'repair' / name) == hash_file(reference / 'repair' / name) for name in actual_report['state_hashes'])
        result['audit_service_report_sha256'] = hash_file(run / 'service_report.json')
        result['measured_service_report_sha256'] = hash_file(reference / 'service_report.json')
        result['charged_worker_lifecycle_seconds'] = json.loads((run / 'persistence_receipt.json').read_text())['charged_total_seconds']
        result['audit_service_snapshot_bytes'] = sum(Path(run / 'repair' / name).stat().st_size for name in actual_report['state_hashes'])
        result['passed'] = result['initial_snapshot_matches_measured'] and result['final_snapshot_matches_measured'] and all(row['passed'] for row in result['checkpoints'])
        result['status'] = 'passed' if result['passed'] else 'state_mismatch'
    except Exception as error:
        result.update(status='audit_failed', exception_type=type(error).__name__, message=str(error), passed=False)
    result['charged_total_audit_seconds'] = time.perf_counter() - started
    result['auditor_process_lifetime_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result['auditor_memory_scope'] = 'whole parent-process lifetime peak; not an isolated stage peak; worker peaks remain in the separate service report'
    result['audit_cost_included_in_measured_method_time'] = False
    result['code_sha256'] = hash_file(Path(__file__))
    result['sha256'] = digest(result)
    write(out / 'state_audit.json', result)
    return result
