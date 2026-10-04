"""State-transition contracts, separate from empirical benchmarks.

Record cases use natural Civil comments and their original labels.
Source cases use a tiny, explicitly algebraic fixture. They are software tests.
FP64 history comparisons are numerical diagnostics, never bitwise canonical claims.
"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from ccu.data import read_natural_jsonl, lexical_engineering_features
from phase6.methods import PRIMARY_METHODS, build_method, load_method
from phase6.state_audit import (array_comparison, compare_coefficients, direct_graph,
                               independent_coefficients, payload_comparison,
                               stable_coefficients, stable_groups, symbolic_state,
                               substitute_symbols)

ATOL, RTOL = 1e-11, 1e-10


def snapshot(state):
    raw = state.snapshot_bytes()
    return {'state.npz': raw} if isinstance(raw, bytes) else raw


def alive_and_horizon(state, method):
    if method in ('P-I', 'P-S', 'P-R'):
        alive = state.state.alive if method == 'P-I' else state.alive
        return {state.unit_ids[i] for i in alive}, state.remaining_horizon
    if method in ('B-E', 'B-A'):
        return {state.unit_ids[i] for i in np.flatnonzero(state.unit_alive)}, state.horizon
    return set(state.alive), state.horizon


def fixtures():
    rows = read_natural_jsonl(ROOT / 'data/civil_comments_engineering_preview.jsonl', require_labels=True)[:20]
    ids = [row['record_id'] for row in rows]
    cx, _ = lexical_engineering_features(rows, 32)
    x, _ = lexical_engineering_features(rows, 16)
    y = np.asarray([row['label'] for row in rows], np.float64)[:, None]
    natural = {'name': 'Civil20_record_lexical_contract', 'unit': 'record', 'ids': ids,
               'sources': ids, 'cx': cx, 'x': x, 'y': y, 'threshold': .6,
               'role': 'natural_preview_software_contract_not_semantic_evidence'}
    # Group sizes are 2, 2, 1, 1. Identical vectors create shared-source blockers.
    tiny = {'name': 'tiny_source_algebra_software_fixture', 'unit': 'source',
            'ids': [f'algebra-record-{i}' for i in range(6)],
            'sources': ['algebra-source-a', 'algebra-source-a', 'algebra-source-b',
                        'algebra-source-b', 'algebra-source-c', 'algebra-source-d'],
            'cx': np.asarray([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [-1, 0]], np.float32),
            'x': np.asarray([[1, 0], [1, 1], [2, 1], [0, 1], [1, 2], [-1, 1]], np.float32),
            'y': np.asarray([[0], [1], [1], [0], [1], [0]], np.float64), 'threshold': .9,
            'role': 'algebraic_source_contract_fixture_no_corpus_or_empirical_provenance_claim'}
    return natural, tiny


def run(destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    checks, cases = [], []

    def check(case, name, passed, **details):
        checks.append({'case': case, 'check': name, 'passed': bool(passed), **details})

    with tempfile.TemporaryDirectory(prefix='phase6-state-contract-') as temporary:
        for fixture in fixtures():
            ids, sources, cx, x, y = [fixture[k] for k in ('ids', 'sources', 'cx', 'x', 'y')]
            unit = fixture['unit']
            owners = ids if unit == 'record' else sources
            units = list(dict.fromkeys(owners))
            graph = direct_graph(cx, ids, sources, fixture['threshold'], 0, 1_000_000)
            initial_selected = {ids[i] for i, b in enumerate(graph.blockers) if not len(b)}
            # Choose a real blocker first, when available. This can activate admissions.
            first = next((owners[int(b[0])] for b in graph.blockers if len(b)), units[0])
            chosen = [first] + [u for u in units if u != first][:2]
            horizon = len(chosen)
            paths = {'singleton': [[u] for u in chosen], 'batch': [chosen],
                     'reverse': [[u] for u in reversed(chosen)],
                     'duplicate': [[chosen[0], chosen[0]], [chosen[0]], chosen[1:]]}
            cases.append({'fixture': fixture['name'], 'role': fixture['role'], 'records': len(ids),
                          'unit': unit, 'horizon': horizon, 'paths': paths,
                          'initial_graph_edges': int(len(graph.indices)), 'initial_selected': len(initial_selected)})
            if unit == 'source':
                check(fixture['name'], 'fixture/own_source_blocker_present',
                      any(sources[i] in {sources[int(j)] for j in b} for i, b in enumerate(graph.blockers)))
                check(fixture['name'], 'fixture/repeated_blocker_source_present',
                      any(len({sources[int(j)] for j in b}) < len(b) for b in graph.blockers))
            for method in PRIMARY_METHODS:
                case = fixture['name'] + '/' + method

                def build(h=horizon):
                    return build_method(method, graph, x, y, h, unit=unit, curator=cx)

                def inspect(state, deleted, remaining, tag):
                    keep = np.asarray([i for i, owner in enumerate(owners) if owner not in deleted], int)
                    retained_ids, retained_sources = [ids[i] for i in keep], [sources[i] for i in keep]
                    fresh_graph = direct_graph(cx[keep], retained_ids, retained_sources,
                                               fixture['threshold'], 0, 1_000_000)
                    alive, observed_horizon = alive_and_horizon(state, method)
                    check(case, tag + '/alive_horizon', alive == set(units) - deleted and observed_horizon == remaining)
                    payload = payload_comparison(state, fresh_graph, x[keep], y[keep], unit,
                                                 remaining, initial_selected, ATOL, RTOL)
                    check(case, tag + '/fresh_moments_membership', payload['passed'],
                          count_exact=payload['count_exact'],
                          max_gram_difference=payload['gram']['maximum_absolute_difference'],
                          max_cross_difference=payload['cross']['maximum_absolute_difference'])
                    if method in ('P-I', 'P-S', 'P-R'):
                        symbols = symbolic_state(fresh_graph, remaining, unit)
                        exact, packed = independent_coefficients(symbols, x[keep], y[keep], retained_ids, 1_000_000)
                        coefficients = compare_coefficients(stable_coefficients(state), exact, packed, ATOL, RTOL)
                        check(case, tag + '/complete_coefficients', coefficients['passed'],
                              coordinates_checked=coefficients['coordinates_checked'])
                    return fresh_graph

                final_states = {}
                for path_name, batches in paths.items():
                    state, deleted = build(), set()
                    symbolic = symbolic_state(graph, horizon, unit)
                    for number, batch in enumerate(batches):
                        new = set(batch) - deleted
                        deleted.update(new)
                        operation = state.delete(batch)
                        remaining = horizon - len(deleted)
                        tag = f'{path_name}/{number}'
                        check(case, tag + '/fresh_deletions', operation['fresh_deletions'] == len(new))
                        fresh_graph = inspect(state, deleted, remaining, tag)
                        symbolic = substitute_symbols(symbolic, new, remaining)
                        check(case, tag + '/exact_record_symbol_substitution',
                              symbolic == symbolic_state(fresh_graph, remaining, unit))
                    final_states[path_name] = state

                anchor = final_states['singleton']
                for path_name, state in final_states.items():
                    a, b = anchor.moments(), state.moments()
                    ok = a.count == b.count and all(array_comparison(left, right, ATOL, RTOL)['passed']
                        for left, right in ((a.gram, b.gram), (a.cross, b.cross)))
                    check(case, path_name + '/history_diagnostic', ok,
                          bitwise_state_history_independence_claim=False)
                    if method in ('P-I', 'P-S', 'P-R'):
                        packed = state.packed_dimension
                        coeff = compare_coefficients(stable_coefficients(state), stable_coefficients(anchor), packed, ATOL, RTOL)
                        check(case, path_name + '/history_coefficients', coeff['passed'])
                        check(case, path_name + '/history_rank_groups', stable_groups(state) == stable_groups(anchor))

                # Same history, with a save/load boundary before the final batch.
                uninterrupted, resumed = build(), build()
                for state in (uninterrupted, resumed):
                    state.delete([chosen[0]])
                path = Path(temporary) / 'state.npz'
                for name, raw in snapshot(resumed).items():
                    (Path(temporary) / name).write_bytes(raw)
                resumed = load_method(method, path)
                uninterrupted.delete(chosen[1:])
                resumed.delete(chosen[1:])
                check(case, 'save_resume/same_history_snapshot_bytes', snapshot(uninterrupted) == snapshot(resumed))
                inspect(resumed, set(chosen), 0, 'save_resume')

                # All invalid requests must be rejected before any persistent mutation.
                state = build()
                for label, request in [('unknown', [chosen[0], '__unknown_contract_identifier__']),
                                       ('non_string', [chosen[0], 17])]:
                    before = snapshot(state)
                    rejected = False
                    try:
                        state.delete(request)
                    except ValueError:
                        rejected = True
                    check(case, label + '/atomic_refusal', rejected and snapshot(state) == before)
                state = build(1)
                before = snapshot(state)
                rejected = False
                try:
                    state.delete(chosen[:2])
                except ValueError:
                    rejected = True
                check(case, 'over_budget_batch/atomic_refusal', rejected and snapshot(state) == before)
                state.delete([chosen[0]])
                before = snapshot(state)
                rejected = False
                try:
                    state.delete([chosen[1]])
                except ValueError:
                    rejected = True
                check(case, 'exhausted_horizon/atomic_refusal', rejected and snapshot(state) == before)
                operation = state.delete([chosen[0], chosen[0]])
                check(case, 'exhausted_duplicate/zero_fresh', operation['fresh_deletions'] == 0)
                inspect(state, {chosen[0]}, 0, 'exhausted_duplicate')
                operation = state.delete([])
                check(case, 'exhausted_empty/zero_fresh', operation['fresh_deletions'] == 0)
                inspect(state, {chosen[0]}, 0, 'exhausted_empty')

                state = build(len(units))
                state.delete(units)
                for name, raw in snapshot(state).items():
                    (Path(temporary) / name).write_bytes(raw)
                restored = load_method(method, path)
                moments = restored.moments()
                alive, remaining = alive_and_horizon(restored, method)
                check(case, 'all_deleted/zero_roundtrip', moments.count == 0 and not alive and remaining == 0
                      and not np.any(moments.gram) and not np.any(moments.cross))
                if unit == 'source':
                    # Match deleted records, while retaining each service's own horizon contract.
                    record_budget = sum(owner in chosen for owner in sources)
                    source_state = build(horizon)
                    record_state = build_method(method, graph, x, y, record_budget, unit='record', curator=cx)
                    removed_records = 0
                    for number, source in enumerate(chosen):
                        expanded = [rid for rid, owner in zip(ids, sources) if owner == source]
                        removed_records += len(expanded)
                        source_state.delete([source])
                        record_state.delete(expanded)
                        a, b = source_state.moments(), record_state.moments()
                        equal = a.count == b.count and all(array_comparison(left, right, ATOL, RTOL)['passed']
                            for left, right in ((a.gram, b.gram), (a.cross, b.cross)))
                        _, source_horizon = alive_and_horizon(source_state, method)
                        _, record_horizon = alive_and_horizon(record_state, method)
                        check(case, f'source_expansion/{number}/same_current_target', equal,
                              equal_coefficient_state_claim=False)
                        check(case, f'source_expansion/{number}/separate_horizons',
                              source_horizon == horizon - number - 1 and record_horizon == record_budget - removed_records,
                              remaining_source_horizon=source_horizon, remaining_record_horizon=record_horizon)
    source_paths = ['phase6/check_state_audit_semantics.py', 'phase6/state_audit.py',
                    'phase6/methods.py', 'phase5/methods.py', 'phase5/payload.py',
                    'ccu/summary.py', 'data/civil_comments_engineering_preview.jsonl']
    result = {'schema': 'ccu-state-transition-contract-checks-1', 'scope': 'software_contract_checks_only',
              'primary_semantic_study_started': False, 'actual_source_empirical_results': False,
              'methods': list(PRIMARY_METHODS), 'absolute_tolerance': ATOL, 'relative_tolerance': RTOL,
              'FP64_history_claim': 'diagnostic coordinate agreement; exact structure and symbolic integer substitutions',
              'cases': cases, 'checks': checks, 'check_count': len(checks),
              'all_passed': all(row['passed'] for row in checks),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}}
    (out / 'checks.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_passed': result['all_passed'], 'check_count': result['check_count'],
                      'failed': [row for row in checks if not row['passed']]}), flush=True)
    if not result['all_passed']:
        raise AssertionError('State-transition contract failures preserved')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    run(parser.parse_args().destination)
