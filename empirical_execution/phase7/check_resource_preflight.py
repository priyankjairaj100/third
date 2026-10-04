"""Symbolic formula and CLI checks only; no empirical data or worker execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import ast
import contextlib
import copy
import io
import itertools
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase7 import resource_preflight as preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.exit(2, 'Preserve existing report: ' + str(args.out) + '\n')
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    def refuses(name, function):
        try:
            function()
        except (ValueError, TypeError):
            check(name, True)
        else:
            raise AssertionError(name)

    contract = preflight.source_contract()
    defaults = contract['default_caps']
    check('defaults_come_from_actual_frozen_sources', defaults == {
        preflight.PAIR_CAP: 100_000_000, preflight.COEFFICIENT_CAP: 4_000_000,
        preflight.CONVEX_CAP: 2_000_000})
    native = preflight.preflight(records=10_000, dimension=768, outputs=[1, 20], contract=contract)
    q1, q20 = native['scenarios']
    check('native_packed_coordinates_are_exact', [q1['packed_coordinates_per_designated_symbol'],
        q20['packed_coordinates_per_designated_symbol']] == [296_065, 310_657])
    a, b = [row['comparisons']['frozen_defaults'] for row in (q1, q20)]
    check('initial_pair_gate_known_refusal', a['graph_audit']['exact_demand'] == 38_396_160_000
          and a['graph_audit']['status'] == 'refused_by_count_cap')
    check('native_maximum_original_rows_510', a['graph_audit']['maximum_original_records_at_this_curator_dimension'] == 510)
    check('native_key_capacity_13_or_12', a['full_coefficients']['maximum_designated_keys'] == 13
          and b['full_coefficients']['maximum_designated_keys'] == 12)
    check('native_selected_capacity_2604_or_130', a['convex_verification']['maximum_selected_records'] == 2604
          and b['convex_verification']['maximum_selected_records'] == 130)
    check('unknown_selected_and_symbolic_counts_stay_unknown', all(
        gate['exact_demand'] is None and gate['status'] == 'unknown_count_required'
        for row in (a, b) for gate in (row['full_coefficients'], row['convex_verification'])))
    check('all_selected_is_a_bound_not_observed_demand',
        a['convex_verification']['all_retained_selected_bound'] == 7_680_000
        and b['convex_verification']['all_retained_selected_bound'] == 153_600_000)
    check('combined_inventory_does_not_admit_or_refuse_individual_jobs',
          a['status_scope'] == 'combined_gate_inventory_not_method_or_job_admission'
          and 'convex-only' in a['applicability'] and 'FP64' in a['graph_audit']['executed_for'])

    # Independent finite combinatorial enumeration, with no record or feature data.
    enumerated = 0
    for n, d in itertools.product(range(26), (1, 3, 768)):
        pairs = len(list(itertools.combinations(range(n), 2))) * d
        value = preflight.preflight(records=n, dimension=d, outputs=[1], contract=contract)
        check('pair_enumeration_' + str(enumerated), value['scenarios'][0]['comparisons']['frozen_defaults']['graph_audit']['exact_demand'] == pairs)
        enumerated += 1
    for d, q in itertools.product(range(1, 16), (1, 2, 20)):
        basis_count = len(list(itertools.combinations_with_replacement(range(d), 2))) + len(list(itertools.product(range(d), range(q)))) + 1
        value = preflight.preflight(records=10, dimension=d, outputs=[q], symbolic_keys=3, selected_records=7, contract=contract)
        row = value['scenarios'][0]
        check('packed_enumeration_' + str(d) + '_' + str(q), row['packed_coordinates_per_designated_symbol'] == basis_count
              and row['comparisons']['frozen_defaults']['full_coefficients']['exact_demand'] == 3 * basis_count)

    # Evaluate only inspected arithmetic predicates over shape placeholders.
    # No verifier, graph constructor, NumPy array, or empirical fixture is run.
    guard_cases = 0
    for expression in contract['source_expressions']:
        if expression['kind'] != 'guard' or expression['function'] == 'audit_service_state':
            continue
        compiled = compile(ast.parse(expression['expression'], mode='eval'), '<frozen arithmetic guard>', 'eval')
        for n, d, q, m in [(0, 1, 1, 0), (1, 1, 1, 1), (7, 3, 2, 4), (10_000, 768, 20, 13)]:
            packed = d * (d + 1) // 2 + d * q + 1
            if expression['function'] == 'direct_graph':
                demand = n * (n - 1) // 2 * d
            elif expression['function'] == 'independent_coefficients':
                demand = m * packed
            elif expression['function'] == 'certify_multioutput':
                demand = n * d * q
            else:
                demand = n * d
            for cap in {max(0, demand - 1), demand, demand + 1}:
                namespace = dict(n=n, d=d, packed=packed, symbols=range(m),
                    x=SimpleNamespace(shape=(n, d), size=n*d), y=SimpleNamespace(shape=(n, q)),
                    cap=cap, maximum_pair_coordinates=cap, maximum_coordinates=cap)
                check('frozen_guard_boundary_' + str(guard_cases), eval(compiled, {'__builtins__': {}, 'len': len}, namespace) == (demand > cap))
                guard_cases += 1

    for cap, d in itertools.product((0, 1, 100_000_000, 38_396_160_000, 2**100), (1, 3, 768)):
        n = preflight.maximum_pair_rows(cap, d)
        check('inverse_pair_' + str(cap) + '_' + str(d), n*(n-1)//2*d <= cap < n*(n+1)//2*d)

    policy = dict(memory_bytes=64 * 1024**2, cpu_seconds=1, threads=1, wall_seconds=1,
        state_audit_max_pair_coordinates=38_396_160_000,
        state_audit_max_coefficient_coordinates=13*310_657, convex_certificate_coordinates=153_600_000)
    supplied = preflight.preflight(records=10_000, dimension=768, outputs=[20], selected_records=10_000,
                                  symbolic_keys=13, policy=policy, contract=contract)
    custom = supplied['scenarios'][0]['comparisons']['supplied_policy']
    check('exact_cap_equality_passes_all_three_predicates', all(custom[name]['status'] == 'passes_count_cap'
          for name in ('graph_audit', 'full_coefficients', 'convex_verification')))
    check('passing_counts_never_approve_tiny_resource_policy', custom['status'] == 'count_caps_only_pass_resource_qualification_absent'
          and supplied['qualification']['runtime_seconds'] is None and supplied['qualification']['peak_rss_bytes'] is None
          and supplied['qualification']['policy_approved'] is False and supplied['qualification']['development_lock_issued'] is False)
    partial = {key: value for key, value in policy.items() if key not in preflight.CAPS}
    value = preflight.preflight(records=10_000, dimension=768, outputs=[1], policy=partial, contract=contract)
    check('unspecified_policy_caps_preserve_frozen_defaults', value['scenarios'][0]['comparisons']['supplied_policy'] == a)
    empty = preflight.preflight(records=0, dimension=768, outputs=[20], contract=contract)
    check('empty_panel_has_exact_zero_counts_without_data', all(empty['scenarios'][0]['comparisons']['frozen_defaults'][name]['exact_demand'] == 0
          for name in ('graph_audit', 'full_coefficients', 'convex_verification')))
    light = preflight.preflight(records=10_000, dimension=768, outputs=[1], audit_scope='light', contract=contract)
    check('light_audit_still_fails_original_graph_gate', light['scenarios'][0]['comparisons']['frozen_defaults']['status'] == 'known_count_refusal'
          and light['scenarios'][0]['comparisons']['frozen_defaults']['full_coefficients']['status'] == 'not_applied_in_light_audit')
    different = preflight.preflight(records=10_000, dimension=64, curator_dimension=768, retained_records=1,
                                   outputs=[20], selected_records=1, symbolic_keys=1, contract=contract)
    diff = different['scenarios'][0]['comparisons']['frozen_defaults']
    check('curator_and_learner_dimensions_stay_distinct', diff['graph_audit']['exact_demand'] == 38_396_160_000
          and diff['graph_audit']['retained_checkpoint_demand'] == 0 and diff['convex_verification']['exact_demand'] == 1280
          and diff['full_coefficients']['exact_demand'] == 3361)
    check('retained_graph_cannot_hide_original_refusal', diff['graph_audit']['status'] == 'refused_by_count_cap')
    for field, value in [('records', True), ('records', -1), ('dimension', 0), ('dimension', 2.5),
                         ('selected_records', -1), ('selected_records', 11), ('retained_records', 11),
                         ('symbolic_keys', True), ('symbolic_keys', 21), ('outputs', [False]), ('outputs', [1, 1]),
                         ('audit_scope', 'waived')]:
        kwargs = dict(records=10, dimension=3, outputs=[1], contract=contract)
        kwargs[field] = value
        refuses('invalid_shape_' + field + '_' + repr(value), lambda kwargs=kwargs: preflight.preflight(**kwargs))
    for field, bad in [(preflight.PAIR_CAP, True), (preflight.COEFFICIENT_CAP, -1), (preflight.CONVEX_CAP, 0.5),
                       ('memory_bytes', 1), ('cpu_seconds', False), ('wall_seconds', float('inf')),
                       ('state_audit_relative_tolerance', 1), ('convex_verifier', 'approximate'), ('unexpected', 1)]:
        value = {**policy, field: bad}
        refuses('invalid_policy_' + field, lambda value=value: preflight.validate_policy(value))
    refuses('duplicate_policy_json_key', lambda: preflight.loads('{"threads":1,"threads":2}'))
    refuses('nonfinite_policy_json', lambda: preflight.loads('{"wall_seconds":NaN}'))
    refuses('missing_policy_fields', lambda: preflight.validate_policy({preflight.PAIR_CAP: 1}))

    source_snapshot = {name: (preflight.ROOT/name).read_bytes() for name in contract['source_sha256']}
    drift = dict(source_snapshot)
    drift['phase6/state_audit.py'] = drift['phase6/state_audit.py'].replace(b'n * (n - 1) // 2 * x.shape[1] > maximum_pair_coordinates', b'n * (n - 1) // 2 * x.shape[1] >= maximum_pair_coordinates')
    refuses('unknown_future_guard_refused', lambda: preflight.source_contract(drift))
    altered_default = dict(source_snapshot)
    altered_default['phase6/state_audit.py'] = altered_default['phase6/state_audit.py'].replace(b"policy.get('state_audit_max_pair_coordinates', 100_000_000)", b"policy.get('state_audit_max_pair_coordinates', 123)")
    check('defaults_are_read_from_bytes_not_only_hardcoded', preflight.source_contract(altered_default)['default_caps'][preflight.PAIR_CAP] == 123)
    check('frozen_sources_unchanged_by_formula_checks', all(preflight.sha256((preflight.ROOT/name).read_bytes()) == value
          for name, value in contract['source_sha256'].items()))

    with tempfile.TemporaryDirectory(prefix='resource-preflight-formula-only-') as temporary:
        directory = Path(temporary)
        path = directory/'policy.json'
        raw = (json.dumps(policy) + '\n').encode()
        path.write_bytes(raw)
        original = preflight.preflight
        def mutate_after_capture(**kwargs):
            path.write_text('{"later_change":true}')
            return original(**kwargs)
        argv = ['--records', '10000', '--dimension', '768', '--outputs', '1', '20',
                '--policy', str(path), '--out', str(directory/'report.json')]
        with patch.object(preflight, 'preflight', side_effect=mutate_after_capture), contextlib.redirect_stdout(io.StringIO()):
            report = preflight.main(argv)
        check('policy_receipt_binds_same_bytes_used_for_parsing', report['policy_file_sha256'] == preflight.sha256(raw)
              and report['supplied_policy'] == policy)
        saved = (directory/'report.json').read_bytes()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                preflight.main(argv)
        except SystemExit as error:
            check('existing_output_refused_without_replacement', error.code == 2 and (directory/'report.json').read_bytes() == saved)
        else:
            raise AssertionError('existing_output_refused_without_replacement')
        check('report_roundtrip_is_exact', json.loads(saved) == report)
    report = {'schema': 'ccu-native-resource-preflight-checks-1', 'status': 'passed',
        'evidence_role': 'symbolic_software_checks_not_empirical_data', 'passed': len(checks), 'checks': checks,
        'enumerated_pair_cases': enumerated, 'frozen_guard_boundary_cases': guard_cases,
        'benchmarks_run': 0, 'corpora_read': 0, 'workers_run': 0,
        'source_sha256': {**contract['source_sha256'],
            'phase7/resource_preflight.py': preflight.sha256(Path(preflight.__file__).read_bytes()),
            'phase7/check_resource_preflight.py': preflight.sha256(Path(__file__).read_bytes())}}
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'passed': len(checks), 'report': str(args.out)}))


if __name__ == '__main__':
    main()
