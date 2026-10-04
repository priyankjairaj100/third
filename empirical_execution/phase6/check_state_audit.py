"""All eight actual services on genuine Civil text. No human or source claims."""
from pathlib import Path
import argparse
import json
import numpy as np

from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase6.run_isolated import prepare_bundle, run_service, DEFAULT_POLICY
from phase6.state_audit import audit_service_state, write
from phase6.methods import PRIMARY_METHODS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'phase6/results/state_audit_release_v2')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'data/civil_comments_engineering_preview.jsonl'
    rows = read_natural_jsonl(source)
    x, _ = lexical_engineering_features(rows, 64)
    cx, _ = lexical_engineering_features(rows, 128)
    y = np.asarray([[row['fields']['toxicity']] for row in rows], np.float64)
    ids = [row['record_id'] for row in rows]
    requests_manifest = json.loads((ROOT / 'phase4/results/execution_engineering_final/d64/requests.json').read_text())
    path = next(p for p in requests_manifest['trajectories'] if p['arm'] == 'R')
    checkpoints = [1, 2, 4, 8]
    requests = [path['deletion_order'][previous:current] for previous, current in zip([0] + checkpoints[:-1], checkpoints)]
    policy = {**DEFAULT_POLICY, 'state_audit_absolute_tolerance': 1e-11,
              'state_audit_relative_tolerance': 1e-10,
              'state_audit_max_pair_coordinates': 100_000_000,
              'state_audit_max_coefficient_coordinates': 4_000_000}
    write(out / 'design_lock.json', {'evidence_role': 'engineering_nonconfirmatory',
          'source_sha256': sha256_file(source), 'methods': list(PRIMARY_METHODS),
          'request_manifest_sha256': requests_manifest['manifest_sha256'], 'trajectory_id': path['trajectory_id'],
          'requests': requests, 'checkpoints': checkpoints, 'policy': policy,
          'feature_contract': 'original lexical engineering vectors; no semantic encoder',
          'genuine_source_arm_available': False, 'all_first_16_primary_R_and_S_completed': False})
    package = prepare_bundle(cx, x, y, ids, out / 'inputs', threshold=.6)
    outcomes = []
    for method in PRIMARY_METHODS:
        reference = out / ('measured_' + method)
        service = run_service(package, method, requests, reference, horizon=8, policy=policy)
        result = audit_service_state(package, method, requests, reference, out / ('audit_' + method),
                    horizon=8, unit='record', lambda_reg=.01, policy=policy, full_state=True)
        outcome = {'method': method, 'service_success': service['success'], 'audit_passed': result['passed'],
                   'audit_status': result['status'], 'audit_report_sha256': sha256_file(out / ('audit_' + method) / 'state_audit.json')}
        outcomes.append(outcome)
        write(out / 'outcomes.json', outcomes)
        print(json.dumps(outcome), flush=True)
    checks = {'all_eight_core_methods_present': [o['method'] for o in outcomes] == list(PRIMARY_METHODS),
              'all_measured_services_completed': all(o['service_success'] for o in outcomes),
              'all_full_state_audits_passed': all(o['audit_passed'] for o in outcomes)}
    all_rows = [row for outcome in outcomes for row in json.loads((out / ('audit_' + outcome['method']) / 'state_audit.json').read_text())['checkpoints']]
    checks['all_32_checkpoint_states_compared'] = len(all_rows) == 32
    checks['actual_worker_snapshots_bound_at_every_checkpoint'] = all(r['reconstructed_actual_snapshot_bound'] for r in all_rows)
    checks['canonical_symbolic_states_equal_at_every_checkpoint'] = all(r['canonical_record_symbol_state_exact'] for r in all_rows)
    checks['measured_and_audit_heads_bitwise_equal'] = all(r['replay_and_measured_heads_bitwise_equal'] for r in all_rows)
    coefficients = [r['coefficients'] for r in all_rows if 'coefficients' in r]
    checks['all_summary_checkpoint_coordinates_compared'] = len(coefficients) == 12 and all(r['passed'] for r in coefficients)
    payload_rows = json.loads((out / 'audit_B-E/state_audit.json').read_text())['checkpoints']
    checks['all_BE_memberships_and_payloads_match'] = all(r['moments_and_membership']['passed'] for r in payload_rows)
    # One genuine record checks total withdrawal, not a synthetic corpus.
    one_package = prepare_bundle(cx[:1], x[:1], y[:1], ids[:1], out / 'one_record_input', threshold=.6)
    one_reference = out / 'one_record_measured_P-R'
    one_service = run_service(one_package, 'P-R', [ids[:1]], one_reference, horizon=1, policy=policy)
    one_audit = audit_service_state(one_package, 'P-R', [ids[:1]], one_reference, out / 'one_record_audit_P-R',
                    horizon=1, unit='record', lambda_reg=.01, policy=policy, full_state=True)
    one_row = one_audit['checkpoints'][0]
    checks['all_deleted_forced_zero_group_quotient_is_explicit'] = bool(one_service['success'] and one_audit['passed'] and
        not one_row['structural_rank_groups_exact'] and one_row['rank_basis_relations_exact'] and
        one_row['rank_groups_actual'] and not one_row['rank_groups_fresh'])
    report = {'schema': 'ccu-complete-state-audit-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'natural_checkpoint_states': len(all_rows), 'full_coefficient_coordinates_compared': sum(r['coordinates_checked'] for r in coefficients),
              'additional_natural_single_record_total_withdrawal_checks': 1,
              'outcomes': outcomes, 'primary_semantic_or_genuine_source_evidence': False,
              'all_16_R_and_16_S_native_paths_completed': False,
              'source_sha256': {p.name: sha256_file(p) for p in [Path(__file__), Path(__file__).with_name('state_audit.py')]}}
    write(out / 'checks.json', report)
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
