"""Compare full original-guard replay and independently reconstruct panels."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import os
import resource
import time
import admission
import common as c
from worker import checked_inputs, admit_runtime


def review_panels(rows, removed, got, plan):
    by_id = {r['record_id']: r for r in rows}
    c.require(len(by_id) == len(rows), 'unique_input_records')
    train = [r for r in rows if r['role'] == 'development_train']
    heldout = {r['record_id'] for r in rows if r['partition'] in ('test', 'calibration')}
    dead = {r['record_id'] for r in removed}
    c.require(len(dead) == len(removed) and dead <= {r['record_id'] for r in train}, 'removed_training_only')
    c.require(all(r['heldout_witness_id'] in heldout for r in removed), 'heldout_witness_scope')
    retained = [r for r in train if r['record_id'] not in dead]
    c.require(got['retained_development_train_ids'] == [r['record_id'] for r in retained], 'retained_ids_in_input_order')
    test = [r['record_id'] for r in rows if r['role'] == 'development_test']
    c.require(got['development_test_ids'] == test, 'only_reserved_development_test')
    groups = defaultdict(list)
    for row in retained:
        groups[row['source_unit_id']].append(row['record_id'])
    ordered = sorted(groups, key=lambda unit: (
        hashlib.sha256((plan['development_order_salt']+'\0'+unit).encode()).hexdigest(), unit))
    c.require(set(got['prefixes']) == {str(n) for n in plan['development_targets']}, 'all_original_targets')
    summaries = {}
    for target in plan['development_targets']:
        selected = []; sources = []
        for unit in ordered:
            if len(selected) >= target:
                break
            sources.append(unit); selected += sorted(groups[unit])
        wanted = {'target': target, 'actual': len(selected), 'shortfall': max(0, target-len(selected)),
                  'overshoot': max(0, len(selected)-target), 'record_ids': selected, 'source_ids': sources}
        c.require(got['prefixes'][str(target)] == wanted, 'whole_source_prefix_exact')
        summaries[str(target)] = {k: wanted[k] for k in ('target', 'actual', 'shortfall', 'overshoot')}
        summaries[str(target)]['source_units'] = len(sources)
    c.require(got['all_reserved_sources_still_excluded_from_evaluation'] is True
              and got['full_calibration_test_membership_unchanged'] is True
              and got['effects_launched'] is False and got['original_primary_acceptance'] is False,
              'unchanged_scope_flags')
    return {'prefixes': summaries, 'retained_train_rows': len(retained), 'removed_train_rows': len(dead),
            'retained_train_sources': len(groups), 'development_test_rows': len(test),
            'retained_unknown_singleton_rows': sum(r['source_kind'] == 'unknown_singleton' for r in retained)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package-sha256', required=True)
    p.add_argument('--admission-sha256', required=True); a = p.parse_args()
    admit_runtime(); c.verify_package(a.package_sha256)
    approved = admission.check_bundle(c.BUNDLE, a.admission_sha256, remote=True)
    _, watched = checked_inputs(); start = time.perf_counter(); reports = {}; payloads = {}; processes = {}
    for mode in ('primary', 'reference'):
        process = c.strict((c.ROOT/(mode+'_process.json')).read_bytes())
        c.require(process['returncode'] == 0 and process['console_cap_exceeded'] is False
                  and process['job_id'] == os.environ['SLURM_JOB_ID'], 'successful_fresh_scan_process')
        report = c.strict((c.ROOT/mode/'result.json').read_bytes())
        c.require(report['passed'] is True and report['mode'] == mode and
                  report['package_sha256'] == a.package_sha256 and report['admission_sha256'] == a.admission_sha256
                  and report['job_id'] == os.environ['SLURM_JOB_ID'], 'scan_result_identity')
        c.require(report['input_bindings'] == {str(k): v for k, v in watched.items()}, 'scan_input_identity')
        c.require(report['pairs_scored'] == 433837782 and report['train_rows'] == 10314
                  and report['heldout_rows'] == 42063 and report['locked_tau'] == approved['locked_tau'], 'full_scope')
        wanted_names = {'guard.private.json', 'panels.private.json'} if mode == 'primary' else {'guard.private.json'}
        c.require(set(report['outputs']) == wanted_names, 'complete_scan_outputs')
        for name, wanted in report['outputs'].items():
            c.require(name in ('guard.private.json', 'panels.private.json') and c.binding(c.ROOT/mode/name) == wanted,
                      'immutable_scan_output')
        payloads[mode] = c.strict((c.ROOT/mode/'guard.private.json').read_bytes())
        reports[mode] = report; processes[mode] = process
    c.require(payloads['primary'] == payloads['reference'], 'full_reference_witness_score_and_keep_parity')
    payload = payloads['primary']; tau = float(admission.fraction(approved['locked_tau']))
    c.require(payload['locked_tau'] == approved['locked_tau'] and payload['pairs_scored'] == 433837782,
              'payload_scope')
    c.require(all(type(r['score']) is float and r['score'] > tau for r in payload['removed']), 'strict_witness_scores')
    rows = c.strict((c.INTAKE/'allocation.private.json').read_bytes())
    panels = c.strict((c.ROOT/'primary/panels.private.json').read_bytes())
    plan = c.strict((c.HERE/'RESERVATION_PLAN.json').read_bytes())
    summary = review_panels(rows, payload['removed'], panels, plan)
    c.require(all(r['removed_rows'] == summary['removed_train_rows'] and
                  r['retained_train_rows'] == summary['retained_train_rows'] for r in reports.values()), 'counts_agree')
    for path, want in watched.items():
        c.require(c.binding(path) == want, 'review_input_changed')
    for mode, report in reports.items():
        for name, wanted in report['outputs'].items():
            c.require(c.binding(c.ROOT/mode/name) == wanted, 'review_output_changed')
    c.verify_package(a.package_sha256); admission.check_bundle(c.BUNDLE, a.admission_sha256, remote=True)
    result = {'schema': 'third-full-development-guard-independent-review-v1', 'passed': True,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': os.environ['SLURM_JOB_ID'],
              'package_sha256': a.package_sha256, 'admission_sha256': a.admission_sha256,
              'model_defined_locked_tau': approved['locked_tau'], 'input_bindings': {str(k): v for k, v in watched.items()},
              'guard_payloads': {mode: c.binding(c.ROOT/mode/'guard.private.json') for mode in reports},
              'scan_reports': {mode: c.binding(c.ROOT/mode/'result.json') for mode in reports},
              'process_receipts': {mode: c.binding(c.ROOT/(mode+'_process.json')) for mode in reports},
              'panel_payload': c.binding(c.ROOT/'primary/panels.private.json'), 'panel_summary': summary,
              'full_original_guard_replay_passed': True, 'pairs_per_scan': 433837782,
              'same_frozen_FP64_scorer_shared': True, 'independent_arithmetic_implementation_of_scorer': False,
              'prefix_memberships_independently_recomputed': True, 'later_evaluation_labels_read': False,
              'original_calibration_and_test_memberships_unchanged': True,
              'all_reserved_sources_excluded_from_later_evaluation': True,
              'post_guard_development_panels_verified': True, 'terminal_accounting_required': True,
              'original_human_and_primary_acceptance': False, 'human_annotation_calls': 0,
              'human_validity_claimed': False, 'effect_experiments_started': False, 'confirmatory_study_ready': False,
              'review_elapsed_seconds': time.perf_counter()-start,
              'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'next_gate': 'Verify terminal Slurm success and all payloads, then seal the smallest natural development request/baseline/resource scope before effects.'}
    c.publish(c.ROOT/'result.json', result)
    print(c.canonical({'passed': True, 'full_reference_replay': True, 'panel_summary': summary}).decode(), flush=True)


if __name__ == '__main__':
    main()
