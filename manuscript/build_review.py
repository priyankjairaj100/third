#!/usr/bin/env python3
"""Record a bounded author claim review and partial compute accounting.

No cluster access, model calls, theorem-fixture reruns, or private-data export.
Existing independent review receipts are checked, not relabeled as a new
independent review of this manuscript. Run before build_evidence.py.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPORTS = 'local_run/local_llm_execution_20261014/'


def bind(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def source(name):
    return {'path': name, **bind(ROOT / name)}


def read(name):
    return json.loads((ROOT / name).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def job(receipt, expected_id, role, expected_state='COMPLETED'):
    parent = [r.split('|') for r in receipt['accounting'].splitlines()
              if r.split('|')[0] == expected_id]
    need(len(parent) == 1, 'one parent accounting row')
    row = parent[0]
    need(receipt['state'] == row[1] == expected_state, 'terminal job state')
    need(receipt['exit'] == row[2], 'exit code identity')
    need(row[2] == ('0:0' if expected_state == 'COMPLETED' else '1:0'), 'expected exit')
    return {'job_id': expected_id, 'role': role, 'state': row[1],
            'exit_code': row[2], 'elapsed_seconds': int(row[3]),
            'compute_node': row[-1], 'saved_observation_utc': receipt['observed_utc'],
            'parent_accounting_row': '|'.join(row)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mistral-launch-receipt', type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    prior_name = REPORTS + 'llm_v4_qwen_audit_stop_20261015.json'
    terminal_name = REPORTS + 'llm_selection_v4_terminal_result.json'
    resources_name = REPORTS + 'llm_node_pool_v2_execution.json'
    prior, terminal, resources = map(read, [prior_name, terminal_name, resources_name])
    need(terminal['execution_complete'] and not terminal['selection_passed'], 'finished negative selection')
    need(terminal['model_requests_attempted'] == 792 and terminal['model_retries'] == 0, 'call count')
    need(resources['gpu_count_per_model'] == 1, 'one Qwen GPU')
    need('--gres=gpu:a100-80gb:1' in resources['launch_resources'], 'Qwen GPU resource')
    private = args.mistral_launch_receipt
    need(private.is_file() and not private.is_symlink(), 'regular launch receipt')
    launch = json.loads(private.read_text())
    need(launch['job_id'] == '15242' and launch['mode'] == 'mistral', 'matching Mistral launch')
    need('--gres=gpu:a100-80gb:1' in launch['command'], 'one Mistral A100')
    jobs = [job(prior['qwen'], '13663', 'Qwen v4 inference'),
            job(terminal['terminal_receipts']['mistral'], '15242', 'Mistral v4 inference'),
            job(prior['audit'], '14187', 'original Qwen audit; failed newline normalization', 'FAILED'),
            job(terminal['terminal_receipts']['audit_qwen'], '15241', 'corrected exact-byte Qwen audit'),
            job(terminal['terminal_receipts']['audit_mistral'], '15276', 'exact-byte Mistral audit'),
            job(terminal['terminal_receipts']['statistics'], '15277', 'v4 selection statistics')]
    need([r['elapsed_seconds'] for r in jobs] == [11836, 7209, 583, 755, 130, 1], 'saved durations')
    for row in jobs[:2]:
        row['allocated_gpu_count'] = 1
        row['allocated_gpu_type'] = 'A100 80GB'
    gpu_seconds = sum(r['elapsed_seconds'] * r['allocated_gpu_count'] for r in jobs[:2])
    cpu_seconds = sum(r['elapsed_seconds'] for r in jobs[2:])
    hours = Fraction(gpu_seconds, 3600)
    compute = {'schema': 'third-partial-compute-accounting-v1', 'created_utc': now,
               'scope': 'Six saved v4 inference, audit and statistics jobs only; not the complete project budget.',
               'jobs': jobs, 'gpu_allocation_seconds': gpu_seconds,
               'gpu_allocation_hours_exact': {'numerator': hours.numerator, 'denominator': hours.denominator},
               'gpu_allocation_hours_display': f'{float(hours):.2f}',
               'cpu_job_elapsed_seconds_sum': cpu_seconds, 'cpu_core_seconds_computed': False,
               'saved_parent_rows_counted_once': True, 'batch_step_times_added': False,
               'failed_audit_cost_retained': True, 'queue_wait_included': False,
               'gpu_utilization_measured': False, 'energy_measured': False,
               'project_total_compute_complete': False, 'new_cluster_queries': 0,
               'source_bindings': [source(n) for n in [prior_name, terminal_name, resources_name]],
               'private_Mistral_launch_binding': bind(private),
               'private_Mistral_launch_export': {'job_id': '15242', 'gpu_count': 1, 'gpu_type': 'A100 80GB'},
               'excluded_scope': ['Earlier pilot and v3 inference', 'Embedding preparation',
                                  'Sampling and request preparation', 'Hardware and runtime qualification',
                                  'Other preparation jobs', 'Local development and manuscript work'],
               'limitations': ['Allocation time includes job overhead and loading; it is not GPU busy time.',
                               'CPU durations sum elapsed time across four jobs, not CPU-core usage.',
                               'Saved scheduler observations are historical; no fresh cluster poll was needed.',
                               'No total-project time, monetary cost, carbon or energy estimate is inferred.'],
               'builder': source('manuscript/build_review.py')}
    write('compute_accounting_20261016.json', compute)

    receipt_name = 'empirical_execution/phase10/results/finite_word_review_final.json'
    receipt = read(receipt_name)
    need(receipt['open_mathematical_findings'] == [], 'historical review open findings')
    verified = []
    for group in ['source_bindings', 'prior_phase9_context_bindings']:
        for name, expected in receipt[group].items():
            actual = bind(ROOT / name)
            need(actual == expected, 'frozen review source changed: ' + name)
            verified.append({'path': name, **actual})
    tex = (HERE / 'third_working_draft.tex').read_text()
    need('all surviving inverse references remain attached to that handle' in tex, 'stable handle description')
    need('rejects repeated, already deleted, invalid, and over-budget IDs' in tex, 'empirical retry semantics')
    need('earlier output, or external store counts toward the private state' in tex, 'private side information')
    need('magnitude at least $2^{-89}$' in tex, 'cancellation range argument')
    need('19,045 GPU-allocation seconds (5.29 hours)' in tex and '1,469 seconds, not CPU-core time' in tex,
         'compute statement matches ledger')
    implementation = (ROOT / 'empirical_execution/ccu/summary.py').read_text()
    need('Surviving inverse references already point to this entry.' in implementation, 'source handle semantics')
    need('duplicate unit IDs within request' in implementation and 'request contains already deleted units' in implementation,
         'source retry semantics')
    review = {'schema': 'third-unified-manuscript-author-claim-review-v1', 'created_utc': now,
              'reviewer_kind': 'Author AI assistant source comparison and receipt arithmetic; not a new independent manuscript review.',
              'review_passed_in_stated_scope': True, 'new_independent_proof_review': False,
              'historical_independent_review': source(receipt_name),
              'historical_source_bindings_reverified': verified,
              'manuscript': source('manuscript/third_working_draft.tex'),
              'builder': source('manuscript/build_review.py'),
              'additional_sources': [source(n) for n in [
                  'output/pdf/counterfactual_curation_theory.tex', 'empirical_execution/ccu/summary.py',
                  'manuscript/compute_accounting_20261016.json', terminal_name,
                  'manuscript/literature_contract_review_20261016.json']],
              'resolved_findings': [
                  {'id': 'R1', 'issue': 'Appendix description could imply rebuilding surviving inverse incidences, contradicting its token accounting.',
                   'resolution': 'Specify stable-handle moves; remove only the deleted incidence unless an entry is retired by merge/pruning.',
                   'source_locator': 'counterfactual_curation_theory.tex indexed algorithm; ccu/summary.py _delete_one'},
                  {'id': 'R2', 'issue': 'Draft conflated mathematical reference retry no-ops with the frozen empirical P-I interface.',
                   'resolution': 'State atomic rejection of duplicate/already-deleted IDs in empirical P-I and preserve the distinct earlier reference policy.',
                   'source_locator': 'ccu/summary.py IndexedSummary.delete; counterfactual_curation_theory.tex algorithm semantics'},
                  {'id': 'R3', 'issue': 'The private-state access model was less explicit than the frozen proof.',
                   'resolution': 'Charge all usable sign-dependent auxiliary stores/outputs and forbid uncharged retained-candidate rereads; fix public contract independently.',
                   'source_locator': 'FINITE_WORD_THEORY.md Section2 and FINITE_WORD_REVIEW.md exact finite-state identification'},
                  {'id': 'R4', 'issue': 'Scorer appendix asserted ordinary-range intermediates without the frozen cancellation-grid argument.',
                   'resolution': 'Include product and accumulation grid bounds, exact-zero exception and overflow bound.',
                   'source_locator': 'SCORER_TRANSFER.md Bound every remaining FP64 operation'}],
              'checked_claim_boundaries': [
                  'Earlier raw-neighbor suppression; fixed representation, graph eligibility, order and threshold.',
                  'Truncated substitution uses remaining cumulative budget and exact arithmetic.',
                  'Indexed work excludes construction, decoding, output serialization and arbitrary-precision bit costs.',
                  'Ridge uses stored raw features, selected-count regularization, no intercept and prescribed empty head.',
                  'Finite family has public binary labels/blocker payloads and bounded FP32 dimension regime.',
                  'Exact mD and fixed-F e(F)D count private information; metadata, padding, outputs and workspace remain additional.',
                  'Expected-error bound is on hard probes; no constant natural task-risk conclusion.',
                  'Finite-output bridge requires an actual exact-target certificate, absent for the natural learner.',
                  'Both calibration negatives, invalid denominators and old human gates remain visible.',
                  'Completed v4 failure blocks downstream experiments; no effect, speed or total-byte advantage is claimed.',
                  'Partial compute record counts each saved parent job once and includes the failed audit.'],
              'remaining_review_limits': [
                  'No proof assistant or new independent unified manuscript review.',
                  'Visual layout remains unverified; native computer-use service was unavailable at the earlier attempt.',
                  'Two close-prior full-text comparisons remain unresolved; no exhaustive novelty clearance.',
                  'No complete project compute/energy budget or raw-data redistribution clearance.',
                  'Original primary empirical study remains unexecuted and blocked under the failed configuration.'],
              'frozen_sources_changed': False, 'new_empirical_experiments': 0,
              'human_annotation_calls': 0, 'submission_ready': False}
    write('claims_review_20261016.json', review)
    print(json.dumps({'review_passed_in_stated_scope': True, 'resolved_description_findings': 4,
                      'frozen_source_bindings_verified': len(verified),
                      'gpu_allocation_seconds': gpu_seconds, 'cpu_job_elapsed_seconds_sum': cpu_seconds}))


if __name__ == '__main__':
    main()
