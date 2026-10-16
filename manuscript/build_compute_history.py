#!/usr/bin/env python3
"""Reconcile saved project receipts without cluster access or job submission.

Default mode uses public reports and a minimal private-receipt extract. Optional
--refresh-private-extract rechecks the original local receipts before exporting
only project-owned job rows and resource counts, never global queue observations.
This is an author accounting check, not a new independent scientific review.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PREFIX = 'local_run/local_llm_execution_'
EXTRACT = 'manuscript/compute_receipts_extract_20261016.json'
OUTPUT = 'manuscript/compute_history_20261016.json'


def bind(name):
    data = (ROOT / name).read_bytes()
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(name):
    return json.loads((ROOT / name).read_text())


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def at(value, pointer):
    for key in pointer.split('/') if pointer else []:
        value = value[key]
    return value


def refresh_extract():
    report = read(PREFIX + '20261014/e5_array_terminal_metadata_result.json')
    name = 'local_workspace/reports/e5-full-sharded-v2-20261013/a01-terminal-metadata/bundle.json'
    binding = bind(name)
    assert {k: binding[k] for k in ['bytes', 'sha256']} == report['raw_bundle']
    data = read(name)
    assert str(data['array_job_id']) == '13498'
    rows = []
    for line in data['scheduler']['accounting_stdout'].splitlines():
        fields = line.split('|')
        if not re.fullmatch(r'13498_\d+', fields[0]):
            continue  # Never add .batch/.extern or unrelated job observations.
        assert len(fields) == 7 and fields[2:4] == ['COMPLETED', '0:0']
        rows.append({'array_task_id': fields[0], 'job_id_raw': fields[1],
                     'state': fields[2], 'exit_code': fields[3],
                     'elapsed_seconds': int(fields[4]), 'node': fields[6]})
    assert {r['array_task_id'] for r in rows} == {f'13498_{i}' for i in range(32)}
    assert len(rows) == len({r['job_id_raw'] for r in rows}) == 32
    resources = []
    for directory, expected_id, mode in [
        ('annotation-pilot-v5-20261011/a01', '13353', 'Qwen successful engineering pilot'),
        ('llm-only-v3-selection-qwen-a01', '13574', 'Qwen v3 selection'),
        ('llm-only-v3-selection-mistral-a01', '13625', 'Mistral v3 selection')]:
        relative = 'local_workspace/reports/' + directory + '/launch_receipt.json'
        launch = read(relative)
        assert str(launch.get('job_id', launch.get('slurm_job_id'))) == expected_id
        assert launch['returncode'] == 0
        command = launch['command']
        assert isinstance(command, list) and command[0] == 'sbatch'
        extra = None
        if expected_id == '13353':
            pilot = read(PREFIX + '20261011/annotation_v5_result.json')
            assert bind(relative)['sha256'] == pilot['source_bindings']['launch_receipt.json']
            script = 'local_run/model_annotation_runtime_v5_20261011/pilot.sbatch'
            assert any(arg.endswith('/' + script) for arg in command)
            assert not any(arg.startswith('--gres') for arg in command)
            assert '#SBATCH --gres=gpu:a100-80gb:1\n' in (ROOT / script).read_text()
            extra = bind(script)
        else:
            assert '--gres=gpu:a100-80gb:1' in command
        resources.append({'job_id': expected_id, 'role': mode, 'gpu_count': 1,
                          'gpu_type': 'A100 80GB', 'private_launch_binding': bind(relative),
                          'public_script_binding': extra})
    save(EXTRACT, {'schema': 'third-minimal-historical-compute-extract-v1',
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'extractor': bind('manuscript/build_compute_history.py'),
                  'review_kind': 'author extraction of saved local receipts',
                  'array_bundle_binding': binding,
                  'array_observed_utc': data['scheduler']['observed_utc'],
                  'array_rows': rows, 'model_launch_resources': resources,
                  'private_queue_observations_exported': False,
                  'private_source_text_exported': False, 'new_cluster_queries': 0})


def build():
    sources = set()
    jobs = {}

    def load(name):
        sources.add(name)
        return read(name)

    def add(job_id, state, exit_code, seconds, role, group, name, pointer, **extras):
        job_id = str(job_id)
        assert job_id not in jobs, 'duplicate parent/task job: ' + job_id
        assert re.fullmatch(r'\d+(?:_\d+)?', job_id), job_id
        assert state in {'COMPLETED', 'FAILED'}
        assert isinstance(seconds, int) and not isinstance(seconds, bool) and seconds >= 0
        assert (exit_code == '0:0') == (state == 'COMPLETED')
        jobs[job_id] = dict(job_id=job_id, state=state, exit_code=exit_code,
                            elapsed_seconds=seconds, role=role, group=group,
                            evidence={'path': name, 'pointer': pointer}, **extras)

    def structured(date, filename, pointer, role, group='other_stage', **extras):
        name = PREFIX + date + '/' + filename + '.json'
        item = at(load(name), pointer)
        def pick(*keys):
            return next(item[k] for k in keys if k in item)
        add(pick('job_id', 'id'), pick('scheduler_state', 'state'),
            pick('scheduler_exit_code', 'exit_code', 'exit'),
            pick('scheduler_elapsed_seconds', 'elapsed_seconds'),
            role, group, name, pointer, **extras)

    def accounting(date, filename, pointer, expected_id, role, group='other_stage', **extras):
        name = PREFIX + date + '/' + filename + '.json'
        records = [line.split('|') for line in at(load(name), pointer).splitlines()
                   if line.split('|')[0] == expected_id]
        assert len(records) == 1
        row = records[0]
        seconds = int(row[4] if ':' in row[3] else row[3])
        add(row[0], row[1], row[2], seconds, role, group, name, pointer, **extras)

    extract = load(EXTRACT)
    assert extract['extractor'] == bind('manuscript/build_compute_history.py')
    launch_resources = {r['job_id']: r for r in extract['model_launch_resources']}
    for item in launch_resources.values():
        assert item['gpu_count'] == 1 and item['gpu_type'] == 'A100 80GB'
        if item['public_script_binding']:
            source = item['public_script_binding']
            assert source == bind(source['path'])
            sources.add(source['path'])

    # Preserve the existing six-job v4 ledger as an explicitly included subset.
    name = 'manuscript/compute_accounting_20261016.json'
    v4 = load(name)
    for i, row in enumerate(v4['jobs']):
        gpu = row.get('allocated_gpu_count')
        extras = {'gpu_count': gpu, 'gpu_type': row['allocated_gpu_type']} if gpu else {}
        add(row['job_id'], row['state'], row['exit_code'], row['elapsed_seconds'],
            row['role'], 'model_inference' if gpu else 'other_stage', name, f'jobs/{i}', **extras)

    accounting('20261011', 'annotation_v5_result', 'scheduler_accounting', '13353',
               'Qwen successful engineering pilot', 'model_inference', gpu_count=1, gpu_type='A100 80GB')
    pilot = load(PREFIX + '20261013/mistral_pilot_result.json')['job']
    assert pilot['gpu'] == 'one A100 80GB'
    structured('20261013', 'mistral_pilot_result', 'job', 'Mistral successful engineering pilot',
               'model_inference', gpu_count=1, gpu_type='A100 80GB')
    name = PREFIX + '20261014/llm_selection_v3_terminal_result.json'
    v3 = load(name)
    for key, row in v3['models'].items():
        assert row['job_id'] in launch_resources
        scheduler = row['scheduler']
        add(row['job_id'], scheduler['state'], scheduler['exit_code'], scheduler['elapsed_seconds'],
            key + ' v3 selection', 'model_inference', name, 'models/' + key + '/scheduler',
            gpu_count=1, gpu_type='A100 80GB')
    for key in ['qwen', 'mistral']:
        structured('20261014', 'llm_selection_v3_terminal_result', 'output_audits/' + key, key + ' v3 output audit')
    structured('20261014', 'llm_selection_v3_terminal_result', 'statistics_job', 'v3 selection statistics')

    for date, filename, pointer, role in [
        ('20261011', 'source_lineage_result', '', 'source lineage replay'),
        ('20261011', 'e5_forward_budget_result', '', 'E5 forward budget pilot'),
        ('20261011', 'mistral_acquisition_v2_result', 'job', 'Mistral asset acquisition'),
        ('20261011', 'e5_shard_qualification_a01_failure', '', 'failed E5 shard qualification'),
        ('20261011', 'e5_scipy_repair_v1_failure', '', 'failed SciPy environment repair'),
        ('20261013', 'e5_scipy_repair_v2_result', 'job', 'SciPy environment repair'),
        ('20261013', 'e5_shard_qualification_v1_a02_failure', '', 'failed second E5 shard qualification'),
        ('20261013', 'e5_cli_import_probe_result', '', 'E5 CLI import probe'),
        ('20261013', 'e5_shard_qualification_v2_result', 'job', 'E5 shard qualification and replay'),
        ('20261014', 'e5_pair_scorer_benchmark_result', '', 'pair-scorer resource benchmark'),
        ('20261014', 'e5_calibration_frame_independent_review', 'job', 'E5 calibration frame'),
        ('20261014', 'llm_selection_terminal_result', '', 'v3 pair census and sampling'),
        ('20261014', 'llm_requests_v2_terminal_result', '', 'v3 request preparation'),
        ('20261014', 'llm_inference_qualification_independent_review', '', 'v3 execution qualification'),
        ('20261014', 'llm_v4_qwen_audit_and_task_locks_20261016', 'native_task_configuration', 'native task calibration'),
        ('20261014', 'llm_development_intake_completed_20261016', '', 'development intake reservation'),
        ('20261014', 'llm_guard_benchmark_completed_20261016', '', 'guard resource pilot'),
        ('20261014', 'llm_node_pool_v2_execution', 'standalone_hardware_probe', 'A100 hardware probe; allocation count not added to six-model-job subtotal')]:
        structured(date, filename, pointer, role)
    for filename, job_id, role in [('source_replay_result', '13357', 'source preparation replay'),
                                   ('e5_token_census_result', '13358', 'E5 token census'),
                                   ('source_mirror_result', '13360', 'source mirror verification')]:
        accounting('20261011', filename, 'scheduler_accounting', job_id, role)
    for filename, pointer, job_id, role in [
        ('llm_tail_census_and_v4_sampling', 'local_review/accounting', '13632', 'tail score census'),
        ('llm_v4_execution_progress', 'sampling/accounting', '13636', 'v4 pair sampling'),
        ('llm_v4_execution_progress', 'requests/accounting', '13639', 'v4 request preparation'),
        ('llm_v4_execution_v1_superseded_snapshot', 'qualification/accounting', '13640', 'superseded execution qualification'),
        ('llm_node_pool_v2_execution', 'qualification/accounting', '13662', 'node-pool execution qualification')]:
        accounting('20261014', filename, pointer, job_id, role)
    name = PREFIX + '20261014/e5_assembly_terminal_metadata_result.json'
    parent = load(name)['terminal_parent']
    add(parent['JobIDRaw'], parent['State'], parent['ExitCode'], int(parent['ElapsedRaw']),
        'E5 cache assembly', 'other_stage', name, 'terminal_parent')

    name = PREFIX + '20261014/e5_array_terminal_metadata_result.json'
    array_report = load(name)
    assert {k: extract['array_bundle_binding'][k] for k in ['bytes', 'sha256']} == array_report['raw_bundle']
    launch = load(PREFIX + '20261013/e5_full_array_v2_launch.json')
    assert launch['job_id'] == '13498' and launch['cpus_per_task'] == 2
    rows = extract['array_rows']
    assert len(rows) == 32 and {r['array_task_id'] for r in rows} == {f'13498_{i}' for i in range(32)}
    raw_ids = {r['job_id_raw'] for r in rows}
    assert len(raw_ids) == 32 and not raw_ids.intersection(jobs)
    assert [min(r['elapsed_seconds'] for r in rows), max(r['elapsed_seconds'] for r in rows)] == array_report['independent_review']['scheduler_elapsed_seconds_range']
    for i, row in enumerate(rows):
        add(row['array_task_id'], row['state'], row['exit_code'], row['elapsed_seconds'],
            'E5 encode then fresh-process replay', 'embedding_array', EXTRACT, f'array_rows/{i}',
            job_id_raw=row['job_id_raw'], requested_cpus_per_task=2)

    by_group = {group: [r for r in jobs.values() if r['group'] == group]
                for group in ['model_inference', 'embedding_array', 'other_stage']}
    gpu_seconds = sum(r['gpu_count'] * r['elapsed_seconds'] for r in by_group['model_inference'])
    array_seconds = sum(r['elapsed_seconds'] for r in by_group['embedding_array'])
    summaries = {k: {'jobs': len(v), 'elapsed_seconds_sum': sum(r['elapsed_seconds'] for r in v)}
                 for k, v in by_group.items()}
    value = {'schema': 'third-expanded-historical-compute-v1',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'review_kind': 'author receipt reconciliation and arithmetic; no new independent audit',
             'scope': 'Explicit saved-job subset spanning successful model pilots, v3/v4 selection, E5 encode/replay and selected preparation/failure jobs.',
             'jobs': sorted(jobs.values(), key=lambda r: (int(r['job_id'].split('_')[0]), r['job_id'])),
             'job_count': len(jobs), 'state_counts': dict(Counter(r['state'] for r in jobs.values())),
             'groups': summaries, 'six_model_job_gpu_allocation_seconds': gpu_seconds,
             'six_model_job_gpu_allocation_hours_display': f'{gpu_seconds / 3600:.2f}',
             'embedding_array_elapsed_seconds': array_seconds,
             'embedding_array_elapsed_hours_display': f'{array_seconds / 3600:.2f}',
             'embedding_requested_cpu_times_elapsed_seconds': 2 * array_seconds,
             'embedding_requested_cpu_times_elapsed_hours_display': f'{2 * array_seconds / 3600:.2f}',
             'existing_v4_jobs_included_once': True, 'parent_and_batch_double_counting': False,
             'array_display_and_raw_ids_double_counted': False,
             'private_extract_replay_required_for_original_receipt_check': True,
             'source_bindings': [bind(n) for n in sorted(sources)],
             'builder': bind('manuscript/build_compute_history.py'),
             'project_total_compute_complete': False, 'new_cluster_queries': 0,
             'new_model_calls': 0, 'energy_or_utilization_measured': False,
             'limits': [
                 'Summed job/task elapsed time is not wall-clock project duration; concurrent tasks overlap.',
                 'GPU subtotal covers six successful model jobs only, not every hardware probe, failed pilot or earlier setup.',
                 'Two requested CPUs times elapsed is a reservation-derived quantity, not measured CPU busy time or verified billed core time.',
                 'Other-stage elapsed sums mix acquisition, preparation, audit and a hardware probe; they are not CPU-core or GPU-hours.',
                 'Queue waits, missing early setup/transfer and annotation-attempt receipts, local work, energy and monetary costs remain unaccounted.',
                 'Known excluded receipt gaps include acquisition13361 and planning13497 without terminal elapsed fields in the inspected public reports.',
                 'Cancelled pending13641 had zero recorded elapsed time and is outside the completed/failed-job subtotal.',
                 'No scientific outcome, model threshold, original gate or frozen protocol is changed by accounting.']}
    assert set(launch_resources) == {'13353', '13574', '13625'}
    assert len(by_group['model_inference']) == 6
    assert gpu_seconds == 19045 + 1603 + 887 + 2634 + 1590
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-private-extract', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert not (args.refresh_private_extract and args.check)
    if args.refresh_private_extract:
        refresh_extract()
    result = build()
    if args.check:
        original = read(OUTPUT)
        for item in [original, result]:
            item.pop('created_utc')
        assert original == result, 'saved historical compute ledger differs from source replay'
    else:
        save(OUTPUT, result)
    print(json.dumps({k: result[k] for k in ['job_count', 'state_counts', 'groups',
                     'six_model_job_gpu_allocation_seconds', 'embedding_array_elapsed_seconds',
                     'embedding_requested_cpu_times_elapsed_seconds', 'project_total_compute_complete']}))


if __name__ == '__main__':
    main()
