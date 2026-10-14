#!/usr/bin/env python3
"""Fixed-array metadata gate. Never reads vector/chunk-log contents or submits jobs."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CAP = 25 * 1024**2
PER_FILE_CAP = 4 * 1024**2
FIELDS = ('JobID', 'JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'MaxRSS', 'NodeList')
SCHEMA = 'third-e5-full-array-metadata-gate-v1'
BUNDLE_SCHEMA = 'third-e5-full-array-metadata-bundle-v1'
FALSE_FLAGS = ('source_authenticity_verified', 'primary_input_acceptance', 'original_route_acceptance',
               'full_population_cache_acceptance', 'confirmatory_study_ready')
DISPATCH_FALSE_FLAGS = ('original_route_acceptance', 'full_population_cache_acceptance',
                        'confirmatory_study_ready', 'automatic_job_submission')
CONTRACT = {'attention_implementation': 'eager', 'deterministic_algorithms': True, 'device': 'cpu',
    'dtype': 'float32', 'input_pad_length': 261, 'interop_threads': 1, 'max_length': 512,
    'mkldnn': False, 'seed': 0, 'threads': 1, 'within_document_chunk_batch_size': 8}


def require(ok, message):
    if not ok: raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate_JSON_key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def binding(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def expectations():
    return strict_json((HERE / 'expectations.json').read_bytes())


def code_bindings():
    return {name: binding((HERE / name).read_bytes()) for name in ('gate.py', 'expectations.json')}


def seal(value):
    require(value.get('sha256') == binding(canonical({k: v for k, v in value.items() if k != 'sha256'}))['sha256'], 'invalid_internal_seal')


def safe_path(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == os.path.normpath(str(path)) and '..' not in path.parts, 'absolute_normalized_path_required')
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'symlink_path_prohibited')
    return path


def file_stat(path):
    path = safe_path(path); value = path.lstat()
    require(stat.S_ISREG(value.st_mode), 'regular_file_required')
    return {name: getattr(value, 'st_' + name) for name in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns')}


def read_small(path, cap=PER_FILE_CAP):
    before = file_stat(path); require(before['size'] <= cap, 'metadata_file_too_large')
    with Path(path).open('rb') as stream:
        opened = os.fstat(stream.fileno()); raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    signature = lambda value: {name: getattr(value, 'st_' + name) for name in before}
    require(before == signature(opened) == signature(after) == file_stat(path), 'metadata_changed_during_read')
    require(len(raw) == before['size'], 'metadata_read_size_mismatch')
    return {**binding(raw), 'base64': base64.b64encode(raw).decode(), 'stat': before}


def directory_inventory(path):
    path = safe_path(path); require(path.is_dir(), 'directory_required')
    for child in path.iterdir(): require(not child.is_symlink(), 'symlink_directory_member')
    return sorted(child.name for child in path.iterdir())


def scheduler_snapshot():
    start = datetime.now(timezone.utc).isoformat()
    queue = subprocess.check_output(['squeue', '-u', 'poonam', '-h', '-o', '%i|%T|%M|%R'], text=True, timeout=10)
    accounting = subprocess.check_output(['sacct', '-j', '13498', '-P', '-n', '--format=' + ','.join(FIELDS)], text=True, timeout=10)
    return {'observed_utc': start, 'queue_stdout': queue, 'accounting_stdout': accounting}


def verify_scheduler(value, pins=None):
    pins = pins or expectations(); job = pins['array_job_id']
    active = [line for line in value['queue_stdout'].splitlines()
              if line.split('|', 1)[0].strip() == job or line.split('|', 1)[0].strip().startswith(job + '_')]
    require(not active, 'array_still_queued_or_running')
    rows = []
    for line in value['accounting_stdout'].splitlines():
        fields = line.split('|')
        if re.fullmatch(re.escape(job) + r'_\d+', fields[0]):
            require(len(fields) in (len(FIELDS), len(FIELDS) + 1), 'bad_accounting_columns')
            rows.append(dict(zip(FIELDS, fields)))
    require(len(rows) == 32 and {r['JobID'] for r in rows} == {f'{job}_{i}' for i in range(32)}, 'exact_32_unique_array_tasks_required')
    require(len({r['JobIDRaw'] for r in rows}) == 32 and all(re.fullmatch(r'\d+', r['JobIDRaw']) for r in rows), 'unique_physical_job_ids_required')
    require(all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' and r['NodeList'] == pins['qualified_node'] for r in rows), 'terminal_success_on_qualified_node_required')
    return {int(row['JobID'].split('_')[-1]): row for row in rows}


def paths(pins):
    root, repo, workspace = pins['root'], pins['repository'], pins['workspace']
    return {'plan': root + '/plan-run/output/plan.json', 'qualification': workspace + '/qualification/e5-sharded-cpu-v2-20261013-a01/qualification/result.json',
        'repair': workspace + '/qualification/e5-scipy-import-repair-v2-20261011-a01/repair/result.json',
        'budget': repo + '/local_run/local_llm_execution_20261013/e5_full_output_budget_v2.json',
        'core': repo + '/local_run/e5_sharded_cpu_v2_20261013/package_manifest.json',
        'dispatch': repo + '/local_run/e5_sharded_dispatch_v2_20261013/package_manifest.json'}


def collect(bundle, pins=None):
    pins = pins or expectations(); root = pins['root']; names = paths(pins)
    wanted = set(pins['pinned_files']) | {root + '/' + name for name in (
        'array_launch_receipt.json', 'array_launch_intent.json', 'plan-run/launch.json',
        'plan-run/dispatch_result.json', 'plan-run/plan.process.json', 'plan-run/output/result.json')}
    def add(path):
        item = read_small(path)
        require(sum(row['bytes'] for row in bundle['metadata'].values()) + item['bytes'] <= CAP, 'metadata_collection_exceeds_cap')
        bundle['metadata'][path] = item
    for path in sorted(wanted): add(path)
    for key in ('core', 'dispatch'):
        manifest = strict_json(base64.b64decode(bundle['metadata'][names[key]]['base64']))
        for member in manifest['files']:
            require(Path(member).name == member and member not in ('.', '..'), 'unsafe_package_member')
            path = str(Path(names[key]).parent / member); add(path)
    shard_root = root + '/shards'; bundle['inventories'][shard_root] = directory_inventory(shard_root)
    for i in range(32):
        folder = shard_root + f'/shard-{i:04d}'
        bundle['inventories'][folder] = directory_inventory(folder)
        for name in ('launch.json', 'dispatch_result.json', 'encode.process.json', 'replay.process.json'):
            path = folder + '/' + name; add(path)
        for mode in ('encode', 'replay'):
            directory = folder + '/' + mode; bundle['inventories'][directory] = directory_inventory(directory)
            path = directory + '/completion.json'; add(path)
            for filename in ('vectors.npy', 'chunk_log.jsonl'):
                path = directory + '/' + filename; bundle['payload_stats'][path] = file_stat(path)
            for channel in ('stdout', 'stderr'):
                path = folder + f'/{mode}.{channel}.txt'; bundle['log_stats'][path] = file_stat(path)
    require(len(canonical(bundle)) <= CAP, 'bundle_exceeds_25MiB')


def verify_bundle(bundle, pins=None):
    pins = pins or expectations(); root = pins['root']; names = paths(pins)
    require(bundle['schema'] == BUNDLE_SCHEMA and bundle['root'] == root and bundle['array_job_id'] == pins['array_job_id'], 'wrong_bundle_identity')
    require(bundle['code_bindings'] == code_bindings(), 'metadata_gate_code_changed')
    scheduler = verify_scheduler(bundle['scheduler'], pins)
    blobs = {}
    for path, item in bundle['metadata'].items():
        require(isinstance(path, str) and path == os.path.normpath(path) and path.startswith('/'), 'unsafe_metadata_path')
        raw = base64.b64decode(item['base64'], validate=True)
        require(len(raw) <= PER_FILE_CAP and binding(raw) == {k: item[k] for k in ('bytes', 'sha256')}, 'metadata_binding_mismatch')
        require(item['stat']['size'] == len(raw), 'metadata_stat_size_mismatch'); blobs[path] = raw
    require(sum(len(raw) for raw in blobs.values()) <= CAP, 'decoded_metadata_exceeds_cap')
    get = lambda path: strict_json(blobs[path])
    bind = lambda path: binding(blobs[path])
    expected_metadata = set(pins['pinned_files']) | {root + '/' + name for name in (
        'array_launch_receipt.json', 'array_launch_intent.json', 'plan-run/launch.json',
        'plan-run/dispatch_result.json', 'plan-run/plan.process.json', 'plan-run/output/result.json')}
    for path, wanted in pins['pinned_files'].items(): require(bind(path) == wanted, 'fixed_metadata_pin_mismatch')
    for key in ('core', 'dispatch'):
        manifest = get(names[key]); require(bool(manifest['files']), 'empty_package_inventory')
        for member, wanted in manifest['files'].items():
            require(Path(member).name == member and member not in ('.', '..'), 'unsafe_package_member')
            path = str(Path(names[key]).parent / member); expected_metadata.add(path)
            require(bind(path) == wanted, 'sealed_package_member_mismatch')
    plan, qualification, repair, budget = [get(names[key]) for key in ('plan', 'qualification', 'repair', 'budget')]
    prior = get(root + '/plan_launch_receipt.json'); launch = get(root + '/array_launch_receipt.json')
    require(binding(canonical(launch))['sha256'] == pins['array_launch_canonical_sha256'], 'array_launch_identity_mismatch')
    require(get(root + '/array_launch_intent.json') == {k: v for k, v in launch.items() if k not in ('returncode', 'stdout', 'stderr', 'job_id')}, 'array_launch_intent_mismatch')
    require(launch['job_id'] == pins['array_job_id'] and launch['returncode'] == 0 and launch['qualified_node'] == pins['qualified_node'], 'array_launch_failed_or_wrong_node')
    require(qualification['passed'] is True and qualification['stage'] == 'complete' and repair['passed'] is True, 'qualification_or_repair_failed')
    seal(plan); seal(qualification)
    require(plan['kind'] == 'full' and plan['documents'] == 140250 and plan['frame_row_indices'] == list(range(140250)) and len(plan['shards']) == 32, 'complete_full_frame_required')
    require(plan['fingerprint'] == qualification['fingerprint'] == pins['fingerprint'], 'wrong_runtime_fingerprint')
    require(plan['qualification_result'] == bind(names['qualification']), 'plan_qualification_binding_mismatch')
    require(all(plan[key] is False for key in ('original_monolithic_writer_executed', 'original_monolithic_replay_executed', 'full_population_cache_acceptance', 'confirmatory_study_ready', 'guaranteed_walltime_bound')), 'plan_claim_boundary_changed')
    expected_bindings = {'core_package': bind(names['core']), 'dispatch_package': bind(names['dispatch']),
                         'qualification_result': bind(names['qualification']), 'full_plan': bind(names['plan'])}
    for key, target in [('full_plan_sha256', 'plan'), ('qualification_sha256', 'qualification'), ('scipy_repair_result_sha256', 'repair'), ('storage_budget_report_sha256', 'budget')]:
        require(launch[key] == bind(names[target])['sha256'], 'array_dependency_binding_mismatch')
    require(prior['core_package_sha256'] == expected_bindings['core_package']['sha256'] and prior['dispatch_package_sha256'] == expected_bindings['dispatch_package']['sha256'], 'plan_package_mismatch')
    plan_result = get(root + '/plan-run/output/result.json'); seal(plan_result)
    require(plan_result['passed'] is True and plan_result['plan_sha256'] == plan['sha256'], 'plan_result_failed')
    plan_dispatch = get(root + '/plan-run/dispatch_result.json'); plan_child = get(root + '/plan-run/plan.process.json')
    require(plan_dispatch['passed'] is True and plan_dispatch['mode'] == 'plan' and plan_child['returncode'] == 0, 'plan_dispatch_failed')
    require(plan_dispatch['children'][0]['artifact'] == bind(root + '/plan-run/output/result.json') and plan_dispatch['children'][0]['process_receipt'] == bind(root + '/plan-run/plan.process.json'), 'plan_child_binding_mismatch')
    common = prior['common_arguments']; require(len(common) == 10 and common[0] == pins['repository'] and common[1] == pins['workspace'] and common[-1] == root, 'common_path_arguments_mismatch')
    worker_common = [v for name, value in zip(('repo','workspace','view','replay-result','census-dir','budget-dir'), common[:6]) for v in ('--'+name, value)]
    worker_common += ['--package-sha256', expected_bindings['core_package']['sha256'], '--qualification-result', names['qualification'], '--qualification-sha256', expected_bindings['qualification_result']['sha256']]
    shard_root = root + '/shards'; inventories = {shard_root: [f'shard-{i:04d}' for i in range(32)]}
    wanted_payloads, wanted_logs, stop, totals = set(), set(), 0, {'rows': 0, 'chunks': 0, 'model_forward_batches': 0, 'content_tokens': 0}
    budget_shards = budget['per_shard_payload']; require(len(budget_shards) == 32, 'budget_shard_count_mismatch')
    for i, shard in enumerate(plan['shards']):
        require(all(type(shard[k]) is int for k in ('index','start','stop')) and shard['index'] == i and shard['start'] == stop and stop < shard['stop'] <= 140250, 'shard_gap_overlap_or_order')
        stop = shard['stop']; folder = shard_root + f'/shard-{i:04d}'
        inventories[folder] = sorted(['launch.json','dispatch_result.json','encode.process.json','replay.process.json','encode.stdout.txt','encode.stderr.txt','replay.stdout.txt','replay.stderr.txt','encode','replay'])
        launch_path, dispatch_path = folder + '/launch.json', folder + '/dispatch_result.json'
        expected_metadata.update((launch_path, dispatch_path)); observed_launch, dispatch = get(launch_path), get(dispatch_path)
        require(observed_launch['mode'] == dispatch['mode'] == 'shard' and type(observed_launch['shard_index']) is int and type(dispatch['shard_index']) is int and observed_launch['shard_index'] == dispatch['shard_index'] == i, 'shard_launch_index_mismatch')
        require(observed_launch['hostname'] == pins['qualified_node'] and observed_launch['slurm_job_id'] == scheduler[i]['JobIDRaw'], 'shard_physical_job_or_node_mismatch')
        require(observed_launch['bindings'] == dispatch['bindings'] == expected_bindings and dispatch['passed'] is True, 'shard_dispatch_binding_or_failure')
        require(all(value.get(key) is False for value in (observed_launch, dispatch) for key in DISPATCH_FALSE_FLAGS), 'dispatch_claim_boundary_changed')
        require([x['name'] for x in dispatch['children']] == ['encode','replay'], 'exact_encode_then_replay_children_required')
        counts = budget_shards[i]; require(counts['index'] == i and counts['start'] == shard['start'] and counts['stop'] == shard['stop'], 'budget_range_mismatch')
        pair = []; processes = []
        for j, mode in enumerate(('encode','replay')):
            directory = folder + '/' + mode; inventories[directory] = ['chunk_log.jsonl','completion.json','vectors.npy']
            completion_path, process_path = directory + '/completion.json', folder + f'/{mode}.process.json'
            expected_metadata.update((completion_path, process_path)); completion, process = get(completion_path), get(process_path)
            require(dispatch['children'][j]['artifact'] == bind(completion_path) and dispatch['children'][j]['process_receipt'] == bind(process_path), 'dispatch_child_hash_mismatch')
            command = [pins['workspace']+'/envs/encoder-cpu/bin/python',pins['repository']+'/local_run/e5_sharded_cpu_v2_20261013/worker.py', *worker_common, '--plan',names['plan'],'--shard-index',str(i),'--mode',mode,'--out',directory]
            if mode == 'replay': command += ['--encoded-dir',folder+'/encode']
            require(process['schema'] == 'third-e5-sharded-dispatch-child-v1' and process['returncode'] == 0 and process['command'] == command, 'child_exit_or_command_mismatch')
            require(all(process.get(key) is False for key in DISPATCH_FALSE_FLAGS), 'child_claim_boundary_changed')
            require(type(process['child_pid']) is int and process['child_pid'] > 0 and type(process['launcher_pid']) is int and process['launcher_pid'] > 0, 'child_PID_required')
            seconds = process['child_wall_seconds_through_exit']
            require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0 and seconds == dispatch['children'][j]['child_wall_seconds_through_exit'] and process['timer_includes_child_artifact_publication_and_exit'] is True, 'child_wall_time_missing_or_mismatch')
            identity = {'schema':'third-e5-cpu-shard-completion-v1','kind':mode,'passed':True,'shard_index':i,'start':shard['start'],'stop':shard['stop'],'rows':shard['stop']-shard['start'],'plan_sha256':plan['sha256'],'fingerprint':pins['fingerprint'],'qualification_result':bind(names['qualification'])}
            require(completion.get('passed') is True and all(completion.get(key) == value for key,value in identity.items()) and all(type(completion.get(key)) is int for key in ('shard_index','start','stop','rows')), 'completion_identity_mismatch')
            require(canonical(completion['runtime']) == canonical(qualification['runtime']) and canonical(completion['numerical_contract']) == canonical(CONTRACT), 'completion_runtime_changed')
            require(all(completion.get(key) is False for key in FALSE_FLAGS), 'completion_claim_boundary_changed')
            require(all(type(completion.get(key)) is int and completion[key] == counts[other] for key,other in [('content_tokens','content_tokens'),('chunks','chunks'),('model_forward_batches','batches')]), 'completion_counters_mismatch')
            require(set(completion['files']) == {'vectors.npy','chunk_log.jsonl'}, 'completion_file_inventory_mismatch')
            for filename, sizekey in [('vectors.npy','vectors_npy_bytes'),('chunk_log.jsonl','chunk_log_bytes')]:
                path = directory + '/' + filename; wanted_payloads.add(path); claim = completion['files'][filename]
                require(set(claim) == {'bytes','sha256'} and type(claim['bytes']) is int and claim['bytes'] == counts[sizekey] and re.fullmatch('[0-9a-f]{64}', claim['sha256']) is not None, 'invalid_payload_hash_reference_or_size')
                require(bundle['payload_stats'][path]['size'] == claim['bytes'], 'actual_payload_size_mismatch')
            for channel in ('stdout','stderr'): wanted_logs.add(folder + f'/{mode}.{channel}.txt')
            pair.append(completion); processes.append(process)
        require(processes[0]['child_pid'] != processes[1]['child_pid'] and processes[0]['launcher_pid'] == processes[1]['launcher_pid'], 'fresh_process_pair_required')
        require(pair[1]['encoded_completion'] == bind(folder+'/encode/completion.json') and pair[1]['exact_vector_bytes_equal'] is True and pair[1]['complete_chunk_log_equal'] is True and pair[0]['files'] == pair[1]['files'], 'replay_claim_or_encoded_binding_mismatch')
        for key in totals: totals[key] += pair[0][key]
    require(stop == 140250 and totals == {'rows':140250,'chunks':312159,'model_forward_batches':147825,'content_tokens':61576539}, 'full_frame_totals_mismatch')
    require(set(blobs) == expected_metadata and bundle['inventories'] == inventories, 'extra_or_missing_metadata_or_directory_inventory')
    require(set(bundle['payload_stats']) == wanted_payloads and set(bundle['log_stats']) == wanted_logs, 'extra_or_missing_payload_or_log_stats')
    return {'schema':SCHEMA,'passed':True,'array_job_id':pins['array_job_id'],'root':root,
        'terminal_task_count':32,'dispatch_receipt_count':32,'child_process_receipt_count':64,'completion_receipt_count':64,
        'metadata_file_count':len(blobs),'payload_file_count':len(wanted_payloads), 'counts_per_pass':totals,
        'full_plan_file':bind(names['plan']),'core_package':bind(names['core']),'dispatch_package':bind(names['dispatch']),
        'qualification_result':bind(names['qualification']),'scipy_repair_result':bind(names['repair']),
        'fingerprint':pins['fingerprint'],'heavy_vector_or_log_hashes_verified':False,
        'requires_unchanged_slurm_assembly_validation':True,'full_population_cache_acceptance':False,'confirmatory_study_ready':False,
        'scope':'Terminal scheduler and metadata bindings only. Payload hashes are recorded claims with actual file sizes/identities; payload contents, numerical equality and source admission are not independently verified here.'}


def recheck_observations(bundle):
    verify_bundle(bundle)
    for path, item in bundle['metadata'].items(): require(read_small(path) == item, 'metadata_changed_after_snapshot')
    for key in ('payload_stats','log_stats'):
        for path, observed in bundle[key].items(): require(file_stat(path) == observed, 'payload_or_log_changed_after_snapshot')
    for path, names in bundle['inventories'].items(): require(directory_inventory(path) == names, 'directory_changed_after_snapshot')


def write_json(path, value):
    raw = canonical(value) + b'\n'; require(len(raw) <= CAP, 'output_exceeds_25MiB')
    with Path(path).open('xb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return binding(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args(); pins = expectations()
    out = safe_path(args.out)
    require(out.parent == Path(pins['root']) and re.fullmatch(r'terminal-metadata-audit-v1-a\d{2}', out.name), 'fixed_root_new_audit_output_required')
    require(not out.exists(), 'audit_output_must_not_exist'); os.umask(0o077); out.mkdir()
    bundle = {'schema':BUNDLE_SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),'root':pins['root'],
        'array_job_id':pins['array_job_id'],'code_bindings':code_bindings(),'metadata':{},'payload_stats':{},'log_stats':{},'inventories':{}}
    result = {'schema':SCHEMA,'passed':False,'array_job_id':pins['array_job_id'],'root':pins['root'],
        'heavy_vector_or_log_hashes_verified':False,'full_population_cache_acceptance':False,'confirmatory_study_ready':False}
    begun = time.monotonic()
    def timeout(_signum, _frame): raise TimeoutError('metadata_gate_exceeded_180_seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(180)
    try:
        bundle['scheduler'] = scheduler_snapshot(); verify_scheduler(bundle['scheduler'])
        collect(bundle); result = verify_bundle(bundle); recheck_observations(bundle)
        verify_scheduler(scheduler_snapshot())
    except Exception as error:
        result.update(passed=False,error_type=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0)
    result['bundle_file'] = {'name':'bundle.json', **write_json(out/'bundle.json',bundle)}
    result['gate_seconds'] = time.monotonic()-begun; result['created_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(out/'metadata_gate_result.json',result)
    print(json.dumps(result,sort_keys=True)); return 0 if result['passed'] else 2


if __name__ == '__main__': sys.exit(main())
