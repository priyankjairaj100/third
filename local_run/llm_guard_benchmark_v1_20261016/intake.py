"""Rebind real inputs and reserve whole sources before any development effects."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/nfs_home/users/poonam/ccu-acl2027/repo')
W = REPO.parent / 'local_workspace'
FRAME = W / 'qualification/e5-calibration-frame-v1-20261014-a01/run/frame'
SOURCE = W / 'qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu/records.jsonl'
CACHE = W / 'qualification/e5-full-sharded-v2-20261013-a01/assembly-run/cache'
TASK = W / 'qualification/askubuntu-task-calibration-v3-20261016-a01/run'
ROOT = W / 'qualification/llm-development-intake-v1-20261016-a01'
CAP = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def strict(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def binding(path):
    path = Path(path)
    require(path.is_file() and path.resolve() == path and not path.is_symlink(), 'regular_absolute_input')
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    require(all(getattr(before, k) == getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')), 'input_changed_during_hash')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def publish(path, value):
    raw = canonical(value) + b'\n'
    require(sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file()) + len(raw) <= CAP, 'aggregate_budget')
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()


def verify_package(pin):
    require(binding(HERE/'package_manifest.json')['sha256'] == pin, 'package_pin')
    manifest = strict((HERE/'package_manifest.json').read_bytes())
    for name, wanted in manifest['files'].items():
        require(Path(name).name == name and name != 'package_manifest.json', 'package_member_name')
        require(binding(HERE/name) == wanted, 'package_member:' + name)
    return manifest


def collect_metadata(source, panels, expected):
    """Decode rows transiently, export no bodies, tags or other task labels."""
    records = []; ids = set(); groups = {}; counts = Counter()
    with source.open('rb') as stream:
        for index, line in enumerate(stream):
            row = strict(line); rid = row.get('record_id'); text = row.get('text')
            require(isinstance(rid, str) and rid and rid not in ids, 'unique_record_identity')
            require(isinstance(text, str) and text and panels.normalized_text(text) == text, 'normalized_source_text')
            ids.add(rid)
            unit, kind = panels.source_unit(row, 'askubuntu', allow_preview_sources=False)
            part = panels.digest_partition(panels.hash_integer('ccu-v1-source-split', unit))
            require((unit, kind, part) == (row.get('source_unit_id'), row.get('source_kind'), row.get('partition')),
                    'native_source_partition_mismatch')
            require(unit not in groups or groups[unit] == (kind, part), 'source_crosses_partitions')
            groups[unit] = (kind, part); counts[part] += 1
            records.append({'original_full_row_index': index, 'record_id': rid,
                            'source_unit_id': unit, 'source_kind': kind, 'partition': part,
                            'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    require(len(records) == expected['full_rows'] and dict(counts) == expected['partition_counts'], 'complete_population')
    require(hashlib.sha256(canonical([r['record_id'] for r in records])).hexdigest() ==
            expected['source_ordered_ids_sha256'], 'source_order')
    return records


def reserve_role(partition, digest):
    require(type(digest) is int and 0 <= digest < 2**256, 'unsigned_digest')
    require(partition in ('train', 'calibration', 'test'), 'known_partition')
    if partition == 'calibration':
        return 'calibration'
    reserved = digest * 10 < 2**256
    return ('development_' if reserved else 'evaluation_') + partition


def whole_prefix(records, target, salt, panels):
    require(type(target) is int and target > 0, 'positive_target')
    groups = defaultdict(list)
    for row in records:
        groups[row['source_unit_id']].append(row['record_id'])
    chosen = []; units = []
    for unit in sorted(groups, key=lambda u: (panels.hash_integer(salt, u), u)):
        if len(chosen) >= target:
            break
        units.append(unit); chosen.extend(sorted(groups[unit]))
    return {'target': target, 'actual': len(chosen), 'shortfall': max(0, target-len(chosen)),
            'overshoot': max(0, len(chosen)-target), 'record_ids': chosen, 'source_ids': units}


def allocate(records, plan, panels):
    require(plan['development_reserve_fraction'] == [1,10], 'fixed_reserve_fraction')
    groups = {}; seen = set(); byrole = defaultdict(list)
    for row in records:
        require(row['record_id'] not in seen, 'duplicate_allocation_record')
        seen.add(row['record_id'])
        unit = row['source_unit_id']
        role = reserve_role(row['partition'], panels.hash_integer(plan['development_reserve_salt'], unit))
        require(unit not in groups or groups[unit] == role, 'source_role_overlap')
        groups[unit] = role
        byrole[role].append({**row, 'role': role,
                            'genuine_source_request_eligible': row['source_kind'].startswith('native_')})
    roles = ('development_train', 'evaluation_train', 'calibration', 'development_test', 'evaluation_test')
    output = [r for role in roles for r in byrole[role]]
    output.sort(key=lambda r: r['original_full_row_index'])
    summary = {role: {'rows': len(byrole[role]),
                     'source_units': len({r['source_unit_id'] for r in byrole[role]}),
                     'unknown_singleton_rows': sum(r['source_kind'] == 'unknown_singleton' for r in byrole[role])}
               for role in roles}
    prefixes = {str(n): whole_prefix(byrole['development_train'], n, plan['development_order_salt'], panels)
                for n in plan['development_targets']}
    return output, summary, prefixes


def main():
    p = argparse.ArgumentParser(); p.add_argument('--package-sha256', required=True); args = p.parse_args()
    require(os.environ.get('SLURM_JOB_ID', '').isdigit() and socket.gethostname() in ('cn1','csis.cn1'), 'compute_node_only')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_single_thread')
    verify_package(args.package_sha256)
    e = strict((HERE/'expectations.json').read_bytes()); plan = strict((HERE/'PLAN.json').read_bytes())
    for name, wanted in e['repository_files'].items():
        require(binding(REPO/name) == wanted, 'frozen_source')
    sys.path.insert(0, str(REPO))
    import numpy as np
    from empirical_execution.phase3 import panels
    from empirical_execution.phase4 import model_selection as ms
    require(platform.python_version() == e['python'] and np.__version__ == e['numpy'], 'runtime')
    require(ROOT.is_dir() and ROOT.resolve() == ROOT, 'launched_root')
    out = ROOT/'run'; out.mkdir(mode=0o700, exist_ok=False); started = time.perf_counter()
    watched = {SOURCE:e['source'], FRAME/'result.json':e['frame_result'],
               CACHE/'manifest.json':e['assembly']['assembly_manifest'], TASK/'result.json':e['task_result']}
    watched.update({CACHE/name:expected for name, expected in e['assembly']['assembly_files'].items()})
    watched.update({TASK/name:expected for name, expected in e['task_outputs'].items()})
    for path, wanted in watched.items():
        require(binding(path) == wanted, 'input_binding:' + str(path))
    frame = strict((FRAME/'result.json').read_bytes()); task = strict((TASK/'result.json').read_bytes())
    qual = strict((HERE/'source_qualification.json').read_bytes())
    review = strict((HERE/'task_independent_review.json').read_bytes())
    require(qual['qualified_facts']['source_bound_v2_development_population_supported'] is True and
            qual['population']['ordered_record_ids_sha256'] == e['assembly']['source_ordered_ids_sha256'], 'documentary_source_scope')
    require(frame['passed'] is True and all(frame[k] is True for k in
            ('actual_cache_payload_hashes_verified','all_32_assembled_slices_exact','all_32_encoded_and_fresh_replay_payloads_verified')),
            'full_sharded_derivation_receipt')
    require(frame['source_cache_manifest'] == e['assembly']['assembly_manifest'] and
            frame['source_cache_files'] == e['assembly']['assembly_files'] and
            frame['prepared_source_records_file'] == e['source'], 'frame_full_source_bindings')
    require(task['passed'] is True and task['outputs'] == e['task_outputs'] and
            review['passed'] is True and review['bindings']['task_result.json'] == e['task_result'], 'native_task_review')
    locks = {name:strict((TASK/name).read_bytes()) for name in e['task_outputs']}
    for lock in locks.values():
        ms.verify(lock)
    require(locks['ridge.json']['vocabulary_lock_sha256'] == locks['vocabulary.json']['sha256'] and
            locks['thresholds.json']['lambda_lock_sha256'] == locks['ridge.json']['sha256'], 'linked_native_task_locks')
    records = collect_metadata(SOURCE, panels, e['assembly'])
    vectors = np.load(CACHE/'vectors.npy', mmap_mode='r', allow_pickle=False)
    require(vectors.dtype == np.dtype('float32') and vectors.shape == (len(records),768), 'full_cache_shape')
    for start in range(0,len(records),1024):
        block = vectors[start:start+1024]
        require(np.isfinite(block).all() and np.any(block != 0, axis=1).all(), 'finite_nonzero_vectors')
    allocated, counts, prefixes = allocate(records, plan, panels)
    design = strict((REPO/'output/empirical_program/study_design.json').read_bytes())
    candidates = [r for r in allocated if r['role'] == 'evaluation_train']
    capacity = {}
    for target in (10000,25000,100000,200000):
        _, info = panels.select_panels(candidates, design, target=target)
        capacity[str(target)] = {pool:{k:info[pool][k] for k in
            ('actual','shortfall','overshoot','available_rows','available_source_units')}
            for pool in ('primary','replication')}
    for path, wanted in watched.items():
        require(binding(path) == wanted, 'input_changed_before_publication')
    verify_package(args.package_sha256)
    publish(out/'allocation.private.json', allocated)
    publish(out/'development_prefixes.private.json', prefixes)
    result = {'schema':'third-llm-development-intake-v1','passed':True,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'job_id':os.environ['SLURM_JOB_ID'],'package_sha256':args.package_sha256,
        'plan_binding':binding(HERE/'PLAN.json'),'input_bindings':{str(k):v for k,v in watched.items()},
        'rows':len(records),'shape':list(vectors.shape),'allocation':counts,
        'pre_guard_development_prefix_counts':{n:{k:v[k] for k in ('target','actual','overshoot','shortfall')} for n,v in prefixes.items()},
        'pre_guard_evaluation_capacity_upper_bounds':capacity,
        'outputs':{n:binding(out/n) for n in ('allocation.private.json','development_prefixes.private.json')},
        'source_disjoint_reservations':True,'all_reserved_sources_excluded_from_evaluation':True,
        'source_body_or_task_labels_exported':False,'task_labels_used_for_allocation':False,
        'source_rows_JSON_decoded_transiently':True,'prior_full_shard_replay_receipt_bound':True,
        'full_cache_bytes_rehashed_this_stage':True,'full_shard_replay_reexecuted_this_stage':False,
        'development_threshold_accepted':False,'semantic_guard_executed':False,'post_guard_panels_frozen':False,
        'effect_experiments_started':False,'human_annotation_calls':0,'human_validity_claimed':False,
        'publisher_cryptographic_authenticity_claimed':False,'original_primary_input_acceptance':False,
        'original_full_population_cache_acceptance':False,'confirmatory_study_ready':False,
        'next_gate':'Passed fresh v4 model agreement validation and separately reviewed guard/experiment execution',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__},
        'elapsed_seconds_before_final_publication':time.perf_counter()-started,
        'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Input byte binding and pre-effect source reservation only; capacity counts are pre-guard upper bounds.'}
    publish(out/'result.json', result)
    print(json.dumps({'passed':True,'rows':len(records),'allocation':counts}), flush=True)


if __name__ == '__main__':
    main()
