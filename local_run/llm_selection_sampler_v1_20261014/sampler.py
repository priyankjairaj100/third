#!/usr/bin/env python3
"""Versioned model-only selection preparation; never an inference activation."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import socket
import sys
import time

FOLDER = Path(__file__).resolve().parent
SCHEMAS = {'qualification': 'third-llm-selection-qualification-v1',
           'selection': 'third-llm-selection-sample-v1'}
STAGES = {'qualification': ('engineering_qualification_selection',
                           'engineering_qualification_selection_swap'),
          'selection': ('selection', 'selection_swap')}
DIAGNOSTIC_SEEDS = {key: hashlib.sha256(('third-llm-only-v3/engineering-first512-v1/' + key).encode()).hexdigest()
                    for key in ('selection', 'selection_swap')}
ROW_KEYS = {'calibration_row_index', 'original_full_row_index', 'record_id',
            'source_unit_id', 'source_kind', 'partition', 'text_sha256'}
FALSE_FLAGS = ('threshold_selected', 'model_calibration_complete',
               'semantic_validity_established', 'original_human_gate_satisfied',
               'inference_activation', 'confirmatory_study_ready',
               'source_authenticity_verified', 'primary_input_acceptance',
               'original_route_acceptance', 'full_population_cache_acceptance')


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical(value, ascii_only=False):
    return json.dumps(value, ensure_ascii=ascii_only, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def strict(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate_json_key')
            value[key] = item
        return value
    def invalid(_):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    return strict(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    require('sha256' not in value, 'already_sealed')
    return dict(value, sha256=digest(value))


def verify_seal(value, ascii_only=False):
    require(isinstance(value, dict) and value.get('sha256') == hashlib.sha256(canonical(
        {k:v for k,v in value.items() if k != 'sha256'}, ascii_only)).hexdigest(), 'invalid_internal_seal')


def safe(path):
    require('..' not in Path(path).parts, 'parent_path_component')
    path = Path(path).absolute()
    require(all(not item.is_symlink() for item in (path, *path.parents)), 'symlink_path')
    return path


def binding(path):
    path = safe(path)
    require(path.is_file(), 'regular_file_required')
    before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    after = path.stat()
    keys = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(all(getattr(before,k) == getattr(after,k) for k in keys), 'changed_during_hash')
    return {'bytes':after.st_size, 'sha256':h.hexdigest()}


class Watch:
    def __init__(self):
        self.files = {}
    def add(self, path, expected=None):
        path = safe(path); actual = binding(path)
        require(expected is None or actual == expected, 'file_binding_mismatch:' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == actual, 'watched_input_changed')
        self.files[str(path)] = actual
        return actual
    def recheck(self):
        for path, expected in self.files.items():
            require(binding(path) == expected, 'input_changed:' + path)


def verify_package(folder, sha, watch):
    folder = safe(folder)
    actual = watch.add(folder/'package_manifest.json')
    require(actual['sha256'] == sha, 'package_manifest_mismatch')
    members = read_json(folder/'package_manifest.json')['files']
    require(isinstance(members,dict) and members, 'empty_package')
    for name, expected in members.items():
        require(Path(name).name == name and name not in ('.','..','package_manifest.json'), 'unsafe_member')
        watch.add(folder/name, expected)
    return actual


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_bytes(path, raw, cap):
    require(len(raw) <= cap, 'output_cap')
    path = Path(path); temporary = path.with_name(path.name+'.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path, value, cap=4 << 20):
    write_bytes(path, canonical(value, True)+b'\n', cap)


def validate_rows(rows, frame, panels):
    require(len(rows) == frame['rows'], 'row_count')
    ids = []; sources = []; indices = []; previous = -1
    for index, row in enumerate(rows):
        require(isinstance(row,dict) and set(row) == ROW_KEYS, 'row_schema')
        require(type(row['calibration_row_index']) is int and row['calibration_row_index'] == index,
                'calibration_order')
        original = row['original_full_row_index']
        require(type(original) is int and previous < original < frame['full_population_rows'], 'full_row_order')
        previous = original
        require(all(isinstance(row[k],str) and row[k] for k in
                    ('record_id','source_unit_id','source_kind','text_sha256')), 'row_identity')
        require(re.fullmatch('[0-9a-f]{64}', row['text_sha256']) is not None, 'text_sha')
        require(row['partition'] == 'calibration' and panels.digest_partition(panels.hash_integer(
                'ccu-v1-source-split',row['source_unit_id'])) == 'calibration', 'noncalibration_row')
        ids.append(row['record_id']); sources.append(row['source_unit_id']); indices.append(original)
    require(len(set(ids)) == len(ids), 'duplicate_id')
    for values, key in ((ids,'calibration_ordered_ids_sha256'),
                        (sources,'calibration_ordered_source_ids_sha256'),
                        (indices,'original_full_row_indices_sha256')):
        require(digest(values) == frame[key], 'ordered_digest:'+key)
    return rows


def load_frame(args, pins, watch, np, panels):
    folder = safe(args.frame_dir)
    require(folder == safe(args.workspace)/pins['frame_relative'], 'wrong_frame_directory')
    require(args.frame_result_sha256 == pins['frame_result']['sha256'], 'unreviewed_frame')
    watch.add(folder/'result.json', pins['frame_result'])
    frame = read_json(folder/'result.json'); verify_seal(frame, True)
    require(frame['sha256'] == pins['frame_result_seal'] and frame['schema'] == 'third-e5-calibration-frame-v1'
            and frame['passed'] is True and frame['stage'] == 'complete', 'frame_not_complete')
    require(frame['frame_package_sha256'] == pins['frame_package_sha256']
            and frame['rows'] == pins['rows'] and frame['shape'] == [pins['rows'],pins['dimension']]
            and frame['dtype'] == 'float32' and frame['unordered_pair_count'] == pins['pairs']
            and frame['within_calibration_same_source_exclusion'] is False, 'frame_contract')
    for key, value in pins['frame_inherited'].items():
        require(frame[key] == value, 'frame_provenance:'+key)
    for key in ('actual_cache_payload_hashes_verified','all_32_assembled_slices_exact',
                'all_32_encoded_and_fresh_replay_payloads_verified','all_calibration_vectors_exact_FP32_gather',
                'input_code_assets_initial_final_equal'):
        require(frame[key] is True, 'frame_payload_admission')
    for key in ('source_authenticity_verified','primary_input_acceptance','original_route_acceptance',
                'full_population_cache_acceptance','semantic_validity_established','confirmatory_study_ready',
                'sampling_performed','threshold_selected','human_validation_claimed'):
        require(frame[key] is False, 'frame_claim_boundary')
    require(frame['files'] == pins['frame_files'] and
            {p.name for p in folder.iterdir()} == {'result.json', *pins['frame_files']}, 'frame_inventory')
    for name, wanted in pins['frame_files'].items(): watch.add(folder/name,wanted)
    inherited = read_json(folder/'input_bindings.json')
    source = safe(args.workspace)/pins['source_relative']
    # Rebind source bytes and small provenance. Do not repeat all encoder forward/replay validation.
    for name, key in (('records.jsonl','prepared_source_records_file'),('audit.json','prepared_source_audit_file')):
        watch.add(source/name, frame[key])
        require(inherited['files'].get(str(source/name)) == frame[key], 'source_path_binding')
    for key, expected in (('full_source_frame_sha256',frame['source_full_rows_sha256']),
                          ('full_text_binding',frame['source_text_binding']),
                          ('source_cache_manifest',frame['source_cache_manifest']),
                          ('repair_result',frame['repair_result'])):
        require(inherited[key] == expected, 'inherited_source_binding:'+key)
    # Small recorded runtime/repair provenance is independently read again; huge encoder payloads are not.
    for path, expected in inherited['files'].items():
        if path.endswith('/models/e5/assets.json') or (expected == frame['repair_result']):
            require(Path(path).is_relative_to(safe(args.workspace)), 'external_provenance_path')
            watch.add(path,expected)
    with (folder/'records.private.jsonl').open('rb') as stream:
        rows = []
        for line in stream:
            require(line.endswith(b'\n') and line.strip(), 'physical_LF_records_required')
            rows.append(strict(line))
    validate_rows(rows,frame,panels)
    vectors = np.load(folder/'vectors.npy', mmap_mode='r', allow_pickle=False)
    validate_vectors(vectors,frame['shape'],np)
    return frame, rows, vectors


def validate_vectors(vectors, shape, np):
    require(vectors.dtype == np.dtype('float32') and vectors.shape == tuple(shape)
            and vectors.flags.c_contiguous, 'vector_shape_dtype')
    require(bool(np.isfinite(vectors).all()) and bool(np.any(vectors != 0,axis=1).all()), 'invalid_vectors')


def closed_inventories(args,pins):
    require({p.name for p in safe(args.frame_dir).iterdir()} == {'result.json', *pins['frame_files']},'final_frame_inventory')
    evidence=safe(args.protocol_evidence); declared=read_json(evidence/'snapshot_manifest.json')['files']
    require({str(p.relative_to(evidence)) for p in evidence.rglob('*') if p.is_file()
             and p != evidence/'snapshot_manifest.json'} == set(declared),'final_snapshot_inventory')
    if args.mode == 'selection':
        require({p.name for p in safe(args.sampling_lock).parent.iterdir()} == {'intent.json','sampling_lock.json'},
                'final_lock_inventory')
        qualifier = safe(args.qualification_result)
        require({p.name for p in qualifier.parent.iterdir()} == {'result.json', *read_json(qualifier)['files']},
                'final_qualification_inventory')


def make_rng(protocol, plan_sha, frame_sha, stage, seed, bin_index):
    require(re.fullmatch('[0-9a-f]{64}', seed) is not None, 'seed256_required')
    return random.Random(int(protocol.domain_digest(protocol.PREFIX+'sampling',
        [plan_sha,frame_sha,stage,seed,bin_index]),16))


def workload(n, dimension=768, block_size=256):
    sizes = [min(block_size,n-i) for i in range(0,n,block_size)]
    entries = sum(a*b for i,a in enumerate(sizes) for b in sizes[i:])
    return {'rows':n,'unordered_pairs':n*(n-1)//2,'blocks':len(sizes),
            'upper_tiles':len(sizes)*(len(sizes)+1)//2,
            'computed_tile_entries':entries,'coordinate_products':entries*dimension}


def stream_line(i,j,score):
    return f'{i}\t{j}\t{score.hex()}\n'.encode('ascii')


def collect(vectors, cal, rngs, progress=None):
    populations = [0]*20; reservoirs = [[] for _ in range(20)]; audit = {}; stream = hashlib.sha256(); count = 0
    started = time.perf_counter()
    for i,j,score in cal.iter_pair_scores(vectors,256,audit):
        require(0 <= i < j < len(vectors) and math.isfinite(score), 'invalid_pair_score')
        b = cal._bin(score); populations[b] += 1; count += 1
        cal._reservoir_insert(reservoirs[b],(i,j,score,b),populations[b],30,rngs[b])
        stream.update(stream_line(i,j,score))
        if progress is not None and count % 4096 == 0:
            progress.update(pairs_observed=count,bin_populations=list(populations))
    elapsed = time.perf_counter()-started
    require(count == len(vectors)*(len(vectors)-1)//2 and sum(populations) == count, 'incomplete_census')
    require(all(len(r) == min(30,n) for r,n in zip(reservoirs,populations)), 'reservoir_count')
    raw = sorted((x for r in reservoirs for x in r),key=lambda x:(x[0],x[1]))
    require(len({(x[0],x[1]) for x in raw}) == len(raw), 'duplicate_sample_pair')
    if progress is not None: progress.update(pairs_observed=count,bin_populations=list(populations))
    return {'populations':populations,'raw':raw,'stream_sha256':stream.hexdigest(),
            'pair_count':count,'numerical_audit':audit,'score_collector_seconds':elapsed}


def fraction(value):
    return {'numerator':value.numerator,'denominator':value.denominator}


def decorate(raw, populations, rows, protocol, plan, frame_sha):
    output = []
    for i,j,score,b in raw:
        population = populations[b]; n = min(30,population)
        require(n > 0 and math.isfinite(score), 'empty_bin_selection')
        ids = [rows[i]['record_id'],rows[j]['record_id']]
        identity = protocol.identities(plan['sha256'],frame_sha,ids)
        canonical_indices = [i,j] if identity['canonical_record_ids'] == ids else [j,i]
        output.append({'frame_row_indices':[i,j], 'original_full_row_indices':[
            rows[i]['original_full_row_index'],rows[j]['original_full_row_index']],
            'record_ids':ids,'source_unit_ids':[rows[i]['source_unit_id'],rows[j]['source_unit_id']],
            'same_source':rows[i]['source_unit_id'] == rows[j]['source_unit_id'],
            **identity,'canonical_frame_row_indices':canonical_indices,
            'score_hex':score.hex(),'bin':b,'bin_population':population,'bin_sample_size':n,
            'inclusion_probability':fraction(Fraction(n,population)),
            'inclusion_weight':fraction(Fraction(population,n)),
            'model_seeds':{m['model_id']:protocol.model_seed(plan['sha256'],frame_sha,
                identity['opaque_pair_id'],m['model_id'],m['revision']) for m in plan['models']}})
    require(len({r['opaque_pair_id'] for r in output}) == len(output), 'opaque_id_collision')
    return output


def swaps(sample, cal, rng):
    capacity = (len(sample)+9)//10; chosen = []
    for index in range(len(sample)):
        cal._reservoir_insert(chosen,index,index+1,capacity,rng)
    return [{'selection_index':index,'opaque_pair_id':sample[index]['opaque_pair_id'],
             'canonical_frame_row_indices':sample[index]['canonical_frame_row_indices'],
             'swapped_frame_row_indices':list(reversed(sample[index]['canonical_frame_row_indices'])),
             'model_seeds':sample[index]['model_seeds'],'diagnostic_only':True}
            for index in sorted(chosen)]


def independent_reference(vectors, panels, rngs):
    """Separate enumeration and reservoir code; full matrix allowed ONLY for bounded qualification."""
    require(len(vectors) <= 512, 'reference_is_bounded')
    normalized = panels.graph_normalize(vectors)
    matrix = panels.reference_cosines(normalized,normalized)
    counts = [0]*20; chosen = [[] for _ in range(20)]; h = hashlib.sha256(); emitted = 0
    edges = [float(Fraction(k,10)) for k in range(-10,11)]
    # Independently reconstruct the registered upper-tile order, without iter_pair_scores/_bin/_reservoir_insert.
    for left in range((len(vectors)+255)//256):
        for right in range(left,(len(vectors)+255)//256):
            for i in range(left*256,min((left+1)*256,len(vectors))):
                for j in range(max(i+1,right*256),min((right+1)*256,len(vectors))):
                    score = float(matrix[i,j]); b = max(0,min(19,sum(score >= e for e in edges)-1))
                    counts[b] += 1; emitted += 1
                    item = (i,j,score,b)
                    if counts[b] <= 30: chosen[b].append(item)
                    else:
                        k = rngs[b].randrange(counts[b])
                        if k < 30: chosen[b][k] = item
                    h.update((str(i)+'\t'+str(j)+'\t'+float.hex(score)+'\n').encode('ascii'))
    pairwise = 0
    for i in range(min(19,len(vectors))):
        for j in range(i+1,min(19,len(vectors))):
            single = panels.reference_cosines(panels.graph_normalize(vectors[i:i+1]),
                                             panels.graph_normalize(vectors[j:j+1]))
            require(float(single[0,0]).hex() == float(matrix[i,j]).hex(), 'pairwise_authority_mismatch')
            pairwise += 1
    return {'populations':counts,'raw':sorted((x for r in chosen for x in r),key=lambda x:(x[0],x[1])),
            'stream_sha256':h.hexdigest(),'pair_count':emitted,'pairwise_pairs':pairwise}


def verify_decoration(sample, swap_rows, raw, populations, rows, protocol, plan, frame_sha, swap_rng):
    """Direct assertions against protocol hashes and rational arithmetic; no decorate/swaps reuse."""
    require(len(sample) == len(raw), 'sample_length')
    for item, (i,j,score,b) in zip(sample,raw):
        ids = [rows[i]['record_id'],rows[j]['record_id']]
        identity = protocol.identities(plan['sha256'],frame_sha,ids)
        require(all(item[k] == v for k,v in identity.items()), 'identity_parity')
        p = Fraction(min(30,populations[b]),populations[b])
        require(item['inclusion_probability'] == {'numerator':p.numerator,'denominator':p.denominator}
                and item['inclusion_weight'] == {'numerator':p.denominator,'denominator':p.numerator}, 'weight_parity')
        require(item['score_hex'] == score.hex() and item['bin'] == b and item['frame_row_indices'] == [i,j]
                and item['record_ids'] == ids and item['bin_population'] == populations[b]
                and item['bin_sample_size'] == min(30,populations[b])
                and item['original_full_row_indices'] == [rows[i]['original_full_row_index'],rows[j]['original_full_row_index']]
                and item['source_unit_ids'] == [rows[i]['source_unit_id'],rows[j]['source_unit_id']]
                and item['same_source'] == (rows[i]['source_unit_id'] == rows[j]['source_unit_id'])
                and item['canonical_frame_row_indices'] == ([i,j] if identity['canonical_record_ids'] == ids else [j,i]),
                'score_identity_orientation_parity')
        for m in plan['models']:
            require(item['model_seeds'][m['model_id']] == protocol.model_seed(plan['sha256'],frame_sha,
                identity['opaque_pair_id'],m['model_id'],m['revision']), 'model_seed_parity')
    indices = []; cap = (len(sample)+9)//10
    for i in range(len(sample)):
        if i < cap: indices.append(i)
        else:
            k = swap_rng.randrange(i+1)
            if k < cap: indices[k] = i
    require([r['selection_index'] for r in swap_rows] == sorted(indices), 'swap_parity')
    for swap in swap_rows:
        row = sample[swap['selection_index']]
        require(swap['opaque_pair_id'] == row['opaque_pair_id'] and swap['model_seeds'] == row['model_seeds']
                and swap['swapped_frame_row_indices'] == list(reversed(row['canonical_frame_row_indices']))
                and swap['diagnostic_only'] is True, 'swap_identity_parity')


def validate_runtime(args, pins, np, watch):
    require(platform.system() == 'Linux' and re.fullmatch('[1-9][0-9]*',os.environ.get('SLURM_JOB_ID','')) is not None,
            'Linux_Slurm_child_required')
    require(platform.python_version() == pins['runtime']['python'] and np.__version__ == pins['runtime']['numpy']
            and socket.gethostname() == pins['runtime']['hostname'], 'unpinned_runtime')
    require(str(Path(sys.executable).absolute()) == str(safe(args.workspace)/'envs/encoder-cpu/bin/python'),
            'wrong_interpreter')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'numeric_threads_not_one:'+key)
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline_required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES','') in ('','-1'), 'GPU_not_disabled')
    for module in (random,np): watch.add(module.__file__)
    return {'python':platform.python_version(),'numpy':np.__version__,'hostname':socket.gethostname(),
            'interpreter':sys.executable,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'numeric_threads':1,'gpu_used':False,'rng':'Python random.Random MT19937',
            'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'concurrent_node_load_not_controlled':True}


def load_context(args, watch):
    pins = read_json(FOLDER/'expectations.json')
    own = verify_package(FOLDER,args.package_sha256,watch)
    verify_package(args.frame_package,pins['frame_package_sha256'],watch)
    verify_package(args.protocol_package,pins['protocol_package_sha256'],watch)
    plan_path = Path(args.protocol_package)/'PLAN.json'; require(binding(plan_path)['sha256'] == pins['plan_sha256'],'plan_pin')
    plan = read_json(plan_path); plan['sha256'] = pins['plan_sha256']  # local identity field, never alters PLAN bytes
    protocol = load_module('third_selection_protocol',Path(args.protocol_package)/'protocol.py')
    evidence = safe(args.protocol_evidence)
    observed = watch.add(evidence/'snapshot_manifest.json')
    require(observed['sha256'] == args.protocol_evidence_sha256 == pins['protocol_evidence_sha256'], 'protocol_evidence_pin')
    snapshot = read_json(evidence/'snapshot_manifest.json')
    require(snapshot.get('files') == plan['bound_files'] and len(plan['bound_files']) == 39, 'snapshot_manifest_inventory')
    actual = {str(p.relative_to(evidence)) for p in evidence.rglob('*') if p.is_file() and p != evidence/'snapshot_manifest.json'}
    require(actual == set(plan['bound_files']), 'snapshot_disk_inventory')
    for name,wanted in plan['bound_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe_evidence_path')
        watch.add(evidence/name,wanted)
    protocol.validate(plan,evidence)
    for name,wanted in pins['frozen_files'].items(): watch.add(safe(args.repo)/name,wanted)
    sys.path.insert(0,str(safe(args.repo)/'empirical_execution'))
    import numpy as np
    from phase3 import calibration as cal, panels
    require(Path(cal.__file__).resolve() == (safe(args.repo)/'empirical_execution/phase3/calibration.py').resolve()
            and Path(panels.__file__).resolve() == (safe(args.repo)/'empirical_execution/phase3/panels.py').resolve(), 'module_shadowing')
    runtime = validate_runtime(args,pins,np,watch)
    frame,rows,vectors = load_frame(args,pins,watch,np,panels)
    return pins,own,plan,protocol,cal,panels,runtime,frame,rows,vectors


def production_seeds(args, pins, own, frame, watch):
    required = ('lock_package','lock_package_sha256','sampling_lock','sampling_lock_sha256',
                'qualification_result','qualification_result_sha256','qualification_review','qualification_review_sha256')
    require(all(getattr(args,k,None) for k in required),'production_lock_arguments_required')
    helper_binding = verify_package(args.lock_package,args.lock_package_sha256,watch)
    require(args.lock_package_sha256 == pins['lock_package_sha256'],'unregistered_lock_helper')
    helper = load_module('third_sampling_lock',Path(args.lock_package)/'sampling_lock.py')
    qual_path = safe(args.qualification_result); review_path = safe(args.qualification_review)
    q = watch.add(qual_path); r = watch.add(review_path)
    require(q['sha256'] == args.qualification_result_sha256 and r['sha256'] == args.qualification_review_sha256,'qualification_review_pin')
    require(qual_path == safe(args.workspace)/'qualification/llm-selection-sampler-v1-20261014-qualify-a01/run/output/result.json'
            and review_path == safe(args.workspace)/'qualification/llm-selection-sampler-v1-20261014-qualify-a01/independent_review.json',
            'alternate_qualification_evidence_path')
    result, review = read_json(qual_path),read_json(review_path)
    helper.check_qualification(result,own); helper.check_review(review,q,own)
    # Also bind every published qualification artifact and its final inventory, without rescoring it here.
    require({p.name for p in qual_path.parent.iterdir()} == {'result.json', *result['files']},'qualification_output_inventory')
    for name,expected in result['files'].items():
        require(Path(name).name == name and name != 'result.json','unsafe_qualification_artifact')
        watch.add(qual_path.parent/name,expected)
    lock_path = safe(args.sampling_lock)
    require(lock_path == safe(args.workspace)/'qualification/llm-only-v3-sampling-lock-a01/sampling_lock.json','alternate_lock_slot')
    lock_binding = watch.add(lock_path); require(lock_binding['sha256'] == args.sampling_lock_sha256,'lock_raw_pin')
    lock = read_json(lock_path)
    snapshot = binding(Path(args.protocol_evidence)/'snapshot_manifest.json')
    expected = helper.contract(frame,own,q,r,snapshot)
    completed, completed_binding = helper.read_completed_lock(lock_path,expected,helper_binding)
    require(completed == lock and completed_binding == lock_binding,'lock_completion_changed')
    watch.add(lock_path.parent/'intent.json')
    helper.validate_lock(lock,expected,helper_binding)
    require(safe(args.out) == safe(args.workspace)/lock['contract']['selection_root_relative']/'run/output',
            'alternate_selection_output')
    return lock['seed_hex'], {'sampling_lock':lock_binding,'qualification_result':q,
                             'qualification_review':r,'lock_helper_package':helper_binding}


def execute(args):
    total_started = time.perf_counter(); out = safe(args.out); workspace = safe(args.workspace)
    require(out.is_relative_to(workspace/'qualification') and not out.exists(),'new_private_output_required')
    os.umask(0o077); out.mkdir(parents=True,exist_ok=False)
    progress = {'stage':'preflight','pairs_observed':0}; watch = Watch()
    try:
        pins,own,plan,protocol,cal,panels,runtime,frame,rows,vectors = load_context(args,watch)
        if args.mode == 'selection':
            seeds, lock_evidence = production_seeds(args,pins,own,frame,watch)
        else:
            require(not any(getattr(args,k,None) for k in ('sampling_lock','sampling_lock_sha256',
                    'qualification_result','qualification_review','lock_package')),'qualification_cannot_use_production_lock')
            seeds, lock_evidence = DIAGNOSTIC_SEEDS, None
        stage,swap_stage = STAGES[args.mode]; n = pins['qualification_rows'] if args.mode == 'qualification' else pins['rows']
        selected_vectors, selected_rows = vectors[:n], rows[:n]
        rng = lambda s,seed,b: make_rng(protocol,plan['sha256'],args.frame_result_sha256,s,seed,b)
        preflight_seconds = time.perf_counter()-total_started
        progress['stage']='complete_score_traversal'
        collected = collect(selected_vectors,cal,[rng(stage,seeds['selection'],b) for b in range(20)],progress)
        sample = decorate(collected['raw'],collected['populations'],selected_rows,protocol,plan,args.frame_result_sha256)
        swap_rows = swaps(sample,cal,rng(swap_stage,seeds['selection_swap'],None))
        parity = None; reference_seconds = 0.0
        if args.mode == 'qualification':
            progress['stage']='independent_bounded_parity'; started=time.perf_counter()
            reference = independent_reference(selected_vectors,panels,[rng(stage,seeds['selection'],b) for b in range(20)])
            require(all(collected[k] == reference[k] for k in ('populations','raw','stream_sha256','pair_count')),
                    'independent_reference_mismatch')
            require(reference['pairwise_pairs'] == 171,'pairwise_reference_count')
            verify_decoration(sample,swap_rows,reference['raw'],reference['populations'],selected_rows,
                              protocol,plan,args.frame_result_sha256,rng(swap_stage,seeds['selection_swap'],None))
            parity = {k:True for k in ('full512_score_stream','full512_bins_and_reservoir','first19_pairwise171','identities_weights_swaps')}
            reference_seconds = time.perf_counter()-started
        require(len(sample) <= 600 and len(swap_rows) == (len(sample)+9)//10 <= 60,'sample_cap')
        progress['stage']='publication_preflight'
        inputs = {'schema':'third-llm-selection-input-bindings-v1','files':watch.files,
                  'protocol_evidence_role':'Exact historical PLAN-bound files; not the executing repository or source acceptance',
                  'frame_payloads_rehashed':True,'original_source_bytes_rehashed':True,
                  'encoder_forward_and_replay_validation_reexecuted':False,'source_frame_result':pins['frame_result'],
                  'sampler_package':own,'lock_evidence':lock_evidence}
        payloads = {'input_bindings.json':canonical(inputs,True)+b'\n',
                    'selection.private.jsonl':b''.join(canonical(x,True)+b'\n' for x in sample),
                    'selection_swap.private.jsonl':b''.join(canonical(x,True)+b'\n' for x in swap_rows)}
        if args.mode == 'qualification':
            import numpy as np
            buffer=io.BytesIO(); np.save(buffer,selected_vectors,allow_pickle=False)
            payloads['diagnostic_frame_vectors.npy']=buffer.getvalue()
            payloads['diagnostic_frame_rows.private.jsonl']=b''.join(canonical(x,True)+b'\n' for x in selected_rows)
            copied=np.load(io.BytesIO(payloads['diagnostic_frame_vectors.npy']),allow_pickle=False)
            require(copied.dtype == selected_vectors.dtype and copied.shape == selected_vectors.shape
                    and copied.tobytes(order='C') == selected_vectors.tobytes(order='C'),'diagnostic_copy_mismatch')
        require(sum(map(len,payloads.values()))+(4 << 20) <= pins['output_directory_cap_bytes'],'prewrite_directory_cap')
        for name,raw in payloads.items(): write_bytes(out/name,raw,8 << 20)
        outputs = {name:binding(out/name) for name in payloads}
        started=time.perf_counter(); watch.recheck(); closed_inventories(args,pins)
        require(all(binding(out/name) == wanted for name,wanted in outputs.items()),'output_changed')
        final_seconds=time.perf_counter()-started
        result = {'schema':SCHEMAS[args.mode],'mode':args.mode,'passed':True,'stage':'complete',
            'source_frame_result':pins['frame_result'],'sampler_package':own,
            'protocol_plan_sha256':plan['sha256'],'frame_package_sha256':pins['frame_package_sha256'],
            'files':outputs,'runtime':runtime,'selected_frame_rows':n,
            'unordered_pair_count':collected['pair_count'],'scoring_block_size':256,
            'workload':workload(n),'full_frame_workload':workload(pins['rows']),
            'score_stream_sha256':collected['stream_sha256'],'score_stream_encoding':'ASCII i TAB j TAB float.hex(score) LF',
            'bin_populations':collected['populations'],'bin_sample_counts':[min(30,x) for x in collected['populations']],
            'selected_pairs':len(sample),'diagnostic_swap_pairs':len(swap_rows),
            'numerical_audit':collected['numerical_audit'],'rng_stages':[stage,swap_stage],
            'diagnostic_seeds':DIAGNOSTIC_SEEDS if args.mode == 'qualification' else None,
            'lock_evidence':lock_evidence,'parity':parity,
            'preflight_seconds':preflight_seconds,'score_collector_seconds':collected['score_collector_seconds'],
            'independent_reference_seconds':reference_seconds,'final_recheck_seconds':final_seconds,
            'elapsed_seconds_before_publication':time.perf_counter()-total_started,
            'peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'sampling_performed':args.mode == 'selection','production_sampling_performed':args.mode == 'selection',
            'engineering_subset_sampled':args.mode == 'qualification',
            'diagnostic_frame_artifacts_are_exact_first512_slice':args.mode == 'qualification',
            'diagnostic_frame_shape':list(selected_vectors.shape) if args.mode == 'qualification' else None,
            'diagnostic_frame_dtype':'float32' if args.mode == 'qualification' else None,
            'engineering_subset_exposure':'First512 existing calibration rows, fixed diagnostic randomness; retained exposure, no labels' if args.mode == 'qualification' else None,
            'same_source_pairs_excluded':False,'availability_filter_applied':False,
            'model_inference_calls':0,'human_annotation_calls':0,'requires_human_annotations':False,
            **{k:False for k in FALSE_FLAGS},
            'inherited_frame_scope':frame['frame_scope'],
            'source_scope':'Replayed prepared engineering/development population; no promotion to publisher authenticity, original-route admission or semantic validity.',
            'limitations':['Sampling success is not inference activation, model accuracy, or scientific/source admission.',
                'Inherited actual frame admission is bound; encoder forwards and 32 replay checks are not rerun here.',
                'MT19937 SRS is the uniform-draw algorithmic model, not a cryptographic or exact-PRNG-randomness claim.',
                'Bounded timing includes scoring/audit/bin/reservoir/stream hash; it is not a full-run guarantee.'],
            'created_utc':datetime.now(timezone.utc).isoformat()}
        result = seal(result)
        raw=canonical(result,True)+b'\n'
        require(sum(p.stat().st_size for p in out.iterdir())+len(raw) <= pins['output_directory_cap_bytes'],'final_directory_cap')
        require({p.name for p in out.iterdir()} == set(payloads),'unexpected_output')
        write_bytes(out/'result.json',raw,4 << 20)
        return result
    except BaseException as exc:
        failure = {'schema':'third-llm-selection-failure-v1','passed':False,'mode':args.mode,
            'progress':progress,'error_type':type(exc).__name__,'error':str(exc),
            'sampling_performed':False,'production_sampling_performed':False,'partial_output_usable':False,
            'model_inference_calls':0,'human_annotation_calls':0,'inference_activation':False,
            'progress_checkpoint_interval_pairs':4096,
            'elapsed_seconds':time.perf_counter()-total_started}
        if not (out/'FAILED.json').exists(): publish(out/'FAILED.json',failure)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=tuple(SCHEMAS))
    for key in ('repo','workspace','protocol-package','protocol-evidence','frame-package','frame-dir','out'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('package-sha256','frame-result-sha256','protocol-evidence-sha256'):
        p.add_argument('--'+key,required=True)
    for key in ('lock-package','sampling-lock','qualification-result','qualification-review'):
        p.add_argument('--'+key,type=Path)
    for key in ('lock-package-sha256','sampling-lock-sha256','qualification-result-sha256','qualification-review-sha256'):
        p.add_argument('--'+key)
    return p


if __name__ == '__main__':
    try:
        result=execute(parser().parse_args())
        print(json.dumps({'passed':True,'mode':result['mode'],'pairs':result['unordered_pair_count'],
                          'result_sha256':result['sha256'],'model_inference_calls':0},sort_keys=True))
    except Exception as exc:
        print(json.dumps({'passed':False,'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),file=sys.stderr)
        sys.exit(1)
