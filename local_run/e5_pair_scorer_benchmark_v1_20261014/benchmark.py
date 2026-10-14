#!/usr/bin/env python3
"""Bounded scorer engineering only: no forwards, sampling, labels or thresholds."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import time

FOLDER = Path(__file__).resolve().parent
SIZES = (256, 260, 512)
BLOCK = 256
MODES = ('score_only', 'boundary_and_bins')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
FALSE_FLAGS = ('human_sample_drawn', 'calibration_sample_drawn', 'threshold_selected',
    'full_population_scored', 'full_population_cache_acceptance',
    'semantic_validity_established', 'confirmatory_study_ready')


def require(value, error):
    if not value:
        raise ValueError(error)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular_file_required')
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'file_changed_during_hash')
    return {'bytes': after.st_size, 'sha256': h.hexdigest()}


def verify_binding(path, expected):
    actual = binding(path)
    require(actual == expected, 'file_binding_mismatch:' + str(path))
    return actual


def verify_package(folder, expected_sha):
    folder = Path(folder)
    manifest = folder / 'package_manifest.json'
    require(binding(manifest)['sha256'] == expected_sha, 'package_manifest_mismatch')
    files = json.loads(manifest.read_text())['files']
    require(isinstance(files, dict) and files, 'empty_package')
    for name, wanted in files.items():
        require(Path(name).name == name and name not in ('.', '..', 'package_manifest.json'),
                'unsafe_package_member')
        verify_binding(folder / name, wanted)
    return files


def new_output(path, workspace):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'output_already_exists')
    path = path.resolve()
    require(path.is_relative_to(Path(workspace).resolve() / 'qualification'), 'private_output_required')
    os.umask(0o077)
    path.mkdir(parents=True, exist_ok=False)
    return path


def publish(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path)  # Atomic, exclusive publication; never replace a result.
    pending.unlink()
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pair_order(n, block_size):
    """Independent integer-only expected order of frozen upper-tile traversal."""
    for i0 in range(0, n, block_size):
        for j0 in range(i0, n, block_size):
            for i in range(i0, min(n, i0 + block_size)):
                for j in range(max(j0, i + 1) if i0 == j0 else j0, min(n, j0 + block_size)):
                    yield i, j


def geometry(n, block_size=BLOCK):
    sizes = [min(block_size, n - i) for i in range(0, n, block_size)]
    entries = sum(x * y for i, x in enumerate(sizes) for y in sizes[i:])
    return {'rows': n, 'unordered_pairs': n * (n - 1) // 2,
        'block_size': block_size, 'blocks': len(sizes), 'last_block_rows': sizes[-1],
        'upper_tiles': len(sizes) * (len(sizes) + 1) // 2,
        'computed_tile_entries_including_diagonal_redundancy': entries,
        'coordinate_products': entries * 768}


def reference_table(vectors, scoring, np):
    require(vectors.dtype == np.dtype('float32') and vectors.shape == (19, 768),
            'expected_19_by_768_FP32')
    normalized = scoring.graph_normalize(vectors)
    require(np.isfinite(normalized).all(), 'nonfinite_normalized_vectors')
    # Independent pair-at-a-time use of the same frozen numerical authority.
    table = {}
    for i in range(19):
        for j in range(19):
            score = float(scoring.reference_cosines(normalized[i:i+1], normalized[j:j+1])[0, 0])
            table[i, j] = struct.pack('<d', score)
    return table


class BudgetExpired(RuntimeError):
    def __init__(self, partial):
        super().__init__('numerical_budget_exhausted')
        self.partial = partial


def measure(vectors, index_map, table, cal, mode, block_size, deadline,
            clock=time.perf_counter, cpu_clock=time.process_time):
    require(mode in MODES, 'invalid_mode')
    require(len(index_map) == len(vectors) and len(vectors) in (19,) + SIZES
            and all(type(i) is int and 0 <= i < 19 for i in index_map), 'invalid_fixture_map')
    start, cpu_start = clock(), cpu_clock()
    audit = {} if mode == 'boundary_and_bins' else None
    bins = [0] * 20 if audit is not None else None
    expected_order = iter(pair_order(len(vectors), block_size))
    digest, count = hashlib.sha256(), 0
    result = {'mode': mode, **geometry(len(vectors), block_size), 'completed': False,
              'pairs_checked': 0, 'collector': 'per-pair order, exact reference bytes and SHA256; included in timing'}
    def expired():
        if clock() >= deadline:
            result.update(pairs_checked=count, elapsed_seconds=clock() - start,
                          process_cpu_seconds=cpu_clock() - cpu_start,
                          partial_score_stream_sha256=digest.hexdigest(), bin_counts=bins,
                          boundary_audit=audit)
            raise BudgetExpired(result)
    expired()
    for i, j, score in cal.iter_pair_scores(vectors, block_size, audit):
        require((i, j) == next(expected_order, None), 'pair_order_gap_duplicate_or_reorder')
        raw = struct.pack('<d', score)
        require(math.isfinite(score) and raw == table[index_map[i], index_map[j]], 'score_hex_parity_failed')
        digest.update(struct.pack('<II', i, j)); digest.update(raw)
        count += 1
        if bins is not None:
            bins[cal._bin(score)] += 1
        if count % 4096 == 0:
            expired()
    require(next(expected_order, None) is None and count == result['unordered_pairs'], 'incomplete_pair_census')
    require(bins is None or sum(bins) == count, 'bin_denominator_mismatch')
    expired()
    result.update(completed=True, pairs_checked=count, elapsed_seconds=clock() - start,
                  process_cpu_seconds=cpu_clock() - cpu_start,
                  score_stream_sha256=digest.hexdigest(), bin_counts=bins, boundary_audit=audit)
    return result


def run_suite(vectors, cal, scoring, np, case_sink, seconds=300):
    start = time.perf_counter(); deadline = start + seconds
    table = reference_table(vectors, scoring, np)
    cases = []
    def run(value, mapping, mode, block_size, kind, repetition):
        try:
            result = measure(value, mapping, table, cal, mode, block_size, deadline)
        except BudgetExpired as error:
            result = {**error.partial, 'kind': kind, 'repetition': repetition}
            cases.append(result); case_sink(result)
            raise
        result.update(kind=kind, repetition=repetition,
            index_map_sha256=hashlib.sha256(canonical(mapping)).hexdigest(),
            natural_records=19 if kind == 'genuine_qualification_correctness' else 0,
            synthetic_index_repetition=kind != 'genuine_qualification_correctness')
        cases.append(result); case_sink(result)
        return result
    for block_size in (1, 7, 19, 256):
        run(vectors, list(range(19)), 'boundary_and_bins', block_size,
            'genuine_qualification_correctness', 0)
    for n in SIZES:
        mapping = [i % 19 for i in range(n)]
        repeated = vectors[mapping]
        for mode in MODES:
            warm = run(repeated, mapping, mode, BLOCK, 'software_shape_warmup', 0)
            for repetition in (1, 2, 3):
                measured = run(repeated, mapping, mode, BLOCK, 'software_shape_measurement', repetition)
                require(measured['score_stream_sha256'] == warm['score_stream_sha256'], 'repeat_score_stream_changed')
    return {'cases': len(cases), 'all_cases_completed': True,
        'genuine_unordered_pairs': 171, 'unique_natural_records': 19,
        'numerical_elapsed_seconds': time.perf_counter() - start,
        'reference_table_entries': len(table), 'full_target_geometry': geometry(20996),
        'full_target_pair_count_executed': 0,
        'projection_scope': 'Measured shape costs only; no guaranteed walltime or natural boundary-distribution estimate',
        'reservoir_sampling_cost_measured': False, 'end_to_end_calibration_cost_measured': False}


def load_core(repo, pins):
    core = repo / pins['core_folder']
    verify_package(core, pins['core_package_sha256'])
    verify_package(repo / pins['dispatch_folder'], pins['dispatch_package_sha256'])
    sys.path.insert(0, str(core))
    module = importlib.import_module('common')
    require(Path(module.__file__).resolve() == core / 'common.py', 'wrong_core_module_import')
    return module


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'workspace', 'view', 'replay-result', 'census-dir', 'budget-dir',
                 'qualification-result', 'repair-result', 'out', 'benchmark-package-sha256'):
        result.add_argument('--' + name, required=True)
    return result


def execute(args):
    start = time.perf_counter(); out = None
    report = {'schema': 'third-frozen-pair-scorer-benchmark-v1', 'passed': False,
              'stage': 'preflight', 'model_forwards': 0, 'new_model_judgments': 0,
              'human_responses': 0, **{key: False for key in FALSE_FLAGS}}
    try:
        verify_package(FOLDER, args.benchmark_package_sha256)
        pins = json.loads((FOLDER / 'expectations.json').read_text())
        repo, workspace = Path(args.repo).resolve(), Path(args.workspace).resolve()
        require(all(os.environ.get(key) == '1' for key in THREAD_ENV), 'single_thread_environment_required')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU_visibility_prohibited')
        out = new_output(args.out, workspace)
        core = load_core(repo, pins)
        args.package_sha256 = pins['core_package_sha256']
        args.qualification_sha256 = pins['qualification_result']['sha256']
        ctx = core.context(args)
        for name, expected in pins['repository_files'].items():
            ctx['watch'].add(repo / name, expected)
        qpath = core.c.inside(args.qualification_result, workspace)
        rpath = core.c.inside(args.repair_result, workspace)
        ctx['watch'].add(qpath, pins['qualification_result'])
        ctx['watch'].add(rpath, pins['repair_result'])
        repair = json.loads(rpath.read_text())
        require(repair['passed'] is True and repair['stage'] == 'complete'
                and repair['all_preexisting_versions_and_RECORDs_unchanged'] is True
                and importlib.metadata.version('scipy') == repair['scipy_version'] == '1.17.1',
                'additive_repair_binding_failed')
        qualification = core.require_qualification(ctx, qpath)
        require(ctx['aggregate']['partition_counts']['calibration'] == 20996, 'calibration_population_count_changed')
        require(qualification['fingerprint'] == pins['fingerprint'], 'qualification_fingerprint_changed')
        cache = qpath.parent / 'assembled/vectors.npy'
        ctx['watch'].add(cache, pins['qualification_vectors'])
        vectors = ctx['np'].load(cache, mmap_mode='r', allow_pickle=False)
        report.update(stage='scoring', package_sha256=args.benchmark_package_sha256,
            qualification_result=binding(qpath), qualification_vectors=binding(cache),
            repair_result=binding(rpath), inherited_derivation_runtime=ctx['runtime'],
            benchmark_runtime={'python': sys.version, 'numpy': ctx['np'].__version__,
                'hostname': socket.gethostname(), 'cpu': core.cpu_identity(),
                'affinity': sorted(os.sched_getaffinity(0)),
                'thread_environment': {key: os.environ[key] for key in THREAD_ENV},
                'scoring_dtype': 'float64', 'input_dtype': 'float32',
                'job_id': os.environ.get('SLURM_JOB_ID'), 'concurrent_load_controlled': False},
            preflight_elapsed_seconds=time.perf_counter() - start)
        publish(out / 'input_bindings.json', {'package_sha256': args.benchmark_package_sha256,
            'qualification_result': binding(qpath), 'repair_result': binding(rpath),
            'qualification_vectors': binding(cache), 'source_package_pins': pins,
            'index_maps': {str(n): [i % 19 for i in range(n)] for n in SIZES}})
        with (out / 'cases.jsonl').open('x') as stream:
            def save_case(value):
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            report['summary'] = run_suite(vectors, ctx['cal'], ctx['cal']._scoring,
                                          ctx['np'], save_case, pins['numerical_budget_seconds'])
        require(report['summary']['cases'] == pins['expected_completed_cases'], 'case_denominator_changed')
        report['stage'] = 'final_immutability'
        final_start = time.perf_counter()
        core.verify_final(ctx)
        verify_package(FOLDER, args.benchmark_package_sha256)
        verify_package(repo / pins['dispatch_folder'], pins['dispatch_package_sha256'])
        report.update(passed=True, stage='complete', initial_final_bindings_equal=True,
                      final_verification_seconds=time.perf_counter() - final_start)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    report.update(elapsed_seconds_before_result_publication=time.perf_counter() - start,
        peak_rss_kib_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing_limit='Slurm wrapper process receipt separately measures through child exit')
    if out is not None:
        report['artifacts'] = {p.name: binding(p) for p in sorted(out.iterdir()) if p.is_file()}
        publish(out / 'result.json', report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(execute(parser().parse_args()))
