"""One full scan per fresh process: new scanner or original semantic_guard."""
import argparse
import datetime
import os
import platform
import resource
import socket
import sys
import time
import numpy as np
import admission
import common as c
from guard_core import scan
import panel_core


def checked_inputs():
    e = c.strict((c.HERE/'expectations.json').read_bytes())
    watched = {c.INTAKE/'result.json': e['intake_result'], c.INTAKE/'independent_review.json': e['intake_review'],
               c.INTAKE/'allocation.private.json': e['allocation'], c.FRAME/'result.json': e['frame_result'],
               c.TASK/'result.json': e['task_result'], c.CACHE/'manifest.json': e['assembly']['assembly_manifest'],
               c.BENCHMARK/'run/result.json': e['benchmark_result'],
               c.BENCHMARK/'process_completion.json': e['benchmark_process'], c.SOURCE: e['source']}
    watched.update({c.CACHE/n: b for n, b in e['assembly']['assembly_files'].items()})
    watched.update({c.TASK/n: b for n, b in e['task_outputs'].items()})
    watched.update({c.REPO/n: b for n, b in e['repository_files'].items()})
    for path, want in watched.items():
        c.require(c.binding(path) == want, 'bound_input:'+str(path))
    intake = c.strict((c.INTAKE/'result.json').read_bytes())
    review = c.strict((c.INTAKE/'independent_review.json').read_bytes())
    bench = c.strict((c.BENCHMARK/'run/result.json').read_bytes())
    c.require(intake['passed'] is True and review['passed'] is True
              and review['intake_result_sha256'] == e['intake_result']['sha256'], 'reviewed_actual_intake')
    c.require(intake['outputs']['allocation.private.json'] == e['allocation'], 'allocation_binding')
    c.require(bench['passed'] is True and bench['package_sha256'] == e['benchmark_package_sha256']
              and all(r['exact_witness_score_keep_parity'] is True for r in bench['exact_original_guard_parity']),
              'actual_resource_pilot')
    return e, watched


def admit_runtime():
    c.require(os.environ.get('SLURM_JOB_ID', '').isdigit() and socket.gethostname() in ('cn1', 'csis.cn1'), 'compute_only')
    c.require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
              and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'single_thread_CPU')


def load_rows():
    rows = c.strict((c.INTAKE/'allocation.private.json').read_bytes())
    c.require(len(rows) == 140250 and panel_core.inventory(rows) == {
        'development_train': 10314, 'evaluation_train': 87873, 'calibration': 20996,
        'development_test': 2138, 'evaluation_test': 18929}, 'complete_original_reservation')
    return rows


def run(mode, package_pin, admission_pin):
    admit_runtime(); c.verify_package(package_pin)
    approved = admission.check_bundle(c.BUNDLE, admission_pin, remote=True)
    e, watched = checked_inputs()
    c.require(platform.python_version() == e['python'] and np.__version__ == e['numpy'], 'qualified_runtime')
    rows = load_rows(); cache = np.load(c.CACHE/'vectors.npy', mmap_mode='r', allow_pickle=False)
    c.require(cache.shape == (140250, 768) and cache.dtype == np.dtype('float32'), 'native_cache')
    train = [i for i, r in enumerate(rows) if r['role'] == 'development_train']
    heldout = sorted((i for i, r in enumerate(rows) if r['partition'] != 'train'), key=lambda i: rows[i]['record_id'])
    c.require(len(train) == 10314 and len(heldout) == 42063, 'full_guard_scope')
    tau = float(admission.fraction(approved['locked_tau']))
    sys.path.insert(0, str(c.REPO)); from empirical_execution.phase3 import panels
    out = c.ROOT/mode; out.mkdir(mode=0o700, exist_ok=False); started = time.perf_counter()
    if mode == 'primary':
        got = scan(cache, train, heldout, [rows[i]['record_id'] for i in heldout], [tau], panels, 256)
        removed = [{'record_id': rows[i]['record_id'],
                    'heldout_witness_id': rows[int(got['witness_indices'][j, 0])]['record_id'],
                    'score': float(got['witness_scores'][j, 0])}
                   for j, i in enumerate(train) if got['witness_indices'][j, 0] >= 0]
        pairs = got['pairs']; gathered_bytes = 0
    else:
        c.require(mode == 'reference', 'known_scan_mode')
        subset = train + heldout
        gathered = cache[subset]; gathered_bytes = gathered.nbytes
        _, kept, audit = panels.semantic_guard([rows[i] for i in subset], gathered, tau, block_size=256)
        removed = audit['removed']; pairs = audit['pairs_scored']
        c.require({rows[subset[i]]['record_id'] for i in kept if rows[subset[i]]['partition'] == 'train'} ==
                  {rows[i]['record_id'] for i in train} - {r['record_id'] for r in removed}, 'reference_keep_consistency')
    scan_seconds = time.perf_counter() - started
    c.require(pairs == 433837782, 'exhaustive_intended_pairs')
    removed.sort(key=lambda r: r['record_id'])
    payload = {'locked_tau': approved['locked_tau'], 'pairs_scored': pairs, 'removed': removed}
    c.publish(out/'guard.private.json', payload)
    if mode == 'primary':
        reservation = c.strict((c.HERE/'RESERVATION_PLAN.json').read_bytes())
        c.publish(out/'panels.private.json', panel_core.build_panels(rows, removed, reservation))
    for path, want in watched.items():
        c.require(c.binding(path) == want, 'input_changed_during_scan')
    c.verify_package(package_pin); admission.check_bundle(c.BUNDLE, admission_pin, remote=True)
    report = {'schema': 'third-full-development-guard-scan-v1', 'passed': True, 'mode': mode,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'package_sha256': package_pin, 'admission_sha256': admission_pin,
              'job_id': os.environ['SLURM_JOB_ID'], 'input_bindings': {str(k): v for k, v in watched.items()},
              'locked_tau': approved['locked_tau'], 'train_rows': len(train), 'heldout_rows': len(heldout),
              'pairs_scored': pairs, 'removed_rows': len(removed), 'retained_train_rows': len(train)-len(removed),
              'outputs': {p.name: c.binding(p) for p in out.iterdir() if p.name.endswith('.private.json')},
              'scan_wall_seconds': scan_seconds, 'gathered_FP32_bytes': gathered_bytes,
              'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'full_cache_memmap_bytes': cache.nbytes, 'block_size': 256,
              'score_block_bytes_max': 256*256*8,
              'resource_limit': 'RSS includes file-backed pages, metadata and temporary arrays; scan timing excludes admission and publication, separately covered by child lifecycle timing.',
              'no_task_labels_or_source_text_decoded': True, 'human_annotation_calls': 0,
              'human_validity_claimed': False, 'effect_experiments_started': False,
              'full_guard_accepted': False, 'requires_separate_full_reference_review': True}
    c.publish(out/'result.json', report)
    print(c.canonical({'mode': mode, 'passed': True, 'pairs': pairs, 'removed': len(removed)}).decode(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('primary', 'reference'), required=True)
    p.add_argument('--package-sha256', required=True); p.add_argument('--admission-sha256', required=True)
    a = p.parse_args(); run(a.mode, a.package_sha256, a.admission_sha256)
