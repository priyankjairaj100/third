"""Exact backend comparison on algebra checks and genuine Civil lexical inputs."""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
import numpy as np

from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase4 import convex as original
from phase5 import convex_multioutput as multi
from phase6 import dyadic_convex as fast

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['gradient_lower', 'gradient_upper', 'gradient_norm_squared_upper',
          'parameter_error_squared_upper', 'parameter_error_norm_upper',
          'objective_gap_upper', 'parameter_tolerance', 'meets_parameter_tolerance', 'work']
JOINT_FIELDS = ['parameter_error_frobenius_squared_upper', 'parameter_error_frobenius_upper',
                'objective_gap_upper', 'parameter_tolerance', 'meets_parameter_tolerance', 'work']
TARGETS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(input_path, backend, output_path):
    with np.load(input_path, allow_pickle=False) as arrays:
        x, y, w = (arrays[k] for k in ('x', 'y', 'w'))
    verifier = fast.certify_multioutput if backend == fast.BACKEND else multi.certify_multioutput
    certificate = verifier(x, y, .01, w)
    result = {'backend': backend, 'input_sha256': sha256_file(input_path),
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'rss_scope': 'whole fresh child process high-water RSS, including imports and input arrays',
              'certificate': certificate, 'pid': os.getpid()}
    write(output_path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'phase6/results/dyadic_convex_release')
    parser.add_argument('--worker-input', type=Path)
    parser.add_argument('--worker-backend')
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    if args.worker_input:
        worker(args.worker_input, args.worker_backend, args.worker_output)
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    tests = [
        (np.asarray([[1, -2], [.125, .5]], np.float32), np.asarray([.2, .8], np.float64), np.asarray([.4, -.7], np.float64)),
        (np.asarray([[np.nextafter(np.float32(0), np.float32(1)), -.125]], np.float32),
         np.asarray([np.nextafter(0., 1.)]), np.asarray([np.nextafter(0., 1.), .25])),
        (np.empty((0, 3), np.float32), np.empty(0, np.float64), np.asarray([1., -2., .5])),
        (np.zeros((3, 3), np.float32), np.asarray([0., .5, 1.]), np.zeros(3)),
    ]
    exact_cases = 0
    for x, y, w in tests:
        for bits in (16, 64, 128):
            a = original.certify_logistic(x, y, .01, w, bits=bits)
            b = fast.certify_logistic(x, y, .01, w, bits=bits)
            assert all(a[k] == b[k] for k in FIELDS)
            exact_cases += 1
    checks['all_algebra_gradient_endpoints_and_bounds_identical'] = True
    checks['empty_target_and_subnormal_values_exact'] = True
    for backend in (original.certify_logistic, fast.certify_logistic):
        try:
            backend(np.zeros((2, 2), np.float32), np.zeros(2), .01, np.zeros(2), max_coordinates=3)
        except original.CertificateBudgetError:
            pass
        else:
            raise AssertionError('coordinate cap bypassed')
    checks['scalar_default_and_custom_work_caps_preserved'] = True
    try:
        fast.certify_multioutput(np.zeros((2, 2), np.float32), np.zeros((2, 3)), .01, np.zeros((2, 3)), max_coordinates=11)
    except original.CertificateBudgetError:
        checks['joint_budget_counts_all_outputs'] = True
    else:
        raise AssertionError('joint coordinate cap bypassed')
    source = ROOT / 'data/civil_comments_engineering_preview.jsonl'
    rows = read_natural_jsonl(source)
    policy = {'schema': 'ccu-dyadic-verifier-development-design-v1',
              'source_sha256': sha256_file(source), 'evidence_role': 'engineering_nonconfirmatory',
              'inputs': [{'records': 100, 'dimension': 64, 'outputs': 7},
                         {'records': 100, 'dimension': 768, 'outputs': 1}],
              'feature_contract': 'existing lexical engineering features; no semantic encoder',
              'candidate_rule': 'original multioutput Newton solve at lambda .01 on all preview records',
              'repetitions': 5, 'order_seed': 202710041,
              'backends': ['original_fraction_v1', fast.BACKEND],
              'same_sigmoid_and_exact_endpoints_required': True,
              'default_coordinate_cap': 2_000_000, 'parameter_tolerance': '1e-8',
              'threads': 1, 'child_wall_timeout_seconds': 120,
              'large_primary_execution_claim': False,
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}}
    write(out / 'design_lock.json', policy)
    candidates = []
    for dimension, outputs in ((64, 7), (768, 1)):
        x, _ = lexical_engineering_features(rows, dimension)
        y = np.asarray([[r['fields'][t] for t in TARGETS[:outputs]] for r in rows], np.float64)
        fit = multi.solve_multioutput(x, y, .01)
        path = out / f'candidate_d{dimension}_q{outputs}.npz'
        np.savez(path, x=x, y=y, w=fit['weights'])
        record = {'dimension': dimension, 'outputs': outputs, 'coordinates': x.size * outputs,
                  'input': path.name, 'input_sha256': sha256_file(path),
                  'fit': {k: v for k, v in fit.items() if k != 'weights'}}
        candidates.append(record)
    write(out / 'candidates.json', candidates)
    plan = [(c, repeat, backend) for c in candidates for repeat in range(5) for backend in policy['backends']]
    random.Random(policy['order_seed']).shuffle(plan)
    write(out / 'run_plan.json', [{'dimension': c['dimension'], 'outputs': c['outputs'], 'repeat': repeat, 'backend': backend} for c, repeat, backend in plan])
    runs = []
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for sequence, (candidate, repeat, backend) in enumerate(plan):
        filename = f"run_{sequence:02d}_d{candidate['dimension']}_q{candidate['outputs']}_r{repeat}_{backend}.json"
        destination = out / filename
        started = time.perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-input', str((out / candidate['input']).resolve()),
                   '--worker-backend', backend, '--worker-output', str(destination.resolve())]
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
            status = 'completed' if completed.returncode == 0 else 'failed'
            detail = {'returncode': completed.returncode, 'stdout': completed.stdout, 'stderr': completed.stderr}
        except subprocess.TimeoutExpired as error:
            status, detail = 'timeout', {'exception': str(error)}
        row = {'sequence': sequence, 'dimension': candidate['dimension'], 'outputs': candidate['outputs'],
               'coordinates': candidate['coordinates'], 'repeat': repeat, 'backend': backend,
               'status': status, 'report': filename, 'process_wall_seconds': time.perf_counter() - started, **detail}
        if status == 'completed':
            report = json.loads(destination.read_text())
            row.update(certificate_seconds=report['certificate']['elapsed_seconds'], peak_rss_bytes=report['peak_rss_bytes'],
                       certificate_passed=report['certificate']['meets_parameter_tolerance'], report_sha256=sha256_file(destination))
        runs.append(row)
        write(out / 'runs.json', runs)
        print(json.dumps({'completed': len(runs), 'planned': len(plan), 'status': status}), flush=True)
    checks['all_planned_fresh_process_runs_retained'] = len(runs) == len(plan) == 20
    checks['all_fresh_process_runs_completed'] = all(r['status'] == 'completed' for r in runs)
    summaries = []
    for candidate in candidates:
        cell = [r for r in runs if r['dimension'] == candidate['dimension'] and r['outputs'] == candidate['outputs']]
        originals = [r for r in cell if r['backend'] == 'original_fraction_v1' and r['status'] == 'completed']
        optimized = [r for r in cell if r['backend'] == fast.BACKEND and r['status'] == 'completed']
        if len(originals) == len(optimized) == 5:
            reference = json.loads((out / originals[0]['report']).read_text())['certificate']
            for row in cell:
                certificate = json.loads((out / row['report']).read_text())['certificate']
                assert all(certificate[k] == reference[k] for k in JOINT_FIELDS)
                for left, right in zip(certificate['scalar_certificates'], reference['scalar_certificates']):
                    assert all(left[k] == right[k] for k in FIELDS)
            summaries.append({'dimension': candidate['dimension'], 'outputs': candidate['outputs'],
                              'coordinates': candidate['coordinates'],
                              'original_median_certificate_seconds': float(np.median([r['certificate_seconds'] for r in originals])),
                              'dyadic_median_certificate_seconds': float(np.median([r['certificate_seconds'] for r in optimized])),
                              'original_max_peak_rss_bytes': max(r['peak_rss_bytes'] for r in originals),
                              'dyadic_max_peak_rss_bytes': max(r['peak_rss_bytes'] for r in optimized),
                              'all_exact_endpoints_and_bounds_equal': True,
                              'all_candidates_passed_certificate': all(r['certificate_passed'] for r in cell)})
    checks['natural_endpoints_identical_at_both_dimensions'] = len(summaries) == 2
    checks['all_natural_candidates_passed_unchanged_radius_gate'] = all(s['all_candidates_passed_certificate'] for s in summaries)
    report = {'schema': 'ccu-dyadic-convex-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'exact_algebra_cases': exact_cases, 'natural_fresh_process_runs': len(runs), 'summaries': summaries,
              'natural_input_sha256': sha256_file(source), 'primary_native_scale_executed': False,
              'source_sha256': {p.name: sha256_file(p) for p in [Path(__file__), Path(fast.__file__), Path(original.__file__), Path(multi.__file__)]}}
    write(out / 'checks.json', report)
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
