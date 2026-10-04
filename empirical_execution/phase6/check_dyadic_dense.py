"""Dense arithmetic check on a fixed rotation of genuine Civil lexical features.

This changes feature coordinates, not corpus records or original target values.
The resulting vectors are not semantic encoder outputs or primary-study data.
"""
from pathlib import Path
import argparse
import json
import os
import random
import subprocess
import sys
import time
import numpy as np

from ccu.data import read_natural_jsonl, lexical_engineering_features, sha256_file
from phase5 import convex_multioutput as multi
from phase6 import dyadic_convex as fast
from phase6.check_dyadic_convex import FIELDS, JOINT_FIELDS, TARGETS, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'phase6/results/dyadic_dense_release')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'data/civil_comments_engineering_preview.jsonl'
    policy = {'schema': 'ccu-dense-dyadic-development-design-v1',
              'source_sha256': sha256_file(source), 'evidence_role': 'engineering_nonconfirmatory',
              'records': 100, 'dimension': 768, 'outputs': 7, 'coordinates': 537600,
              'feature_rule': 'existing lexical FP32 vectors multiplied by fixed Gaussian QR orthogonal matrix, then FP32; no renormalization',
              'rotation_seed': 202710042, 'repetitions': 5, 'order_seed': 202710043,
              'backends': ['original_fraction_v1', fast.BACKEND],
              'same_sigmoid_and_exact_endpoints_required': True,
              'default_coordinate_cap': 2_000_000, 'parameter_tolerance': '1e-8',
              'threads': 1, 'child_wall_timeout_seconds': 120,
              'semantic_encoder_used': False, 'new_corpus_or_labels_created': False,
              'large_primary_execution_claim': False}
    write(out / 'design_lock.json', policy)
    rows = read_natural_jsonl(source)
    sparse, _ = lexical_engineering_features(rows, 768)
    generator = np.random.default_rng(policy['rotation_seed'])
    rotation, diagonal = np.linalg.qr(generator.standard_normal((768, 768)))
    rotation *= np.where(np.diag(diagonal) < 0, -1., 1.)
    x = (sparse.astype(np.float64) @ rotation).astype(np.float32)
    y = np.asarray([[row['fields'][target] for target in TARGETS] for row in rows], np.float64)
    fit = multi.solve_multioutput(x, y, .01)
    candidate = out / 'candidate_dense_d768_q7.npz'
    np.savez(candidate, x=x, y=y, w=fit['weights'])
    write(out / 'candidate.json', {'input_sha256': sha256_file(candidate),
                                 'natural_source_sha256': sha256_file(source),
                                 'rotation_sha256': multi.scalar._hash_array(rotation),
                                 'orthogonality_max_abs_error': float(np.max(np.abs(rotation.T @ rotation - np.eye(768)))),
                                 'input_nonzero_fraction': float(np.count_nonzero(sparse) / sparse.size),
                                 'rotated_nonzero_fraction': float(np.count_nonzero(x) / x.size),
                                 'fit': {key: value for key, value in fit.items() if key != 'weights'}})
    # Release construction arrays before launching isolated measurements.
    del rotation, diagonal, sparse, x, y, fit
    plan = [(repeat, backend) for repeat in range(5) for backend in policy['backends']]
    random.Random(policy['order_seed']).shuffle(plan)
    write(out / 'run_plan.json', [{'repeat': repeat, 'backend': backend} for repeat, backend in plan])
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    runner = Path(__file__).with_name('check_dyadic_convex.py').resolve()
    runs = []
    for sequence, (repeat, backend) in enumerate(plan):
        filename = f'run_{sequence:02d}_r{repeat}_{backend}.json'
        destination = out / filename
        command = [sys.executable, str(runner), '--worker-input', str(candidate.resolve()),
                   '--worker-backend', backend, '--worker-output', str(destination.resolve())]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
            status = 'completed' if completed.returncode == 0 else 'failed'
            detail = {'returncode': completed.returncode, 'stdout': completed.stdout, 'stderr': completed.stderr}
        except subprocess.TimeoutExpired as error:
            status, detail = 'timeout', {'exception': str(error)}
        row = {'sequence': sequence, 'repeat': repeat, 'backend': backend, 'coordinates': policy['coordinates'],
               'status': status, 'report': filename, 'process_wall_seconds': time.perf_counter() - started, **detail}
        if status == 'completed':
            result = json.loads(destination.read_text())
            row.update(certificate_seconds=result['certificate']['elapsed_seconds'], peak_rss_bytes=result['peak_rss_bytes'],
                       certificate_passed=result['certificate']['meets_parameter_tolerance'], report_sha256=sha256_file(destination))
        runs.append(row)
        write(out / 'runs.json', runs)
        print(json.dumps({'completed': len(runs), 'planned': len(plan), 'status': status}), flush=True)
    checks = {'all_planned_runs_retained': len(runs) == 10,
              'all_runs_completed': all(row['status'] == 'completed' for row in runs)}
    summary = None
    if checks['all_runs_completed']:
        reference = json.loads((out / runs[0]['report']).read_text())['certificate']
        for row in runs:
            certificate = json.loads((out / row['report']).read_text())['certificate']
            assert all(certificate[key] == reference[key] for key in JOINT_FIELDS)
            for left, right in zip(certificate['scalar_certificates'], reference['scalar_certificates']):
                assert all(left[key] == right[key] for key in FIELDS)
        checks['all_dense_gradient_endpoints_and_bounds_identical'] = True
        checks['all_dense_candidates_pass_unchanged_certificate'] = all(row['certificate_passed'] for row in runs)
        summary = {'dimension': 768, 'outputs': 7, 'records': 100, 'coordinates': policy['coordinates']}
        for backend in policy['backends']:
            subset = [row for row in runs if row['backend'] == backend]
            summary[backend] = {'median_certificate_seconds': float(np.median([r['certificate_seconds'] for r in subset])),
                                'maximum_certificate_seconds': max(r['certificate_seconds'] for r in subset),
                                'maximum_peak_rss_bytes': max(r['peak_rss_bytes'] for r in subset)}
    report = {'schema': 'ccu-dyadic-dense-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'summary': summary,
              'primary_native_scale_executed': False,
              'source_sha256': {p.name: sha256_file(p) for p in [Path(__file__), runner, Path(fast.__file__), Path(multi.__file__), Path(multi.scalar.__file__)]}}
    write(out / 'checks.json', report)
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
