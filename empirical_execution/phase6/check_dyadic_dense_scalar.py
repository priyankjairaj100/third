"""Dense Civil toxicity certificate cost at its actual single-output contract."""
from pathlib import Path
import argparse
import json
import os
import random
import subprocess
import sys
import time
import numpy as np

from ccu.data import sha256_file
from phase6 import dyadic_convex as fast
from phase6.check_dyadic_convex import FIELDS, JOINT_FIELDS, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'phase6/results/dyadic_dense_scalar_release')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'phase6/results/dyadic_dense_release/candidate_dense_d768_q7.npz'
    policy = {'schema': 'ccu-dense-scalar-dyadic-development-design-v1',
              'source_candidate_sha256': sha256_file(source),
              'source_candidate_rule': 'same dense rotated Civil vectors; original toxicity column and its independent fitted head',
              'evidence_role': 'engineering_nonconfirmatory', 'records': 100, 'dimension': 768,
              'outputs': 1, 'coordinates': 76800, 'repetitions': 5, 'order_seed': 202710044,
              'backends': ['original_fraction_v1', fast.BACKEND], 'threads': 1,
              'default_coordinate_cap': 2_000_000, 'child_wall_timeout_seconds': 120,
              'semantic_encoder_used': False, 'native_primary_acceptance': False}
    write(out / 'design_lock.json', policy)
    with np.load(source, allow_pickle=False) as arrays:
        x, y, w = arrays['x'], arrays['y'][:, :1], arrays['w'][:, :1]
    candidate = out / 'candidate_dense_d768_q1.npz'
    np.savez(candidate, x=x, y=y, w=w)
    write(out / 'candidate.json', {'input_sha256': sha256_file(candidate), 'source_candidate_sha256': sha256_file(source),
                                 'rotated_nonzero_fraction': float(np.count_nonzero(x) / x.size),
                                 'target': 'original Civil toxicity fraction'})
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
        row = {'sequence': sequence, 'repeat': repeat, 'backend': backend, 'coordinates': 76800,
               'status': status, 'report': filename, 'process_wall_seconds': time.perf_counter() - started, **detail}
        if status == 'completed':
            result = json.loads(destination.read_text())
            row.update(certificate_seconds=result['certificate']['elapsed_seconds'], peak_rss_bytes=result['peak_rss_bytes'],
                       certificate_passed=result['certificate']['meets_parameter_tolerance'], report_sha256=sha256_file(destination))
        runs.append(row)
        write(out / 'runs.json', runs)
    checks = {'all_planned_runs_retained': len(runs) == 10,
              'all_runs_completed': all(row['status'] == 'completed' for row in runs)}
    summary = None
    if checks['all_runs_completed']:
        reference = json.loads((out / runs[0]['report']).read_text())['certificate']
        for row in runs:
            certificate = json.loads((out / row['report']).read_text())['certificate']
            assert all(certificate[key] == reference[key] for key in JOINT_FIELDS)
            assert all(certificate['scalar_certificates'][0][key] == reference['scalar_certificates'][0][key] for key in FIELDS)
        checks['all_dense_scalar_endpoints_and_bounds_identical'] = True
        checks['all_candidates_pass_unchanged_certificate'] = all(row['certificate_passed'] for row in runs)
        summary = {'dimension': 768, 'outputs': 1, 'records': 100, 'coordinates': 76800}
        for backend in policy['backends']:
            subset = [row for row in runs if row['backend'] == backend]
            summary[backend] = {'median_certificate_seconds': float(np.median([r['certificate_seconds'] for r in subset])),
                                'maximum_certificate_seconds': max(r['certificate_seconds'] for r in subset),
                                'maximum_peak_rss_bytes': max(r['peak_rss_bytes'] for r in subset)}
    report = {'schema': 'ccu-dyadic-dense-scalar-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'summary': summary,
              'primary_native_scale_executed': False,
              'source_sha256': {p.name: sha256_file(p) for p in [Path(__file__), runner, Path(fast.__file__)]}}
    write(out / 'checks.json', report)
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
