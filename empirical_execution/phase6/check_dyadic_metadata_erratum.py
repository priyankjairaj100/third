"""Metadata-only correction check. Historical timing evidence stays unchanged."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

from ccu.data import sha256_file
from phase6 import dyadic_convex as corrected
from phase6.check_dyadic_convex import FIELDS, JOINT_FIELDS, write

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'phase6/history/dyadic_metadata_v1/dyadic_convex.py'
OLD_SHA = 'eaf003048efdf9e59975f5a1e2ba26e646a362b8fc68862c1aab0447448e13f1'


def main():
    checks = {}
    assert sha256_file(HISTORY) == OLD_SHA
    spec = importlib.util.spec_from_file_location('historical_dyadic_metadata_v1', HISTORY)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    checks['original_source_bytes_preserved'] = True
    shared = int('1000000')
    references = [shared, shared]
    shallow = corrected.integer_bytes(references)
    unique = sys.getsizeof(references) + sys.getsizeof(shared)
    checks['shared_reference_counterexample_is_explicit'] = shallow > unique
    checks['metric_is_descriptive_shallow_sum'] = shallow == sys.getsizeof(references) + 2 * sys.getsizeof(shared)
    candidate_path = ROOT / 'phase6/results/dyadic_dense_scalar_release/candidate_dense_d768_q1.npz'
    with np.load(candidate_path, allow_pickle=False) as candidate:
        x, y, w = (candidate[key] for key in ('x', 'y', 'w'))
    before = historical.certify_multioutput(x, y, .01, w)
    after = corrected.certify_multioutput(x, y, .01, w)
    checks['natural_joint_bounds_unchanged'] = all(before[key] == after[key] for key in JOINT_FIELDS)
    checks['natural_gradient_endpoints_unchanged'] = all(before['scalar_certificates'][0][key] == after['scalar_certificates'][0][key] for key in FIELDS)
    old_work = before['scalar_certificates'][0]['dyadic_work']
    new_work = after['scalar_certificates'][0]['dyadic_work']
    checks['counter_value_unchanged_only_description_corrected'] = old_work['tracked_integer_list_bytes_lower_bound'] == new_work['tracked_integer_list_shallow_byte_sum']
    checks['incorrect_lower_bound_key_removed'] = 'tracked_integer_list_bytes_lower_bound' not in new_work
    checks['new_scope_disclaims_total_memory_bounds'] = 'neither a lower bound nor an upper bound' in new_work['memory_scope'] and new_work['memory_metadata_revision'] == 2
    empty_x, empty_y, empty_w = np.empty((0, 2), np.float32), np.empty(0), np.zeros(2)
    old_empty = historical.certify_logistic(empty_x, empty_y, .01, empty_w)
    new_empty = corrected.certify_logistic(empty_x, empty_y, .01, empty_w)
    checks['empty_shared_zero_arithmetic_unchanged'] = all(old_empty[key] == new_empty[key] for key in FIELDS)
    historical_releases = []
    for name in ('dyadic_convex_release', 'dyadic_dense_release', 'dyadic_dense_scalar_release'):
        path = ROOT / 'phase6/results' / name
        report = json.loads((path / 'checks.json').read_text())
        assert report['source_sha256']['dyadic_convex.py'] == OLD_SHA
        for run in json.loads((path / 'runs.json').read_text()):
            result = json.loads((path / run['report']).read_text())
            if run['backend'] == corrected.BACKEND:
                assert result['certificate']['code_sha256'] == OLD_SHA
                assert all(c['input_bindings']['code_sha256'] == OLD_SHA for c in result['certificate']['scalar_certificates'])
            else:
                # Original comparator reports correctly bind their own unchanged code.
                assert result['certificate']['code_sha256'] == sha256_file(ROOT / 'phase5/convex_multioutput.py')
                assert all(c['input_bindings']['code_sha256'] == sha256_file(ROOT / 'phase4/convex.py') for c in result['certificate']['scalar_certificates'])
        historical_releases.append({'directory': str(path.relative_to(ROOT)),
                                    'checks_sha256': sha256_file(path / 'checks.json'),
                                    'runs_sha256': sha256_file(path / 'runs.json')})
    checks['all_40_historical_timing_runs_bind_preserved_source'] = True
    report = {'schema': 'ccu-dyadic-memory-metadata-erratum-v1',
              'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'historical_source_path': str(HISTORY.relative_to(ROOT)),
              'historical_source_sha256': OLD_SHA,
              'corrected_source_sha256': sha256_file(Path(corrected.__file__)),
              'focused_check_sha256': sha256_file(Path(__file__)),
              'natural_candidate_sha256': sha256_file(candidate_path),
              'correction': 'Rename the shallow referenced-byte sum; shared objects can count repeatedly. It is not a lower bound on memory.',
              'arithmetic_or_sigmoid_changed': False, 'timing_benchmarks_repeated': False,
              'historical_timing_reports_modified': False,
              'historical_reports': historical_releases,
              'focused_checks_are_new_timing_evidence': False}
    write(ROOT / 'phase6/results/dyadic_metadata_erratum.json', report)
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
