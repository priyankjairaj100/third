"""Algebra-only selection checks. No invented corpus or empirical labels."""
from pathlib import Path
from fractions import Fraction
import copy
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from empirical_execution.phase4 import model_selection as ms
from empirical_execution.phase4.check_model_selection import fixtures, provenance
from phase6.logistic_selection import select_logistic_decision_thresholds, apply_logistic_thresholds

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = {}
    rows = fixtures('askubuntu')
    prov = provenance('askubuntu')
    x = np.asarray([[i / 10, (i % 3) - 1, 1] for i in range(10)], np.float32)
    vocabulary = ms.select_tag_vocabulary(rows, prov)
    ridge = ms.select_ridge_regularization(x, rows, prov, vocabulary_lock=vocabulary)
    lock = select_logistic_decision_thresholds(x, rows, prov, ridge, vocabulary)
    checks['same_frozen_ridge_lambda'] = lock['lambda'] == ridge['lambda']
    checks['five_separate_logistic_folds'] = len(lock['logistic_fit_ledger']) == 5 and len(lock['thresholds']) == 20
    checks['source_folds_unchanged'] = lock['record_folds'] == ridge['record_folds'] and lock['source_sha256'] == ridge['source_sha256']
    checks['not_a_primary_or_optimizer_certificate'] = not lock['confirmatory_study_ready'] and not lock['optimizer_error_certificate']
    checks['stable_input_order'] = lock == select_logistic_decision_thresholds(x[::-1].copy(), rows[::-1], prov, ridge, vocabulary)
    p = np.asarray(lock['selected_oof_probabilities'])
    y = np.asarray(ridge['calibration_targets'])
    folds = np.asarray(ridge['record_folds'])
    independent = np.empty_like(p)
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        a, b = x[tr].astype(float), y[tr]
        for c in range(20):
            def objective(w):
                logits = a @ w
                loss = np.mean(np.logaddexp(0, logits) - b[:, c] * logits) + lock['lambda'] * (w @ w) / 2
                gradient = a.T @ (expit(logits) - b[:, c]) / len(a) + lock['lambda'] * w
                return loss, gradient
            candidate = minimize(objective, np.zeros(x.shape[1]), jac=True, method='BFGS', options={'gtol': 1e-11, 'maxiter': 1000})
            independent[va, c] = expit(x[va].astype(float) @ candidate.x)
    discrepancy = float(np.max(np.abs(independent - p)))
    checks['independent_BFGS_probability_oracle'] = discrepancy < 1e-7
    decisions = apply_logistic_thresholds(p, lock)
    for c, entry in enumerate(lock['thresholds']):
        positives = int(y[:, c].sum())
        candidates = []
        for threshold in sorted(set(p[:, c]), reverse=True):
            prediction = p[:, c] >= threshold
            candidates.append((Fraction(2 * int((prediction * y[:, c]).sum()), positives + int(prediction.sum())), float(threshold)))
        expected = max(candidates)
        assert entry['threshold'] == expected[1]
    checks['independent_exact_F1_and_strictest_tie'] = True
    checks['probabilities_not_ridge_scores'] = not np.array_equal(p, np.asarray(ridge['selected_oof_scores']))
    def rejects(name, call):
        try:
            call()
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    ridge_thresholds = ms.select_tag_decision_thresholds(ridge)
    rejects('ridge_threshold_lock_rejected', lambda: apply_logistic_thresholds(p, ridge_thresholds))
    rejects('out_of_range_probability_rejected', lambda: apply_logistic_thresholds(np.full_like(p, 1.01), lock))
    altered = copy.deepcopy(ridge)
    altered['lambda'] = 0.123
    altered = ms.seal({k: v for k, v in altered.items() if k != 'sha256'})
    rejects('rehashed_wrong_lambda_rejected', lambda: select_logistic_decision_thresholds(x, rows, prov, altered, vocabulary))
    test_rows = copy.deepcopy(rows)
    test_rows[0]['partition'] = 'test'
    rejects('test_partition_rejected_before_selection', lambda: select_logistic_decision_thresholds(x, test_rows, prov, ridge, vocabulary))
    zero = ms._tag_threshold(np.asarray([.1, .7, .7]), np.zeros(3))
    checks['zero_positive_never_positive'] = zero['kind'] == 'never_positive'
    # The genuine Civil preview has no genuine source groups. It must not pass.
    natural_path = ROOT / 'data/civil_comments_engineering_preview.jsonl'
    natural = [json.loads(line) for line in natural_path.read_text().splitlines()]
    prepared = [{**r, 'partition': 'calibration', 'original_fields': r['fields']} for r in natural]
    for row in prepared:
        row['source_unit_id'] = ms.source_unit(row, 'civil_comments')[0]
    natural_prov = {**provenance('civil_comments'), 'evidence_role': 'engineering_nonconfirmatory',
                    'population_scope': 'Civil100 original lexical engineering preview'}
    rejects('actual_Civil_missing_sources_cannot_supply_task_lock', lambda: ms.select_ridge_regularization(np.zeros((100, 2), np.float32), prepared, natural_prov))
    report = {'schema': 'ccu-logistic-selection-checks-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'independent_probability_max_abs_error': discrepancy,
              'genuine_Civil_input_sha256': hashlib.sha256(natural_path.read_bytes()).hexdigest(),
              'genuine_empirical_logistic_configuration_selected': False,
              'software_fixtures_are_empirical_data': False,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), Path(__file__).with_name('logistic_selection.py')]}}
    (ROOT / 'phase6/results/logistic_selection_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    main()
