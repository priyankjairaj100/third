"""Separate calibration-only logistic decisions at the frozen ridge lambda.

No new lambda search occurs. Source folds and tags retain their original locks.
The selected OOF scores are probabilities from logistic fits, never ridge scores.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
from scipy.special import expit

from empirical_execution.phase4 import model_selection as ms
from phase5.convex_multioutput import solve_multioutput

SOLVER_POLICY = {'gradient_tolerance': 1e-11, 'max_iterations': 100,
                 'max_backtracks': 40}


def select_logistic_decision_thresholds(fp32_features, calibration_records, provenance,
                                       ridge_lambda_lock, vocabulary_lock):
    """Recompute input bindings before five independent logistic fold fits."""
    ms.verify(ridge_lambda_lock)
    ms.verify(vocabulary_lock)
    expected = ms.select_ridge_regularization(fp32_features, calibration_records,
                                              provenance, vocabulary_lock=vocabulary_lock)
    if ridge_lambda_lock != expected:
        raise ValueError('frozen ridge lambda lock does not match calibration inputs')
    if provenance['dataset_id'] not in {'askubuntu', 'english_stackexchange'}:
        raise ValueError('logistic tag thresholds require the original Stack task')
    ids = ridge_lambda_lock['record_ids']
    lookup = {row['record_id']: i for i, row in enumerate(calibration_records)}
    x = np.asarray(fp32_features)[[lookup[rid] for rid in ids]]
    y = np.asarray(ridge_lambda_lock['calibration_targets'], np.float64)
    folds = np.asarray(ridge_lambda_lock['record_folds'])
    lam = ridge_lambda_lock['lambda']
    oof = np.empty_like(y)
    ledger = []
    for fold in range(5):
        train, validation = folds != fold, folds == fold
        fit = solve_multioutput(x[train], y[train], lam, **SOLVER_POLICY)
        outputs = fit['output_results']
        if any(result['status'] != 'numerically_converged' or
               not math.isfinite(result['gradient_norm_diagnostic']) or
               result['gradient_norm_diagnostic'] > SOLVER_POLICY['gradient_tolerance']
               for result in outputs):
            raise RuntimeError('a logistic calibration fold failed its frozen numerical gate')
        logits = x[validation].astype(np.float64) @ fit['weights']
        oof[validation] = expit(logits)
        if not np.isfinite(oof[validation]).all():
            raise RuntimeError('nonfinite logistic OOF probabilities')
        bce = np.logaddexp(0.0, logits) - y[validation] * logits
        ledger.append({'fold': fold, 'training_rows': int(train.sum()),
                       'validation_rows': int(validation.sum()),
                       'mean_binary_cross_entropy_diagnostic': math.fsum(map(float, bce.ravel())) / bce.size,
                       'head_sha256': ms._array_hash(fit['weights']),
                       'output_statuses': [r['status'] for r in outputs],
                       'output_gradient_norms': [r['gradient_norm_diagnostic'] for r in outputs],
                       'output_iterations': [r['iterations'] for r in outputs]})
    result = {'schema': 'ccu-logistic-tag-decisions-v1',
              'kind': 'logistic_tag_decision_thresholds',
              'provenance': provenance, 'ridge_lambda_lock_sha256': ridge_lambda_lock['sha256'],
              'vocabulary_lock_sha256': vocabulary_lock['sha256'],
              'lambda': lam, 'lambda_policy': 'same frozen ridge lambda; no logistic retuning',
              'source_sha256': ridge_lambda_lock['source_sha256'],
              'feature_sha256': ridge_lambda_lock['feature_sha256'],
              'label_sha256': ridge_lambda_lock['label_sha256'],
              'input_sha256': ridge_lambda_lock['input_sha256'],
              'record_ids': ids, 'record_folds': folds.tolist(),
              'fold_ledger': ridge_lambda_lock['fold_ledger'],
              'logistic_fit_ledger': ledger, 'solver_policy': dict(SOLVER_POLICY),
              'learner_objective': 'sum_over_outputs_mean_row_BCE + lambda/2 * Frobenius_squared; no intercept',
              'selected_oof_probabilities': oof.tolist(), 'selected_oof_sha256': ms._array_hash(oof),
              'thresholds': [{'output_coordinate': c, **ms._tag_threshold(oof[:, c], y[:, c])} for c in range(20)],
              'decision_rule': 'probability >= threshold; exact rational F1; strictest tied threshold; zero-positive predicts none',
              'selection_scope': 'calibration only; all calibration questions retained including zero-target and unknown-source rows',
              'performance_scope': 'OOF values are selection diagnostics, not unbiased post-selection performance estimates',
              'optimizer_error_certificate': False, 'test_labels_accessed': False,
              'confirmatory_study_ready': False, 'external_provenance_verified_by_this_module': False,
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_sha256': {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                    for p in [ms.__file__, Path(__file__).resolve().parents[1] / 'phase5/convex_multioutput.py',
                                              Path(__file__).resolve().parents[1] / 'phase4/convex.py']}}
    return ms.seal(result)


def apply_logistic_thresholds(probabilities, thresholds_lock):
    ms.verify(thresholds_lock)
    a = np.asarray(probabilities, np.float64)
    if thresholds_lock.get('kind') != 'logistic_tag_decision_thresholds' or a.ndim != 2 or a.shape[1] != 20 or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError('finite N by 20 probabilities and a logistic threshold lock required')
    entries = thresholds_lock.get('thresholds')
    if not isinstance(entries, list) or len(entries) != 20 or {e.get('output_coordinate') for e in entries} != set(range(20)):
        raise ValueError('exactly twenty unique logistic decision rules required')
    out = np.zeros(a.shape, np.int8)
    for entry in entries:
        kind = entry.get('kind')
        if kind == 'score_at_least':
            threshold = entry.get('threshold')
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError('logistic probability threshold must lie in [0,1]')
            out[:, entry['output_coordinate']] = a[:, entry['output_coordinate']] >= threshold
        elif kind != 'never_positive':
            raise ValueError('unknown logistic threshold rule')
    return out
