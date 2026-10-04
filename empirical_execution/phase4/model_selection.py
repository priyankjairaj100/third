"""Calibration-only task configuration; never a provenance or readiness certificate.

The original protocol fixes top-20 tags, five source folds, average-loss ridge,
the six lambda values and the larger-lambda tie rule. Fold assignment and the
tag-threshold F1 rule below are explicit prospective completion conventions.
All software-fixture results remain software evidence; this module cannot verify
whether supplied source metadata, labels or feature provenance are authentic.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np

from empirical_execution.phase3 import panels as _panels
from empirical_execution.phase3.panels import source_unit

VERSION = "ccu-calibration-task-selection-v1"
LAMBDA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FOLD_SALT = "ccu-v1-task-cv"
ROLES = {"confirmatory_calibration", "engineering_nonconfirmatory", "software_fixture_only"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    result = dict(value)
    result["sha256"] = digest(value)
    return result


def verify(value):
    if not isinstance(value, dict) or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Missing or altered sealed selection lock")


def _array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _base(kind, provenance):
    return {"schema": VERSION, "kind": kind, "provenance": provenance,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_derivation_code_sha256": hashlib.sha256(Path(_panels.__file__).read_bytes()).hexdigest(),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "external_provenance_verified_by_this_module": False,
            "confirmatory_study_ready": False, "test_labels_accessed": False}


def _records(records, provenance, *, partitions=("calibration",)):
    # Partition rejection precedes ANY inspection of labels/original_fields.
    if not records or any(r.get("partition") not in partitions for r in records):
        raise ValueError("Only explicitly permitted calibration/train partitions accepted; test labels stay sealed")
    required = ("dataset_id", "evidence_role", "population_scope")
    if not isinstance(provenance, dict) or any(not isinstance(provenance.get(k), str) or not provenance[k].strip() for k in required):
        raise ValueError("Dataset, population and evidence provenance required")
    if provenance["evidence_role"] not in ROLES:
        raise ValueError("Unrecognized evidence role")
    ids = [r.get("record_id") for r in records]
    if any(not isinstance(x, str) or not x or "\0" in x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Nonempty unique stable record IDs required")
    sources = []
    for row in records:
        source, kind = source_unit(row, provenance["dataset_id"])
        if not (kind.startswith("native_") or kind == "unknown_singleton"):
            raise ValueError("Native or explicitly derived unknown-singleton sources required")
        if row.get("source_unit_id") != source:
            raise ValueError("Prepared source ID must match original metadata")
        sources.append(source)
    return ids, sources


def _source_counts(sources):
    return {"genuine_source_groups": len({s for s in sources if not s.startswith("unknown:")}),
            "unknown_singleton_groups": len({s for s in sources if s.startswith("unknown:")}),
            "unknown_singleton_rows_retained": sum(s.startswith("unknown:") for s in sources)}


def _tags(row):
    value = row.get("original_fields", {}).get("Tags")
    if isinstance(value, str):
        parts = re.findall(r"<([^<>]+)>", value)
        if "".join("<" + x + ">" for x in parts) != value:
            raise ValueError("Tags must be the original fully bracketed XML tag string")
    elif isinstance(value, list) and all(isinstance(x, str) and x and "<" not in x and ">" not in x for x in value):
        parts = value
    else:
        raise ValueError("Original question-local tags required")
    return sorted(set(parts))


def select_tag_vocabulary(calibration_records, provenance):
    ids, sources = _records(calibration_records, provenance)
    if provenance["dataset_id"] not in {"askubuntu", "english_stackexchange"}:
        raise ValueError("Top-20 vocabulary is defined only for Stack question tasks")
    rows = sorted(zip(ids, sources, map(_tags, calibration_records)))
    counts = Counter(tag for _, _, tags in rows for tag in tags)
    if len(counts) < 20:
        raise ValueError("Fewer than 20 observed calibration tags: prescribed 20-output task cannot be locked")
    ranking = sorted(counts, key=lambda x: (-counts[x], x))
    vocab = ranking[:20]
    chosen = set(vocab)
    return seal({**_base("tag_vocabulary", provenance), "vocabulary": vocab,
                 "selection_rule": "top20_question_presence_counts_descending_then_Unicode_lexicographic",
                 "calibration_records": len(rows), "source_units": len(set(sources)),
                 **_source_counts(sources),
                 "input_sha256": digest(rows), "source_sha256": digest(sorted(zip(ids, sources))),
                 "calibration_record_ids": [r[0] for r in rows],
                 "ranking_ledger": [{"tag": t, "questions": counts[t], "selected": t in chosen} for t in ranking],
                 "all_zero_target_questions": sum(not (set(tags) & chosen) for _, _, tags in rows),
                 "targets_include_all_calibration_questions": True})


def encode_tag_targets(records, vocabulary_lock):
    verify(vocabulary_lock)
    if vocabulary_lock.get("kind") != "tag_vocabulary":
        raise ValueError("Expected vocabulary lock")
    _records(records, vocabulary_lock["provenance"], partitions=("train", "calibration"))
    vocab = vocabulary_lock["vocabulary"]
    return np.asarray([[int(t in set(_tags(row))) for t in vocab] for row in records], dtype=np.float64)


def encode_heldout_tag_targets_for_evaluation(test_records, vocabulary_lock):
    """Apply an already sealed vocabulary inside a separate evaluation process.

    This reads held-out gold tags solely to construct evaluation targets. It does
    not rank tags, select lambda, fit thresholds, amend the vocabulary or generate
    a new selection lock. Selection APIs themselves continue to reject test rows.
    """
    verify(vocabulary_lock)
    if vocabulary_lock.get("kind") != "tag_vocabulary":
        raise ValueError("Expected a previously frozen vocabulary lock")
    _records(test_records, vocabulary_lock["provenance"], partitions=("test",))
    vocab = vocabulary_lock["vocabulary"]
    return np.asarray([[int(t in set(_tags(row))) for t in vocab] for row in test_records], dtype=np.float64)


def fixed_source_folds(record_ids, source_ids):
    """Hash-sort whole sources, then assign consecutive sources cyclically to 0..4.

    This guarantees five nonempty folds when >=5 groups exist. It balances source
    counts, not record counts; every record retains equal weight in the CV risk.
    """
    if len(record_ids) != len(source_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Aligned unique IDs required")
    groups = sorted(set(source_ids), key=lambda s: (hashlib.sha256((FOLD_SALT + "\0" + s).encode()).digest(), s))
    if len(groups) < 5:
        raise ValueError("Five nonempty source-group folds require at least five genuine source groups")
    allocation = {s: i % 5 for i, s in enumerate(groups)}
    folds = np.asarray([allocation[s] for s in source_ids], dtype=np.int8)
    if any(not np.any(folds == i) or not np.any(folds != i) for i in range(5)):
        raise ValueError("Empty training or validation fold")
    return folds, [{"source_unit_id": s, "fold": allocation[s]} for s in sorted(groups)]


def ridge_head(features, targets, regularization):
    """FP64 average-loss ridge, no intercept, all coordinates penalized.

    Use primal solve for D<=N, algebraically equivalent dual solve otherwise;
    the selected solver and numerical residual are logged for every fold.
    """
    z, y = np.asarray(features, np.float64), np.asarray(targets, np.float64)
    if z.ndim != 2 or y.ndim != 2 or z.shape[0] != y.shape[0] or z.shape[1] == 0 or y.shape[1] == 0 or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Finite aligned feature and target matrices required")
    if isinstance(regularization, bool) or not isinstance(regularization, (int, float)) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive finite ridge regularization required")
    n, d = z.shape
    if not n:
        return np.zeros((d, y.shape[1]), np.float64), "empty_zero", 0.0
    shift = regularization * n
    if d <= n:
        a, h = z.T @ z + shift * np.eye(d), z.T @ y
        theta = np.linalg.solve(a, h)
        solver = "primal_numpy_solve"
    else:
        theta = z.T @ np.linalg.solve(z @ z.T + shift * np.eye(n), y)
        solver = "dual_numpy_solve"
    residual = z.T @ (z @ theta - y) + shift * theta
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(z.T @ y), np.finfo(float).tiny))
    if not np.isfinite(theta).all() or not math.isfinite(relative) or relative > 1e-8:
        raise ValueError("Calibration ridge solve failed normalized residual gate 1e-8")
    return theta, solver, relative


def select_ridge_regularization(fp32_features, calibration_records, provenance, *, vocabulary_lock=None):
    ids, sources = _records(calibration_records, provenance)
    if any(not isinstance(provenance.get(k), str) or not provenance[k].strip() for k in ("encoder_id", "encoder_revision")):
        raise ValueError("Pinned feature encoder declarations required")
    z = np.asanyarray(fp32_features)
    if z.ndim != 2 or z.dtype != np.dtype("float32") or z.shape[0] != len(ids) or z.shape[1] < 1 or not np.isfinite(z).all():
        raise ValueError("Aligned finite N by D stored FP32 features required")
    counts = _source_counts(sources)
    if counts["genuine_source_groups"] < 5:
        raise ValueError("At least five genuine source groups required to lock primary source-CV; unknown records are retained but cannot supply missing genuine groups")
    if provenance["evidence_role"] != "software_fixture_only":
        contract = provenance.get("feature_contract")
        if contract == "native_normalized_fp32":
            norms = np.linalg.norm(z.astype(np.float64), axis=1)
            if not np.all(np.abs(norms - 1) <= 1e-5):
                raise ValueError("Stored normalized FP32 learner vectors required; no normalization performed here")
        elif contract == "fixed_projection_fp32":
            lineage = provenance.get("projection_lineage", {})
            if (not isinstance(lineage, dict) or
                    any(not isinstance(lineage.get(k), str) or re.fullmatch(r"[0-9a-f]{64}", lineage[k]) is None
                        for k in ("projection_matrix_sha256", "source_feature_cache_sha256")) or
                    lineage.get("output_dimension") != z.shape[1] or
                    not isinstance(lineage.get("source_dimension"), int) or
                    isinstance(lineage.get("source_dimension"), bool) or lineage["source_dimension"] < z.shape[1]):
                raise ValueError("Fixed projection requires explicit source/cache/projection lineage and dimensions")
            # Do not renormalize the projection: the stored FP32 values define
            # this learner. Orthogonality and derivation remain provenance gates.
        else:
            raise ValueError("Declare native_normalized_fp32 or fixed_projection_fp32 feature contract")
    dataset = provenance["dataset_id"]
    if dataset == "civil_comments":
        raw = [r.get("original_fields", {}).get("toxicity") for r in calibration_records]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= 1 for t in raw):
            raise ValueError("Original crowd toxicity fractions required")
        y = np.asarray(raw, np.float64)[:, None]
        vocabulary_sha = None
    elif dataset in {"askubuntu", "english_stackexchange"}:
        if vocabulary_lock is None:
            raise ValueError("Sealed calibration vocabulary required")
        expected = select_tag_vocabulary(calibration_records, provenance)
        verify(vocabulary_lock)
        if vocabulary_lock["input_sha256"] != expected["input_sha256"] or vocabulary_lock["vocabulary"] != expected["vocabulary"]:
            raise ValueError("Vocabulary must come from these same calibration records and original tags")
        y = encode_tag_targets(calibration_records, vocabulary_lock)
        vocabulary_sha = vocabulary_lock["sha256"]
    else:
        raise ValueError("Unlabeled corpus cannot select task regularization")
    # Stable summation/solver row order; reversing an aligned input changes neither
    # folds nor the actual matrices passed to the FP64 solver.
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    z, y = z[order], y[order]
    ids, sources = [ids[i] for i in order], [sources[i] for i in order]
    folds, fold_ledger = fixed_source_folds(ids, sources)
    all_scores, ledger = {}, []
    for lam in LAMBDA_GRID:
        oof = np.empty_like(y)
        fold_rows = []
        for fold in range(5):
            train, val = folds != fold, folds == fold
            theta, solver, residual = ridge_head(z[train], y[train], lam)
            oof[val] = z[val].astype(np.float64) @ theta
            if not np.isfinite(oof[val]).all():
                raise ValueError("Nonfinite calibration prediction")
            squares = (oof[val] - y[val]) ** 2
            sse = math.fsum(float(t) for t in squares.ravel())
            fold_rows.append({"fold": fold, "training_rows": int(train.sum()), "validation_rows": int(val.sum()),
                              "sum_squared_error": sse, "mean_squared_loss": sse / squares.size,
                              "solver": solver, "normalized_solve_residual": residual})
        mse = math.fsum(float(t) for t in ((oof - y) ** 2).ravel()) / y.size
        ledger.append({"lambda": lam, "lambda_hex": float(lam).hex(), "mean_squared_loss": mse,
                       "mean_squared_loss_hex": mse.hex(), "folds": fold_rows})
        all_scores[lam] = oof
    selected = min(ledger, key=lambda entry: (entry["mean_squared_loss"], -entry["lambda"]))
    lam = selected["lambda"]
    return seal({**_base("ridge_regularization", provenance), "lambda": lam, "lambda_hex": lam.hex(),
                 "grid": list(LAMBDA_GRID), "selection_ledger": ledger,
                 "loss": "unweighted_mean_over_all_calibration_records_and_output_coordinates",
                 "tie_rule": "exact_FP64_loss_equality_choose_larger_lambda_no_tolerance",
                 "fold_rule": "SHA256(UTF8(ccu-v1-task-cv + NUL + source_unit_id)); sorted_sources_round_robin_5",
                 "fold_ledger": fold_ledger, "record_ids": ids, "source_ids": sources,
                 **counts, "genuine_group_gate": "at_least_five_genuine_groups_required; unknown_singletons_retained_in_CV",
                 "record_folds": folds.tolist(), "feature_sha256": _array_hash(z), "label_sha256": _array_hash(y),
                 "source_sha256": digest(list(zip(ids, sources))), "input_sha256": digest([ids, sources, _array_hash(z), _array_hash(y)]),
                 "vocabulary_lock_sha256": vocabulary_sha,
                 "selected_oof_scores": all_scores[lam].tolist(), "calibration_targets": y.tolist(),
                 "selected_oof_sha256": _array_hash(all_scores[lam]),
                 "calibration_only_oof_is_not_an_unbiased_post_selection_performance_estimate": True,
                 "learner_contract": "average_loss_ridge_no_intercept_all_coordinates_penalized_lambda_times_current_count"})


def _tag_threshold(scores, targets):
    n, positives = len(scores), int(np.sum(targets))
    if positives == 0:
        return {"kind": "never_positive", "threshold": None, "positives": 0,
                "rule": "zero_positive_calibration_tag_predicts_no_positives", "f1": {"numerator": 0, "denominator": 1},
                "candidate_count": len(set(map(float, scores))) + 1}
    # >= a score predicts that tie block and all larger blocks. Sweep descending,
    # keeping the first (strictest) threshold when exact rational F1 ties occur.
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    tp, predicted, best, chosen = 0, 0, Fraction(-1), None
    cursor = 0
    while cursor < n:
        threshold = float(scores[order[cursor]])
        while cursor < n and float(scores[order[cursor]]) == threshold:
            tp += int(targets[order[cursor]])
            predicted += 1
            cursor += 1
        f1 = Fraction(2 * tp, positives + predicted)
        if f1 > best:
            best = f1
            chosen = {"kind": "score_at_least", "threshold": threshold, "threshold_hex": threshold.hex(),
                      "positives": positives, "predicted_positives": predicted, "true_positives": tp,
                      "f1": {"numerator": f1.numerator, "denominator": f1.denominator}}
    return {**chosen, "candidate_count": len(set(map(float, scores))) + 1,
            "rule": "maximize_per_tag_F1_on_selected_lambda_OOF_scores_exact_count_ratio_ties_larger_threshold"}


def select_tag_decision_thresholds(lambda_lock):
    verify(lambda_lock)
    if lambda_lock.get("kind") != "ridge_regularization" or not lambda_lock.get("vocabulary_lock_sha256"):
        raise ValueError("Stack vocabulary-bound ridge lock required")
    scores, targets = np.asarray(lambda_lock["selected_oof_scores"], float), np.asarray(lambda_lock["calibration_targets"], float)
    if scores.shape != targets.shape or scores.ndim != 2 or scores.shape[1] != 20 or not np.isfinite(scores).all() or not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Twenty finite OOF score columns and binary original tag targets required")
    if _array_hash(scores) != lambda_lock["selected_oof_sha256"] or _array_hash(targets) != lambda_lock["label_sha256"]:
        raise ValueError("OOF scores/labels do not match locked hashes")
    return seal({**_base("tag_decision_thresholds", lambda_lock["provenance"]),
                 "lambda_lock_sha256": lambda_lock["sha256"], "vocabulary_lock_sha256": lambda_lock["vocabulary_lock_sha256"],
                 "source_sha256": lambda_lock["source_sha256"], "feature_sha256": lambda_lock["feature_sha256"],
                 "input_sha256": lambda_lock["input_sha256"], "label_sha256": lambda_lock["label_sha256"],
                 "thresholds": [{"output_coordinate": i, **_tag_threshold(scores[:, i], targets[:, i])} for i in range(20)],
                 "status": "calibration_configuration_only_not_confirmatory_performance",
                 "prospective_completion_convention": "per_tag_F1_on_source_group_OOF_scores_at_selected_lambda; >= comparison; strictest_ties; zero_positive_never_positive"})


def apply_tag_thresholds(scores, thresholds_lock):
    verify(thresholds_lock)
    a = np.asarray(scores, np.float64)
    if a.ndim != 2 or a.shape[1] != 20 or not np.isfinite(a).all() or thresholds_lock.get("kind") != "tag_decision_thresholds":
        raise ValueError("Finite N by 20 score matrix and threshold lock required")
    out = np.zeros(a.shape, dtype=np.int8)
    for entry in thresholds_lock["thresholds"]:
        if entry["kind"] == "score_at_least":
            out[:, entry["output_coordinate"]] = a[:, entry["output_coordinate"]] >= entry["threshold"]
    return out
