"""Task-selection branch/algebra checks; no synthetic empirical dataset is saved.

Small in-memory matrices below are explicitly software fixtures, not corpus or
human evidence. The genuine Civil preview is checked ONLY for required rejection
because its source metadata are absent. No empirical lambda is selected here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from empirical_execution.phase3.panels import source_unit
from empirical_execution.phase4 import model_selection as ms

ROOT = Path(__file__).resolve().parents[2]
CHECKS = []


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    CHECKS.append(name)


def rejects(name, function):
    try:
        function()
    except ValueError:
        check(name, True)
    else:
        raise AssertionError(name + ": accepted invalid input")


def provenance(dataset):
    return {"dataset_id": dataset, "evidence_role": "software_fixture_only", "population_scope": "in_memory_algebra_fixtures_not_empirical_data",
            "encoder_id": "software_fixture_identity_coordinates_not_encoder", "encoder_revision": "software_fixture_not_model_revision"}


def fixtures(dataset, count=10):
    rows = []
    owners = [1, 1, 1, 1, 2, 2, 3, 4, 5, 5][:count]
    for i, owner in enumerate(owners):
        fields = ({"publication_id": "SOFTWARE_FIXTURE_NOT_REAL_PUBLICATION", "article_id": owner, "toxicity": (i % 3) / 2}
                  if dataset == "civil_comments" else {"site": "SOFTWARE_FIXTURE_NOT_REAL_SITE", "OwnerUserId": owner,
                  "Tags": "".join(f"<tag-{j:02d}>" for j in range(20)) if i < 8 else ("<rare>" if i == 8 else "")})
        row = {"record_id": f"SOFTWARE_FIXTURE_{i:02d}", "partition": "calibration", "original_fields": fields}
        row["source_unit_id"] = source_unit(row, dataset)[0]
        rows.append(row)
    return rows


def main():
    natural_path = ROOT / "empirical_execution/data/civil_comments_engineering_preview.jsonl"
    natural = [json.loads(s) for s in natural_path.read_text().splitlines()]
    # Metadata stays absent: never add fictional groups to natural observations.
    prepared = [{**r, "partition": "calibration", "original_fields": r["fields"]} for r in natural]
    for row in prepared:
        row["source_unit_id"] = source_unit(row, "civil_comments")[0]
    prov = {**provenance("civil_comments"), "evidence_role": "engineering_nonconfirmatory", "population_scope": "known_nonconfirmatory_Civil100_preview"}
    rejects("natural_Civil100_missing_sources_rejected_without_selecting_lambda", lambda: ms.select_ridge_regularization(np.zeros((100, 2), np.float32), prepared, prov))
    rows = fixtures("civil_comments")
    p = provenance("civil_comments")
    z = np.asarray([[i / 10, (i % 3) - 1, 1] for i in range(10)], np.float32)
    chosen = ms.select_ridge_regularization(z, rows, p)
    ms.verify(chosen)
    check("six_lambdas_five_folds_logged", len(chosen["selection_ledger"]) == 6 and all(len(c["folds"]) == 5 for c in chosen["selection_ledger"]))
    check("all_readiness_false", chosen["confirmatory_study_ready"] is False and chosen["external_provenance_verified_by_this_module"] is False)
    check("every_source_in_exactly_one_fold", all(len({f for s, f in zip(chosen["source_ids"], chosen["record_folds"]) if s == source}) == 1 for source in set(chosen["source_ids"])))
    check("five_nonempty_folds", set(chosen["record_folds"]) == set(range(5)))
    check("stable_aligned_input_permutation", chosen == ms.select_ridge_regularization(z[::-1].copy(), rows[::-1], p))
    unknown = copy.deepcopy(rows[0]); unknown["record_id"] = "SOFTWARE_FIXTURE_UNKNOWN"
    unknown["original_fields"]["article_id"] = None
    unknown["source_unit_id"] = source_unit(unknown, "civil_comments")[0]
    with_unknown = ms.select_ridge_regularization(np.vstack((z, z[:1])), rows + [unknown], p)
    check("unknown_singleton_record_retained_with_five_genuine_groups", len(with_unknown["record_ids"]) == 11 and with_unknown["unknown_singleton_groups"] == 1 and with_unknown["genuine_source_groups"] == 5)
    projected_provenance = {**p, "evidence_role": "engineering_nonconfirmatory", "feature_contract": "fixed_projection_fp32",
                            "projection_lineage": {"projection_matrix_sha256": "a" * 64, "source_feature_cache_sha256": "b" * 64,
                                                   "source_dimension": 8, "output_dimension": 3}}
    # This is still an in-memory interface fixture, not measured projections.
    projected = ms.select_ridge_regularization(z, rows, projected_provenance)
    check("declared_projection_not_renormalized", projected["feature_sha256"] == chosen["feature_sha256"])
    rejects("projection_missing_lineage_rejected", lambda: ms.select_ridge_regularization(z, rows, {**projected_provenance, "projection_lineage": {}}))
    max_error = 0.0
    y = np.asarray([r["original_fields"]["toxicity"] for r in rows])[:, None]
    folds = np.asarray(chosen["record_folds"])
    for candidate in chosen["selection_ledger"]:
        pred = np.empty_like(y)
        for fold in range(5):
            train, val = folds != fold, folds == fold
            # Independent augmented least-squares oracle, no production solver.
            aug_z = np.vstack((z[train].astype(float), np.sqrt(candidate["lambda"] * train.sum()) * np.eye(z.shape[1])))
            aug_y = np.vstack((y[train], np.zeros((z.shape[1], 1))))
            theta = np.linalg.lstsq(aug_z, aug_y, rcond=None)[0]
            pred[val] = z[val].astype(float) @ theta
        mse = float(np.mean((pred - y) ** 2))
        max_error = max(max_error, abs(mse - candidate["mean_squared_loss"]))
    check("independent_augmented_least_squares_CV_oracle", max_error < 1e-10)
    tie = ms.select_ridge_regularization(np.zeros_like(z), rows, p)
    check("exact_loss_tie_chooses_largest_lambda", tie["lambda"] == max(ms.LAMBDA_GRID))
    rejects("fewer_than_five_genuine_source_groups", lambda: ms.select_ridge_regularization(z[:8], rows[:8], p))
    malformed = copy.deepcopy(rows); malformed[0]["source_unit_id"] = "unknown:fixture"
    rejects("mismatched_source_id_rejected", lambda: ms.select_ridge_regularization(z, malformed, p))
    bad = copy.deepcopy(rows); bad[0]["original_fields"]["toxicity"] = True
    rejects("boolean_original_label_rejected", lambda: ms.select_ridge_regularization(z, bad, p))
    rejects("FP64_cache_rejected", lambda: ms.select_ridge_regularization(z.astype(float), rows, p))
    rejects("non_normalized_primary_features_rejected", lambda: ms.select_ridge_regularization(z, rows, {**p, "evidence_role": "confirmatory_calibration"}))
    class SealedHeldout(dict):
        def get(self, key, default=None):
            if key == "original_fields":
                raise AssertionError("Heldout label fields accessed")
            return super().get(key, default)
    heldout = SealedHeldout({"partition": "test", "record_id": "test"})
    rejects("test_partition_rejected_before_label_access", lambda: ms.select_ridge_regularization(np.vstack((z, z[:1])), rows + [heldout], p))
    stack = fixtures("askubuntu")
    sp = provenance("askubuntu")
    vocab = ms.select_tag_vocabulary(stack, sp)
    check("twenty_tags_selected_lexicographic_ties", vocab["vocabulary"] == [f"tag-{i:02d}" for i in range(20)])
    targets = ms.encode_tag_targets(stack, vocab)
    check("zero_target_questions_retained", targets.shape == (10, 20) and np.all(targets[8:] == 0) and vocab["all_zero_target_questions"] == 2)
    stack_unknown = copy.deepcopy(stack[-1]); stack_unknown["original_fields"]["OwnerUserId"] = None
    stack_unknown["source_unit_id"] = source_unit(stack_unknown, "askubuntu")[0]
    unknown_stack = stack[:-1] + [stack_unknown]
    unknown_vocab = ms.select_tag_vocabulary(unknown_stack, sp)
    check("unknown_owner_zero_target_question_preserved", unknown_vocab["unknown_singleton_groups"] == 1 and ms.encode_tag_targets(unknown_stack, unknown_vocab).shape == (10, 20))
    check("vocabulary_order_independent", vocab == ms.select_tag_vocabulary(stack[::-1], sp))
    rejects("test_tags_not_accessed", lambda: ms.select_tag_vocabulary(stack + [heldout], sp))
    eval_rows = copy.deepcopy(stack)
    for row in eval_rows:
        row["partition"] = "test"
    vocab_before = copy.deepcopy(vocab)
    eval_targets = ms.encode_heldout_tag_targets_for_evaluation(eval_rows, vocab)
    check("heldout_gold_tags_only_applied_after_vocabulary_lock", np.array_equal(eval_targets, targets) and vocab == vocab_before)
    rejects("ordinary_target_encoder_refuses_test", lambda: ms.encode_tag_targets(eval_rows, vocab))
    rejects("heldout_evaluation_encoder_refuses_calibration", lambda: ms.encode_heldout_tag_targets_for_evaluation(stack, vocab))
    few_tags = copy.deepcopy(stack)
    for r in few_tags:
        r["original_fields"]["Tags"] = "<only-one>"
    rejects("less_than_twenty_tags_blocks_lock", lambda: ms.select_tag_vocabulary(few_tags, sp))
    stack_lambda = ms.select_ridge_regularization(z, stack, sp, vocabulary_lock=vocab)
    thresholds = ms.select_tag_decision_thresholds(stack_lambda)
    check("twenty_frozen_tag_thresholds", len(thresholds["thresholds"]) == 20)
    predictions = ms.apply_tag_thresholds(np.asarray(stack_lambda["selected_oof_scores"]), thresholds)
    check("finite_binary_threshold_outputs", predictions.shape == (10, 20) and set(predictions.ravel()) <= {0, 1})
    zero = ms._tag_threshold(np.asarray([0.0, 1.0, -1.0]), np.zeros(3))
    check("zero_positive_rule_never_positive", zero["kind"] == "never_positive" and zero["threshold"] is None)
    tied = ms._tag_threshold(np.asarray([3., 2., 1., 0.]), np.asarray([1, 0, 0, 1]))
    check("F1_tie_chooses_strictest_threshold", tied["threshold"] == 3.0)
    duplicates = ms._tag_threshold(np.asarray([2., 2., 1.]), np.asarray([1, 0, 1]))
    check("score_ties_are_indivisible_prediction_block", duplicates["threshold"] == 1.0 and duplicates["predicted_positives"] == 3)
    damaged = copy.deepcopy(chosen); damaged["lambda"] = 42
    rejects("altered_lock_rejected", lambda: ms.verify(damaged))
    for lam in (0, -1, float("nan"), True):
        rejects("invalid_regularization_" + repr(lam), lambda lam=lam: ms.ridge_head(z, y, lam))
    # Independent dual/primal equality on a wide matrix; still algebra only.
    wide = np.column_stack((z, z * 2, z * 3, z * 4))
    theta, solver, _ = ms.ridge_head(wide[:4], y[:4], 0.01)
    reference = np.linalg.solve(wide[:4].astype(float).T @ wide[:4].astype(float) + .04 * np.eye(12), wide[:4].astype(float).T @ y[:4])
    check("dual_matches_independent_primal", solver == "dual_numpy_solve" and np.max(np.abs(theta - reference)) < 1e-10)
    report = {"passed": True, "checks": CHECKS, "check_count": len(CHECKS), "maximum_independent_CV_MSE_difference": max_error,
              "software_fixture_scope": "Transient explicit algebra/metadata fixtures only; no synthetic empirical dataset or measurements",
              "natural_input": {"rows": len(natural), "sha256": hashlib.sha256(natural_path.read_bytes()).hexdigest(),
                                "outcome": "rejected_for_missing_genuine_source_metadata_no_lambda_selected"},
              "empirical_task_settings_selected": False, "confirmatory_study_ready": False,
              "module_sha256": hashlib.sha256(Path(ms.__file__).read_bytes()).hexdigest()}
    destination = Path(__file__).parent / "results/model_selection_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(CHECKS), "maximum_independent_CV_MSE_difference": max_error,
                      "natural_task_selection": "blocked_missing_genuine_sources"}))


if __name__ == "__main__":
    main()
