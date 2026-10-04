CALIBRATION-ONLY TASK CONFIGURATION
==================================

Status
------
This module completes executable selection rules, not actual empirical task
selection. The available Civil100 preview lacks genuine article/publication IDs;
it is rejected without selecting lambda. No real tag vocabulary, lambda or task
decision threshold is claimed. Existing phase2 engineering lambda stays a declared
engineering constant, not a retrospectively calibrated hyperparameter.

Authoritative original choices
------------------------------
output/empirical_program/counterfactual_curation_empirical_protocol.tex specifies:
  * Top 20 question tags by calibration-only frequency, lexicographic ties.
  * All original questions remain, including all-zero chosen-tag targets.
  * Five fixed whole-source calibration folds.
  * Lambda grid 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1.
  * Unweighted squared loss; ties choose larger lambda.
  * Average-loss ridge, no intercept, every coordinate penalized, lambda*n shift.
  * Calibration-only tag decision thresholds and an explicit zero-positive rule.

Prospective completion conventions (new, not falsely attributed to the protocol)
-------------------------------------------------------------------------------
1. SHA256(UTF8("ccu-v1-task-cv" + NUL + source_unit_id)); sort sources by digest
   and then ID, assign cyclically to five folds. This balances group counts, not
   row counts. CV loss pools every heldout record/output equally; it does not
   average fold means with equal fold weights.
2. A primary calibration lock requires at least five genuine source groups.
   Correctly derived missing-source singleton records are RETAINED in vocabulary,
   fitting, scoring and folds; they cannot satisfy the genuine-group requirement.
   All folds must contain rows and all training complements must contain rows.
   The genuine-group requirement is separate from fold allocation; a fold can
   consist of unknown singleton groups. No source occurs in multiple folds.
3. Exact equality of computed FP64 CV loss triggers the larger-lambda tie rule;
   no floating tolerance or near-tie discretion is used. Stable record-ID order
   pins each solver input, but this is not a cross-platform arithmetic theorem.
4. At the selected lambda, use calibration out-of-fold scores to maximize each
   tag's F1 over every observed-score >= boundary and the never-positive option.
   Score tie blocks are indivisible. F1 comparisons use exact integer ratios.
   Ties choose the strictest threshold. Zero-positive calibration tags always
   predict no positives, represented by a distinct never_positive kind (not an
   infinity pretending to be JSON). Thresholds never depend on heldout test data.
5. These OOF losses are selection criteria, not unbiased post-selection utility
   estimates. The vocabulary itself uses the prescribed whole calibration pool.

Representation contract
-----------------------
Features are finite aligned stored FP32 values; products/solves use FP64 without
renormalizing. For non-fixture roles, provenance.feature_contract is either:
  native_normalized_fp32: unit norms within 1e-5 storage tolerance; or
  fixed_projection_fp32: projection_lineage containing projection_matrix_sha256,
    source_feature_cache_sha256, source_dimension, output_dimension.
The latter permits norms below one after the protocol's fixed orthonormal
projection. Hash declarations do not prove that a matrix was orthonormal or an
encoder was run; these remain external provenance gates. Orthogonality and actual
cache derivation must be checked by the artifact-intake/execution workflow.

Numerical solver
----------------
Each fold uses primal numpy.linalg.solve if feature dimension <= training rows,
and the algebraically equivalent dual solve otherwise. This calibration-only
policy is explicit; it does not change the main experiment's common decoder.
Require normalized normal-equation residual <=1e-8. Log solver and residual per
candidate/fold. The check is numerical, not the exact-rational release certificate.

Interfaces
----------
Import empirical_execution.phase4.model_selection from repository root.

select_tag_vocabulary(calibration_records, provenance) -> sealed vocabulary lock
encode_tag_targets(records, vocabulary_lock) -> FP64 multi-hot matrix
encode_heldout_tag_targets_for_evaluation(test_records, vocabulary_lock)
    -> FP64 heldout multi-hot matrix, only in the separate evaluation process
select_ridge_regularization(fp32_features, calibration_records, provenance,
                            vocabulary_lock=None) -> sealed lambda lock
select_tag_decision_thresholds(lambda_lock) -> sealed decision-threshold lock
apply_tag_thresholds(score_matrix, thresholds_lock) -> binary prediction matrix
verify(lock) checks a lock's canonical SHA256 against its current contents.

Selection inputs must contain only partition="calibration" rows. encode_tag_targets
allows calibration/train only. Selection rejects any test/training row BEFORE
reading its original fields. Required record keys: record_id, partition,
source_unit_id and original_fields. Source IDs must agree with frozen phase3 source
derivation; missing sources use unknown:<record_id>. Civil labels come from original
toxicity; Stack tags come from original Tags (bracketed XML string or explicit list).

The explicit heldout evaluation encoder is a separate application-only API. It
requires a previously sealed vocabulary and exclusively test-partition rows,
constructs their original multi-hot targets without changing the vocabulary, and
does not select any setting. It intentionally reads gold heldout tags when the
evaluation process is run; it must never be invoked by a selection workflow.
Unknown/zero-target heldout questions remain present. Selection APIs themselves
continue to refuse all test rows before reading test tags. This distinction makes
eventual heldout evaluation possible without allowing test-tuned configuration.

Minimum provenance: dataset_id, evidence_role, population_scope; lambda additionally
requires encoder_id and encoder_revision, plus the feature contract above.
Allowed roles: confirmatory_calibration, engineering_nonconfirmatory,
software_fixture_only. The last is exclusively for algebraic software checks.

Every lock includes source/input hashes; lambda locks additionally bind the exact
stored feature and target arrays, source folds, complete candidate ledger and
selected OOF scores. Decision locks bind lambda and vocabulary locks. Code hashes
include this module and phase3 source derivation. Locks containing calibration
labels/OOF scores are private selection artifacts, not blinded annotation packs.

Every lock sets external_provenance_verified_by_this_module=false and
confirmatory_study_ready=false. A checksum is integrity metadata, not authentication.
No metadata declaration, selected vocabulary or good CV loss can authorize the
entire study or establish actual dataset/model provenance.

Verification
------------
python3 -m empirical_execution.phase4.check_model_selection
Writes results/model_selection_checks.json. The checks use explicit tiny transient
algebra/metadata fixtures, never presented as an empirical corpus. Independent
augmented least squares verifies every CV candidate; natural Civil100 is used only
to verify rejection for absent genuine source IDs. Actual model selection remains
blocked pending source-rich calibration data and pinned representation caches.
