# Frozen-input task and sensitivity execution

`task_program.py` executes task effects and initial utility references for the applicable labeled `B_relevance`, `D_replication` and `G_boundary` cells of `study_registry.py`. It goes beyond enumerating jobs: supplied cells build the requested exact curator, fit heads, evaluate immutable diagnostic/test arrays and save predictions. Every planned task job is retained as completed, structurally empty/unavailable, missing input, unsupported by this evaluator or failed. No missing encoder is replaced by a lexical one and no favorable configuration is selected after inspecting results.

This is an **engineering evaluator**, not a primary-study acceptance gate, unlearning repair service or systems benchmark. `evidence_role='primary'` is rejected before output. Artifact checks do not establish authentic source provenance, legitimate human collection or original encoder derivation. Those whole-study checks remain separate. Filesystem presence or a registry entry cannot promote a run.

## What is executed

For each applicable configuration:

1. Verify the complete ordered training/evaluation record IDs, their disjointness, original FP64 task targets, and stored FP32 feature arrays. Each curator and learner carries explicit ordered row IDs. Verify the native request manifest against the original exact graph. Calibration records cannot overlap either evaluated population.
2. Resolve the particular curator and learner encoders, fixed priority, threshold and dimension. Native encoder-cross cells require both real encoders' separately supplied caches and learner-specific lambda locks; all four registry cells survive. All alternatives use the same post-E5 population rather than adding an MPNet filter.
3. For genuine calibration locks, recompute the calibration-only lambda selection or curator selection/quality from their actual supplied arrays, rows and responses. Match encoder names/revisions. Separate engineering constants use their own unmistakable schema and are explicitly **not calibration results**.
4. Fit initial curated ridge, uncurated ridge and same-size deterministic hash-selected ridge. The hash salt is `ccu-v1-utility-hash`; its selection never reads labels. The hash baseline uses exactly the curated count.
5. Fit a regularized logistic utility head on the same initial curated set and frozen features, using the same frozen lambda without test tuning. The prospective policy fixes gradient tolerance `1e-11`, at most 100 Newton iterations, 60 backtracks and a rigorous Frobenius optimizer-radius gate of `1e-8`. The exact verifier has a declared 2,000,000 output-row-coordinate budget. A budget exceedance or failed certificate remains a failed utility reference; already computed heads, predictions, optimizer statuses and costs are saved. This fixed budget is not a claim of feasibility for the registered native 10k panel.
6. At every frozen path checkpoint, expand whole-source requests where applicable, rebuild the correct selected IDs from the exact fixed graph, and compare fresh ridge against deletion from the originally selected set. All zero admissions, empty targets and signed adverse effects remain.

Ridge heads use the same fresh FP64 moments, `lambda * selected_count` regularization and Cholesky decoder. Each solve checks the fixed normalized residual gate `eta <= 1e-10`. This is a numerical gate, not a strict rational state certificate. Empty selection produces the prescribed zero head. Predictions promote frozen FP32 evaluation features without further normalization, and ridge scores are never clipped.

Raw predictions and both heads are saved per checkpoint. The program records task metrics, signed normalized loss, absolute normalized loss, prediction RMS, normalized prediction RMS, both denominators, exceedances, selected IDs, deleted counts and admissions. Zero normalizing denominators remain undefined through `task_metrics`/`statistics`; they are not given favorable floors. These raw aligned files support the separate trajectory/source bootstrap implementation. The program does not claim its checkpoints are independent statistical replicates.

## Sensitivities

- Encoder cross: distinct supplied curator/learner entries implement the complete 2×2 grid. Every cell has its own proper curator threshold and representation-specific lambda/evaluation cache.
- Priority: use the registry's fixed seeds with the exact Phase 3 priority formula.
- Threshold: implement exactly `max(0, tau - 0.02)` and `min(1, tau + 0.02)`. Genuine semantic alternatives require their own fresh quality audit. `prepare_sensitivity_validation` derives a clearly named secondary lock from the parent calibration lock, uses the unchanged calibration frame and reservoir-samples up to 200 from **all** above-alternative-threshold pairs. It uses the existing pair/assignment and three-human-majority protocol. It never pretends the alternate value won a new threshold selection. `sensitivity_quality` reconstructs that manifest and computes the exact finite-population lower bound from genuine responses. Missing responses block the semantic cell; a failed alternative quality audit is reported as failed and cannot replace the main gate. Engineering lexical alternatives explicitly carry no audited semantic quality claim.
- Dimension: retain the curator and change only learner features. The supplied projection must equal the recorded PCG64 Gaussian matrix's reduced QR with positive-diagonal sign convention, have orthonormal columns and match the fixed seed. Cast projected coordinates to FP32 and **do not renormalize**. Recompute the corresponding calibration features with the same matrix and require a separate projection-specific lambda lock.
- Conditioning: apply the locked 0.1/10 factors to the representation's calibrated lambda without retuning.

Shared graph-independent R/S and source-volume-matched paths remain identical across alternatives. Graph-dependent U/A paths cannot silently be moved to a different curator. A clipped empty source frame is recorded, not supplied with invented source identities.

FP32-state and CG service comparisons remain explicitly outside this task evaluator and retain their registry jobs with that status. They have their separate method implementations. Projection/conditioning registry groups may also request repair methods beyond O-G/B-F; the group ledger says these methods were not run by this evaluator. Running this task slice does not complete those systems cells. The full-method block's existing oracle/frozen effects should reference the matching relevance execution, not count duplicate fits as independent evidence.

## Multilabel rules

Original zero-target questions remain. The lambda-specific tag threshold lock is reconstructed from the selected calibration OOF scores, including the never-positive rule. No test-set threshold is selected. If the decision lock is missing, threshold-dependent F1 metrics are `null`; MSE and ranking average precision remain available. Ridge thresholds are not silently applied to logistic probabilities. The logistic utility reference therefore leaves F1 unavailable absent a separately specified calibration-only logistic decision rule.

`task_metrics.py` uses the recorded noninterpolated average-precision convention, zero AP for a zero-positive target, undefined one-class AUROC, and `>=` for locked tag thresholds. Civil uses original toxicity fractions and the fixed `toxicity >= 0.5` binary endpoint.

## Python interface and file runner

```python
summary = run_task_program(registry, jobs, bundles, new_output_directory)
```

`bundles` maps each registry `request_family` to:

- `dataset_id`, task (`civil` or `multilabel`), `train_ids`, `evaluation_ids`, `source_ids`;
- original FP64 `train_y` and `evaluation_y`;
- `base_curator`, optional `base_priority_seed`, and the frozen Phase 4 `requests` manifest;
- `curators[name]`: stored `train` FP32 array, ordered `train_ids`, threshold lock, and its calibration/quality dossier;
- `learners[name]`: stored `train`/`evaluation` FP32 arrays, ordered IDs, lambda lock, calibration dossier and optional calibration tag threshold lock;
- for projections, `projections[str(d)]` with `matrix`, fixed `seed`, and `data_independent=True`, plus each learner's `projection_parameters[str(d)]` containing its actual corresponding calibration data and lock.

Real encoder entries also provide the full encoder ID and immutable revision. Dossiers use the existing Phase 3/4 row, provenance, selection-response and validation-response structures. Every scalar, artifact, array, registry and source file is bound before model evaluation; code/input bindings are rechecked afterward.

The CLI accepts JSON bundles with file descriptors of exactly `{"path": "relative/file.npy", "sha256": "...", "kind": "npy"}`. `json` and `jsonl` kinds are also supported. It verifies byte hashes, rejects paths escaping the bundle directory, loads arrays with `allow_pickle=False`, and handles compressed registry job JSONL.

```bash
python3 -m empirical_execution.phase5.task_program \
  --registry registry.json --jobs jobs.jsonl.gz \
  --bundles bundles.json --out new-task-output
```

## Recorded check scope

The authoritative check is `results/task_program_release/`. It reuses the historical Civil82/18 split and true comment labels with explicitly named lexical feature maps. The lexical 2×2 engineering grid is **not** the semantic E5/MPNet grid. The diagnostic evaluation set has been reused and earlier leakage limitations remain; its metrics do not establish untouched-test task value.

The official registry's genuine semantic task cells are retained separately under `registered_missing_inputs/`, all blocked by missing inputs. No real model cache, source identity, calibrated threshold or human rating has been fabricated. The first `task_program_engineering/` run is a preserved earlier draft; use the final release and its source bindings for continuation.
