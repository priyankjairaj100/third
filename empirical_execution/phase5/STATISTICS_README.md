# Trajectory and crossed-source analysis

`statistics.py` implements the analysis contract for future locked inputs. It does not start confirmation or supply missing native sources, ratings, model caches, or hypothesis-test assumptions.

## Registry and unconditional effects

`lock_registry(cells, seed=20271004, replicates=10000)` accepts only outcome-free identity fields: `dataset_id`, `panel_id`, `configuration_id`, `arm`, `trajectory_id`, and `checkpoint`. Every trajectory in one dataset/panel/configuration/arm must have the same planned checkpoints. The checksum binds content; external evidence must establish that it was fixed before outcomes. Separate configurations/panels/corpora/arms remain separate analysis groups.

`reconcile_registry(lock, records)` requires one explicit status and metric mapping per supplied cell, refuses unplanned or duplicate cells, and inserts absent planned cells with `status="missing"`. Failed, timeout, skipped and infeasible statuses remain visible. A structural oracle metric may still be valid when a separate prediction method failed; metrics must therefore be populated only where their recorded acquisition succeeded. A timeout's elapsed time is a censored lower bound: store it in a separately named bound field and leave the unknown completed-lifecycle metric null. The generic numeric API cannot infer censoring from an arbitrary metric name. Zero values are measurements, never exclusion rules. Missing/undefined values are `None`, never zero or a favorable denominator floor.

`summarize_registry(lock, records, metrics=[...])` uses 10,000 seed-locked resamples of independent complete trajectories. The same sampled trajectories serve every checkpoint and metric within a group. Independently sampled request arms are not artificially paired. If a metric has any missing value, its full-registry mean and primary interval are undefined; an explicitly named observed-only descriptive mean and missing counts remain available. Bootstrap draws containing a missing metric remain undefined. No interval based on silently dropping these draws is returned.

`prediction_metrics(y,before,frozen,oracle)` computes the protocol's signed normalized loss difference, absolute normalized loss, raw prediction RMS, normalized RMS, original MSE and score standard deviation, and the fixed 0.01 exceedance indicators. Negative signed effects remain negative. Zero normalizers return null. Optional locked output thresholds compute macro-F1 with the same `>=` convention as Phase 4 task selection; positive infinity implements the declared never-positive rule. Zero-positive/zero-prediction F1 contributes zero by explicit convention.

`admission_metrics` preserves zero admissions and returns undefined per-unit rates at zero deleted volume. `exact_activation_interval` supplies Clopper–Pearson two-sided and one-sided limits for an independent, identically distributed Bernoulli trajectory population at one checkpoint. Deterministic adversarial stress trajectories do not acquire an iid sampling interpretation simply by calling this routine.

`paired_method_bootstrap` uses the same locked trajectories for paired mean differences and **ratio of means**, recomputing the ratio in each replicate. A zero mean baseline makes that replicate undefined; no denominator floor or deletion of such replicates is used. It does not substitute a mean of per-trajectory ratios. `equal_rs_lifecycle_difference` weights the R and S means equally regardless of unequal sample sizes and makes no deployment-mixture claim. `equal_rs_lifecycle_bootstrap` independently resamples the complete paired within-arm differences for R and S, applies the fixed half weights in every replicate, and reports its percentile interval without inventing a p-value. Callers must supply every planned trajectory from the locked registry; missing/nonfinite arrays are refused, not filtered.

## Secondary crossed bootstrap

`crossed_source_bootstrap` expects complete finite `[trajectory, checkpoint, test record, output]` predictions plus fixed labels and predeletion predictions. It separately samples whole trajectories and whole test provenance groups with replacement. All records of a sampled group inherit its multiplicity, method pairing is preserved, and every checkpoint shares that trajectory draw. The MSE and score-variance denominators are recomputed from each resampled test corpus. Source-level sufficient aggregates avoid expanding duplicated test records. Weighted within/between-group variance avoids a cancellation-based positive floor at exactly constant predictions.

Optional macro-F1 recomputes group-weighted TP/predicted-positive/true-positive totals in each replicate before taking F1 and averaging output coordinates; it never bootstraps precomputed F1 scores as independent cells. Complete prediction failure makes this crossed analysis unavailable rather than licensing successful-trajectory selection. The registry retains all failures separately.

This secondary interval concerns the specified held-out source-group sampling population, conditional on the training corpus and curator. It does not represent training-corpus uncertainty or corpus replication. Genuine source declarations require external provenance review. Explicit unknown singleton groups remain identified, must contain exactly one record, and cannot substitute for the required minimum two genuine test groups. Algebra tests can opt into `software_fixture_only=True`; that is not a scientific source analysis. The Civil100 preview lacks genuine source fields and no natural crossed-source bootstrap is executed.

## Multiplicity and decision limits

`holm_family(..., family="signed_loss")` fixes four tests: Civil R/S and AskUbuntu R/S. `family="equal_RS_lifecycle"` fixes two corpus-level equal-R/S systems comparisons. Every family member is required, with `None` for unperformed tests; missing tests remain in the family with conservative effective p-value one. Input p-values must come from a separately justified, predeclared test. The module does not manufacture significance p-values from percentile bootstrap intervals, and valid Holm adjustment does not make invalid input p-values valid. Null-significance and mean equivalence do not establish negligible per-request impact.

## Checks and actual evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python empirical_execution/phase5/check_statistics.py
```

Algebra checks exercise shared trajectories/checkpoints, method pairing, all-zero outcomes, missing cells, atomic registry rejection, undefined ratios, negative effects, exact zero-event limits, full Holm family accounting, equal-arm weighting and crossed denominator/nonlinear metric behavior. These arrays are software fixtures, not empirical datasets.

The compatibility check reconstructs all 448 saved Phase 2 d64 checkpoint predictions on the original 18-record diagnostic evaluation set and reanalyzes their existing registry without changing any original result. It retains known cross-split leakage and other development limitations. Its registry is retrospectively constructed for software verification, explicitly not a preoutcome confirmatory registration. Files `results/statistics_checks.json` and `results/statistics_natural_compatibility.json` record this scope. No new semantic or source evidence results from this compatibility check.
