# Three-arm convex execution

`convex_program.py` supplies the formerly missing three-arm driver. It accepts fixed local curator/learner arrays, original fractional/binary targets, source ownership/kinds, a sealed request manifest, fixed lambda, and a separately declared evaluation matrix. The standard schedule chooses the first 32 R and first 32 S trajectory IDs; smaller engineering schedules must be explicit. Every missing planned path remains in `missing_trajectories.json`. Unknown/proxy ownership cannot supply a genuine-source S trajectory.

The methods are:

| Method | Curation and optimization access |
|---|---|
| `fresh_optimum` | Independently rebuild the graph from all actual retained curator rows, select, gather original selected learner rows and solve from zero. Full retained raw access and fresh curation cost are disclosed. |
| `eligible_payload_retraining` | Keep only finite-horizon eligible learner features/targets and maintenance metadata. Update own CSR membership, gather current selected payload and solve from zero. |
| `certified_convex_repair` | Same eligible payload, counts, compaction policy and solver/certificate as the preceding method; start at its own previously released head. All retained gradient/Hessian/certificate queries are charged. |

Every successful head gets the same exact stored-value optimization certificate from the frozen scalar/matrix verifier. The matrix objective and Frobenius error contract are defined in `CONVEX_MULTIOUTPUT_README.txt`. The default radius remains 1e-8. Numerical convergence alone does not release a head. Each pair of released heads is checked using exact rational squared distance against the sum of certified radii. An unsuccessful method retains a failure record and all later checkpoints as unexecuted; the other methods continue. If fewer than two heads release, pairwise agreement is undefined, not an affirmative check.

The inherited exact-verification budget defaults to 2,000,000 output-row-coordinates. A native 10k-by768-by20 problem can exceed it substantially. The Python API exposes `certificate_budget` for a prospectively locked resource choice; it must be fixed using development feasibility before confirmation. An over-budget result is preserved as a failure, never replaced by an ordinary numerical residual or an after-the-fact tolerance change. The preview checks do not establish that full native certificates fit the eventual time/memory allowance.

## Actual eligible payload

`LogisticPayload` uses the frozen packed CSR antijoin for source/record membership, degree buckets, finite-horizon pruning and compaction. Its input to that counter kernel is a single zero dummy coordinate. Thus the kernel never allocates an irrelevant real d-by-d ridge moment matrix. Actual eligible FP32 feature and FP64 target rows are separately owned and aligned to those packed rows. Constructor arrays and the original graph/curator are not retained. Selected gathers read only these retained rows.

The zero dummy arrays and tiny associated moments still cost memory/work and are accounted. Real payload moves with the kernel's compaction; live/stale real bytes, selected gathered bytes and real compaction read/write bytes are reported. Tombstones remain physically allocated until the fixed compaction trigger, and they remain charged. This is a logical-state service, not physical erasure. It avoids forcing a logistic baseline to perform unnecessary real ridge outer products.

Each trajectory gets independent cold and warm states. Final snapshots include the complete owned counter state, actual retained payload, warm head and operation metadata. A hash-checked restore loads only that snapshot, validates counter invariants and payload shapes and reproduces the saved selected IDs/head. Saved candidates from failed methods are explicitly distinguished from heads released for the current target. Serialization/read buffers, times and bytes are recorded. Snapshot restoration does not recreate discarded original rows or depend on the original graph.

This driver executes in a shared evaluator process. Kernel no-reaccess enforcement and the primary isolated worker timing design are separate; this program makes neither an isolated-process nor a systems-speedup claim. Fresh graph computation, state construction, maintenance, full numerical data passes, exact certification, task prediction and persistence are distinguishable stages. Independent oracle work is not available to the eligible services, although it is present in the evaluator process for checks. Global OS cache/allocator behavior and process peak memory are not inferred from named arrays.

Each released model is written as its own FP64 NPY file, with timed output and exact file-byte count, including the initial head. A fresh-optimum release costs `curator_rebuild_seconds + method_seconds_before_output + head_release_seconds`; the maintained methods omit the fresh-curator term. `release_total_seconds` records that sum. Initial setup and final persistence are additional lifecycle stages. Combined audit JSON, re-reading outputs for their hashes and independent membership comparisons are evaluator bookkeeping outside these method times. Those exclusions are explicit; the shared process does not establish a deployment no-reaccess boundary.

## Local API and CLI

```python
run_convex_program(curator_fp32, learner_fp32, targets_fp64,
    record_ids, source_ids, source_kinds, request_manifest, out_dir,
    lambda_reg=locked_lambda, threshold=locked_tau,
    eval_features=evaluation_fp32, eval_targets=evaluation_fp64)
```

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 empirical_execution/phase5/convex_program.py \
  --bundle /absolute/local/bundle.json --out /absolute/new/run
```

The JSON bundle contains identity/source/request fields named as in the API; `lambda_reg`, `threshold`, `evaluation_scope`; and five array descriptors named `curator`, `learner`, `targets`, `evaluation_features`, `evaluation_targets`. Each descriptor is `{ "path": "relative.npy", "sha256": "actual-file-hash" }`. Path escapes and changed bytes are refused. Arrays must have the registered FP32 feature and FP64 target types; labels are not synthesized or rounded into artificial binary targets. Optional `allocations` explicitly alters only an engineering test schedule.

The API/CLI accepts authentic arrays once supplied, but it does not authenticate them itself. It requires role `engineering_nonconfirmatory`; a primary role or a proposed dossier bypass is refused because the full study-wide acceptance/activation workflow is absent. This refusal protects the difference between a reusable executable branch and a completed primary study. A later audited activator must verify original assets, encoder derivation, semantic quality, source partition, held-out evaluation and the frozen task/resource plan before promoting an actual scientific run. Hashes alone are insufficient.

## Evidence and checks

`check_convex_program.py` runs two natural Civil100 R trajectories at four checkpoints, with all three methods: 8 checkpoint rows, 24 method releases and 30 certificates including the six original fits. The requested two S trajectories remain unavailable because native source IDs are absent. Only the original toxicity target is used. The lexical64 learner/lexical128 curator and same original Civil100 evaluation pool make these implementation checks, not semantic or held-out task results.

Additional tiny algebraic software fixtures exercise multioutput full-source withdrawal, final empty-target zero and preserved certificate-budget failures. They are not synthetic empirical corpora. Source methods use declared artificial fixture ownership only inside these checks. Constructor-array mutation, finite-horizon selection, actual selected payload, snapshot restoration and primary/proxy-source refusal are checked separately.

The initial engineering run and first release are preserved with source snapshots. `results/convex_program_release_v2/checks.json` names the authoritative final code and run; duplicate reruns are not additional evidence. The actual prescribed Civil/Stack32R/32S program still requires original source-rich data, pinned semantic caches, calibration/quality pass, locked tasks and genuine held-out evaluation. No existing engineering check establishes those prerequisites.
