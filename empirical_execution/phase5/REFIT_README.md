# Full-refit boundary branch

`refit.py` implements the separate full-refit experiment described in the empirical protocol. It maintains its own original curation and original ridge head, then compares:

* **R:** recluster retained records, reorder within new clusters, suppress, rebuild moments and refit the head.
* **F:** restrict the original fitted cluster membership and original within-cluster priority to retained records, suppress again and refit the head.
* **B:** retain only surviving members of this branch's original selected set and refit the head.

R/F/B do not use the main global curator's initial model. F retains fitted objects influenced by withdrawn records, so reproducing F is not full-pipeline unlearning. The experiment measures this applicability boundary; R is not a same-target speed baseline for the main fixed curator.

## Pinned official code and license

The official Meta repository is pinned at commit `6b4194511202c29b0e1ac8c730996777449ea2a4`:
https://github.com/facebookresearch/SemDeDup/tree/6b4194511202c29b0e1ac8c730996777449ea2a4

Ten original source/configuration/license files are stored under `vendor/SemDeDup/`. The retrieval used the connected GitHub repository API at this commit. Their Git blob SHA-1 values were checked against its recursive tree API; `UPSTREAM_MANIFEST.json` also records byte counts and SHA-256. Upstream files are unmodified and attributed to Meta Platforms, Inc. and affiliates. Their **Creative Commons Attribution-NonCommercial 4.0 International** license is included in full. This is a noncommercial research integration; this vendor license must remain with redistributed source. The repository does not represent the upstream authors as endorsing this project.

Backend `pinned_faiss_and_upstream_rank_suppression_cpu` uses:

1. The upstream `faiss.Kmeans` constructor and training/search sequence with fixed dimension, cluster count, iteration count, seed, spherical=True, gpu=False; the current workspace has no usable GPU.
2. The exact AST body of upstream `rank_within_cluster`, loaded only after vendor integrity verification. Original input row order resolves equal-distance ties. Hard/easy/random policies reproduce the upstream rank/reversal/shuffle conventions, with one sequential process traversing clusters and one seeded Python random stream.
3. The exact AST body of `SemDeDupJob.semdedup`, with CPU FP32 tensors, its zero-filled upper triangular matrix, and strict `maximum > 1 - epsilon` predicate.

It is **direct upstream-kernel integration with local orchestration**, not execution of the complete upstream SLURM CLI. Local orchestration avoids known upstream wrapper problems: discarded tensor `.to()` return, `append_cluster` on a Python list, and GPU-index conversion during CPU persistence. None of these is silently patched in the vendor files. The adapter serializes centers and fitted orders itself and handles empty clusters. Spherical clustering is this version's explicit scope. The original configuration's cluster count for a vastly larger corpus is not silently reused on a 5k panel.

The official backend requires local `torch`, `faiss`, `pandas`, and `tqdm`. The available runtime lacks torch, faiss and tqdm. No official run or official-versus-reference equality test has been performed. Missing dependencies fail before a run lock can claim execution. Assets/runtime installation cannot be replaced with metadata declarations.

## Executable independent CPU reference

Backend `separate_numpy_spherical_lloyd_reference` supplies a deliberately separate development implementation: seeded sample initialization, fixed-count Lloyd iterations, FP64 centroid sums normalized and stored FP32, retained previous centroids for empty clusters, FP32 similarity/assignment, stable distance ordering, and the same structural earlier-raw-neighbor suppression convention.

**It is not claimed equivalent to Faiss.** Initialization, training subsampling, centroid updates, convergence and floating arithmetic can differ. Its natural Civil100 run establishes branch orchestration and head reconstruction only. It is not a standard SemDeDup scientific result, a semantic experiment, an original source-withdrawal result, or a speed comparison. Lexical embeddings remain lexical even if another feature dimension is supplied.

## Configuration and requests

All cluster/iteration/policy/epsilon/seed choices are explicit in `Config`, with a required development configuration identity. There is no outcome-driven selection inside the runner. Confirmatory execution remains blocked pending the study-wide authenticated inputs and gates. The supplied configuration identity is a declaration, not evidence of developmental selection.

Epsilon is strictly between zero and one. This guarantees a positive similarity threshold and avoids altering upstream zero-triangular semantics at negative thresholds. The cluster count is fixed: fewer nonzero retained records than clusters produces a preserved failure, never adaptive cluster reduction. An empty corpus has explicitly defined empty selection and zero ridge head.

The request sampler receives this branch's own original cluster-local blocker graph. The generic Phase 4 sampler's historical graph-binding label describes its abstract oriented-graph interface; `request_origin.json` binds the actual branch fit and explicitly identifies cluster-local provenance. This graph is not the main global semantic curator G. A paths depend only on this branch's original graph, never on labels or head outcomes. R/S use the existing graph-independent random streams. Source ownership and kinds must be supplied; unknown singleton metadata cannot create a genuine-source arm. Matched-volume R paths are preserved when the sampler creates them.

For protocol runs supply allocations R=32, S=32, A=16 and the frozen registered horizons/checkpoints. The first two R and first two S paths are evaluated after **every individual service deletion**, even when the main checkpoint schedule is sparse. Source requests expand to all raw records owned by the source. Five no-deletion fits repeat the original declared seed; three distinct alternative declared seeds are reported separately. These comparisons use selected IDs and prediction disagreements, not arbitrary cluster labels.

## Head and accounting

All branches use the same stored FP32 learner inputs and FP64 targets, no intercept, fixed positive lambda and average-loss regularization `M + lambda * current_selected_count * I`. The empty target returns zero. The reference fixture independently reconstructs every saved natural-data head using direct normal equations.

Each full fit writes a fresh directory. Existing centroid or rank directories cannot silently skip fresh work. Ledgers include backend setup, reclustering, reordering/policy, suppression, centroid/order persistence, retained gather allocation/time, moment construction, head solving, and checkpoint elapsed time before report output. Input arrays, centroid arrays, pairwise matrix element counts, per-fit persisted bytes and whole-output bytes are recorded. Checkpoint failures are preserved; a later checkpoint may be attempted independently because each R is a new full fit.

The current boundary runner uses a shared evaluator process and warm caches. It does not claim optimized no-reaccess service execution, measured process-tree peak RAM, cold-cache timing, canonical arithmetic, numeric certification, or speedup. Report-writing bytes and elapsed costs are recorded separately from algorithm work where available; results from this runner cannot supply the primary systems comparison.

## Run locally

From repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 empirical_execution/phase5/check_refit.py --output-dir /path/to/new/audit-directory
```

The check includes the actual reused Civil100 natural preview and a six-vector algebraic software fixture. The fixture is only for source-service expansion, empty targets and refusal behavior; it is not a synthetic empirical dataset or source-withdrawal evidence. It is kept out of the natural-data result summaries. The natural run uses two R and two A trajectories, eight record deletions, checkpoints 1/2/4/8, and all-step R oracles. There are no genuine Civil source units, so it runs zero S trajectories.

For supplied authentic local inputs, the CLI is:

```sh
python3 empirical_execution/phase5/refit.py --bundle /path/to/bundle.json --output-dir /path/to/new/run
```

The bundle contains `record_ids`, `source_ids`, `source_kinds`, `config` matching Config fields, `lambda_reg`, `design`, `allocations`, three `seed_variants`, and `curator`, `learner`, `targets` descriptors `{ "path": "relative.npy", "file_sha256": "..." }`. Optional `prediction` supplies a fixed held-out prediction array with its own descriptor. The CLI refuses path escapes and altered array bytes. `evidence_role` must remain `engineering_nonconfirmatory`. Scientific activation needs the external study integration; this flag cannot promote a lexical run.

## Remaining scientific inputs and acceptance

Missing: actual locally installed official backend dependencies, official-backend execution/acceptance checks, authentic source-complete Civil/AskUbuntu 5k panels, pinned semantic embeddings, genuine calibration ratings, development-only locked cluster/selection choices and the study resource policy. The natural reference release is an implementation acceptance check only. Neither vendoring code nor a passed CPU-reference run completes the prescribed SemDeDup scientific boundary study.
