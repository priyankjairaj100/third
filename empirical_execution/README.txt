COUNTERFACTUAL CURATION: LOCAL ENGINEERING EXECUTION
Recorded run: 4 October 2026
Numerical revision: v1.1_fixed_graph_strict_requests
Documentation / portable replay amendment: 4 October 2026

SCOPE

This directory contains runnable software and completed local engineering
checks for the proposed ACL research program. It is not the completed paper's
confirmatory experiment. All computation was performed in this conversation's
workspace; no external training job or remote compute service was used.

The data are natural text with original labels where available. No synthetic
corpus, injected duplicate, generated paraphrase, or fabricated target was used.
Tiny mathematical unit fixtures mentioned in code comments are software checks,
not empirical datasets or evidence about real-data prevalence.

The local pilot uses unfitted lexical word/bigram hashing. It does not use E5,
MPNet, a trained text encoder, or a human-validated semantic threshold. In
particular, 768-dimensional hashing is a dimension-matched lexical stress
check; it is not an experiment with the native E5 representation.

DATA AND PORTABLE CONTENTS

- 100 Civil Comments preview records with original toxicity fractions support
  numerical checks. The available preview omits article, publication, parent,
  date, and original record identifiers. This slice supports record deletion
  only; row provenance IDs are not article or author identities.
- 90 complete CC-News preview articles across five observed host groups were
  used locally for structural checks. Ten requested rows could not be acquired.
  This is an incomplete convenience preview, not a census of source groups.
  News labels remain absent; no News prediction task was invented.
- Both previews were acquired through connector-rendered pages, not original
  corpus binary files. Original byte fidelity and immutable source revisions
  are unverified. See DATA_ACCESS_NOTE.txt and acquisition manifests.
- The portable ZIP omits News article text and News raw connector response files.
  Those remain in the original local workspace. The Civil-only replay below
  requires no News cache. Saved structural result tables may be inspected,
  but News structural recomputation requires its original local cache.
- Filenames ending in _raw.json refer to untouched connector responses, not
  original source-corpus bytes. They should never be described otherwise.

COMPLETED v1.1 CHECKS

The curation graph is fixed at 128-dimensional lexical hashing for every
learner dimension. Its cosine threshold is 0.6, selected before this engineering
run without semantic calibration. All dimensions therefore use the same
record universe, curation decisions, priority, and matched request paths.
The 100-record graph initially selects 65 records; 83 records are eligible
under a cumulative horizon of eight record deletions.

The independent oracle recomputes retained earlier-raw-neighbor scores, then
builds moments and solves ridge afresh. Both repair implementations share FP32
learner values promoted to FP64, original fractional targets, lambda=0.01,
average-loss normalization, and an FP64 Cholesky decoder.

Learner dimension | Trajectories | Passed checkpoints | Failed trajectories
64                | 40           | 160                | 0
128               | 40           | 160                | 0
768               | 1            | 4                  | 0
Total             | 81           | 324                | 0

At dimensions 64 and 128 each allocation consists of 16 uniform-record paths,
16 initially excluded-record paths, and eight label-blind blocker stress paths.
The 768-dimensional allocation uses the first matched uniform-record path.
Checkpoints occur after 1, 2, 4, and 8 cumulative deletions.

Every listed checkpoint checked selected membership for B-E, selected count,
Gram/cross moments, and both heads against the independent oracle. The maximum
absolute head difference across these checks was 6.661338147750939e-16; the
maximum absolute moment difference was 8.881784197001252e-16. These are ordinary
FP64 diagnostics under this implementation, not verified interval certificates,
bitwise history-independence guarantees, or proofs about arbitrary inputs.

Complete coefficient-map reconstruction was additionally checked at eight
checkpoints: the first uniform-record trajectory at dimensions 64 and 128.
It was not performed at all 324 checkpoints or at dimension 768.

An aggregate-state checkpoint was loaded in a fresh process and replayed for
four additional checkpoints. Heads matched its in-process replay exactly, and
the active Python audit hook observed no attempted file/network/subprocess I/O
during repair. This checks an instrumented code path, not an OS security
boundary, allocator erasure, transcript privacy, or absence of covert channels.

Nine local structural configurations were evaluated: three lexical thresholds
(0.35, 0.60, 0.85), with record deletion on Civil and record/host deletion on
News. These are engineering censuses of the previews. They do not establish
semantic redundancy prevalence or representative source-withdrawal behavior.

Additional recorded audits:
- check_real_boundaries.py: 13 event checks passed, including atomic invalid
  requests, order/batching agreement, allocation rejection, and 100 additional
  releases ending in full deletion. The empty target releases the defined zero
  head while retaining raw cancellation drift for inspection.
- run_structure_audit.py: 96 News source-subset comparisons and 18 exhaustive
  expected-admission comparisons passed; 576 Civil R/U/A trajectories produced
  2,112 structural checkpoints. Negative controls passed 640 executions covering
  453 distinct curator/deletion-set cases. Repeated requests are not counted as
  independent real-world evidence.
Their saved outputs are results/boundary_audit.json and structure_audit.json.
The boundary audit replays with `python check_real_boundaries.py` using Civil
only. The structural audit replays with `python run_structure_audit.py` and
requires the News cache retained in the original workspace.

INITIAL ARRAY BYTES: OBSERVED TRADEOFF

Dimension | P-I coefficient arrays | B-E feature+target+moment arrays
64        | 1,698,840              | 55,192
128       | 6,640,920              | 175,256
768       | 234,483,480            | 4,980,376

These are bytes of the actual initial numerical arrays, not peak process
memory, serialized size, or a full service memory comparison. P-I uses packed
moment coefficients; B-E stores eligible payload plus a maintained Gram and
cross moment. Dense coefficients are much larger in this pilot; there is no
storage-saving claim.

Python object-size estimates are also logged. Their identifier representations
are not fully matched: P-I uses positional integers, while B-E stores string
IDs. A string-to-position service mapping must be charged or representations
aligned before making a total-metadata comparison. P-I serialization allocates
a second coefficient matrix; the pilot does not compare persistence costs for
both methods. Whole-parent-process peak RSS is logged, not attributed to either
method. Recorded timings are unreplicated engineering timings, not publishable
speedup or lifecycle-performance evidence.

METHODS AND ENTRY POINTS

ccu/core.py
  canonical_features: immutable FP32 learner copy, without renormalization.
  build_blocker_graph: exhaustive FP64 normalized cosine graph; strict > edge.
  direct_oracle: fresh retained-score and ridge oracle, independent of updates.
  EligiblePayloadState: B-E eligible payload, source/record blocker-count indexes,
    signed moments, and decreasing-horizon pruning.
  solve_ridge: common FP64 scipy.linalg.cho_factor/cho_solve decoder; failures
    propagate with no jitter, PSD correction, or silent fallback.

ccu/summary.py
  preflight: structural key/rank and dense coefficient-byte forecast.
  IndexedRidgeSummary.build/delete/statistics/decode/save/load: P-I finite-horizon
    coefficient repair and current-state persistence, without feature-row or
    original-graph retention. The common service releases a head, not selected
    record IDs. Floating histories need not have identical state bytes.

ccu/structure.py
  Independent incidence-rank census, exact expected admission counts for
  declared uniform request frames, trajectory generation, and observations.

ccu/data.py
  Strict natural-row ingestion, provenance checks, fixed lexical engineering
  features, and a row-ID-checked loader for future local canonical features.

run_engineering_pilot.py
  Executes numerical, isolated-replay, and optional structural checks. Fresh
  request kernels reject repeated IDs, within-batch duplicates, unknown IDs,
  and requests beyond the cumulative remaining horizon before mutation.

repair_worker.py
  Replays an already constructed aggregate state in a fresh process with the
  scoped Python audit hook. It does not construct its state from source rows.

Results and manifests
  results/execution_summary.json: complete recorded v1.1 run summary.
  results/engineering_design_lock.json: engineering configuration and input hashes.
  results/civil_d*/checkpoint_registry.jsonl: per-checkpoint diagnostics and work.
  results/civil_d*/requests.json: trajectory IDs and requested row positions.
  results/civil_d*/features.json: separate fixed curator and learner feature hashes.
  results/civil_d*/memory_preflight.json and initial_accounting.json: byte evidence.
  results/isolation_summary.json: scoped fresh-process replay evidence.
  results/structure_census.json: the nine local structural censuses.
  environment_audit.json: observed execution environment and resource limitations.

REPLAY INSTRUCTIONS

Dependencies recorded here are Python 3.12.14 and the pinned versions in
requirements.txt. The pilot needs NumPy, SciPy, and scikit-learn; it does not
need PyTorch, Transformers, a GPU, model weights, or network access at runtime.
Installing dependencies is unnecessary in the original environment where these
versions are already available. If installing elsewhere, use an authorized
package source; this file itself downloads nothing.

Run from inside the extracted empirical_execution directory. Preserve the
recorded results before a replay because the runner replaces its output files.
For a clean replay, copy the directory and move that copy's existing results
folder to a different name. A --skip-structure replay leaves the old structural
file untouched if it is still present, so use a clean results directory to avoid
confusing historical outputs with newly computed outputs.

Exact Civil-only replay command (all three learner dimensions):

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_engineering_pilot.py --skip-structure

This command loads/hashes only the included Civil data, regenerates numerical
checks, and executes the fresh-process aggregate-state replay. It skips both
Civil and News structural censuses. The design records News as unused rather
than presenting an absent News cache as an observed empty dataset.

Faster Civil-only replay, omitting the 768-dimensional lexical stress check:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_engineering_pilot.py --skip-structure --skip-native

The legacy CLI name --skip-native only omits dimension 768; it does not mean an
E5 experiment. Numerical results can vary slightly across BLAS/library builds.
Passing diagnostics, rather than identical floating-point text, is the relevant
replay criterion. Thread environment settings above support reproducible local
replays; they are not a claim of controlled performance benchmarking.

Full local replay, only where the original News cache is available:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_engineering_pilot.py

A successful run exits 0. Numerical trajectory failures are retained in the
JSONL registry and final summary, and produce a nonzero exit status. Exceptions
in acquisition, isolated replay, or structural construction propagate explicitly.
Do not silently remove failed rows or reinterpret diagnostic checks as results
from the final preregistered study.

PACKAGING-ONLY AMENDMENT

After the recorded v1.1 run, --skip-structure was corrected to avoid both loading
and hashing the unused News cache. Help text now explains the scope of that flag
and the legacy --skip-native name. This changes optional data loading and replay
packaging only; no numerical algorithm, threshold, feature definition, request,
or recorded result was changed. The amendment received syntax and CLI-help
checks. The full pilot was not rerun for this packaging-only change.

WHAT REMAINS BEFORE THE PRIMARY STUDY

- Acquire complete original metadata-rich Civil Comments and Stack Exchange
  snapshots through a permitted local import route. Current previews cannot
  instantiate the planned genuine-source experimental populations.
- Make pinned E5 and MPNet weights, tokenizers, and their inference runtime
  locally available. Their absence is not repaired by calling hashing semantic.
- Complete independent human semantic-quality calibration and validation;
  thresholds in this pilot are not substitutes for that gate.
- Construct source-disjoint task/calibration/test populations, guards, and
  outcome-blind request manifests; select the frozen regularization using the
  planned calibration procedure rather than the pilot's engineering constant.
- Perform full planned task, source, convex-extension, and full-refit SemDeDup
  comparisons, with robust numerical/error interpretation and human audits.
- Establish a measured memory/time envelope before larger native-model panels.
  The environment reported about 9.73 GiB of physical memory, not a proven
  process/container allowance. The 512 MiB guard covers coefficient allocations
  only, excluding metadata, original rows, reconstruction, and decoder workspace.
- Align identifier accounting, isolate benchmark workers, and measure actual
  persistence, construction, peak memory, repeated runtime, and lifecycle costs
  for all competitors before making any efficiency claim.

This package records implemented and executed local work. It neither reports
completion of the ACL empirical program nor substitutes convenience-preview
engineering evidence for the intended confirmatory NLP study.
