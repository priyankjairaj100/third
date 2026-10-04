# Phase 5: method services and experiment integrations

This phase completes substantial remaining algorithm and experiment software.
It does **not** complete the primary semantic study. No original source-rich
corpus, real E5/MPNet cache, passed human semantic-quality gate or completed human
interpretation study has appeared. All computations were local. No annotation
messages, remote jobs or paid compute were launched.

Read [COMPLETION_LEDGER.md](COMPLETION_LEDGER.md) for the item-by-item mapping from
the frozen Phase 4 TODOs. The machine-readable current status is one directory
above. Historical Phase 3/4 implementations and results remain unchanged.

## Implemented components

| Component | Entry points | Evidence and boundary |
|---|---|---|
| Compact payload maintenance | `payload.py`, `PAYLOAD_README.md` | CSR incidences, eligibility buckets, signed batched BLAS, source expansion, compaction and resumed snapshots; 8,817 checks, including 80 natural Civil states |
| Summary methods and access | `methods.py`, `run_isolated.py`, `WORKERS_README.md` | B-E/B-A/P-I/P-S/P-R, joint-span/compact variants, O-G/O-T; separately constructed method snapshots and fresh repair processes |
| Numerical release | `decoders.py`, `DECODERS_README.md`, `run_isolated_v2.py` | Versioned normalized-residual/conditioning gate and common zero-start CG; does not retroactively certify v1 results |
| Full fitted-curator branch | `refit.py`, `REFIT_README.md` | Official SemDeDup commit pinned; full/frozen-fitted/frozen-selection branches. NumPy reference executes; official torch/Faiss backend remains unexecuted |
| Convex outputs and comparison | `convex_multioutput.py`, `convex_program.py` | Independent logistic outputs, rigorous joint parameter bounds, actual eligible payload, three-arm local driver; no primary semantic/source claim |
| Boundary studies | `boundary.py`, `BOUNDARY_README.md` | LSH missed-pair sampling, FP32 state, conditioning factors and conservative graph/model envelopes; approximate retrieval failures remain visible |
| WCEP and chronological News | `replication.py`, `replication_encoding.py` | Original event/instance parsing, whole-event panels, fixed real encoder bridge, inherited News threshold and complete date-group requests; no WCEP data acquired |
| Human interpretation | `context_audit.py`, `human_analysis.py` | Complete former-blocker and nearest-surviving-selected contexts, blank assignments, weighted/missingness/agreement analysis; zero actual responses |
| Task outcomes and configuration runs | `task_metrics.py`, `task_program.py` | Original fractional/multihot metrics, utility references and shared configuration/path accounting; no tuning on held-out labels |
| Statistical analysis | `statistics.py`, `STATISTICS_README.md` | Full planned-cell accounting, 10,000 trajectory resamples, signed effects, crossed source uncertainty and fixed Holm families |
| Study inventory | `study_registry.py`, `STUDY_REGISTRY_README.md` | 43 groups, 16,933 planned jobs, 158 artifact keys, 21 visible unresolved recipe/input obligations; inventory checks never authenticate inputs |

## Executed natural-data integration

`results/isolated_natural_engineering_final/` is the authoritative v1 isolated
matrix: nine methods, four frozen record-request paths, dimensions 64/768,
five d64 repetitions and one d768 feasibility pass. All **216 service jobs** and
**864 released heads** passed. Independent reconstruction additionally checked
the 216 original heads: **1,080 saved heads**, maximum absolute difference
**9.43689570931383e-16**. There are 648 zero-admission method/checkpoint rows.

The repair process loads only its own accounted snapshot, closes input file
descriptors and installs seccomp TSYNC before deletion. Actual existing-file
reads fail at the kernel boundary, including tests with existing native threads.
Landlock was unavailable; failed capability attempts are retained. This is a
trusted-program no-new-input boundary, not a proof against malicious native
code, a full environment sandbox, or physical erasure.

The memory result remains unfavorable to a blanket summary-efficiency claim:
at d768 the compact eligible-payload initial snapshot is approximately 0.29 MB,
versus 1.15 MB for joint-span and 232 MB for dense P-I. Exact tables are in
`results/isolated_natural_engineering_final/RESULTS.md`. Child RSS and serialized
state are separate; all state, runtime and workspace categories matter. Shared
host contention means these timings support no paper speedup claim.

The journal initially contained 213 lines although all 216 per-job artifacts
were intact. `journal_reconciliation.json` records recovery of three exact
audit rows, preserving the original journal. No method outcomes were rerun.

The separate v2 numerical integration completes 18 services (nine methods × two
solvers), 36 restricted releases and 216 checks. Its maximum independent head
Frobenius error is 1.458e-13. Both initial joint-span symmetry failures and their
source snapshot are preserved; the packed-upper reconstruction fix did not
relax the numerical gate. V2 also records loaded-library hashes and scoped read/
write byte counters, retaining unknown metadata/hardware traffic as null.

Other authoritative evidence directories are identified in each branch README.
The task/utility/sensitivity release retains all 31 planned engineering jobs: 26
completed, five explicitly blocked/outside the task slice, 104 checkpoints, 86
zero admissions and no failed checkpoints. Independent replay checked 244 ridge
heads; 12 logistic utility certificates passed. All 4,064 registered semantic task
jobs remain blocked.

The natural seven-output Civil check has 13 joint/91 scalar optimizer-error
certificates, maximum radius 1.071e-9 against the fixed 1e-8 tolerance. It uses
the seven original Civil fractions, not fabricated Stack labels. The refit
reference has 24 natural checkpoints/72 post-deletion heads, while its official
backend remains an acceptance requirement. The ANN audit recovers only 101 of
332 reference edges on this fixture; poor retrieval is retained as a result.

## Restore and replay

The verbose v1 process results are stored losslessly in `jobs.tar.gz` beside
`jobs_archive.json`: **3,888 files**, including every released head and report.
The manifest binds all original file bytes. Restore before independent auditing:

```sh
python3 tools/phase5_result_archive.py restore
python3 tools/verify_backup.py
OPENBLAS_NUM_THREADS=1 python3 empirical_execution/phase5/audit_independent.py \
  --output /absolute/new_audit/independent_review.json
```

Reproduce the full v1 engineering matrix into a **new** directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 empirical_execution/phase5/run_program.py --output-dir /absolute/new/results
python3 empirical_execution/phase5/reconcile_program.py /absolute/new/results
python3 empirical_execution/phase5/summarize_program.py /absolute/new/results
```

Large method-state snapshots were temporary runtime products, with their exact
hashes/bytes retained; generation and resume software is saved. These snapshots
are not part of the result archive. Tests use small algebra/schema fixtures only
for software validation. No synthetic empirical corpus is introduced.

## Concrete inputs needed to proceed scientifically

1. Source-rich original Civil records, corrected April 2024 Ask Ubuntu/English
   archives with Posts and PostLinks, original dated News with evidenced calendar
   coverage and a pinned PSL; original WCEP event/article files for replication.
2. Pinned local E5/MPNet model/tokenizer files and transformer runtime, or real
   caches whose row, text, preparation and encoder lineage can be independently
   accepted. Official SemDeDup also needs torch/Faiss/tqdm and an acceptance run.
3. Genuine independently collected calibration and fresh validation responses,
   followed by interpretation ratings. Current pair/context packs are blank;
   response validators cannot prove human authenticity from a declaration.
4. Disjoint development measurements and reviews to freeze semantic thresholds,
   task choices, precision targets, resource limits and refit settings before
   confirmation. The 21 registry obligations remain visible until their actual
   recipes, inputs and acceptance are resolved.

Neither a checksum inventory nor a `primary_ready` field enables confirmation.
The final acceptance/activation integration must be exercised on the real
backend and input chain. More repetitions of this preview cannot establish the
missing semantic phenomenon or complete an ACL submission.
