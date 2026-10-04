# Packed eligible-payload service

`payload.py` supplies `PayloadState`, the optimized B-E comparator, and B-A through `retain_all=True`. It is a new implementation; the Phase 4 and `ccu` implementations and results remain unchanged.

```python
from phase5.payload import PayloadState
state = PayloadState(graph, learner_fp32, targets_fp64, horizon=128,
                     unit="record", retain_all=False,
                     compaction_fraction=0.5, batch_rows=1024)
log = state.delete(["record-id"])
head = state.decode(lambda_reg=0.01).weights
persisted = state.snapshot("state.npz")
resumed = PayloadState.load("state.npz")
```

Source mode uses `unit="source"` and the complete fixed disjoint partition in `graph.source_ids`. It does not authenticate those source IDs or decide which are genuine-source workload units. The native-source S sampler and provenance gate are external requirements. Unknown singleton source units remain part of the service universe, even when a scientific S workload excludes them.

## Algorithms and scope

B-E retains FP32 learner vectors and FP64 targets for exactly its live finite-horizon eligible records. Blockers are all earlier *raw* neighbors, including suppressed records. In source mode, blocker sources are deduplicated; a record blocked by its own owner cannot activate while surviving and is excluded from B-E. B-A retains every live record, including unreachable records. Its extra retained-payload capability is explicitly disclosed and it is not a same-access comparator.

The state packs forward and inverse blocker/source incidence in int32 values and int64 CSR offsets, with integer ownership CSR, live/selected masks, and integer doubly linked degree buckets. Requests gather owned rows and inverse blocker incidences, decrement counts, and update affected buckets. Selected removals and admissions both use bounded signed batched BLAS products `Z.T @ Z` and `Z.T @ Y`. No per-record outer product is retained. Degree buckets expiring above the decremented horizon are discarded without a full-record scan. Initial construction validates a strict earlier-record graph and copies only permitted payload; it retains no reference to the original graph, curator vectors, complete learner arrays, or targets.

The count/antijoin maintenance and signed ridge sufficient statistics are established incremental-maintenance mechanisms. The protocol's DBSP and incremental-ridge references apply; the implementation is not a claim of algorithmic novelty for the baseline.

Every request validates the entire submitted ID batch before mutating state. Unknown or malformed IDs and fresh units beyond the remaining horizon fail atomically. Duplicate IDs within a request and retries across requests are idempotent; only new unique units consume the initial cumulative budget. This is the prospective Phase 5 service contract and differs from historical services that rejected retries. There is no budget replenishment. Allocator/numerical failures terminate a worker; arbitrary hardware or allocation failure is not promised to roll back.

Ridge releases use `(G + lambda * current_count * I) W = H`. Empty selected data produces exactly zero statistics and head. The moments are FP64 signed accumulations. Agreement within floating tolerance is not the separate exact-rational history-independent target or a rigorous numerical certificate. `selected_ids()` and `logical_membership()` are internal audit interfaces, not additional head-only service outputs.

## Memory and persistence

Rows and CSR entries become logical tombstones before physical compaction. Compaction is locked at construction: after a request, compact if either the dead-row fraction or the obsolete-blocker-incidence fraction is at least `compaction_fraction`. A compaction gathers live rows, removes withdrawn-unit incidences, and rebuilds packed CSR and buckets. Its scanned rows/incidences and event count are charged. B-E and B-A follow the same schedule. A separate worker may lock a different fraction before outcomes; this is not an adaptive result-driven setting.

`accounting()` exposes actual currently allocated NumPy array bytes by field, live and stale payload bytes, stale CSR-value and row-slot metadata bytes, encoded registry content, a recursive Python-owned size estimate, and operation counts. These components are not disjoint totals to sum indiscriminately: allocated bytes already include the stale portions. CSR pointer arrays and bucket arrays remain charged in full rather than claiming their slack is absent. The complete unit registry and deleted-unit mask are retained for ID validation and idempotent retries. No physical erasure, allocator erasure, RSS, peak-memory or privacy claim follows.

BLAS gathering/products are bounded by `batch_rows` and the maximum sum of explicitly allocated BLAS workspace arrays is recorded. The whole-request affected-ID arrays can scale with the requested source batch. Compaction temporarily holds old/new state and construction uses additional temporary graph/index arrays. Those allocations and allocator overhead require the isolated worker's process measurements; the BLAS workspace field is not a full peak-memory bound.

`snapshot_bytes()` serializes to an in-memory byte buffer without filesystem access for restricted workers; the full buffer must be charged as transient memory. `snapshot(path)` writes one uncompressed NPZ at the exact supplied filename, reports its measured file length and SHA256, and preserves all currently allocated tombstones. `load(path)` needs only that checkpoint and implementation/runtime, not the original arrays, graph or corpus. It validates reciprocal indexes, membership, budget and moment consistency. It is a trusted research-checkpoint loader, not a parser for hostile files. Snapshot SHA256 should be bound by the worker manifest; NPZ structural checks alone do not authenticate a checkpoint.

No original text is used by this service. Retained learner vectors/labels may themselves contain information about records. Snapshot history, OS page cache, allocator remnants and caller-owned original inputs are outside a logical-state deletion claim and must be disclosed in deployment accounting.

## Verification

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python empirical_execution/phase5/check_payload.py
```

The checks exhaust every earlier-neighbor graph on four ordered vertices, all finite budgets/deletion subsets, both record and disjoint-source services, and both access strata: 8,704 small algebra configurations. These are software fixtures, not synthetic empirical corpora or source-withdrawal data. Additional checks cover fresh retained reconstruction, own-source blockers, sequential/reversed/batched requests, retries, malformed/over-budget atomic rejection, a real new request after exhaustion, snapshots/resume, stale-byte disclosure, and a 2,051-record single-source algebra fixture processed in bounded BLAS blocks.

Natural integration reuses the actual Civil100 lexical preview: five record trajectories, eight checkpoints, both B-E and B-A, totaling 80 states. Native Civil sources are absent and no source arm is run. This is implementation verification, not a new semantic, source, utility, systems-speedup or confirmatory result. The machine-readable audit and implementation hashes are in `results/payload_checks.json`. Isolated system workers and independent audit are separate modules.
