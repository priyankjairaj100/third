# Actual method services and filesystem no-reaccess

This document describes the original worker version. The subsequently completed
numerically gated service is documented in `WORKERS_V2_README.md`; its separate
files and evidence preserve this version unchanged.

This version supplies executable construction and repair workers, rather than a
generic subprocess wrapper. It supports record and source units, idempotent
retries, atomic unknown-ID/budget rejection, current-count regularization,
head-only releases, final persistence, and the separately selected every-release
persistence sensitivity. The source service is software capability: the Civil
preview still has no authentic contributor/source metadata.

`run_isolated.prepare_bundle` builds and accounts for the shared reference graph
and saves canonical input arrays. `run_isolated.run_service` starts a construction
process, which loads these inputs, builds its own state, solves its initial head,
and persists a snapshot. That process exits. A different `exec` process receives
only its method snapshot, public identifier requests, and locked parameters.

## Implementations and labels

| ID | Implementation and access |
|---|---|
| B-E | New integer CSR eligible-payload antijoin, signed batched BLAS moments, horizon pruning and charged tombstone compaction |
| B-A | Same CSR kernel with all live payload; greater future capability is explicit |
| B-E-compact | Frozen compact eligible-payload dictionary implementation; no persistent Gram, selected rows gathered at each common decode |
| P-I | Dense packed FP64 indexed coefficient map, retaining no record feature/target rows |
| P-S | Same dense coefficients, complete sorted-map substitution/merge/truncation on each update |
| P-R | Incidence-rank coordinate representation, full reconstruction/scan and canonical coordinate reselection |
| P-I-jointspan | Frozen indexed feature/response joint-span coefficient compression; separately named, not relabeled P-R |
| O-G | Greater-access oracle retaining original arrays with a live mask, repeating independent scalar retained pair scoring and fresh moments |
| O-T | Greater-access oracle retaining fixed CSR and original learner arrays, computing selected IDs internally and fresh moments; graph work excluded by definition |

Every branch uses the same FP64 Cholesky solver for `(M + lambda*n*I) W = H`.
There is no alternative compact/dual solver in this comparison. At count zero,
the prescribed empty objective produces the zero head; pre-canonicalization
moment drift is reported explicitly. No near-zero coefficient pruning occurs.
These floating states do not inherit the exact rational certificate or bytewise
history-independence claims of the separate strict service.

P-I is explicitly the dense indexed sensitivity. The memory-qualified joint-span
branch and strongest compact-payload comparator remain visible. All current
implementations use Python/NumPy/SciPy; the tiny development run cannot establish
optimized native-scale systems superiority or a practical memory advantage.

## Genuine kernel enforcement

This runtime exposes no Landlock syscall (`ENOSYS`) and denies unprivileged
user-namespace mapping (`EPERM`). Those failures are retained. The executable
fallback is a **stricter file-opening restriction**, not a Python audit hook.

The repair process imports its reviewed code and loads only its own method
snapshot. It closes all file descriptors numbered at least three. It then opens
only declared output files **write-only**, plus its private directory for fsync.
Before the first deletion, it installs an additional seccomp-BPF kernel filter
with `SECCOMP_FILTER_FLAG_TSYNC`, covering all existing native threads. All new
file opens (including `openat2` and open-by-handle), process-memory access,
ptrace, new sockets, SysV IPC, exec, forks/clones and io_uring are denied. The
x86-64 architecture is checked and the x32 syscall ABI is rejected. Any failed
filter installation aborts the run. Existing-file read probes must fail with
`EACCES`. Original input paths are never passed into the repair worker.

Repair works on the deserialized method state and identifier requests in RAM.
Heads and snapshots are serialized into memory buffers and written only through
the preopened write-only descriptors. Their full buffers count towards RSS.
Existing state files are replaced through those descriptors; the requested
release schedule and fsync work are charged. The remaining directory descriptor
refers only to that worker's private output directory. Standard input is closed
onto `/dev/null`; stdout/stderr are the declared output logs.

This contract starts **after own-state loading**. It is not physical erasure,
transcript privacy, a proof against malicious native code or a VM boundary.
Parent/evaluator processes retain data for independent verification and are not
claimed to be confined. O-G/O-T explicitly retain their declared greater-access
payload inside their own states. Historical oracle arrays are not a valid
payload-free deployment representation.

## Resource and cost scope

The prospective development default is 1 GiB address-space limit, 60 CPU seconds,
60 elapsed seconds and one BLAS/OpenMP thread, equally for every method. A policy
is saved before process launch. The parent kills timed-out process groups and
retains failed cells. Child address-space/CPU limits are installed before data
loading; interpreter/import startup is included in process timings and RSS.

The ledger separates shared graph/input preparation, method construction and
initial solve, snapshot transfer, fresh repair loading, each update/moment
recovery/solve/head release, serialization, fsync and process teardown. Parent
bookkeeping and final fsync have a separate inclusive lifecycle receipt. Snapshot
bytes, all artifacts, temporary serialization buffers and process high-water RSS
are different reported quantities. `wait4` block-operation counts are **not**
logical bytes read/written. Full Python allocator attribution and exact per-read
logical byte metering are not claimed. Snapshot transfer and duplicate evaluator
copies count in the lifecycle/artifact ledger; deployment-state size must use the
designated snapshot, not every audit artifact.

Fresh execution is a cold process, not cold disk: no OS page-cache flush occurs.
Shared encoding/acquisition remain external reported work. The caller must use
the prescribed repetitions and paired trajectory order for an actual systems
panel. This module does not choose thresholds, make up source units, authenticate
corpora, collect human ratings or unlock the primary semantic study.

## Why P-R is a real rank-coordinate method

The structural coefficient matrix is the oriented incidence matrix on monomial
keys, with a distinguished ground node for truncated negative terms. Within an
ungrounded component, coefficient vectors sum to zero. A grounded component has
no such constraint. Store every vector in a grounded component; in each other
component omit the lexicographically smallest key and reconstruct its vector as
the negative sum of the others. The number stored is exactly `M - c0`, the
incidence rank, while component metadata is charged separately.

Substitution maps each key to its surviving key, or to ground if the remaining
degree exceeds the new horizon. Apply this map to each component, merge images
that overlap, and mark any component touching ground as grounded. This is the
image of the entire component zero-sum subspace, so the same reconstruction rule
remains valid. No original record edges or payload rows are needed. The method
retains structural zero keys in its component partition; it does not claim
minimal rank of the observed numeric coefficient values or minimum total bytes.

Canonical basis *selection* is deterministic; floating summation can depend on
history. Complete reconstructed coefficient comparisons are software audit work
outside the timed head-only service. Full prescribed 16 R/16 S semantic
canonical-state audits and primary-scale comparisons still require actual data.
