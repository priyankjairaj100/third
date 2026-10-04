# Version 2: common numerical gate inside the isolated service

`run_isolated_v2.run_service` preserves the original API and adds a locked
`solver="cholesky" | "cg"` parameter. It launches `workers_v2.py` in actual
separate construction/repair processes. These are explicitly versioned files;
the original `methods.py`, `payload.py`, `workers.py` and `run_isolated.py` remain
the implementations bound to the earlier large engineering run.

Version 2 enforces `decoders.py`'s common normalized residual gate on the initial
head and every release. Failures write `numerical_failure.json`, produce no new
approved head, and terminate with a retained nonzero status. No state rebuild,
regularizer change, tolerance relaxation or fallback solver is performed. Both
solvers have the same input, moment extraction, regularization and head contract.
The same seccomp TSYNC enforcement and declared resource envelope apply.

## Explicit joint-span interface repair

The first v2 integration run exposed an actual interface defect: independently
rounded full-matrix reconstruction from joint-span factors can yield an
antisymmetric difference in the last bits. The strict numerical decoder rejected
both joint-span cells. These failures remain in `results/worker_v2_engineering/`,
including a reconstructed original worker source whose SHA256 exactly matches
the original run's source binding.

The subsequent version-2 interface uses the joint summary's **existing packed
upper-triangle Gram contract**, mirroring those entries once before common
decoding. It does not average independently rounded entries, clip eigenvalues,
change the exact-arithmetic aggregate or relax the gate. Diagnostics explicitly
record `jointspan_existing_packed_upper_Gram_mirrored_v2`. This is a declared
finite-precision extraction convention; the two failed runs are not relabeled.

## Dependency and read/write accounting

Both process stages hash each ELF file currently mapped according to
`/proc/self/maps`, and record Python/NumPy versions. A representative final check
had 45 mapped ELF files. This pins loaded binary bytes, not unobserved optional
dependencies, package supply-chain authenticity or a complete environment.
Hashing time is charged before kernel confinement; new library file opens are
denied once repair begins.

Linux `/proc/self/io` counters measure all process read-call bytes inside the
named input-load/hash or snapshot-load/hash stage. The first counter read's own
known byte length is subtracted; the ending counter read has not yet been added
to its returned value. Kernel storage-read bytes are separately reported and
can legitimately be zero with cached files. Serialized logical input-file bytes
remain separate from read-call bytes. These counters do not measure memory-bus
traffic or assign every metadata read to an individual algorithmic operation.

After confinement, every release/state write passes through the preopened
write-only descriptor routine, which sums actual successful `os.write` byte
counts. The report states that its own later write is excluded from that field;
the parent persists and measures the final report separately. B-E/B-A signed
update counters additionally yield exact declared payload-value reads:
`rows_read * (4*d + 8*C)`. This excludes compaction copying, metadata traversals
and cache/hardware traffic, which remain explicitly null where unmetered.
Summary methods have zero record-payload reads by their state contract. No
unavailable measurement is converted to a zero.

## Completed checks and scope

`check_workers_v2.py` runs all nine methods with both solvers on twenty original
Civil preview comments and original labels, using explicitly lexical features.
Two deletion checkpoints per service give 18 services, 36 restricted releases,
and 216 passed checks. Initial heads also passed their numerical gates. The
maximum independent fresh-oracle head Frobenius discrepancy was
`1.4571047612668122e-13`. The authoritative verification is
`results/worker_v2_engineering_final/verification.json`; its source hashes bind
the final implementation. The earlier failed directory is retained separately.

Run:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=empirical_execution python3 -m phase5.check_workers_v2 --destination NEW_DIRECTORY
```

This is a working prospective numerical/access route for a future authenticated
study dispatcher. It does not waive corpus/model/human gates, create source
provenance, establish a primary semantic result, or retroactively add the new
gate to version-1 measurements. No primary-scale efficiency conclusion follows
from this small integration check.

The operating-system interfaces are documented by the primary Linux sources:
[seccomp filtering](https://kernel.org/doc/html/latest/userspace-api/seccomp_filter.html)
and the [`/proc` filesystem counters](https://www.kernel.org/doc/html/latest/filesystems/proc.html).
Those documents explain interface semantics; the saved local negative-access
checks establish which mechanisms actually worked in this execution environment.
