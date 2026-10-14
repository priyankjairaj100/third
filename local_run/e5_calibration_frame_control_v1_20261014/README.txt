One bounded cache-to-calibration-frame admission, 2026-10-14.

The user explicitly requires fully LLM annotation, with zero human reviews.
This stage performs no sampling, model inference or human annotation. It makes
the next source-bound LLM sampling stage possible if actual payload validation
passes. It does not activate the scientific study or legacy human protocol.

The launcher verifies its external manifest, the frame package, completed
assembly/benchmark metadata and accounting, a clear relevant queue, an unchanged
cluster checkout and fresh local quota. It records exclusive intent and submits
once; failure or uncertain submission requires read-only reconciliation.

Only small metadata is read on the login node. The frame's actual source,
weights, census, full vectors/logs, 32 encodes and 32 fresh replay checks execute
on csis.cn1 through Slurm: CPU-short, 2 CPUs, 48 GiB, 20-minute cap, no GPU.
The frame child uses the existing qualified CPU runtime and creates no new
embeddings. The existing hash-verified scorer-control module supplies only
metadata helpers. No prior package is modified.

Output allowance: 256 MiB, separately from the unchanged 8 GiB free reserve.
Frame output is capped at 96 MiB (64,499,840 byte calibration NPY plus bounded
metadata); console allowance 16 MiB; filesystem margin 16 MiB; contingency
128 MiB. Test logs capped at 1 MiB; combined child stdout/stderr at 14 MiB.
Slurm wrapper normally produces no output; its errors share remaining margin.
Quota is a fresh observation, not an exclusive reservation of shared storage.

Child stdout/stderr and process completion are preserved, including failure.
runner_children_maxrss_kib is getrusage(RUSAGE_CHILDREN), which covers completed
children of the runner; it is distinct from Slurm's batch sampled MaxRSS.
Never infer numerical admission or LLM calibration from submission alone.
