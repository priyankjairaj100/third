Two-stage sampler controller, 14 October 2026.

Qualification uses a fixed first-512-row engineering subset and its separate
diagnostic RNG domain. It consumes no production lock or entropy. Selection
requires a completed independently reviewed qualification and one separately
created immutable four-seed sampling lock. Neither stage performs model inference,
human annotation, threshold choice or scientific source acceptance.

Only bounded metadata/package/evidence reads and payload size checks occur on
the login node. Actual frame and source payload validation/scoring occur in
Slurm on csis.cn1, CPU-short, 2 CPUs, 8 GiB. Qualification has 20 minutes.
Selection must choose 60, 120 or 240 minutes based on observed qualified cost;
these are scheduler caps, not guaranteed runtime bounds. The observed CPU-short
partition permits up to 12 hours. Record concurrent-load limitations.
Source-file hashes stream through 1 MiB buffers. Vectors map about 65 MB,
row metadata is bounded, and the largest diagnostic score matrix is about 2 MB.
An OOM remains a recorded failure; there is no fallback numerical algorithm.

Every submission requires new private stage root, clear relevant queue, completed
frame accounting, exact package/result pins, clean executing checkout and fresh
quota of 8 GiB reserve plus 128 MiB output allowance. That allowance includes
bounded numeric sample/result outputs, 16 MiB console, filesystem margin and
contingency. It does not authorize copying a full vector/source/score population.
Quota snapshots are not storage reservations.

Use exact 39-file historical protocol evidence snapshot with its own manifest.
The snapshot includes earlier local analysis-lock and report bindings absent in
the intentionally older executing checkout. It never stands in for executing
scorer code: sampler verifies and imports that separately from the actual repo.
No historical user lock or the executing checkout is changed by this workflow.

The previously reviewed frame-control package supplies bounded capture helpers:
1 MiB test console and 14 MiB combined child console. Slurm wrapper is quiet apart
from terminal failures. Keep process getrusage RSS distinct from Slurm MaxRSS.
Durable intent precedes one sbatch call. Failed/unknown submissions are retained;
never retry, overwrite or infer completion from submission alone.
