Third: post-v3 upper-tail population census, 14 October 2026

Motivation: the completed v3 LLM selection was negative (0/90 conservative
joint positives). Its original uniform-width sample included one pair above
0.95 and none above 0.96. This justifies investigating score support, not
changing the frozen result or presuming that higher-score pairs are positive.

This diagnostic counts every calibration pair strictly above each of the
unchanged 51 v3 threshold cuts. It uses the exact qualified 20,996-row E5
frame, coordinate-ordered scorer and tile order. The complete 220,405,510-pair
score stream and original 20 bin populations must match the sealed v3 census.
A separate first-512 matrix reference directly counts strict exceedances.
Boundary tests cover equality, adjacent binary floats, nonfinite values,
incomplete enumeration and invalid cut grids.

No sampling, entropy, LLM calls, human annotations or threshold selection occur.
No source-authenticity, semantic-validity or confirmatory-admission gate changes.
Any future tail-stratified sample requires a new prospective version, fresh
selection/validation randomness and unchanged explicit negative-result history.
No automatic downstream launch is authorized by this package.

Execution: one Slurm CPU job on csis.cn1, 2 CPUs, 8 GiB RAM, 60 minute cap,
128 MiB output allowance plus an 8 GiB free-space reserve. Login activity
checks metadata, package source, queue, terminal receipts and fresh quota only.
The Slurm worker verifies the frame, source bindings, input hashes and runtime.
An occupied stage or unknown submission outcome must be reconciled, never retried.
Raw source/model text remains private. Outputs are aggregate counts and hashes.
