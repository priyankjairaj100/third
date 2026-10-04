# Common numerical decoder and CG ablation

`decoders.py` is a new prospective module. The already frozen isolated-worker
run continues to use its original common FP64 decoder; its results must not be
retroactively described as passing this new numerical gate.

`decode(moments, lambda_reg, solver="cholesky" | "cg")` returns a `DecodeResult`
with `weights` and a JSON-safe `diagnostics` dictionary. The version-1 policy is
fixed: the normalized residual gate is `eta <= 1e-10`, with the denominator
`||A||_2 ||W||_F + ||H||_F`, where `A = M + lambda*n*I`. Zero numerator and zero
denominator mean eta zero. Stored FP64 moments must be finite and exactly
symmetric; nonzero moments with count zero fail explicitly. No PSD projection,
jitter, coefficient rounding, precision downgrade or silent reset is permitted.

Cholesky is the primary solver. Its reported matrix 2-norm and condition number
come from FP64 symmetric eigendecomposition; this work is charged. CG is a
separately labeled ablation, initialized to zero independently for each output.
Its prospective iteration limit is `10*d` per output. It uses no preconditioner
or warm start. Every iteration checks the true compensated residual, and the
full matrix residual is recomputed before release. Zero RHS columns take zero
iterations. Squared-norm underflow on a nonzero RHS is a retained failure.

Matrix-vector and final residual products use fixed coordinate-order Neumaier
accumulation of ordinary rounded products. Iteration dot products use `fsum`.
Matrix formation, conditioning, solve and residual times are separate. The
condition number, absolute residual, normalized residual, hashes, iteration
counts and release decision are recorded. Failures request an independent audit,
remain failures, and produce no fallback head. Versioning is required to change
the tolerance or iteration budget; callers cannot loosen this policy in place.

`extended_audit` is **evaluator-only**. On preselected audit cases and all numerical
failures with a candidate, it rebuilds Gram/cross moments from canonical retained
FP32/FP64 rows using compensated extended-precision sums and recomputes the
candidate residual. The platform must expose a genuinely wider mantissa than
FP64. The audit binds the rows, targets and candidate by hashes; reports moment
differences, diagnostic residual/error quantities, workspace and elapsed cost;
and never replaces the service state or changes a failed release decision.

Neither the ordinary nor extended residual is a rigorous interval certificate.
The displayed `residual/(lambda*n)` parameter diagnostic assumes the target PSD
Gram and excludes unbounded moment-accumulation error. The separate exact-rational
certificate implementation remains the source of rigorous numerical guarantees.
No projection-based or reduced-space residual is substituted for the full
ambient ridge residual here.

Run `OPENBLAS_NUM_THREADS=1 PYTHONPATH=empirical_execution python3 -m phase5.check_decoders`.
Checks compare tiny dyadic algebraic fixtures against a separate rational
Gauss-Jordan solution and exact residual/error inequality. Natural Civil100
lexical checks exercise both solvers and the extended audit; these are software
checks, not new semantic research evidence. The separate `workers_v2.py` and
`run_isolated_v2.py` now enforce this gate for every method's original head and
every released head, with both Cholesky and CG supported. Its natural-preview
integration passed 216 checks across 18 services and 36 restricted releases;
see `WORKERS_V2_README.md`. The original worker results remain unchanged.
