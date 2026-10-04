PHASE 4 FIXED-INPUT RIDGE COMPARISON HARNESS

Scope
  This is executable software integration on the existing natural Civil100
  preview. It does not start the semantic confirmation study or add a new
  scientific effect, utility, speedup, or memory-efficiency result. No synthetic
  empirical corpus is generated. Lexical features are identified explicitly.

Run from the repository root (NumPy, SciPy and scikit-learn required):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 empirical_execution/phase4/check_execution.py
  To reproduce without overwriting frozen outputs, append:
    --output-dir /absolute/path/to/a/new/execution_engineering

The callable run_comparison accepts aligned fixed FP32 curator and learner
arrays, FP64 targets, unique record IDs, a verified phase4 request manifest,
and threshold/lambda. It writes the complete execution lock before method
construction or outcome computation. Output directories must be new/empty.
The evaluator owns arrays for independent audits; services are not process
isolated. An evidence-role declaration or supplied primary lock cannot promote
this harness to primary-study execution; primary calls fail before writing.

Implemented mappings
  O-G: independently rerun the retained pair-local graph, rebuild moments,
       solve the common FP64 primal system.
  O-T: correct selected IDs, fresh moments, same head solve. Its timing is the
       moment/decode part of that exact O-G execution, not a separate replicate.
  B-E-signed-python: existing eligible payload with Python string dictionaries,
       inverse incidences, eligibility pruning and signed batched moment updates.
       This is not the protocol's optimized CSR systems implementation.
  B-E-compact-common: integer-indexed eligible FP32 rows and FP64 targets; no
       persistent moments; recover selected moments at release. This is the
       previously developed memory comparator with a common decoder, not the
       original protocol B-E definition and not its optional dual decoder.
  P-I-jointspan-common: indexed joint-span aggregate coefficient state, repair
       keys/expiry, recover current moments and solve the common system. It
       does not decode selected IDs or preserve bitwise canonical state.
  B-F: rebuild on retained originally selected records only. This intentionally
       wrong-target diagnostic is never an eligible unlearning competitor.

Unsupported/incomplete branches
  Source service in this runner; B-A; P-S; P-R; O-R/SemDeDup refit; optimized CSR
  baseline; isolated/no-reaccess workers; optional native compact decoders;
  full registered lifecycle systems trial; LoRA/model extensions.
  Such branches are not silently renamed, simulated, or marked complete.
  A separate single-output fractional-logistic extension now exists in convex.py;
  see CONVEX_README.txt. It has a different objective and its own exact optimizer
  certificates, and is not folded into the ridge timing comparison.

Fixed engineering run
  Reused Civil100 original toxicity labels; lexical curator dimension 128;
  tau=0.6; learner dimensions 64 and 768; lambda=0.01 with lambda*n at every
  release; all coordinates penalized, no appended intercept. Shared sealed
  requests: 4 R + 4 U + 2 A paths, record horizon 8, checkpoints 1/2/4/8 where
  supported. A is stress targeting, not a population-effect estimator.
  No source withdrawal is fabricated from singleton preview records.

Numerical and state checks
  Every retained graph is checked against a separately coded scalar FP64
  normalization/dot-product implementation using the fixed coordinate order.
  All maintenance methods use the same supplied values and common primal
  Cholesky decoder. Integer counts, selected memberships (payload methods),
  remaining horizon and live service identifiers are checked at every release.
  Gram/cross relative Frobenius discrepancies and head maximum absolute error
  must be <=1e-9. Counts must match exactly. This tolerance is a diagnostic,
  not a rigorous roundoff certificate. Empty count always decodes to the
  declared zero head, while any moment cancellation remains visible in audits.
  A zero-admission path must agree with B-F. Zero outcomes are retained.
  Canonical full coefficient bytes, strict certificates and no-reaccess are
  not asserted by this harness; earlier strict-mode audits are separate.

Outputs
  execution_lock.json: input/method/code bindings and cost scope.
  requests.json: unchanged sealed input request manifest.
  construction.jsonl: each state construction, elapsed seconds and state bytes.
  checkpoints.jsonl: all checkpoints, every method's raw timings/accounting,
       selection audit, disagreements and failures.
  heads.jsonl: exact recorded FP64 output weights for independent recomputation.
  audits.jsonl: independent edge and selected-moment checks.
  summary.json: engineering-only flags, failures and discrepancy maxima.
  Failed paths retain the failure row; subsequent checkpoints explicitly state
  not executed after path failure. No successful-path-only filtering or reroll.

Cost interpretation
  Construction, repair, moment recovery and decode are separately timed.
  Graph building is shared initial preprocessing and separately recorded;
  retained graph rebuilding is charged to O-G. Audit computations are outside
  method timers. Execution is one warm shared Python process; O-G caches and
  evaluator arrays remain available. No latency significance or speedup claim.
  State accounting reports numeric buffers and recursive Python byte estimates.
  It excludes peak process RSS, transient BLAS workspace, allocator slack,
  caller/oracle arrays, external ID adapters and persisted outputs. Those
  omitted objects must be measured in isolated future systems experiments.

Frozen result directories
  results/execution_engineering_final/ is authoritative for this phase. The
  earlier results/execution_engineering/ run is retained only as an intermediate
  record; it predates removal of graph identity from request RNG seeds. Do not
  pool runs or count them as independent research evidence.
  Final frozen replay: 80 checkpoints, 0 failures, 54 zero-admission rows;
  maximum same-target head absolute discrepancy 8.326672684688674e-16;
  maximum Gram/cross absolute discrepancy 4.440892098500626e-15.
  These describe implementation agreement, not semantic accuracy or utility.
