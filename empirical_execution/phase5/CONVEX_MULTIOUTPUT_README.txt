MULTIOUTPUT CONVEX EXTENSION — VERSIONED SOFTWARE AND NATURAL PREVIEW CHECK

The Phase 4 single-output solver/certificate is frozen. This module wraps it for
q independent fractional or binary outputs and combines its exact certificates.
The manuscript's primary Civil task remains original toxicity alone. The seven
original Civil toxicity-related fields here provide a natural-data integration
check for vector targets, not a change to that task and not a Stack result.

Objective and normalization (prospective completion convention)
--------------------------------------------------------------
For W in R^(d x q), lambda>0, no intercept and n selected records:

 F(W) = sum_c [ (1/n) sum_i BCE(y_ic, x_i^T w_c)
                         + (lambda/2) ||w_c||_2^2 ].

For n=0 the data term is zero and the unique target is W=0. This is the SUM
over independent output losses, averaged over records. It exactly reuses each
scalar optimum with lambda unchanged. Reporting a mean-over-output task loss
does not change training normalization. If instead BCE is divided by q while
the regularizer is unchanged, the optimum corresponds to scalar regularization
q*lambda, a different target. Dividing the ENTIRE displayed objective by q
preserves its optimizer but changes strong-convexity constant to lambda/q.
This choice fills an unspecified protocol detail before a primary Stack run;
it must remain explicit rather than silently changing lambda semantics.

The Hessian is block diagonal by output and each block is at least lambda*I.
The unique optimum minimizes each scalar objective independently. For exact
scalar certificates ||w_c-w*_c||^2 <= b_c and objective gap <= a_c,

 ||W-W*||_F^2 = sum_c ||w_c-w*_c||^2 <= sum_c b_c,
 F(W)-F(W*) <= sum_c a_c.

All sums are exact Fraction arithmetic. The Frobenius radius is an upward
rational square root; the matrix tolerance is checked against the SUM of
squared scalar bounds. Merely passing q individual tolerances is insufficient.
The total verification budget is n*d*q and is rejected BEFORE scalar work
when over budget. Output-wise status/timing/certificates remain inspectable.
No exponential/sigmoid float approximation is trusted by the certificate;
see frozen phase4/CONVEX_README.txt for the interval construction and proof.

Count change, source requests and resumption
------------------------------------------
The Phase 4 signed-gradient identity applies columnwise at the SAME W:
 G_new = (n/m)G_old + (sum_added gradients - sum_removed gradients)/m
                           + (1-n/m)*lambda*W.
The last term is required whenever selection size changes. m=0 returns lambda*W.
retained_selection validates the sealed manifest and expands a source prefix
to ALL owned records before independently evaluating B(v) subset deleted.
It supports both units; genuine original source provenance is still external.

save_resume/load_resume retain weights, selected/deleted IDs and exact caller
input bindings; corruption and foreign inputs are rejected. A resumed head is
only a warm candidate. Retained payload, graph and future gradient/Hessian
queries are still required. Writes/reads record elapsed time and stored bytes;
payload reload, graph reload, transient JSON/Python memory and system peak RAM
are not those counters. Every subsequent released result must be recertified.
The resume artifact does not prove erasure, canonical state, privacy, or a
no-reaccess scheme. Callers must bind all relevant inputs and validate the
saved selected/deleted IDs against the current request state.

Cost and algorithm scope
------------------------
Outputs are solved sequentially using the same frozen scalar solver. Row
counters are output-row evaluations and are added across outputs; forward,
gradient and Hessian counters describe overlapping operations, not distinct
independent full passes to be added again. Shared feature/target/head bytes,
one promoted feature workspace and one Hessian are named, but are not peak RSS.
Both warm and cold fits reread complete selected data. Certification itself
recomputes exact logits/gradients and its work/time is separately recorded.
This supplies a correct extension beyond quadratic loss, not a proven repair
speedup and not sufficient-statistic logistic decoding from ridge moments.

Reproduction
------------
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python3 empirical_execution/phase5/check_convex_multioutput.py \
   --output-dir /absolute/path/to/new/output

The frozen release is results/convex_multioutput_engineering/. It uses original
Civil100 fields toxicity, severe_toxicity, obscene, threat, insult,
identity_attack and sexual_explicit; lexical learner64/curator128, tau0.6,
lambda0.01, the first frozen R/U/A paths at checkpoints1 and8. Six checkpoint
comparisons plus the initial fit produce13 joint and91 scalar certificates.
Maximum joint radius is1.070102464743579e-9 at unchanged1e-8 tolerance.
Initial and checkpoint heads, certificates and resume artifacts are saved.
This reuses the existing natural preview and supplies implementation evidence,
not new semantic prevalence or primary-task results. Native source identifiers
remain absent, so no Civil source-withdrawal claim is made.

A twenty-output binary matrix and a three-record source partition are explicitly
algebraic software fixtures only; no synthetic empirical dataset or source
provenance is manufactured. They check twenty-output shape, exact empty-target
radius, full-source expansion, summed objective and whole-output budget refusal.
check_convex_source_software.py adds a separate four-record algebraic check of
three sequential source checkpoints. Each checkpoint verifies the20-output
changed-count gradient, fresh/warm certificates and saved/resumed head; deleting
all three artificial sources produces the exact empty-target zero matrix. Its
six joint certificates are software checks, not additional natural checkpoints.
The registered 32 R/32 S native semantic Civil/Stack comparison, independent
eligible-payload arm and isolated measured resource program still require real
primary assets and orchestration; this module does not declare them complete.
