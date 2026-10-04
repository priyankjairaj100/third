FRACTIONAL-TOXICITY LOGISTIC EXTENSION
====================================

Purpose and scope
-----------------
This standalone module extends the downstream learner beyond quadratic ridge.
It implements a complete retained-selected-data refit, a warm-start repair,
charged data evaluations, and an exact residual-based optimization certificate.
It is single-output fractional toxicity only. The Stack multioutput extension
remains unimplemented. Neither the current Civil100 lexical pilot nor the
certificate is semantic confirmation, statistical/privacy unlearning, a speedup
claim, canonical model bytes, or a payload-free sufficient-statistic service.
The previously frozen ridge comparison outputs remain unchanged.

Target and normalization
------------------------
For selected rows S, n=|S|>0, x_i in R^d and y_i in [0,1], define

  f_S(w) = (1/n) sum_i [log(1+exp(x_i^T w))-y_i x_i^T w]
           + (lambda/2) ||w||_2^2,             lambda>0.

All d coordinates are penalized; no intercept is appended. Each supplied x is
exactly its stored FP32 value, each y and candidate w exactly its stored FP64
value. Lambda is the exact rational represented by the supplied binary64
lambda. The certificate targets this real-valued function of those fixed values,
not an unrecorded pre-rounding feature or ideal encoder output. Empty data has
zero data loss: f_empty(w)=(lambda/2)||w||² and its minimizer is zero.

The equivalent sum-loss normal equation includes n*lambda; simply retaining an
old count-dependent penalty after deletion would change the target. In average
form used here, lambda stays fixed and every data term is divided by the new n.
For one output, with p_i=sigmoid(x_i^T w),

  g_S(w) = X^T(p-y)/n + lambda*w,
  H_S(w) = X^T diag(p_i(1-p_i)) X/n + lambda*I.

There is no averaging over output coordinates. This module deliberately rejects
multioutput targets rather than silently changing the regularization scale.

Why ridge moments cannot simply be reused
----------------------------------------
The label cross moment X^T y is reusable, but X^T sigmoid(Xw) and the Hessian
change with w. Counts, Gram matrices and label cross moments do not determine
this objective in general. An exact mathematical counterexample (not an
empirical dataset) uses scalar rows

  A: x=(-5,5,0,0),         y=1/2+x/16;
  B: x=(-3,3,-4,4),       y=1/2+x/16.

All values are exactly representable in FP32/FP64. Both have n=4, sum(x)=0,
sum(x²)=50, sum(x*y)=25/8. Their fourth moments are 1250 and 674. Since
sigmoid'''(0)=-1/8, the third derivatives of their data gradients at zero differ.
Thus the ridge moments, even augmented with the first feature moment, cannot
identify both logistic objectives. This is a failure of these particular
sufficient statistics, not a universal impossibility of special data-dependent
compression schemes or finite representations retaining more information.

Solver and charged work
-----------------------
solve_fractional_logistic uses damped Newton with a positive-definite Cholesky
solve and Armijo backtracking. Stable fractional BCE is evaluated as
  softplus(-s)+(1-y)*s for s>=0,
  softplus(s)-y*s for s<0.
Cold refitting starts at zero; warm repair starts at the previously certified
head. Both use the identical objective, complete repaired selected payload,
solver, numeric gradient tolerance, maximum iterations and line-search limit.
A 'numerically_converged' status is not itself a rigorous certificate.

Each joint objective/gradient/Hessian evaluation charges all selected rows;
each backtracking objective evaluation charges another complete row pass.
The ledger records forward rows, gradient rows, Hessian rows, objective rows,
linear solves, accepted steps, line-search backtracks, and named array bytes.
One joint evaluation shares its forward products, so the separate row-operation
counters must not be added and called independent passes. A promoted FP64
feature matrix, Hessian, factorization and temporary arrays incur work/memory;
named allocations are not process peak RSS. Certification rereads the full
selected data and its exact arithmetic work/time is separately charged.

Signed warm-gradient identity
-----------------------------
Let old and new selected sets have sizes n and m, with removed rows R and added
rows A. For m>0, evaluated at the SAME w,

  g_new(w) = (n/m) g_old(w)
           + [sum_A x_i(sigmoid(x_i^T w)-y_i)
              -sum_R x_i(sigmoid(x_i^T w)-y_i)]/m
           +(1-n/m)*lambda*w.

Proof: substitute sum_old individual_gradient = n(g_old-lambda*w), then divide
the updated sum by m and add lambda*w. The last term cannot be omitted when
selection size changes. m=0 uses lambda*w directly. The implementation checks
this identity numerically on natural deletion/admission transitions, including
an explicit negative control omitting the last term. It does not assert that
this first gradient update obviates later retained-data passes; every subsequent
Newton iterate requires its own full gradient and Hessian evaluation here.

Exact certificate: theorem
--------------------------
For finite data and lambda>0, f is differentiable, coercive and lambda-strongly
convex: H(w) >= lambda*I since p(1-p)>=0. It has a unique minimizer w*. Strong
monotonicity gives

  lambda ||w-w*||² <= (g(w)-g(w*))^T(w-w*)
                     <= ||g(w)|| ||w-w*||.

Therefore ||w-w*|| <= ||g(w)||/lambda (also valid at w=w*). Strong convexity also
gives f(w)-f(w*) <= ||g(w)||²/(2lambda) by minimizing the quadratic lower bound
f(w)+g(w)^T(v-w)+(lambda/2)||v-w||² over v.

Suppose exact rational interval arithmetic establishes g_j(w) in [L_j,U_j].
Let B²=sum_j max(|L_j|,|U_j|)². Then

  ||w-w*||² <= B²/lambda²,
  f(w)-f(w*) <= B²/(2lambda).

The module returns these rational bounds, an integer-arithmetic upward-rounded
rational square root, and an exact rational comparison with the requested
parameter tolerance. Display floats are not the authoritative upper bounds.
Two certified warm/cold heads obey ||w_warm-w_cold|| <= r_warm+r_cold. These
bounds concern optimization fidelity on the declared retained-data target.
They do not prove a statistical distributional unlearning guarantee, generalize
to missing/unverified corpus data, or match finite-iteration reference bytes.

Rigorous sigmoid enclosure without trusting platform exp
-------------------------------------------------------
Every logit is recomputed as an exact Fraction dot product of the stored values.
For t=0, sigmoid(t)=1/2 exactly. Otherwise choose s>=0 such that
u=|t|/2^s <= 1/2. For k>=0, put a_j=u^j/j!, S_k=sum_{j=0}^k a_j.
The ratios after the first omitted term obey

  a_{j+1}/a_j = u/(j+1) <= u/(k+2) < 1,   j>=k+1.

All terms are nonnegative, hence

  S_k <= exp(u) <= S_k + a_{k+1}/(1-u/(k+2)).

The routine increases k until this rational tail is <=2^(-bits-16), or rejects
when its configured term budget is exhausted. Taking reciprocals reverses
positive endpoints, giving an interval for exp(-u). Endpoints are rounded
OUTWARD to a dyadic grid: floor lower and ceiling upper by integer arithmetic.
Repeated squaring is monotone on nonnegative intervals and yields an enclosure
of exp(-|t|); every squaring is again rounded outward. If that interval is [a,b],

  t>0: sigmoid(t) in [1/(1+b), 1/(1+a)],
  t<0: sigmoid(t) in [a/(1+a), b/(1+b)].

A final outward dyadic rounding controls exact rational denominator size.
For each coordinate, multiplication by x_ij reverses endpoints if x_ij<0.
The exact FP64 target, division by n, and lambda*w term are accumulated with
Fraction arithmetic. Thus every gradient enclosure is rigorous, conditional
only on exact integer/Fraction operations and the specified stored inputs.
No Decimal, numpy.exp, scipy.expit or BLAS residual is trusted by this verifier.
Their use in independent numerical crosschecks does not establish the theorem.

Resource qualification
----------------------
The default precision is 128 bits, Taylor budget 256 terms, range-reduction
budget 64 squarings, and total row-coordinate budget 2,000,000. The verifier
rejects over-budget requests explicitly with CertificateBudgetError; it never
substitutes a numerical residual as a successful exact certificate. Arithmetic
bit lengths and transient Python allocations are not constant. The resulting
certificate is not a promise of bounded wall-clock time or peak memory. Extremely
large logits can fail the squaring budget even when the numerical optimizer can
evaluate their sigmoid. This is a documented verification limit, not data loss.

Reproduction and output interpretation
--------------------------------------
From the repository root, use a new output directory:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 empirical_execution/phase4/check_convex.py \
    --output-dir /absolute/path/to/new/convex_engineering

The frozen natural pilot uses existing Civil100 original fractional toxicity,
lexical learner d=64, lexical curator d=128 and tau=0.6, lambda=0.01, and first
R/U/A trajectories from the already frozen phase4 request manifest. There are
four checkpoints per trajectory. The first initial fit is shared and charged
separately. Both warm and cold heads receive exact residual certificates;
failures and zero-admission cases are preserved. Solver ordering alternates
according to fixed checkpoint indices, but timings remain engineering diagnostics.
No test labels, synthetic empirical corpus, remote computation, or paid jobs
are introduced. Small algebraic sigmoid/empty-target checks are software tests.

Outputs include design_lock.json, initial_fit.json, checkpoints.jsonl,
heads.jsonl, certificates.jsonl and convex_checks.json. Full metadata and method
limits remain attached so an optimization certificate cannot be mistaken for
completed semantic experiments or a privacy/data-erasure guarantee.

Observed numerical issue and prospective engineering amendment
-------------------------------------------------------------
The first engineering pilot is preserved in results/convex_engineering/,
including its source snapshots and a reproduced failure_diagnosis.json. Its
A000 checkpoint 4 warm solver stalled at the iteration limit: ordinary FP64
objective comparisons repeatedly accepted tiny steps near an objective plateau.
The exact parameter-error bound was 1.3940385712210056e-8, which FAILED the fixed
1e-8 tolerance. The later checkpoint was explicitly not executed. This failure
is retained; neither the tolerance nor the reported result was edited.

The revised candidate generator introduces a narrowly scoped resolution check.
When the predicted decrease (-step*g^T*delta) is at most
32*eps_binary64*max(1,|objective|), and the trial objective is within that same
window above the current objective, it evaluates the complete trial gradient.
Only a gradient norm reduction by at least a factor of two permits the step.
That additional full gradient/forward pass is counted. This numerical safeguard
is NOT a theorem that the objective decreases monotonically. Its purpose is
to generate a better finite candidate when objective differences are unresolved.
The exact stored-value residual certificate, with the unchanged 1e-8 radius
threshold, remains the release authority. Every finite candidate is eligible
for verification regardless of its diagnostic solver status; failed certificate
or budget checks remain failures. No cold fallback or threshold relaxation is
introduced. Both cold and warm methods use the same revised solver.

The authoritative complete rerun is results/convex_engineering_release_v2/.
It includes the initial head so all 25 certificates can be independently
replayed. The earlier successful convex_engineering_final/ and
convex_engineering_before_initial_head_fix/ directories are historical and
omit that initial-head artifact. Do not pool any of these reruns with the
historical failed run as independent empirical evidence. The single-output,
lexical, no-system-speedup scope remains unchanged.

Final artifact completeness correction
-------------------------------------
An independent reviewer found that the first successful replay omitted the
initial fitted weight vector even though its certificate bound the vector by
hash. The check script now writes that vector into initial_fit.json. A complete
replay with matching current code hashes is frozen in
results/convex_engineering_release_v2/, which is the authoritative convex result.
The earlier solver-corrected directories are marked historical; their duplicated
observations are not additional evidence. The 1e-8 tolerance and solver are
unchanged by this artifact-only correction. All 25 certificates can now be
independently checked against saved candidate heads and the original inputs.
