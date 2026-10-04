STRICT CANONICAL REPAIR AND CERTIFIED RELEASE
4 October 2026 | Local implementation and natural-text engineering evidence

Outcome
The new exact mode closes two specific qualifications of the previous fast
mode: aggregate checkpoint bytes are history independent, and numerical head
error has a rigorous certificate rather than only an oracle comparison.
The old floating implementation and its recorded claims remain unchanged.

Contract
Features are first frozen as FP32, responses as FP64. Their exact binary
values define rational ridge sufficient statistics. Curation graph, priority,
ownership and upstream feature map are fixed. The repair target is the fresh
exact aggregate on retained records under the same specification and remaining
horizon. This is not bit equality to an ordered BLAS floating retraining run.

Each coefficient stores the unique pivot-identity chart of range([G H]), the
packed pivot Gram core A, pivot cross core B, and an exact signed count. Pivot
identity rows are implicit; there is no dense/compact mode or historical basis.
Normalized rationals, sorted keys and canonical JSON determine unique bytes.
Repair substitutes deletion variables, merges exactly and decreases horizon.
It requires no original features, targets or graph after initialization.
No separate payload rows are retained, but moments can reveal information;
this is not a privacy guarantee or physical-memory erasure claim.

The current Python reference scans/sorts all K keys and stages the next state.
Its work includes O(K(h+1) log(K+1)) metadata work plus exact factor merges.
It does not claim an affected-key-only runtime. A one-time exact rebuild was
performed here from authorized retained inputs. Lost rounding information
cannot in general be recovered from an old approximate checkpoint alone.

Certificate
For alpha=lambda*n, an arbitrary finite numerical candidate W_hat is checked
against the exact target via R=G*W_hat+alpha*W_hat-H. Exact rational PSD checking
and residual arithmetic prove
    ||W_hat-W_exact||_F^2 <= ||R||_F^2 / alpha^2.
The reported decimal radius uses integer-square-root upward rounding. The
check covers all candidate solver/coordinate errors. Empty data uses exact
zero instead of division by zero. A requested tolerance that fails the exact
test produces no release. Certificates hash both target and candidate.

Evidence actually run
100 unchanged real Civil Comments records and original toxicity labels.
Frozen 128-dimensional lexical curator, threshold 0.6; learner dimensions
16,64,768; seven-output panel uses all original toxicity fields. Horizon eight,
lambda is the exact binary64 value of 0.01. No synthetic empirical data,
remote computation, semantic encoder, or new NLP-utility claim.

- 40 regular release checkpoints and 100 full-deletion checkpoints matched
  fresh retained-only, rescored-graph exact builds byte for byte.
- Ten final-set comparisons covered one batch, singletons, reverse order,
  fresh builds and strict checkpoint reload. Same-runtime heads also agreed.
- All 140 head certificates met exact tolerance 1/10^10. Largest reported
  rigorous Frobenius error radius was below 4.34e-14. Final deletion yielded
  empty coefficients, exactly zero head and exactly zero certificate bound.
- Invalid/duplicate/over-budget/retried requests preserved input bytes.
  The bytes-only service accepted the stated tolerance and rejected zero.
- Independent checks: 243 exact rational solution checks, 1,984 upward-sqrt
  checks, 66 ambient-RREF comparisons, three additional large-integer cases,
  and six malformed-loader cases under python -O. Twenty native/Python
  algebra fixtures and 50 certificate module checks also passed.
Small algebraic fixtures are software checks, not empirical datasets.

Measured cost (100 records, d=768, one response)
Initial exact build 1.869 s. Initial canonical checkpoint 1,514,487 bytes.
Rational numerator/denominator integer payload 184,525 bytes; recursive Python
live estimate 11,053,348 bytes. These are different accounting layers, not
process peak memory. Maximum numerator/denominator lengths 94/92 bits.
Median batch repair 0.919 s and certificate 0.638 s across 12 checkpoints;
fresh exact construction median 1.577 s. Batches have sizes 1,1,2,4 and these
are unreplicated observations. Timers exclude some loading/serialization and
candidate solve work, so they do not establish end-to-end speedup.
Rational-entry bound O(E_h(d+c)) does not mean fixed-width byte storage.
Bit growth and exact arithmetic can dominate on other corpora.

Main API (run with this directory on Python's import path)
    from fractions import Fraction
    from ccu.exact_canonical import ExactCanonicalSummary
    from ccu.strict_service import repair_and_release
    state = ExactCanonicalSummary.build(x,y,blockers,horizon=8)
    release = repair_and_release(state.canonical_bytes(), [record_id],
        lambda_reg=0.01, tolerance=Fraction(1,10**10))
    state_bytes = release.canonical_state
    weights = release.weights
    certificate = release.certificate.to_dict(40)

Strict loading defaults to recanonicalization validation. validate=False is
only for trusted state; mandatory shape/type/encoding checks still run.
The candidate is outside the authoritative checkpoint. Hardware-independent
floating-head bytes are not part of the canonical-state theorem.

Replay (preserve recorded canonical_results before rerunning)
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python3 run_canonical_validation.py
    python3 audit_certified_ridge.py
    python3 audit_certificates_independent.py
    python3 audit_exact_chart_independent.py
    python3 audit_canonical_loader_independent.py
    python3 check_exact_algebra.py

Dependencies are pinned in requirements.txt. No network is used by replay.
Native acceleration uses local GMP 6.3.0 on Linux LP64; source, verified
binary and exact platform restriction are in ccu/native/. Its README provides
the compile command. Pure Python Fraction fallback uses backend='python'.
It is exact but can be much slower; binary interchange is backend independent.

Files
ccu/exact_canonical.py          exact state, repair, canonical serialization
ccu/certified_ridge.py          exact PSD/residual and upward-radius checker
ccu/strict_service.py           bytes-only certified release boundary
canonical_theory_addendum.tex   proofs, scalar and bit/work bounds
run_canonical_validation.py    natural-data execution and fresh oracle checks
canonical_results/             raw results, reviewed hashes, audit records
STRICT_SHA256SUMS.txt           companion archive integrity manifest

Scope for the paper
These results close the stated implementation qualifications for fixed
blocker curation and ridge statistics. They do not finalize semantic/NLP
utility evaluation, establish source-withdrawal effects on this source-free
fixture, show superiority to compact payload retention, or claim novelty
for RREF or classical residual/coercivity certification.
