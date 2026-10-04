# Independent review of the finite-word activation frontier

This review concerns the finite algebraic family in Phase 10, not natural text embeddings or an empirical dataset.
Earlier phases remain frozen.
The construction closes two specific qualifications of Phase 9: its private inputs are exactly representable in binary32, and its exact private sign state admits a sequential canonical codec.
It does not establish total-byte optimality, a practical compression advantage, or global novelty.

No open mathematical finding remains in the final full proof, LaTeX fragment, scorer proof, or inspected codec state contract.
This is a scoped independent review, not proof-assistant verification or exhaustive software certification.
Final review status and exact reviewed source hashes are recorded in `results/finite_word_review_final.json`.
The observations below distinguish the mathematical information claim, the frozen scorer's arithmetic contract, and the service's serialization contract.

The review read `FINITE_WORD_THEORY.md`, `finite_word_theory.tex`, `SCORER_TRANSFER.md`, `finite_word_codec.py`, `CODEC.md`, and `CODEC_REVIEW.md`, and the earlier Phase 9 proof and qualification stack.
The codec implementation's separate independent review owns its detailed malformed-state and accounting tests.
The checks reported here implement independent geometry, exact normal equations, private-state cardinality, and algebraic boundary controls.

## Geometry and access model

The coordinate blocks are the anchor coordinate, a common-blocker coordinate, a public signature block of length `k`, and a private sign block of length `D`.
Both block lengths are powers of four, with `2+k+D <= 2^20`.
The public signatures satisfy the stated pairwise coherence bound.
The private signs may vary independently without changing any blocker payload or public signature.
All labels are fixed public binary labels: one for hidden candidates and zero otherwise.

The stored candidate has squared norm

\[
\nu=\frac{16641}{16384}=\left(\frac{129}{128}\right)^2.
\]

It is not a unit vector.
The raw stored vector must feed the ridge learner unchanged; normalization is used only by the curator's cosine score.
Renormalizing the learner features would define a different theorem.

The smallest edge cosine is greater than `0.43`; the largest nonedge is less than `0.372`.
These statements hold for every private sign assignment, including identical private sign blocks on distinct candidates.
The latter case is accounted for by the full upper bound on their private inner product.
Every controlling inequality can be verified with rational arithmetic after squaring positive radical comparisons.

Under the priority order anchor, common blockers, private blockers, candidates, the hidden candidate's earlier raw blocker set is precisely

\[
B_i=S_0\cup\{p_i\}.
\]

At `F_i=B_i`, exactly the anchor and candidate `i` are selected.
Other private blockers survive and continue to suppress their candidates even though those blockers are themselves excluded by the anchor.
This uses **earlier raw neighbors**, not greedy suppression against previously selected records.
The frozen hash priority is handled by assigning the public payload roles after sorting IDs, rather than assuming the hash order honors role names.

All private candidates remain excluded through cumulative budget `b`, and the original head is zero.
At budget `b+1`, only a complete blocker deletion can admit one candidate.
No valid request at that horizon admits two candidates.
Every hard request contains only fixed public payloads, so access to forgotten records does not reveal the private signs.

## Exact head and finite-state identification

The hard target is the actual two-record averaged-ridge solution

\[
\theta_i=\frac{w_i}{t},\qquad t=\nu+2\lambda.
\]

The anchor is orthogonal to the candidate, and its label is zero.
Substitution into the normal equations verifies this formula without treating Gram or cross-moment entries as independently variable data.
The private coordinates of this head equal

\[
\frac{\alpha}{t\sqrt D}\,z_{ij},\qquad
\alpha=\frac1{128},\quad z_{ij}\in\{-1,+1\}.
\]

Thus the collection of hard probes identifies all `mD` private bits.
Two different private sign arrays cannot share one exact deterministic state when a decoder must answer every possible first hard request.
The alphabet must contain at least `2^(mD)` distinguishable states.
This is a finite-state counting argument and needs no continuous-encoder assumption.
It counts every private-dependent persistent model, cache, ticket, seed, external store, or other auxiliary input available to repair.
Public randomness must be independent of the private signs.
A `B`-bit information state means at most `2^B` possible states conditional on the public contract and coins.
An uncharged variable-length delimiter is not free private capacity: arbitrary strings of length at most `B` have `2^(B+1)-1` possibilities.
The information-curve proof needs the explicit state-alphabet convention.

The matching sign encoding uses exactly `mD` private bits at initialization.
Its header, public codebook, IDs, deleted-set metadata, byte padding, output values, and workspace are separate resources.
The exact claim therefore concerns the private information conditional on the fixed public contract, not the size of the complete Python process or every serialized byte.
Fixed binary32 and a bounded dimension range do not support an unqualified asymptotic statement as dimension tends to infinity.

## Frozen scorer transfer

The complete `SCORER_TRANSFER.md` proof was reviewed independently.
Its argument covers all admitted private assignments and dimensions under explicit IEEE execution assumptions.
The associated small executions support source correspondence rather than proving those universal arithmetic assumptions.

All raw coordinates are short dyadic values exactly representable in binary32.
Their binary64 squares and ordered norm sums are exact on the common `2^-34` grid.
Square-root and division errors are bounded coordinatewise; the ordered dot error is bounded absolutely, including cancellation.
The resulting error is less than `2^-32`, far below both threshold margins.

The underflow argument is also needed and valid.
Nonzero rounded products are at least `2^-37` in magnitude and lie on the binary64 `2^-89` grid.
Addition and rounding preserve that grid; cancellation either gives zero or a value far above the subnormal range.
The intermediate magnitude bound excludes overflow.
The actual frozen path has no clipping operation.
The stored binary64 threshold and its strict comparison are used explicitly.

The proof does not certify arbitrary hardware modes, fast-math changes, different reduction orders, or images of an actual semantic encoder.

## Sequential state and canonicality

Let the initial horizon be `K=b+1`, the cumulative deleted set be `F`, and the remaining budget be `K-|F|`.
Candidate `i` remains relevant precisely when it survives and

\[
|B_i\setminus F|\le K-|F|.
\]

For this family, `|B_i|=K`, so the inequality is equivalent to `F subseteq B_i`.
It follows immediately that eligibility cannot increase along a valid deletion history with a decreasing remaining budget.
Every eligible candidate remains independently probeable by completing `F` to `B_i`.
Consequently the exact conditional private minimum is `D` times the number of eligible candidates.
This fixes one admissible `F` and requires correctness for all private inputs and allowed continuations; it does not assert the same conditional entropy after an externally imposed, sign-revealing request policy.
Ineligible signs can be discarded permanently for this fixed horizon.
Replenishing the budget would invalidate that conclusion.

The inspected codec assigns roles after frozen priority sorting, derives the eligible-block order from public metadata, packs signs in a fixed order, and enforces zero tail padding.
It stores no private bitmap, materialized head, decoded sign cache, or original feature matrix.
Its transition reads only the existing bytes, public contract, and request IDs.
The fresh retained initializer may read surviving input signs, validates them, and persists only the eligible blocks.
It is an oracle for canonical-state comparison, not a retained-data access path used by transition.

The authoritative state depends only on the fixed public contract, current retained private signs that remain relevant, cumulative deleted set, and remaining horizon.
It does not depend on the order or grouping of deletions.
This is abstract serialized-state canonicality.
It does not establish physical erasure, transcript privacy, or invisibility of deletion identities.
Old immutable byte objects or returned heads retained by a caller remain additional deployment storage and must be charged.

## Exact targets, certified releases, and expected error

Exactly representable inputs do not imply exactly representable ridge outputs in a fixed floating format.
The exact target is rational for rational `lambda`, including a stored binary64 value interpreted exactly.
Parameter fidelity and raw output-byte equality are separate contracts.

An actual exact-target certificate with radius strictly below

\[
a_H=\frac{\alpha}{t\sqrt D}
\]

preserves every hidden sign.
A coordinate with the wrong sign alone would contribute at least `a_H^2` to squared error.
Under `0<lambda<=1` and the dimension cap, `a_H>2^-19`; a certificate at radius at most `2^-20` is therefore sufficient.
This does not promise that a numerical solver will obtain the certificate.
With per-hard-probe certificate failure at most `delta`, block Fano gives the corresponding `m[(1-delta)D-h2(delta)]` information bound.

The expected squared-error curve uses expectation over each hard probe, or the uniform average over those `m` probes.
It must not be replaced by average error over all valid deletion sets: many of those targets are zero and would dilute the objective.
For a posterior sign probability `p`, the minimal squared coordinate risk is the scaled variance `4p(1-p)`.
Summing the `D` coordinates gives the dimensionless variance level

\[
v=\frac{t^2\varepsilon}{4\alpha^2}.
\]

There is no extra factor of `D` in this normalization, because each feature sign has magnitude `1/sqrt(D)`.
The entropy function

\[
g(v)=h_2\!\left(\frac{1-\sqrt{1-4v}}2\right),\qquad 0\le v\le\frac14,
\]

is increasing and concave.
One direct check differentiates through `v=p(1-p)`; the second derivative is negative because `atanh(x)<x/(1-x^2)` for `0<x<1`.
Conditional entropy subadditivity and Jensen therefore give `B>=mD[1-g(v)]`.
For larger error levels the nonnegative zero bound is the appropriate statement.

The prospective matching rate via a Hamming covering code is a query-summary existence bound.
With covering radius fraction `rho`, output shrinkage `1-2rho` produces normalized squared sign error at most `4rho(1-rho)`.
A data-independent public mask and coordinate permutation are needed to distribute error uniformly across every requested block for every fixed input.
A permutation alone would not justify that claim, because its error locations could remain correlated with the input.
The enormous public codebook, enumeration, and computational cost cannot be silently omitted from a total-memory or practical-algorithm claim.
This approximate summary is not the sequential canonical exact codec.

## Limits and retained checks

The checker `check_finite_word_review.py` is an independent exact-algebra and scalar-format check, not an empirical experiment.
Its final 22 assertions cover all 100 admitted exponent pairs, 600 binary32 round trips, all 256 private sign arrays in a small family, 512 fresh exact normal equations, 7,424 low-budget queries, 64 conditional state alphabets, and 39,424 private-state drop transitions.
It also checks the expected-error scaling, release-radius inequality, and covering-code shrinkage identity.
For `lambda=1`, it verifies a hidden output coordinate is `64/49409`, which is not exactly binary64 despite the exact binary32 input.
The initial 19-assertion and intermediate 21-assertion attempts and their exact sources are preserved separately.

The final receipt also binds three separately executed component reports, without pooling their counts as independent empirical evidence:

| Component report | Passed checks | Bounded scope |
| --- | ---: | --- |
| `results/scorer_transfer_final/checks.json` | 85 | Exact scalar constants and exponent pairs; actual frozen graphs on eight small payloads. |
| `results/codec_checks_corrected.json` | 51 | 5,888 fresh graphs and exact heads; 19,780 sequential transitions. |
| `results/codec_independent_final/checks.json` | 51 | 1,560 state cases and 20,280 exact normal-equation coordinates. |

These reports were inspected and source bindings checked; their executions were not repeated for this final review.

A release-semantics counterexample is retained deliberately.
For the exactly representable public value `lambda=2^150`, hidden head coordinates round to numerical binary32 zero.
Positive and negative zeros remain different IEEE byte strings, however.
Numerical-vector equality, canonical positive-zero release, and raw-byte equality therefore must not be conflated.
The example does not refute the exact rational theorem or its certified-error bridge.

Cross-format review also identified two qualifications that must agree between the full proof and LaTeX fragment: the soft reconstruction is the affine value `q_n+(1-2q_n)c`, and the displayed block-Fano substitution uses `0<=delta<1/2`.
The failure-probability range is material: without it, substituting `delta=1,D=1` would incorrectly give a positive bound although the success requirement is vacuous.
The final source-bound receipt records their resolution; the previously announced fragment is preserved as superseded evidence.

The useful novelty claim is this specific finite-format curation-and-ridge realization and its qualified canonical state.
Finite-state injection, binary entropy bounds, covering codes, and bit packing are standard tools.
The result is not a new general information theorem, nor evidence that real NLP datasets approach the constructed worst case.
Global priority and empirical relevance require separate literature and natural-data evidence.
