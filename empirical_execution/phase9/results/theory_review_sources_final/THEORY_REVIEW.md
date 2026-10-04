# Independent review of the realizable ridge-head strengthening

The private-feature activation-frontier theorem is mathematically sound under its stated access, precision, and query contracts.
It strengthens the earlier binary-label obstruction to feature-plus-response dimension for actual ridge heads.
The finite-bit result follows from a separate packing argument; it does not rely on the continuous-encoder theorem.
This review found no open mathematical objection to that construction.

This is a scoped mathematical review, not proof-assistant verification or an exhaustive novelty claim.
The accompanying checks are bounded exact algebra, not empirical datasets or benchmarks.
No frozen Phase 3–8 source was changed.

## Sources and reviewed claim

The review read the current theorem addendum, `THEORY_EXTENSION.md`, and the prior sources below:

- `round2_lower_bounds.md`: finite-bit activation frontier and explicit forgotten-payload access.
- `round2_structural.md`: selector rank, source ownership, and query-state qualifications.
- `round3_lower.md`: approximate query fidelity, information bounds, and upper-bound limitations.
- `round3_compression.md`: arbitrary-statistic rank and the sharp factor-two coordinate comparison.
- `empirical_execution/memory_theory_addendum.tex`: realizable joint-span upper bounds.
- `empirical_execution/canonical_theory_addendum.tex`: exact rational state and bit-growth qualifications.
- `empirical_execution/certified_ridge_contract.txt`: the actual averaged ridge target and numerical certificate.

The main reviewed claim is deliberately a worst-case family statement:
at adjacent cumulative budgets, private information needed for prescribed ridge-head queries changes from zero to feature-scale memory.
It is not a lower bound for every fixed dataset or every numerical representation.

## 1. The private-feature construction survives the geometric audit

Use the addendum's fixed rational centers, with coefficients `8/17` and `15/17`, threshold `2/5`, and cap radius `r=1/100`.
All noncandidate features and labels remain public and fixed.
Only each retained candidate's feature and response parameters vary.

The controlling margins are exact:

\[
\frac{120}{289}-\frac25=\frac{22}{1445}>r,
\qquad
\frac25-\frac{203}{578}=\frac{141}{2890}>2r.
\]

Changing one unit vector by less than `r` changes its inner product with a fixed unit vector by less than `r`.
Changing two candidates changes their inner product by less than `2r`.
The remaining own-blocker, wrong-blocker, and anchor inequalities have larger margins.
Thus the graph stays fixed throughout the independent candidate caps, uniformly in `m,d,q`.
The signature-existence requirement still constrains the relation between `m` and `d`.
The result does not support arbitrarily many candidates at fixed dimension.

For a hidden candidate, the earlier raw blocker set is exactly `S_0 ∪ {p_i}`.
At request `F_i=S_0 ∪ {p_i}`, the anchor survives and the candidate becomes selected.
Every other hidden candidate retains its own private blocker.
That blocker remains excluded by the anchor, but still blocks the candidate under the earlier-raw-neighbor curator.
This distinction is essential; replacing the curator with greedy suppression against selected records would invalidate the argument.

The selected set is therefore exactly `{a,w_i}`.
Any request of size at most `b+1` admitting a candidate must equal its full blocker set.
All other allowed queries have only zero-label selected records and return the zero ridge head.
At budget `b`, no hidden candidate can be admitted at all.
The original trained head is also exactly zero and contains no hidden-feature or hidden-label information.

The private blocker `p_i` must remain fixed when `w_i` varies.
Varying both using the same hidden signature would disclose that signature in the forgotten payload.
The reviewed construction avoids this problem.

## 2. The claimed heads are realizable and identify the full local parameters

For two selected records, define

\[
C=aa^T+2\lambda I,
\qquad c=1+2\lambda,
\qquad \beta_0=\frac{\lambda}{2c}.
\]

The eigenvalues of `C` are `c` along the anchor and `2λ` elsewhere.
With unit `w` and `β∈[β0,2β0]^q`, set

\[
y=(1+w^T C^{-1}w)\beta.
\]

Since `1+w^TC^-1w ≤ c/(2λ)`, every response coordinate lies in `(0,1/2]`.
The open-parameter theorem uses open intervals and hence strict upper inequalities.
These are coordinatewise bounds; response norms may grow with `sqrt(q)`.

The actual normal equations are `(C+ww^T)W=wy^T`.
Direct substitution gives

\[
W=C^{-1}w\beta^T.
\]

This remains valid when the candidate acquires an anchor-direction component.
The construction therefore gains a full `d−1` sphere-direction parameters, instead of imposing anchor orthogonality.
Multiplying by the known `C` gives a unit vector times a positive response row.
The direction and every positive response coordinate are uniquely recovered by the addendum's column-norm formulas.
No arbitrary covariance or unconstrained moment tensor is substituted for a realizable ridge dataset.

The optional fixed-label corollary also passes review.
Give every hidden candidate the public label one and every blocker/anchor label zero.
Now `β(w)=1/(1+w^TC^-1w)≥2λ/c`, and the same positive-direction recovery identifies `w`.
The exact continuous minimum becomes `m(d−1)` with no private label variation.
The distance identity implies

\[
\|W-W'\|_2\ge\frac{2\lambda}{c^2}\|w-w'\|_2.
\]

Thus the feature-memory obstruction survives ordinary binary labels and does not depend on coupled private responses.
This corollary has a separate response class from the main theorem's coordinates bounded by `1/2`.

## 3. The continuous-coordinate minimum has the necessary hypotheses

The tangent chart and positive response intervals form an open domain of dimension `M=m(d−1+q)`.
Equal stored states would give equal answers to every fixed probe `F_i`.
Those probes have identical public payloads on every input in the family.
Head identifiability forces equal private parameter tuples, so an exact deterministic encoder must be injective.
Invariance of domain then excludes a continuous encoder into fewer than `M` real coordinates.

The matching chart-storage upper bound is valid for the initial-state **model-query summary**.
Store every private chart coordinate and response parameter.
Decode on a hard probe, and return zero on every other valid initial-corpus query.
This proves the stated continuous-coordinate minimum.

It does not prove an arbitrary-real encoding lower bound.
Continuity of the decoder alone would not suffice.
Finite floating-point input sets are discrete, so this topological proof does not itself lower-bound their bytes.
The exact chart upper bound is also not a canonical deletion-clean update algorithm.
The existing exact polynomial state has that stronger repair contract and separate resource costs.

Every exact sequential service supporting each probe as a possible first request inherits the lower bound on its initial state.
The proof compares possible branches; it does not require the service to expose rollback or clone operations.
After a deletion, the remaining budget must decrease.
An upper bound for a fixed initial budget does not permit budget replenishment.

For fixed `b`, the high-budget eligible population is `E=N=1+b+2m`, so the numerical order is `Θ(E(d+q))`.
If `b` grows much faster than `m`, the lower bound remains proportional to `m`, not automatically to all of `E`.
This comparison concerns numerical coordinates with public metadata; it does not establish total-memory or byte optimality.

## 4. The finite-bit packing proof is independent and valid

For unit directions and positive response parameters,

\[
\|w\beta^T-w'\beta'^T\|_F^2
=\|\beta-\beta'\|_2^2
 +(\beta^T\beta')\|w-w'\|_2^2.
\]

The smallest singular value of `C^-1` is `1/c`, and `β^Tβ'≥β0²`.
This proves the claimed output-separation inequality without assuming orthogonal candidates.
The tangent chart cannot shrink distances, because orthogonal projection recovers its coordinates.

A maximal packing of the tangent ball with spacing `4cη/β0` covers that ball with balls of that radius.
The volume comparison yields

\[
L_w\ge\left(\frac{r\beta_0}{8c\eta}\right)^{d-1}.
\]

The coordinate grid for responses has the addendum's stated `Lβ` points.
Their Cartesian product produces at least `4η` separation between distinct prescribed heads.
An output within `η` uniquely identifies its symbol.
The closed sets used for finite packing remain inside the permitted feature caps and satisfy the bounded-response condition.
They are distinct from the open-domain condition needed only by the continuous theorem.

Let the `m` candidate symbols be independent and uniform.
Fixed forgotten payloads reveal none of them.
Per-query recovery with failure at most `δ` bounds each conditional entropy by
`h2(δ)+δ log2(L−1)`.
Conditional subadditivity and the `B`-bit state bound then give

\[
B\ge m\big[(1-\delta)\log_2L-h_2(\delta)\big].
\]

Fresh repair randomness cannot improve upon the optimal symbol decoder from the retained state and public coins.
Average correctness over uniform symbols and probe indices also suffices, using concavity of binary entropy.
There is no simultaneous-success requirement or output union bound.

The radius `η<rβ0/(16c)` is positive and independent of `m,d,q` when `λ` is fixed.
It is numerically small; “constant error” must not be interpreted as a large practical tolerance.
For example, with `λ=1` and `η=rβ0/(32c)`, the feature-packing volume ratio is exactly four.
The response grid has 801 levels per coordinate.
These constants support the feature-plus-response bit scaling without borrowing a real-coordinate lower bound.

This is not a matching finite-bit upper bound.
Rational precision, quantization, public codebook size, and implementation complexity remain separate.
The actual fixed FP32 representation needs its own margin-and-separation argument before inheriting this packing result.

## 5. Counterexamples that delimit broader claims

### Forgotten-label payloads can defeat a stacked-rank lower bound

Take three scalar constant-feature records and no curation, with binary labels and `λ=1`.
The exact head map for the empty request and all singleton deletions has stacked rank three.
Yet the single sum `S=y1+y2+y3`, together with the deleted label supplied by the request, answers every query exactly.
The sum has only four possible values.
This contradicts transferring an identifier-only rank bound unchanged to free forgotten-label payloads.

For linear stored state `Ty` and query side information `P_Fy`, with labels varying over an open real domain, the exact condition is

\[
\ker\!\begin{bmatrix}T\\P_F\end{bmatrix}\subseteq\ker R_F,
\quad\text{equivalently}\quad
\operatorname{row}(R_F)\subseteq\operatorname{row}(T)+\operatorname{row}(P_F).
\]

It follows by comparing labels with identical observations.
Without side information, this reduces to the stacked observable-rank condition.
For a finite label alphabet with an arbitrary decoder, that real-domain necessity must not be assumed; a single unbounded integer can encode many labels.
The new frontier theorem uses fixed public probes specifically to avoid the side-information loophole.

### Moment dimension need not equal head dimension

Two two-record ridge datasets, with `λ=1`, can have different Gram matrices and the same initial head `(1/10,1/10)`.
One uses features `(1,0),(0,1)` and labels `3/10,3/10`.
The other uses `(1/2,0),(0,1)` and labels `9/20,3/10`.
Their single-record retained heads differ after deleting the second record.
Thus moments are not identified by one head, and a current-head summary need not support a positive future deletion horizon.
The new lower bound uses a family of future heads whose joint identifiability is proved explicitly.

### Exact rank does not imply a uniform approximate obstruction

Take shared unit features `z_i=e_i`, a positive cosine threshold, and binary labels.
The graph is edgeless and the exact initial ridge-head map has rank `n`.
For `λ=1`, after deleting at most one record, each retained coordinate equals `y_i/(1+n−|F|)`.
At `n=25`, every target is within Frobenius/Euclidean error `1/5` of zero.
Zero private state therefore meets that tolerance despite full exact observable rank.
Any approximate lower bound needs quantitative output separation, as supplied by the frontier packing.

### Coordinates are not bounded words

Eight binary labels can be packed in one integer and recovered by bit extraction.
The coordinate count is one; the state still has 256 possibilities and requires eight bits.
The same issue scales arbitrarily.
All data-dependent model weights, caches, lookup tables, private coins, and external stores must be charged.
The new theorem explicitly counts them, while its public fixed graph and cap codebooks remain independent of the hidden symbols.

## 6. Verification evidence and claim boundary

The retained script `check_theory_review.py` checks exact `Fraction` algebra and the counterexamples above.
Its final algebra run passed 28 assertions, including:

| Bounded verification | Cases |
| --- | ---: |
| Scalar queries with forgotten-label side information | 32 |
| Independent feature-cap graph instances | 27 |
| All low-budget requests in those small instances | 1,242 |
| Exact fresh two-record ridge heads at hard probes | 81 |
| Rank-one distance identities and inverse-matrix bounds | 81 |
| Exact hard-query heads with fixed public binary labels | 81 |
| Uniform distance bounds with fixed public binary labels | 81 |
| Finite integer codewords | 256 |

The small geometry uses orthogonal signatures and explicit rational sphere rotations.
It checks the actual new centers and full-sphere anchor-direction variation.
It does not test the logarithmic-dimensional signature existence proof by sampling.
The topology, packing, and entropy claims were reviewed as proofs, not established by these finite checks.

The earlier 21-check and 26-check versions and their source snapshots are preserved as historical review evidence.
The final source-bound receipt `results/theory_review_final.json` identifies the reviewed theorem files and checker.
The executed final checker result is `results/theory_review_checks_binary_corollary.json`; exact final review sources are preserved under `results/theory_review_sources_final/`.
No claim of global novelty follows from this review.
The separate literature audit must determine how this construction differs from earlier unlearning and data-structure lower bounds.
