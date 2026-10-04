# Third-pass adversarial audit

Audited `output/pdf/counterfactual_curation_theory.tex` (version 2 as received) and `algorithm_round2.py`. Scope: theorem correctness, algorithm accounting, and state/metadata contracts. This is an independent mathematical/code audit, not machine verification or novelty certification.

## Findings

No concrete counterexample or algebraic error was found in the stated core results. In particular:

- Earlier-raw-neighbor suppression is correctly distinguished from greedy independent-set selection. The activation equation, eligible-set criterion, and monotonicity of surviving originally selected records follow directly from restriction consistency.
- Truncated multilinear coefficients survive substitution exactly at the *remaining* horizon. The proof correctly excludes omitted high-degree monomials, and deleted-record cancellation still works when one monomial was truncated.
- Sequential canonical equality is a property of the designated abstract state; it does not erase history or establish physical memory canonicality. The document already states this limitation.
- The incidence graph rank, untruncated forest, reduced-root profile, and source-level incidence extension are correct. Source collapsing destroys the forest property but not incidence rank.
- The randomized lower bound correctly reduces each allowed high-budget request to one independent hidden bit. Its Fano argument requires neither simultaneous success nor a union bound. Full forgotten-record payloads are independent of those bits in the hard family. The low-budget state contains no hidden-label dependence, while public metadata may still be large.
- The fixed-threshold geometry and shared-feature ridge equations are consistent. The ambient dimension grows logarithmically with the number of hidden records and is not falsely advertised as fixed.
- The convex residual and mismatch certificates have the correct average-loss normalization and gradient/Hessian error terms. Their unconditional strong-convexity implications are correct; their input error bounds still need certification.

## Conclusive improvement: fast storage is already within factor two

Let the incidence graph have p incident non-ground nodes and c_0 ungrounded components. Every ungrounded component has at least two nodes, since same-source loops have been canceled and unused isolated nodes are omitted. Therefore c_0 <= p/2, and

    r = p - c_0 >= p/2.

The canonical sparse map stores at most one statistic vector for each incident non-ground node; zero coefficients only reduce its size. Hence at every retained state,

    M <= p <= 2r.

For s arbitrary independent statistic coordinates, the existing indexed implementation therefore stores at most 2sr statistic-value coordinates, against the sr linear-query optimum. This does not count keys, IDs, incidence indices, or candidate metadata; it does not establish a lower bound for constrained ridge moments or nonlinear real encodings. The compression agent independently derived this result and owns the sharpness construction and checks.

This is more useful than claiming that the fast implementation and rank theorem have no quantitative connection: they have a sharp constant-factor connection without modifying the fast algorithm.

## Rank-optimal canonical alternative

A canonical current incidence graph permits omission of one deterministically selected node value per ungrounded component. Recover the missing value as minus the sum of the other coefficients in that component. Grounded components retain all non-ground node values. This stores exactly r scalar coordinates per statistic dimension.

To make the metadata state deletion-consistent, retain current membership and current blocker lists only for eligible records with b_v <= h. Previously ineligible records cannot become eligible as the horizon decreases: after r deletions their blocker count remains greater than h-r. The IDs of such records can still appear in eligible records' blocker lists, so discarding their own candidate metadata does not discard their blocking role. After a request, remove deleted IDs from these lists, drop deleted candidates and candidates above the new horizon, construct the current incidence graph, and apply the same deterministic component convention. Decompress-specialize-recompress needs no retained statistics beyond the old compressed state.

This yields a rank-optimal statistic-value representation with a canonical metadata builder, but may require a scan of the candidate metadata and full coefficient reconstruction. It does not inherit the indexed update bound. The compression agent owns the final formalization.

## Implementation audit and new check

The stable-entry-handle implementation correctly avoids reindexing surviving incidences during moves. New destinations do not contain the just-deleted identifier, so they cannot be revisited during that atomic deletion. Collision cleanup removes the source's surviving incidences; cancellation additionally removes the destination's incidences. Pruning the old degree-h bucket implements the horizon decrement. No operation creates an incidence token.

The vector-cost bound charges additions and zero tests only to collisions. Tuple shrinking charges its O(d) cost to the decrease d in d(d+1)/2. Whole-map canonical exports and ridge solves are correctly excluded from the update theorem.

I added an independent check that does not rely on curation-generated coefficients: 2,000 random sparse vector-valued Boolean polynomials, 2,805 intermediate batch comparisons against a direct cumulative substitution/truncation oracle, including 275 empty-result maps. All comparisons, state invariants, and token/potential bounds passed. Files:

- `round3_audit_check.py`
- `round3_audit_check.json`

This broadens coverage of the generic map engine but is not a replacement for the proof.

## Caveats to preserve in the finalized text

1. Expected dictionary runtime is an explicit abstract data-structure assumption. Python's deterministic tuple hashes and built-in dictionaries are a reference implementation; the tests do not prove an adversarial expected-time hash-table theorem. State this once if making a formal runtime claim for the delivered implementation.
2. The factor-two statement is about statistic-value coordinates, not total physical memory or arbitrary constrained ridge sufficient statistics.
3. Rank-optimal canonical metadata must describe the current eligible records, not retain the original fitted graph or original-record metadata while calling it erased.
4. A degree-zero coefficient vector requires at least O(s) work to output or decode. The update bound is not a constant-time release theorem.
5. Full refitting, history-independent allocator state, and a replenished horizon remain outside the exact core. The present qualifications are necessary and should not be weakened to make the story broader.

## Recommendation

The core algebra and indexed algorithm can be frozen after adding the sharp factor-two statement and the canonical rank-optimal baseline. Keep the latter as a slower optimal-statistic benchmark. Do not expand the main claim to generic reclustering or deep-model unlearning in this pass.

## Follow-up audit: new prediction, coding, and graph-envelope modules

Independently audited `round3_lower.md` and `round3_envelope.md` after the initial report. No mathematical correction is required.

- For the squared-error information converse, g(v)=h_2((1-sqrt(1-4v))/2) is indeed increasing and concave. Conditional Bernoulli variance is the minimum MSE, and the stated Jensen/entropy chain gives B >= m[1-g(D)].
- The covering-code construction attains the matching asymptotic rate. Uniform masking makes the transformed word independent of the random permutation; the queried transformed coordinate is uniform independently of that word. Thus a per-word mean-distortion bound becomes an expected bound for every fixed original input and query. The codebook can be public and fixed. Its costs and the exclusion of adaptive queries must remain explicit.
- With c=1+2 lambda, ridge excess objective is at least (Yhat-Y)^2/(4c), with equality along the target direction. Therefore the zero-information excess threshold 1/(16c) and the rate argument R_sq(4c epsilon) are correct.
- Both proposed fixed test distributions have the stated positive, dimension-independent separation. They certify fidelity to the retained-data predictor, not arbitrary supervised population risk.
- The envelope TV optimum Delta=1-a/max(p,a+c) is correct; its extremizers are feasible under the abstract ordered-edge interval relaxation. The exact coefficient-norm bound H(a,p,c), covariance refinement, gradient-diameter argument, and strict score-envelope inequalities are correct.

One concrete implementation issue was found in the new gauge-compression reference: repeated blocker/component sorting, tuple-key disjoint-set operations, and per-scalar tuple-key lookups would not meet the proposed linear key-processing bound. This was reported to the compression agent and root; the agent is replacing these operations with sorted tuple metadata, interned integer DSU handles, one-pass component minimum selection, and one lookup per vector. This issue does not affect the mathematical gauge construction, the original indexed implementation, or the lower/envelope modules.
