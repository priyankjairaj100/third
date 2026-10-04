# Certified graph envelopes: a constructive curation-mismatch module

This is a supporting extension of the fixed-curator theory. It gives a computable model-error certificate when the actual refitted blocker relation is certified to lie between two **fixed**, preprocessed ordered graphs. It does not prove that arbitrary reclustering or learned priorities satisfy those envelopes. The certificate can be informative without the actual refitted selected corpus being computed.

## 1. Contract and selection sandwich

Let the original public record identifiers be V, with fixed strict total order. For each v, let B_lo(v), B_0(v), B_hi(v) be subsets of earlier identifiers, with

    B_lo(v) ⊆ B_0(v) ⊆ B_hi(v).

B_0 is the prescribed proxy graph. Assume that for every allowable cumulative deletion request F, |F|≤k, the actual post-request curation uses the same earlier-raw-neighbor suppression rule and has retained blocker sets satisfying

    B_lo(v) \ F ⊆ B_actual,F(v) ⊆ B_hi(v) \ F     (v∉F).

The actual graph may depend arbitrarily on F within those intervals. The lower/upper graphs themselves are fixed before requests. Define

    I_F = {v∉F : B_hi(v)⊆F},
    P_F = {v∉F : B_0(v)⊆F},
    O_F = {v∉F : B_lo(v)⊆F}.

Then, writing T_F for the actual selected corpus,

    I_F ⊆ P_F ⊆ O_F,     I_F ⊆ T_F ⊆ O_F.

Proof: A vertex with no possible retained blocker must be selected; a vertex with a mandatory retained blocker cannot be selected. The proxy graph satisfies the same envelope. If V\F is nonempty, its earliest vertex belongs to I_F, so |I_F|≥1. Empty retained corpora use the predeclared empty-training convention; all three sets are then empty.

The fixed order matters for this nonemptiness guarantee. Arbitrary union envelopes over changing orders may contain cycles, leave I_F empty, and require a separate version of the bound.

## 2. Sharp count-only total variation certificate

Suppress the request subscript. Let

    a=|I|≥1,       p=|P|,       c=|O\P|=|O|−p.

Then the exact worst case over the interval relaxation is

    max_{I⊆T⊆O} TV(Unif(P), Unif(T))
       = 1 − a / max{p, a+c} =: Δ.

This is sharper than the generic symmetric-difference bound min(1, |O\I|/p). It depends only on the three selected-corpus counts.

Proof. For a feasible T set q=|T∩P| and r=|T\P|. Its TV is 1−q/max(p,q+r), with a≤q≤p and 0≤r≤c. For fixed q, selecting r=c cannot increase the common mass q/max(p,q+r). For r=c that common mass is nondecreasing in q: it is q/p where q+c≤p and q/(q+c) otherwise. The minimum is therefore a/max(p,a+c). If a+c≤p, T=I attains the bound; if a+c>p, T=I∪(O\P) attains it. ∎

Sharpness is not merely algebraic for arbitrary ordered graph intervals. Each uncertain vertex v∈O\I has no mandatory retained predecessor and at least one optional retained predecessor. To select it, remove all optional incoming edges; to exclude it, keep one. Incoming edge choices of distinct v are disjoint. Hence every I⊆T⊆O is realizable by some intermediate ordered graph. This does **not** assert realizability by every particular cosine perturbation model, whose edges may be geometrically coupled. In that setting Δ remains valid but can be conservative.

Immediate exactness case: if a=|O|, the proxy and actual selections agree exactly. A wide score-uncertainty band need not hurt the certificate if every uncertain vertex still has a mandatory surviving blocker.

## 3. Consequence for model repair

Use the full-refit mismatch theorem already proved in the main note. Assume shared records have the same training losses, the actual objective is λ-strongly convex, and the repaired proxy model w has validated proxy residual ≤r. If ||∇ell_v(w)||≤G for every v∈O, then

    ||w−w_actual*|| ≤ (r+2GΔ)/λ,
    F_actual(w)−F_actual(w_actual*) ≤ (r+2GΔ)^2/(2λ).

More generally, replace 2G by any certified diameter bound

    D_w ≥ max_{u,v∈O} ||∇ell_u(w)−∇ell_v(w)||.

Then c_w≤D_w Δ. Proof: couple the uniform empirical measures by matching their common probability mass identically. The unmatched mass is TV, and each unmatched gradient pair differs by at most D_w. Bounded norms imply D_w≤2G.

The learning guarantee concerns model approximation. Maintaining lower/upper proxy summaries does not establish that the internal state equals the state of an actual refitted pipeline. It is not an exact full-refit unlearning guarantee.

The fixed-loss hypothesis is separate from curation-score stability. If curation embeddings change while downstream features remain fixed, the theorem applies directly. If the same changing encoder also changes training losses, add a certified term β satisfying ||∇ell_v^actual(w)−∇ell_v^proxy(w)||≤β for v∈O. Then c_w≤D_wΔ+β. Without such a term, the selection certificate alone does not bound model error.

## 4. Task-sensitive second-moment refinement

The count-only bound can be pessimistic when uncertain records have nearly identical gradients. A computable refinement uses gradient second moments over O.

For fixed w let g_v=∇ell_v(w), g_bar=|O|^{-1}Σ_{v∈O}g_v, and

    C_O=Σ_{v∈O}(g_v−g_bar)(g_v−g_bar)^T.

Let

    H(a,p,c) =
      1/a−1/p,                                  if 2a≤p;
      1/p+(p−2a)/(p(a+c)),                       if 2a≥p.

The expressions agree at equality. Then for every I⊆T⊆O,

    ||mean_T(g)−mean_P(g)||
       ≤ sqrt( λ_max(C_O) · H(a,p,c) ).

Proof. Let h_v=1_T(v)/|T|−1_P(v)/p. Because Σh_v=0, the gradient difference is the centered gradient matrix times h. Its squared norm is at most λ_max(C_O)||h||². If q=|T∩P| and r=|T\P|, then

    ||h||² = 1/p + (p−2q)/(p(q+r)).

At fixed r this decreases in q because its derivative is −(p+2r)/(p(q+r)^2), so q=a maximizes it. At q=a it decreases in r if p≥2a and increases in r if p≤2a. The appropriate endpoint is r=0 or c, giving H. ∎

The coefficient-norm optimization is exact. The covariance inequality is an upper bound, not a claim that all gradient sets attain equality. Using tr(C_O) instead of λ_max(C_O) is valid and avoids an eigensolve if only squared norms are available.

For dynamic w, fixed record-local summaries cannot normally evaluate all current gradient moments. A legitimate implementation picks an externally fixed reference w_0, stores moments of g_v(w_0), and assumes per-record gradients are L-Lipschitz between w_0 and w. It obtains

    c_w ≤ sqrt(λ_max(C_O(w_0)) H(a,p,c))
           + 2L ||w−w_0|| Δ.

Proof: Apply the second-moment bound at w_0. For h_v=∇ell_v(w)−∇ell_v(w_0), ||h_v||≤L||w−w_0||. Apply the bounded-gradient TV inequality to the change. ∎

Take the minimum of this bound and 2GΔ when both are valid. For bounded-feature squared loss, L can be the squared feature-norm bound. A trained original-data-dependent w_0 is not automatically deletion-clean auxiliary state; use a public anchor or expressly weaken the state contract.

## 5. Concrete construction from score-drift bounds

Suppose eligibility and relative order are fixed, all original and refitted embedding vectors are unit norm, and certified bounds hold uniformly for every allowed F:

    ||z_u^F−z_u||≤ε_u,   |τ_uv^F−τ_uv|≤ε_τ,uv.

Define the original pair margin m_uv=z_u^T z_v−τ_uv and d_uv=ε_u+ε_v+ε_τ,uv. Then

    |(z_u^F)^T z_v^F − τ_uv^F − m_uv| ≤ d_uv,

because the dot-product change is at most ε_u+ε_v using unit norms. With the strict duplicate test score>threshold, use

    u∈B_lo(v) iff u≺v, eligible, and m_uv>d_uv;
    u∈B_hi(v) iff u≺v, eligible, and m_uv>−d_uv.

The strict inequalities handle equality correctly: m_uv+d_uv=0 cannot yield an edge; m_uv−d_uv=0 does not force an edge. The proxy graph m_uv>0 lies between them. More generally any certified interval [l_uv,h_uv] for the actual margin gives mandatory edges at l_uv>0 and possible edges at h_uv>0.

This construction is useful for a genuinely bounded encoder/threshold perturbation. It does not bound an arbitrary encoder refit for free. Learned partition changes require a certified envelope for eligibility, and learned priority changes fall outside this fixed-order module. An incomplete nearest-neighbor search does not certify the upper graph unless its recall over the required score range is itself guaranteed.

## 6. Exact maintenance and cost

Preprocess count-valued deletion-horizon polynomials for B_lo and B_hi:

    T_j(F)=Σ_v (1−x_v) Π_{u∈B_j(v)} x_u,       j∈{lo,hi}.

Apply the existing indexed canonical update to both summaries. After a cumulative request, their constant coefficients give |O_F| and |I_F|, respectively. The proxy summary supplies p. This gives Δ in O(1) arithmetic beyond those updates, without computing the actual refitted selection.

For each envelope let M_j be its initial nonzero coefficient count and Φ_j=Σ_{J∈keys_j}|J|(|J|+1)/2. Under the indexed algorithm's same hashing and arithmetic model, the total extra work across the cumulative horizon is

    O(Q + Φ_lo + Φ_hi + M_lo + M_hi),

with count-map state O(M_lo+M_hi+L_lo+L_hi), apart from shared identifier bookkeeping. Initial graph construction and polynomial construction are separate costs. Count coefficients require integer bit costs if those are being charged.

For the optional gradient-moment refinement, enrich the outer summary with g_v(w_0), its symmetric outer product, and count; the dimension is 1+d+d(d+1)/2. C_O follows from the resulting sums. Evaluating λ_max has its own cost, typically O(d^3) for an exact dense eigensolve, or use an explicitly certified upper bound. A cheaper trace-only summary uses count, d gradient coordinates, and ||g_v(w_0)||², and computes tr(C_O) in O(d) arithmetic.

Fixed envelope graphs valid for all requests are essential to these costs. Query-specific envelope changes can require rebuilding or additional dynamic-graph work and are not covered. Maintaining proxy envelopes does not certify that graph construction itself is cheap or that a real learned curator meets the assumed drift budget.

Canonical deletion-clean state additionally requires that the envelope graphs and anchor statistics are prescribed external inputs or otherwise have their own valid state-repair argument. If originally trained envelopes are retained merely as predictors of a refitted target, the numerical model certificate is conditional on their certified validity, but those historical fitted objects are not thereby certified as the fresh retained-pipeline state.

## 7. Verification

`round3_envelope.py` ran successfully with NumPy. Exhaustive ordered graphs through four vertices checked 760 envelopes, 23,788 polynomial count queries, 61,901 intermediate graph selections, 11,134 interval-realizability identities, and 15,812 sharp TV bounds. All TV comparisons were exact in the numerical check. Another 648 count configurations checked the exact coefficient-norm maximization. Random gradient sets gave 4,665 second-moment checks with largest floating-point overshoot 4.44e−16. There were 2,000 vector/threshold drift checks. These are theorem/implementation sanity checks, not corpus experiments.

## Recommendation for main note

Add the selection sandwich and sharp Δ formula to the full-refit section as a compact, constructive **sufficient-condition module**. Include the optional second-moment refinement only if it earns its space; it is valuable because it reflects task-relevant ambiguity rather than merely set churn. Do not promote graph-envelope validity to a proved property of SemDeDup refitting. The exact fixed-curator theorem stack remains unchanged.
