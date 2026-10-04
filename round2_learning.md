# Round 2: certified convex repair with explicit computational errors

This note audits and strengthens the convex module in the current TeX. It does not edit the main document. The fixed-curator contract should remain the core. The full-refit discrepancy result below is a conditional robustness statement, not a new clustering algorithm or a claim that arbitrary refits remain stable.

## 1. Audit conclusion

The normalized derivative formulas, strong-convexity argument, and Gaussian marginal-output comparison in the current TeX are mathematically correct. The logistic Hessian-Lipschitz bound B³/4 is conservative but valid; the sharper global constant is B³/(6√3). The following details should be explicit in an implementation:

- A certificate uses the target average-loss objective with its new sample count. Approximate old optimality contributes the factor n/n' times the old gradient error.
- An approximate inverse is acceptable, but its actual linear-system residual or a valid operator-error bound must enter the certificate.
- A small gradient residual provides parameter and objective-error guarantees. It becomes a retraining-distribution guarantee only for the prescribed noisy learner, with independent output smoothing of the repaired center.
- Derivative statistics at a fitted original anchor preserve information about deleted data. Canonical state deletion applies only if anchors/bases are fixed independently of deletable data, or if their state is separately repaired.
- Reanchoring a nonlinear derivative summary is not possible from one stored gradient/Hessian pair in general. A fallback requires enough surviving raw/feature payload to recompute the needed statistics, or an explicit retained-data reread contract.

The robust hidden-copy lower bound currently in the TeX avoids the forgotten-payload issue. Its approximate decoding inequality also checks out: the recovered integer error is at most (2m+3)(1+λ)η.

## 2. A single certificate covering approximate gradients, curvature, inverses, and solves

Let F be the intended target empirical objective after exact counterfactual curation, including regularization. Assume F is λ-strongly convex, λ>0, and its Hessian is ρ-Lipschitz on the segment from w to w+s. Suppose computed quantities satisfy certified bounds

||ĝ−∇F(w)|| ≤ a,

||B−∇²F(w)||op ≤ e,

||ĝ+Bs|| ≤ ξ.

Here s may be obtained by an approximate inverse, conjugate gradients, a low-rank preconditioner, or any other procedure. The certificate itself does not require that procedure to be exact.

**Theorem 1 (unified residual certificate).** Define

r = a+ξ+e||s||+(ρ/2)||s||².

Then

||∇F(w+s)|| ≤ r,

||w+s−w*|| ≤ r/λ,

F(w+s)−F(w*) ≤ r²/(2λ).

**Proof.** Taylor expansion gives

∇F(w+s) = (∇F(w)−ĝ)+(ĝ+Bs)+(∇²F(w)−B)s+R,

where ||R||≤ρ||s||²/2. This proves the residual bound. Strong monotonicity yields the distance bound. Strong convexity also gives, for v=w*,

F(v)≥F(w+s)+⟨∇F(w+s),v−(w+s)⟩+(λ/2)||v−(w+s)||².

Maximizing the right-hand upper bound on F(w+s)−F(v) over the displacement yields ||∇F(w+s)||²/(2λ).

If B is positive definite, the numerical solve is well behaved, but positivity is not an extra premise of the certificate once a finite s and valid a,e,ξ are available. Damping B by τI changes the operator-error allowance from e to at most e+τ. Do not omit that cost.

### Approximate inverse

If s=−Pĝ and a known bound gives ||I−BP||op≤κ, then ξ≤κ||ĝ||. Computing the actual residual ||ĝ+Bs|| is usually sharper and avoids proving an inverse approximation theorem. A cached inverse of the old Hessian is merely a step generator; its use does not excuse evaluation/bounding of the retained target's linear residual.

For an iterative solve with a Hessian-vector-product oracle, stop when the validated linear residual plus the other terms fits the desired residual budget. There is no need to form or certify the entire inverse.

### Numerical meaning of “certified”

The quantities a and e include errors from coefficient summation, division by the new sample count, evaluating gradients/Hessians, and sketching. ξ must upper-bound the actual linear residual, including the residual computation's own numerical error. Norm and smoothness constants must be conservative bounds. Exact/rational arithmetic for additive statistics, directed-rounding intervals, or a documented forward-error analysis can provide these allowances.

A standard floating-point residual printed by NumPy/PyTorch is a useful diagnostic but is not, by itself, a rigorous floating-point certificate. The mathematical theorem does not require bitwise equality with a differently ordered fresh retraining calculation. An empirical paper can explicitly report an analytic certificate with a controlled numerical tolerance, but it should avoid calling unaccounted roundoff a formal proof.

## 3. Two implementable repair modes with exact curation

The choice is a storage/access contract, not simply a choice of optimizer.

### Mode A: fixed-anchor, derivative-summary repair

Choose an anchor a₀ and retain the curation-statistic summary for

t(v)=(∇ℓ_v(a₀), ∇²ℓ_v(a₀), 1).

For the current exact selected set S_t, query its aggregate (G_t,H_t,n_t). Compute

g_t=G_t/n_t+λa₀,

K_t=H_t/n_t+λI.

Generate a step s_t and apply Theorem 1 with w=a₀. If the statistics are exact and the solve is exact, r_t=ρ_t||s_t||²/2. Neither a₀ being the original optimum nor the deletion requests being nonadaptive is required for this deterministic per-target parameter theorem. A trained original anchor simply makes small steps more plausible initially.

Each request is certified afresh against the current target. There is no hidden accumulation of derivative-approximation error from previous model steps because the anchor remains fixed. Numeric summation errors and any sketch error still enter a,e.

Storage with full dense derivatives is Θ(M(d+d²)) real numbers for M coefficient keys, plus key metadata; queries sum this many coordinates per visited key. This is substantially larger than storing a d-dimensional feature payload per eligible record unless many documents genuinely share keys. The fixed-anchor derivative summary is not automatically a memory-efficient method.

**Fallback limitation.** If its residual exceeds the budget, reoptimizing on S_t requires access to S_t's surviving payloads. Rebuilding a derivative summary at a new fitted anchor for future requests generally requires evaluating every record in the current activation-eligible set E_h, not only the currently selected records. Cached first and second derivatives at the old anchor do not determine logistic derivatives at an arbitrary new anchor. A no-reread system that discarded those payloads must declare this failure mode; it cannot promise an unrestricted retraining fallback.

### Mode B: moving-anchor repair with a retained eligible-payload buffer

Store enough payload to maintain the exact curator under the allowed cumulative budget. At time t, maintain candidate w_t and approximate derivative information satisfying

||ĝ_t−∇F_t(w_t)||≤a_t,

||B_t−∇²F_t(w_t)||op≤e_t.

Let exact curation identify additions P and removals R, with selected counts n_t,n_{t+1}>0. Put f_i(w)=ℓ_i(w)+λ||w||²/2. At w_t compute

ḡ=(n_t/n_{t+1})ĝ_t + [Σ_P∇f_i(w_t)−Σ_R∇f_i(w_t)]/n_{t+1},

B̄=(n_t/n_{t+1})B_t + [Σ_P∇²f_i(w_t)−Σ_R∇²f_i(w_t)]/n_{t+1}.

If those changed-record evaluations are exact, the corresponding errors at w_t are ā=(n_t/n_{t+1})a_t and ē=(n_t/n_{t+1})e_t. Add explicit allowances for inexact changed-record evaluations or a compression operation.

Choose a step with ||ḡ+B̄s||≤ξ. Then a valid new gradient residual is

a_{t+1}=ā+ξ+ē||s||+(ρ_{t+1}/2)||s||².

Store ĝ_{t+1}=0. If B̄ is retained as the next Hessian estimate, a valid next curvature error is

e_{t+1}=ē+ρ_{t+1}||s||.

These recurrences follow directly from Theorem 1 and Hessian Lipschitz continuity. They account for drift instead of silently treating the old Hessian as current. If B̄ needs damping, include it in ē. Empty selected sets use the specified zero-model convention and reset the corresponding state appropriately.

**Residual fallback.** Given a public allowed residual r_max, accept the candidate only if the bound is at most r_max. Otherwise perform an exact-current-objective gradient pass at the candidate. Its validated norm may certify the candidate immediately despite the conservative Taylor bound. If still too large, optimize the retained selected objective until a validated residual meets the budget, and either refresh curvature or add the Hessian-Lipschitz allowance for the further move to the stored curvature error. A fallback merely to a finite solver tolerance supplies that tolerance as the next a_t; do not reset it to zero without justification.

With a full dense Hessian, evaluating q=|P|+|R| changed examples costs O(qd²), and solving costs O(d³) in a dense implementation. A direct gradient audit costs O(n_td). These costs must be reported alongside curation and state-maintenance costs. Matrix-free solving can reduce storage, but generally needs repeated target-data passes unless a sufficient HVP summary exists.

This mode has a clear and honest advantage: many small changes may use only changed payloads and cached curvature, while failures have an explicit retained-data path. It does not inherit the no-payload model-query advantage of a polynomial derivative summary.

## 4. A conclusive compact curvature option

This is a deterministic optional implementation lemma, not a proposed new research branch. It replaces a full d×d Hessian per coefficient by an r×r projected matrix and one nonnegative scalar.

For binary logistic loss at a fixed anchor a₀, write

C=(1/n)Σ_i q_i z_i z_iᵀ,   q_i=σ(a₀ᵀz_i)σ(−a₀ᵀz_i)≥0,

so the true target Hessian is λI+C. Let V have orthonormal columns, P=VVᵀ, and define

K=VᵀCV,

τ=tr[(I−P)C]=(1/n)Σ_i q_i||(I−P)z_i||²,

α=||K||op.

The approximate Hessian is B=λI+VKVᵀ. All entries of K and τ are additive statistics at the fixed anchor. They therefore admit exactly the same curation coefficient query and substitution operations as the full Hessian.

**Theorem 2 (projected-curvature certificate).**

||C−PCP||op ≤ e_proj := [τ+sqrt(τ²+4ατ)]/2.

**Proof.** In the decomposition range(P)⊕ker(P), write the PSD matrix C as blocks [[K,L],[Lᵀ,D]]. Positivity implies ||L||²≤||K||·||D||≤ατ, and ||D||≤tr(D)=τ. For a unit vector split into components x,y,

|[x;y]ᵀ(C−PCP)[x;y]| ≤2sqrt(ατ)||x||||y||+τ||y||².

The largest eigenvalue of the 2×2 scalar matrix [[0,sqrt(ατ)],[sqrt(ατ),τ]] is exactly e_proj. Taking the supremum proves the claim.

The approximate Newton step can be computed by

s=−λ^{-1}(I−P)g−V(λI_r+K)^{-1}Vᵀg.

Thus the certificate in Theorem 1 becomes

r≤ξ+a+e_proj||s||+(ρ/2)||s||²,

with additional numerical/sketch errors included if K,τ are inexact. A practical implementation may use a certified upper bound on τ and α. The scalar τ must bound the true nonnegative off-subspace energy; setting a numerically negative value to zero is not a proof that its true value vanishes.

An optional sharper directional error is available. Let s_parallel=Vᵀs and s_perp=(I−P)s. Then

||(C−PCP)s|| ≤ sqrt{ ατ||s_perp||² + [sqrt(τ s_parallelᵀK s_parallel)+τ||s_perp||]² }.

This follows from ||Ls_perp||≤sqrt(ατ)||s_perp|| and ||Lᵀs_parallel||≤sqrt(τ s_parallelᵀK s_parallel). Replace e_proj||s|| by the smaller of these two valid bounds.

**Costs and limits.** Including an exact d-dimensional gradient, the statistic dimension becomes O(d+r²), with a global O(dr) basis and O(r³+dr) approximate solve. The bound can be loose when retained curvature has appreciable energy outside the chosen subspace. It does not promise a speedup or useful certificate for every embedding distribution. A public basis and public anchor keep the statistic record-local for canonical-state results. A data-dependent original basis or anchor has the same state-deletion caveat as before.

I checked this operator bound and directional bound on 100 random PSD logistic-weighted matrices; all inequalities held. This finite check is not evidence of sharpness on NLP data; the largest observed actual-operator-error/bound ratio was about 0.27.

For the main exact-ridge experiment, a simpler option is to specify a lower-dimensional public feature projection as part of the learner from the outset. That changes the reference learner but retains exactness and often gives a clearer experiment than adding approximate-curvature machinery.

## 5. Full-refit discrepancy can be bounded without claiming cluster stability

Let Ŝ denote the exact output of the frozen-curator repair and S* the actual full-refit curator output. Assume the learner uses the same fixed loss for each surviving record and the same regularization λ in both cases. Let μ and ν be their normalized empirical measures. Suppose a repaired center w has validated residual

||∇F_μ(w)||≤r.

Define the task-gradient discrepancy

c_w=||E_ν∇ℓ_z(w)−E_μ∇ℓ_z(w)||.

**Theorem 3 (modular curation-mismatch bound).** For the full-refit optimum w*_ν,

||w−w*_ν|| ≤ (r+c_w)/λ,

F_ν(w)−F_ν(w*_ν) ≤ (r+c_w)²/(2λ).

**Proof.** The regularization gradients are identical and cancel in the objective-gradient difference. Thus ||∇F_ν(w)||≤r+c_w. Apply strong convexity as in Theorem 1.

Any certified upper bound on c_w can be substituted. Several useful choices have precise assumptions:

1. **Total variation.** If every relevant record has ||∇ℓ_z(w)||≤G, then c_w≤2G TV(μ,ν). For binary logistic loss with feature norm at most B, G=B. For uniform sets with fixed record losses and sizes n,m>0 and intersection size h,

   TV(μ,ν)=1−h/max(n,m).

   This identity follows by summing the common probability mass h·min(1/n,1/m). It is the correctly normalized mismatch; |Ŝ△S*| divided by an arbitrary denominator is generally not exact.

2. **Known uncertain set.** If a proved argument establishes |Ŝ△S*|≤u, then TV(μ,ν)≤min(1,u/|Ŝ|). Indeed if a records are added and r records removed, TV equals a/(n−r+a) when a≥r and r/n otherwise. Each is at most (a+r)/n. This converts a certified bound on potentially changed selection flags into a model guarantee, without demanding zero changes.

3. **Task-aware transport.** If ||∇ℓ_z(w)−∇ℓ_{z'}(w)||≤L_g d(z,z'), then c_w≤L_g W₁(μ,ν;d). Any explicit feasible coupling provides an upper bound; exact optimal transport is unnecessary. The metric must include training labels or some other assumption guaranteeing gradient closeness.

4. **Anchor discrepancy plus a movement term.** If each loss has L-Lipschitz gradient on the segment from a₀ to w, then

   c_w ≤ c_{a₀}+2L·TV(μ,ν)||w−a₀||.

   To prove it, apply the signed measure ν−μ to the gradient change between the two anchors. Its total variation norm is 2TV(μ,ν), and every integrand norm is at most L||w−a₀||. For logistic loss L≤B²/4.

These bounds do not assume that a clustering fixed point survives. Conversely, without a proved bound on the mismatch or an actual full-refit oracle, none of them certifies that c_w is small. Computing the full oracle solely to evaluate c_w is a valid retrospective audit, but its cost is not an efficient repair algorithm. A small empirical mismatch observed on a sample of requests is not a uniform future-request guarantee.

The total-variation intersection formula assumes each shared identifier carries the same loss in both pipelines. If a full refit changes embeddings or labels, couple the actual labeled representations and include those changes in the transport discrepancy; shared IDs no longer contribute automatically zero gradient discrepancy. If the regularization changes too, add |λ_ν−λ_μ|·||w|| to the mismatch residual.

### Task consequence

For a test feature z with ||z||≤B_test, the difference in linear logits is at most B_test(r+c_w)/λ. Binary classification is guaranteed unchanged relative to the full-refit optimum whenever the repaired model's absolute logit exceeds that bound. The sigmoid probability difference is at most B_test(r+c_w)/(4λ). These are deterministic consequences of parameter proximity; they do not require a noisy learner.

## 6. Distributional claim that is justified—and its limits

If the reference learner from inception releases w*+N(0,σ²I) for fixed public σ, and repair independently releases w+N(0,σ²I), any proved parameter bound η gives the current TeX's valid two-sided (ε,δ) comparison with

ε=(η/σ)²/2+(η/σ)sqrt(2log(1/δ)).

Use η=r/λ for exact curation, or η=(r+c_w)/λ for the conditional full-refit target. The fixed σ belongs to the reference algorithm; selecting σ based on an original-data-dependent repair certificate changes the comparison unless separately analyzed. A deterministic tolerance fallback preserves the bound if every accepted branch satisfies η and smoothing occurs independently after the branch is selected.

For the fixed request considered in the TeX, no adaptive-transcript claim is needed. A sequence would require per-history kernel bounds and explicit composition. Keeping the previously released model or fitted anchor in the state still prevents a claim that the entire historical system has been erased.

A narrowly stronger state statement is available when the anchor, feature extractor, and projection basis are public and independent of deletable data: the canonical derivative coefficient map is exactly the retained-data map, while the current released model can satisfy the smoothed approximation guarantee. The joint pair (canonical map, current noisy model) then obeys the same comparison to the corresponding budgeted retained-data learner, provided no additional original-dependent state is retained. This is a conditional extension of the canonical-state theorem, not a claim about the usual trained-anchor implementation.

## 7. Recommended placement in the paper

Keep the main algorithm/theorem stack exact and curator-specific. The unified numerical residual theorem is a useful supporting proposition because it closes a real implementation gap. The streaming recurrences provide a fully specified baseline with honest data access and fallback costs. Put Theorem 3 in a scoped appendix or robustness section; it quantifies an audited full-refit mismatch but does not replace the fixed-curator assumption. The compact curvature lemma is complete and optional; it should not become a new central claim before experiments show that its certificate is useful.
