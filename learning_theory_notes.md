# End-to-end learning theory for counterfactual semantic curation

These are derivations for the proposed project, not claims of literature novelty. The mathematical statements below distinguish exact algebraic results, consequences of standard convex analysis, and research directions. The strongest contribution should come from the curation-specific representation and repair problem; a signed Newton update by itself is a standard extension of existing certified-removal techniques.

## 1. Contract, state, and assumptions

Let a fixed curation rule map the raw collection D to a labeled training set S=C(D). After deleting raw records F, the intended target is S'=C(D\\F), followed by rerunning the specified learner on S'. It is not generally C(D)\\F.

Write K=S∩S', R=S\\S', A=S'\\S, n=|S|, and n'=|S'|. R need not equal the raw deletion request. A consists of retained records that did not participate in the old learner. The following theory assumes frozen, corpus-independent text features and fixed labels. If the encoder, label generation, graph, ordering, or hyperparameters were learned from D, their counterfactual repair is an additional obligation.

For n>0 consider

F_S(w) = (1/n) Σ_{i∈S} ℓ_i(w) + (λ/2)||w||²,    λ>0.

Assume each loss is convex and twice differentiable. Then F_S is λ-strongly convex and has a unique optimum w_S. For the one-Newton-step results, assume the target Hessian is ρ-Lipschitz along the correction segment:

||∇²F_{S'}(u)-∇²F_{S'}(v)||op ≤ ρ||u-v||.

Define the empty-set objective explicitly as λ||w||²/2, with optimum zero; the formulas involving 1/n' apply only when n'>0.

Three different certification contracts must not be conflated:

1. Parameter approximation: ||w̃-w_{S'}||≤a.
2. Model-output unlearning: the distribution of the released repaired model is close to the released retained-data retraining model.
3. State unlearning: the joint distribution of every retained/exposed state component agrees with a fresh prescribed stateful learner on the retained corpus.

Keeping the old optimum or its derivative cache generally permits (1) and (2), not (3). Exact canonical intrinsic-statistic states can sometimes achieve (3), as described below. Previously disclosed models are not retroactively erased by any of these statements.

## 2. Correct signed update, including normalization

Let h_i(w)=∇ℓ_i(w)+λw and J_i(w)=∇²ℓ_i(w)+λI. For any reference point w₀,

g' := ∇F_{S'}(w₀)
   = (n/n')∇F_S(w₀) + [Σ_{i∈A}h_i(w₀)-Σ_{i∈R}h_i(w₀)]/n',

H' := ∇²F_{S'}(w₀)
   = (n/n')∇²F_S(w₀) + [Σ_{i∈A}J_i(w₀)-Σ_{i∈R}J_i(w₀)]/n'.

Proof: represent each objective as the average of f_i(w)=ℓ_i(w)+λ||w||²/2, multiply by the corresponding set cardinality, and apply the additive set identity ΣS'=ΣS+ΣA−ΣR.

At an exact old optimizer, ∇F_S(w₀)=0, and therefore

g' = [Σ_A∇ℓ_i(w₀)−Σ_R∇ℓ_i(w₀)+(n'−n)λw₀]/n'.

The final normalization term is necessary. Using only additions minus removals of unregularized gradients silently changes the target objective whenever n'≠n. The same issue appears in the Hessian:

H' = (n/n')H₀ + [Σ_A∇²ℓ_i(w₀)−Σ_R∇²ℓ_i(w₀)]/n' + [(n'−n)/n']λI.

If the old solve is approximate and ||∇F_S(w₀)||≤e₀, its residual contribution is at most ne₀/n'. Recording the actual old gradient permits an exact target gradient at w₀; recording only its norm gives an uncertainty term.

## 3. A cache-computable one-step certificate

Theorem 1. Suppose g' and H' above are exact, let s=−(H')^{-1}g', and release the deterministic candidate w̃=w₀+s. Then

||∇F_{S'}(w̃)|| ≤ (ρ/2)||s||²,

||w̃−w_{S'}|| ≤ (ρ/(2λ))||s||²
                  ≤ ρ||g'||²/(2λ³).

Proof. By the integral form of Taylor's theorem,

∇F_{S'}(w₀+s)=g'+H's+∫₀¹[∇²F_{S'}(w₀+ts)−H']s dt.

The first two terms cancel. The integrand norm is at most ρt||s||², giving the residual bound. Strong monotonicity gives

λ||w̃−w_{S'}||² ≤ ⟨∇F_{S'}(w̃),w̃−w_{S'}⟩,

and Cauchy–Schwarz yields distance at most residual/λ. Finally H'≽λI implies ||s||≤||g'||/λ.

The most useful bound is the step-dependent ρ||s||²/(2λ), rather than replacing ||s|| by ||g'||/λ. It captures favorable conditioning and is computable from cached derivatives plus the changed examples. No target-gradient evaluation over every retained example is necessary, provided a valid global or segment-local ρ is available.

A useful extension handles approximate gradient/Hessian information. If ||ĝ−g'||≤a and ||B−H'||op≤η, B is positive definite, and s=−B^{-1}ĝ, then

||∇F_{S'}(w₀+s)|| ≤ a+η||s||+(ρ/2)||s||².

The proof is the same Taylor identity with g'+H's=(g'−ĝ)+(H'−B)s. Divide the result by λ for the optimizer-distance certificate. If B needs damping, include the damping matrix norm in η; do not call a damped inverse the exact target Hessian inverse.

### Logistic specialization

For binary logistic loss ℓ(w;u,y)=log(1+exp(−ywᵀu)), y∈{−1,+1}, with ||u||≤R,

∇ℓ=−yu σ(−ywᵀu),

∇²ℓ=σ(ywᵀu)σ(−ywᵀu) uuᵀ.

The absolute third derivative of scalar logistic loss is at most 1/(6√3), so the target objective has globally Lipschitz Hessian with

ρ≤R³/(6√3).

Consequently the retained-data optimizer-distance bound is

||w̃−w_{S'}|| ≤ R³||s||²/(12√3 λ).

If vectors are clipped to a public norm R, this bound requires no retained-data pass. It may be conservative. A numerical check using random norm-bounded features verified the exact normalization and Hessian identities to ≈10^{-16}, and observed residual 1.61×10^{-6} below the predicted 3.62×10^{-4}; this is only a sanity check, not evidence of practical tightness.

## 4. A clean distribution certificate through output smoothing

Specify the learning algorithm from the beginning as

Aσ(S)=w_S+σZ,    Z~N(0,I_d),

for a fixed public σ>0. The repair algorithm releases

Uσ(D,F)=w̃+σZ',    Z'~N(0,I_d),

with fresh independent noise. Assume the approximation w̃ is deterministic conditional on the dataset/curator seed, or the distance bound holds uniformly over any internal randomness. If ||w̃−w_{S'}||≤a, set q=a/σ.

Theorem 2. For every ε≥0, Uσ(D,F) and Aσ(S') satisfy both directions of approximate max-divergence with

δ(ε,q)=Φ(q/2−ε/q)−exp(ε)Φ(−q/2−ε/q),    q>0,

and δ(ε,0)=0. Here Φ is the standard normal CDF. Thus for every measurable event E,

Pr[Uσ∈E] ≤ exp(ε)Pr[Aσ∈E]+δ,

Pr[Aσ∈E] ≤ exp(ε)Pr[Uσ∈E]+δ.

Proof. For equal-covariance Gaussians with mean difference of standardized norm d, the log density ratio under the first distribution is normal with mean d²/2 and variance d². Integrating over the Neyman–Pearson rejection set where the log ratio exceeds ε gives precisely the displayed hockey-stick divergence. The reverse divergence is identical by symmetry. The expression is nondecreasing in d, so the upper bound d≤q suffices. Mixtures preserve both inequalities when the bound holds conditionally with the same q.

At ε=0, this yields the exact total-variation bound

TV ≤ 2Φ(q/2)−1 ≤ q/√(2π).

Combining Theorems 1 and 2 gives the explicitly computable certificate

q ≤ ρ||s||²/(2λσ).

This is an output-distribution certificate, not merely closeness of losses or predictions. It also has costs: the expected squared norm of output noise is dσ². Report utility at the preregistered σ and the resulting certificate jointly. A small norm approximation without added noise does not imply distributional unlearning relative to deterministic retraining: two different point masses have TV distance one.

The same construction works with a fixed public positive-definite covariance Σ, using ||Σ^{-1/2}(w̃−w*)||. A covariance estimated from the old private model cannot simply be reused while comparing against a retained-data oracle that estimates a different covariance.

### Why we should not casually invoke objective perturbation

Objective perturbation can be effective, but a bounded gradient residual alone is insufficient to justify a general approximate-solver distribution certificate under noisy objectives. For example, with F(w)=w²/2 and perturbation bw, the exact optimizer is −b and is continuous when b is Gaussian. Round −b to a grid of spacing a. The perturbed gradient residual has magnitude at most a/2, but the approximate output is discrete. It cannot be close to the exact continuous optimizer in TV, no matter how small a is. A valid objective-perturbation theorem must control the actual algorithm-induced probability transformation and its regularity; do not assume residual boundedness alone proves it.

The proposed simple Gaussian-output theorem avoids this gap by independently smoothing the final approximate optimizer. It is not claimed as a new contribution.

## 5. The task-relevant size of curation change is not |A|+|R|

Let P_S and P_{S'} be uniform empirical distributions over the selected labeled records. Define g_w(z)=∇ℓ(w;z). At an exact old optimum,

∇F_{S'}(w₀)=E_{P_{S'}}g_{w₀}(z)−E_{P_S}g_{w₀}(z).

For any coupling π of P_S and P_{S'},

||∇F_{S'}(w₀)|| ≤ E_π ||g_{w₀}(z')−g_{w₀}(z)||.

Taking the infimum yields

||∇F_{S'}(w₀)|| ≤ W₁(P_S,P_{S'};d_{w₀}),

where d_{w₀}(z,z')=||g_{w₀}(z)−g_{w₀}(z')|| is a pseudometric. This automatically handles changing sample counts through the probability masses. It is often more informative than treating every changed record as an unrelated unit contribution.

If gradients are L_z-Lipschitz under a task-relevant record metric d_z, then

||∇F_{S'}(w₀)|| ≤ L_z W₁(P_S,P_{S'};d_z),

and Theorem 1 gives quadratic one-step error in this task transport distance. An exact optimal-transport computation is not needed for certification: any explicit feasible coupling gives a valid upper bound. Computing the actual signed gradient from statistics can be still tighter because vector cancellation is then retained.

For equal sizes n'=n and a bijection π:R→A,

∇F_{S'}(w₀)=(1/n)Σ_{r∈R}[g_{w₀}(π(r))−g_{w₀}(r)],

so semantic replacement can be much easier than the support size suggests. For unequal sizes, pairing only A and R misses the renormalization of the common records. Use the probability-distribution coupling or the exact signed regularized-gradient formula.

### Why semantic similarity alone is insufficient

For logistic regression, put a=yu. The gradient is −aσ(−wᵀa). On ||u||≤R it is (1+R||w||/4)-Lipschitz in the signed feature a. Therefore

||g_w(u,y)−g_w(v,y')|| ≤ (1+R||w||/4)||yu−y'v||.

For identical labels this provides a semantic-feature distance bound. For opposite labels on the very same feature u, the gradient difference is exactly −u, with norm ||u||. Near-duplicate text can therefore create substantial task changes when labels differ. This is a direct reason to include contradiction/negation/entity-sensitive NLP examples and to measure label consistency inside duplicate groups.

Also, a connected threshold-similarity component can contain arbitrarily long chains. Small neighbor distances do not bound the diameter or imply that newly revived representatives are close to the old representative. Component splitting and task-gradient transport must be analyzed separately.

## 6. Exact ridge endpoint and the normalization trap

For ℓ_i(w)=(1/2)(u_iᵀw−y_i)², store

M_S=Σ_{i∈S}u_i u_iᵀ,    h_S=Σ_{i∈S}u_i y_i,    n_S=|S|.

Then

w_S=(M_S+λn_S I)^{-1}h_S.

Obtain M_{S'},h_{S'},n' by signed additions/removals, or by the curation-specific polynomial-statistic query described below. This exactly reproduces the retained-data learner in real arithmetic; no smoothing is needed for exact equality. The one-step Newton correction is also exact because the target objective is quadratic.

When n changes, the ridge matrix changes by λ(n'−n)I in addition to the low-rank data modifications. A naive rank-|A|+|R| Sherman–Morrison update that omits this shift targets the wrong regularization strength. One may use a cached eigendecomposition of the old unregularized M to handle the scalar shift, followed by low-rank signed updates. The oracle convention must match the implementation exactly.

This endpoint is valuable as an exact controlled experiment, but ridge algebra itself is not a publication novelty. The curation-specific query state, its storage tradeoff, and its deletion-horizon behavior are the potentially distinctive contributions.

## 7. Curation polynomial statistics: algebra and scope

This section reviews the root agent's proposed representation. It applies to a fixed total order and fixed pairwise graph, where a record v is selected iff none of its earlier RAW neighbors remain. Let B(v) be its earlier-neighbor set, and let x_u indicate deletion. This is not the greedy independent-set rule comparing v only with earlier SELECTED records.

The selection indicator is

s_v(x)=(1−x_v)∏_{u∈B(v)}x_u.

For any intrinsic additive statistic t(v),

T(x)=Σ_v t(v)s_v(x)=Σ_J α_J∏_{u∈J}x_u,

where

α_J=Σ_{v:B(v)=J}t(v)−Σ_{v:B(v)∪{v}=J}t(v).

For a deletion set F, x_u=1 iff u∈F, hence

T(F)=Σ_{J⊆F}α_J.

With a maximum total deletion budget k, retaining only |J|≤k is exact for every |F|≤k. Each record contributes at most two vector terms before grouping, so a sparse implementation has at most 2|D| contributed keys before zero cancellation. A model-only query uses at most 2^{|F|} coefficient lookups, independent of the number of revived records. The vector-addition cost must still include the statistic dimension.

For ridge, use t(v)=(u_vu_vᵀ,u_vy_v,1). For one-shot logistic correction, use t(v)=(∇ℓ_v(w₀),∇²ℓ_v(w₀),1). The latter provides exact target derivatives at w₀ without reading revived payloads but is not a canonical retained-data state if w₀ was fitted to D.

### Exact budgeted state repair for intrinsic statistics

Let the stored canonical coefficient map have horizon k. After deleting F with r=|F|, substitute x_F=1 and collect coefficients:

β_K=Σ_{J:J\\F=K}α_J.

Then discard all |K|>k−r. The resulting map equals the canonical map constructed afresh on D\\F with horizon k−r.

Proof. In the full polynomial, substitution removes every deleted record term because its factor 1−x_v becomes zero; every surviving record has blocker set B(v)\\F. An omitted original monomial of degree at least k+1 retains degree at least k+1−r after substitution, so it cannot contribute at degree at most k−r. Thus truncation loses no coefficient relevant to the remaining budget. Collecting terms produces exactly the fresh retained-corpus polynomial.

Therefore a state consisting of the canonical intrinsic-statistic map, the current ridge solution, and a public remaining horizon admits exact sequential state repair in real arithmetic. This does NOT reconstruct raw documents. The target is the budgeted learner A_{k−r}(D\\F), not A_k(D\\F) with a newly reset budget. The ridge weights are horizon-independent and match ordinary retained-data retraining.

Updating the complete canonical map may require scanning and combining its stored keys; it is not automatically a 2^r operation. Report cold model-only query cost separately from canonical state-update cost. Canonicalization must erase removed identifiers from keys and remove zero entries if fresh construction does. If a corpus-dependent embedding, label, order, or graph is used, intrinsic t(v) or B(v) can change and the proof no longer applies.

For finite-precision software, mathematical equality does not automatically imply bitwise equality to a fresh reduction with a different summation order. Exact rational/fixed-point statistics or an explicitly bounded numerical-error contract resolve this implementation distinction.

### Storage tradeoff is real, but not uniformly favorable

A dense ridge statistic has Θ(d²) entries, compared with Θ(d) for a labeled embedding payload. Moment grouping saves memory only for sufficiently large groups or suitable low-rank representations. For k=1, many records with blocker set {u} can be aggregated into a revival bucket. For k=2, their individual negative coefficients α_{u,v}=−t(v) may need to be stored, eliminating the first-order compression.

A useful budget-separation example is a star: hub u is initially selected and each leaf v has B(v)={u}. One deletion only needs the sum of all leaf statistics. Two deletions can request {u,v} and require the sum minus the particular leaf. For scalar unit-feature ridge with binary leaf labels, the family of two-deletion queries can recover all individual labels from the total and leave-one-out sums. Thus the state can require Ω(number of leaves) bits for horizon two even though horizon one needs only a count and total. This is a potential lower-bound illustration, subject to the exact chosen computation/storage model.

An especially clean generalization separates horizons k and k+1. Give m leaves a shared blocker set B of size k and no leaf–leaf edges; keep blocker data fixed. Give leaves task feature 1 and balanced binary labels y_v∈{−1,+1}, so Σ_v y_v=0. At budget at most k, either some blocker remains and every leaf is excluded, or F=B and every leaf revives with the same aggregate statistics (M,h,n)=(m,0,m). No possible answer depends on the individual balanced labeling. At budget k+1, the query F=B∪{v} gives

w(F)=−y_v/[(m−1)(1+λ)],

which reveals y_v. Any deterministic finite-bit summary supporting all such exact queries must distinguish the binomial(m,m/2) balanced label assignments, requiring at least log₂ binomial(m,m/2)=m−O(log m) bits. The same lower bound holds for deterministic approximate outputs with absolute error strictly below 1/[(m−1)(1+λ)], because their signs recover y_v. A standard one-way INDEX argument extends the conclusion to constant-error randomized query schemes, with an explicit success-probability convention.

This family has only O(log m) label-dependent state at horizon k, but Ω(m) state at horizon k+1. It is model-repair hardness, not raw-document reconstruction hardness. It is also an accuracy-sensitive result: the revealed model differences scale as 1/m, so fixed coarse error can erase the separation.

The curation graph is geometrically realizable: choose identical unit blocker embeddings e₀; leaf embeddings a e₀+√(1−a²)e_v; and a cosine threshold τ with a²<τ<a. Blockers form a clique and connect to every leaf, while leaves have no mutual edges. The classifier feature can be a separate fixed representation or a public scalar projection. If the paper requires curation and task features to be identical full vectors, this particular constant-statistic construction does not establish the same separation and must not be claimed without another construction.

## 8. Sequential approximate learning: what is and is not free

Two practical strategies are legitimate.

**Fixed reference strategy.** Keep w₀ and target derivatives evaluated at w₀. Maintain current additive statistics exactly as raw deletions arrive. For every request, compute a new Newton correction from the same w₀. Theorem 1 remains valid for the current target regardless of the path. The certificate may deteriorate as the current optimum moves away. This is model-output certification only when w₀ depends on deleted data.

**Moving reference strategy with error accounting.** Maintain a current candidate w, an approximate gradient ĝ with error bound a, and an approximate Hessian B with error bound η for the current selected objective. A signed curation update n→n' at the same w gives

ĝ_new=(n/n')ĝ+[Σ_A h_i(w)−Σ_R h_i(w)]/n',

B_new=(n/n')B+[Σ_A J_i(w)−Σ_R J_i(w)]/n',

a_new=(n/n')a,    η_new=(n/n')η.

Choose s=−B_new^{-1}ĝ_new when B_new is positive definite. Then the new candidate has residual bound

a_next=a_new+η_new||s||+(ρ_new/2)||s||².

One may store the next gradient estimate as zero with error a_next, and retain B_new as a Hessian estimate at w+s with error

η_next=η_new+ρ_new||s||.

These recurrences explicitly account for approximation accumulation. Refresh derivatives or retrain when the resulting output certificate exceeds a preset budget. Signed update evaluations at a moving reference require changed-record payloads or a richer functional summary; the one-anchor polynomial derivative cache cannot provide them automatically.

For adaptively chosen deletion requests, a per-release guarantee requires the bound to hold conditional on every possible prior released transcript and request policy. Fresh output noise with a fixed public σ gives such a conditional kernel comparison if the deterministic bounds hold uniformly. Standard sequential composition then controls the transcript. One-shot ε,δ must not be reused as a transcript-level claim without composition. This still does not erase already disclosed pre-deletion models.

## 9. Why the ACL contribution must live above routine influence functions

The convincing paper should connect three levels:

1. Structural counterfactual selection: which retained records revive, under which actual curation rules, and which cases cannot be inferred from the selected corpus alone.
2. Query representation and storage: exact or approximate sufficient state for the selected learner, with an explicit deletion horizon and costs; distinguish raw corpus recovery from model recovery.
3. Task effect and certification: signed gradient/curvature repair, label-sensitive transport, exact ridge or a correct smoothed convex learner, and a demonstrated NLP phenomenon that support churn alone fails to explain.

Potential central claim: a large curation change need not require enumerating every revived record to repair a learner, but avoiding enumeration requires a deletion-query summary whose information demand can rise sharply with the deletion horizon. Task-relevant cancellation determines whether approximate convex repair remains local.

Claims to avoid:

- That deletion of an untrained document is intrinsically harmless.
- That every semantic deduplicator behaves like connected components or the earlier-raw-neighbor rule.
- That a gradient residual is already an unlearning certificate for deterministic models.
- That near-duplicate text implies near-duplicate task gradients without label assumptions.
- That the current-model guarantee covers hidden derivative caches or previous model releases.
- That a model-only 2^k query bound also bounds full state repair.
- That the frozen-feature theorem certifies end-to-end LLM finetuning or a feature encoder trained on the raw corpus.

## 10. Verified primary reference relevant to this module

Guo, Goldstein, Hannun, and van der Maaten (ICML 2020), *Certified Data Removal from Machine Learning Models*: https://proceedings.mlr.press/v119/guo20c.html and https://proceedings.mlr.press/v119/guo20c/guo20c.pdf. The official abstract and paper establish certified removal for linear classifiers; the paper explicitly uses a Newton correction and loss perturbation. This makes a generic Newton deletion update an unsuitable novelty claim. The original repository also describes accumulating residual bounds over multiple removals: https://github.com/facebookresearch/certified-removal.

Search references available to the root agent after opening: turn14search0 (official proceedings page), turn14search13 (official PDF), turn14search2 (official repository). The output-smoothing proof above is derived directly and should be attributed to the standard equal-covariance Gaussian likelihood-ratio calculation, not presented as a new discovery.
