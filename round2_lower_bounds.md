# Second-pass lower bounds: a conclusive randomized, constant-accuracy theorem

This note audits and strengthens Sections 7–8 of the first theory draft. The principal improvement is a new activation-frontier construction: the model-storage lower bound now holds for **bounded binary labels, constant parameter error, randomized preprocessing and repair, per-query success, complete forgotten-record payloads, and cosine curation in O(log m) dimensions**. It also holds when the downstream learner uses exactly the same full embeddings as the curator. The previous aggregate-to-individual construction remains useful, but its 1/m accuracy scale is intrinsic under bounded labels.

## 1. Explicit information/access model

Fix a public identifier universe, deterministic ordered threshold graph, all curation embeddings, and all zero-label payloads. The only varying data are m labels Y=(Y_1,...,Y_m). A preprocessing algorithm sees the full dataset and stores a possibly randomized message M in a state space of at most 2^B possibilities (a fixed-length B-bit encoding). Count **all** label-dependent retained information in M: model parameters, auxiliary state, caches, private random coins if retained, and remotely accessible label-dependent storage. A variable-length representation must count its self-delimiting length or permit the corresponding additive overhead; an uncounted length is a side channel. Arbitrarily many exact real numbers are not a finite-bit model. Public randomness R is allowed but must be independent of Y. The repair algorithm sees M, R, the query identifiers, and the full original payloads of all requested forgotten records. It cannot reread retained hidden labels from an uncounted service. Repair can use fresh private randomness.

Correctness is only a per-query requirement: for every fixed Y and every permitted F, the probability of parameter error at most eta is at least 1-delta, over preprocessing and repair randomness. A guarantee under a uniformly random independent label vector and uniformly random query index is already enough for the lower bound. No simultaneous-correctness event or union bound over m queries is needed.

Public graph and feature metadata do not count toward B; consequently B is a lower bound on total state but the achievability comparison counts only hidden-label-dependent storage. Graph metadata are not free in an implementation.

## 2. New frontier theorem: one additional deletion changes zero label memory to linear label memory

Let b>=1 and m>=1. The ordered vertex set consists of an anchor a, b common blockers S={s_1,...,s_b}, m private blockers p_1,...,p_m, and m hidden candidates w_1,...,w_m. The order places a first, every common/private blocker next, and every candidate last. Give a, S, and all p_i label zero and each w_i label Y_i in {0,1}. Arrange the threshold graph so that a is adjacent to every common/private blocker but to no candidate, and

    B(w_i) = S union {p_i}.

Edges within S are harmless; the geometric realization below has S as a clique. Initially the curator selects exactly a, so no hidden candidate and no requested blocker trained the original model.

Use the scalar constant-feature learner z_v=1 with average-loss ridge objective

    L_T(theta) = (1/(2|T|)) sum_{v in T}(theta-y_v)^2 + (lambda/2)theta^2,

where lambda>0 and an empty corpus returns zero.

**Theorem F (randomized constant-accuracy frontier).** For this family:

1. Every deletion request of size at most b produces model zero. Hence zero hidden-label-dependent bits suffice for exact repair at cumulative budget b, including exact remaining-horizon state.
2. Let the budget be b+1. If randomized repair satisfies per-query absolute parameter error eta with probability at least 1-delta, where

       eta < 1/[4(1+lambda)],     0<=delta<1/2,

   then

       B >= m [1-h_2(delta)] bits.

   The forgotten records' complete payloads may be supplied. The result concerns a single requested model, not reconstruction of raw records.

**Proof.** A hidden w_i has b+1 blockers. At budget b none can activate; all other labels are zero. At budget b+1 query

    F_i = S union {p_i}.

The retained selected set is exactly {a,w_i}: every other candidate retains its private blocker, and every retained private blocker remains suppressed by a. Thus

    theta_i(Y) = Y_i/[2(1+lambda)].

Every member of F_i has a fixed zero label and a fixed feature/identifier payload independent of Y. The request supplies no hidden information. Threshold an accurate repaired scalar at 1/[4(1+lambda)] to decode Y_i. This gives an index decoder with bit-error probability at most delta.

Take independent uniform bits Y_i. Let R be the public randomness. The optimal decoder of Y_i from (M,R) and its fixed index does at least as well as the repair decoder; fresh decoder coins cannot improve over a Bayes-optimal deterministic decision. Binary Fano gives H(Y_i|M,R)<=h_2(delta). Independence and conditional subadditivity give

    B >= H(M|R) >= I(Y;M|R)
      = m-H(Y|M,R)
      >= m-sum_i H(Y_i|M,R)
      >= m[1-h_2(delta)].

If only the mean bit error over a uniform index is bounded, apply concavity of h_2 to the last sum. This proves the claim without reconstructing all labels simultaneously. For the low-budget canonical state, after r deletions each surviving hidden candidate has at least b+1-r blockers, exceeding the remaining horizon b-r, so hidden labels can still be discarded. QED.

**Exact and multilevel form.** For labels independently in a public alphabet of L distinct values, exact repair requires m log_2 L bits. If labels are the bounded grid {0,1/(L-1),...,1}, a parameter error

    eta < 1/[4(1+lambda)(L-1)]

permits nearest-level decoding. With per-query failure probability delta, Fano gives

    B >= m[log_2 L - h_2(delta) - delta log_2(L-1)].

The approximation accuracy is independent of m, but necessarily shrinks as the alphabet becomes denser.

**Interpretation.** The transition is zero-to-individual information at an activation frontier, not the earlier aggregate-to-individual transition among candidates already reachable at budget b. Both mechanisms matter and must not be conflated. For fixed b, the total number of records is N=1+b+2m and the lower bound is Omega(N) bits. For growing b the precise bound is Omega(N-b), not automatically Omega(N).

## 3. Fixed-threshold cosine realization with near-optimal dimension

Take unit signatures u_1,...,u_m in R^d with |<u_i,u_j>|<=1/6 for distinct i,j. Let e_0,e_1 be orthogonal to the signature subspace. Define unit curation embeddings

    a   = e_0,
    s_j = (1/2)e_0 + (sqrt(3)/2)e_1       (all common blockers identical),
    p_i = (1/2)e_0 + (sqrt(3)/2)u_i,
    w_i = (1/2)e_1 + (sqrt(3)/2)u_i.

Use strict cosine adjacency at threshold tau=2/5. The relevant inner products are:

| Pair | Inner product | Consequence |
|---|---:|---|
| a,s_j and a,p_i | 1/2 | Anchor suppresses every blocker |
| a,w_i | 0 | Anchor does not suppress candidates |
| s_j,w_i | sqrt(3)/4 > 2/5 | Every common blocker suppresses every candidate |
| p_i,w_i | 3/4 | The private blocker suppresses its candidate |
| p_i,w_j, i!=j | at most 1/8 | No wrong private blocker |
| p_i,p_j and w_i,w_j, i!=j | at most 3/8 | No private/private or candidate/candidate edges |
| s_j,p_i | 1/4 | No common/private edges |
| s_j,s_l | 1 | Harmless common-blocker clique |

Consequently B(w_i)=S union {p_i}, as required.

Existence in d=O(log m) is elementary. Draw each u_i independently from {+/-1/sqrt(d)}^d. For a fixed pair, Hoeffding yields

    Pr(|<u_i,u_j>|>1/6) <= 2 exp(-d/72).

A union bound gives failure probability at most m(m-1) exp(-d/72). For m>=2 it is therefore sufficient to take

    d >= ceil(72 log(2m(m-1))).

For m=1 one signature coordinate suffices. The realization is a valid fixed-threshold cosine instance. Its dimension depends on m; calling it fixed dimensional would be false.

**Matching dimension obstruction for this architecture.** For b>=1 all w_i share a common neighbor s and are pairwise nonneighbors at one cosine threshold tau<1. Set r=sqrt(2(1-tau)). The w_i lie within radius r of s and are pairwise at distance at least r. Disjoint balls of radius r/2 around w_i fit within the radius-3r/2 ball around s. Comparing d-dimensional volumes gives m<=3^d. Hence d>=log_3 m. The O(log m) dimension order is optimal for this shared-neighbor, pairwise-independent-candidate construction, not a universal lower bound for every possible curation reduction.

The construction uses fixed features and independent labels, a legitimate worst-case supervised-learning family. It does not establish natural-text frequency or that pretrained semantic encoders realize this family on actual language.

## 4. The same embeddings can be used downstream

The scalar theorem can use an intercept feature, standard in linear models. Two further versions eliminate any concern that curation and training rely on incompatible feature views.

**Full-embedding ridge.** Use the very same unit vectors a,p_i,s_j,w_i as downstream ridge features, with a vector parameter theta. At F_i the selected set is {a,w_i}; a is orthogonal to w_i. Therefore the average-loss ridge optimum is exactly

    theta_i(Y) = Y_i w_i/(1+2lambda).

The proof follows by solving

    (aa^T+w_iw_i^T+2lambda I)theta = Y_i w_i.

The two possible target vectors are separated by 1/(1+2lambda), so Euclidean error

    eta < 1/[2(1+2lambda)]

allows bit decoding. The same B>=m[1-h_2(delta)] bound follows. Thus unit-norm shared representations and bounded binary labels still give a constant-accuracy randomized storage obstruction. The model dimension is O(log m), whereas the required state is Omega(m) bits.

**One-parameter head on the same vectors.** If retaining the one-parameter result while using one common feature map is important, append a common coordinate:

    e'_v = alpha e_* + sqrt(1-alpha^2)e_v,  0<alpha<1.

These are unit vectors. At threshold tau'=alpha^2+(1-alpha^2)(2/5), they induce exactly the old graph. Fix the scalar feature z_v=<e_*,e'_v>=alpha, a public linear projection of the same curation embedding. Then

    theta_i(Y) = alpha Y_i/[2(alpha^2+lambda)].

The scalar lower bound holds for eta<alpha/[4(alpha^2+lambda)]. With alpha=1/sqrt(2), tau'=7/10. This is mathematically legitimate but the intercept-only variant is easier to explain, while full-embedding ridge is more natural experimentally.

## 5. Retain the old aggregate-to-individual theorem, and close its approximation gap

The original strengthened construction includes anchor a, common blockers S (|S|=b), private blockers p_i, and two copies v_i,w_i of each label Y_i, with B(v_i)=S and B(w_i)=S union {p_i}. It has the exact budget-b aggregate summary Y_sum=sum_i Y_i, and exact budget-(b+1) model queries recover all labels. That exact result is sound.

For bounded binary labels, its 1/m accuracy dependence is **inherent**, not merely a weak proof.

**Proposition A (randomized aggregate lower bound).** In that construction, assume full forgotten payloads, scalar constant-feature average-loss ridge, and per-query error eta<1/[2(m+2)(1+lambda)] with failure probability delta<1/2. Then

    B >= m[1-h_2(delta)] - log_2(m+1).

**Proof.** Give the decoder the exact sum Y_sum as additional side information, costing at most log_2(m+1) bits. For query F_i=S union {p_i}, the target model is

    theta_i=(Y_sum+Y_i)/[(m+2)(1+lambda)].

Multiplying an accurate answer by (m+2)(1+lambda), subtracting the supplied sum, and thresholding at 1/2 decodes Y_i. Apply the entropy proof to augmented state (M,Y_sum), then subtract the sum's entropy. There is one randomized query per recovered coordinate; no union bound over queries is needed. QED.

**Proposition B (matching small-memory upper bound above that accuracy).** Store only the exact sum Y_sum and public graph metadata. For every size-at-most-(b+1) request, full forgotten payloads allow exact recovery except when the request is S union {p_i}. In that case output

    theta_hat_i=(Y_sum+1/2)/[(m+2)(1+lambda)].

The worst-case error is exactly 1/[2(m+2)(1+lambda)]. Thus O(log m) hidden-label-dependent bits suffice for deterministic uniformly accurate repair at that tolerance.

**Proof of the exhaustive request classification.** If a common blocker remains, no v_i or w_i enters and the model is zero. If all common blockers are removed, at most one additional record is removed. With no extra deletion, all v_i plus a are selected. If the extra record is p_i, the selected set additionally contains w_i, producing the sole unknown contribution. If it is v_i, the supplied forgotten payload gives Y_i and the sum is updated exactly. If it is w_i, that w_i would otherwise remain blocked by p_i and has no effect. If it is a, all v_i and the surviving public-zero p_i become selected; the known denominator changes and the label sum remains Y_sum. This last case has 2m selected records, not m: deleting the anchor admits the p_i, which are public-zero and must be included in the normalized denominator. Every case uses known counts and the stored sum. QED.

This supplies an essentially sharp accuracy threshold for the old construction under full forgotten payload access. Do not assert that it yields a constant-accuracy linear-space obstruction as m grows. Use Theorem F for that stronger claim.

## 6. What can be finalized, and what remains a novelty question

The mathematical result can be finalized under the explicit no-reaccess, finite-bit, fixed-curation contract. Recommended main theorem:

> Even with bounded labels and a shared unit-norm cosine representation, increasing the cumulative deletion budget from b to b+1 can change the required label-dependent storage from zero to Omega(N) bits for constant-accuracy randomized repair of a ridge learner. The hard requests delete only originally excluded records and may reveal their full payloads. The target model has one parameter with a fixed projection, or O(log N) parameters when using the full shared embedding.

Keep the exact aggregate-to-individual result as a second mechanism, now with its matching accuracy caveat. The lower bound is a reduction to classical random-access encoding/one-way INDEX, not a new information-theoretic inequality. The novelty question is the curation construction, its transition, and its integration with the exact-summary upper bounds. Existing learning-unlearning memory lower bounds remain essential comparison literature.

Primary source for the standard random-access entropy bound: Ashwin Nayak, *Optimal lower bounds for quantum automata and random access codes* (FOCS 1999), arXiv:quant-ph/9904093, https://arxiv.org/abs/quant-ph/9904093. The classical bound used here has the self-contained elementary entropy proof above; do not describe the entropy inequality as novel. Search result verified the primary paper and its (1-H(p))n statement (retrieval reference turn36academia12; parent must open before citation).
