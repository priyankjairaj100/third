# Counterfactual semantic curation: graph theory notes

These are derived results, not literature-validated novelty claims. All graphs and priorities below are fixed before the deletion request. The results do **not** silently cover refitting embeddings, k-means, centroid-based ordering, ANN indexes, or random initializations. The literature agent verified that the released SemDeDup pairwise suppression code checks all earlier rows, including rows it subsequently removes. Consequently, Section 2 is relevant to a fixed-state version of that released rule. Sections 3 and 4 concern different deduplicators.

## 1. Setup and why selectors must be distinguished

Let G=(V,E) be a simple undirected similarity graph. Each vertex is a raw record with immutable unique priority; write u<v when u is earlier. A deletion request F is a subset of V, and the exact curation target runs the same selector on the induced graph G[V\F], with the restricted original priority. Documents outside the selected corpus can still participate in curation.

Three superficially similar selectors differ sharply:

1. Earlier-neighbor suppression (ENS): select v iff it has no earlier neighbor in the raw retained graph.
2. Component minimum (CC): select the earliest vertex in each connected component.
3. Greedy maximal independent set (GMIS): process vertices in priority order, selecting v iff it has no earlier **selected** neighbor.

ENS and CC preserve every surviving originally selected vertex. GMIS need not. ENS can reactivate records when an originally unselected blocker is deleted. GMIS cannot change at all when the deletion request contains only originally unselected records. A connected-component bridge example is not an example of SemDeDup unless its actual selector is separately changed.

## 2. Earlier-neighbor suppression: exact local algebra

Define B(v)={u in V: u<v and {u,v} in E}. Then

    C_ENS(V\F) = {v in V\F : B(v) subseteq F}.                 (1)

Proof: earlier retained neighbors of v are exactly B(v)\F; v survives suppression iff that set is empty. This also proves that all surviving originally selected records remain selected.

The exact newly selected set is

    A(F)={v notin F : B(v) nonempty and B(v) subseteq F}.

The exact selected removals are C_ENS(V) intersect F. No previously selected surviving document is removed. All additions have at least one neighbor in F, so curation repair is one hop in this fixed graph. There is no recursive ENS cascade: a newly retained document was already a blocker before it was retained.

### 2.1 An untrained record can change the model

On the priority-ordered path r<f<v with edges r--f and f--v, ENS initially retains only r. Removing f retains r and v. Thus deletion of f, which never trained the downstream model, changes its training set. This graph is realizable with three unit vectors at angles 0, alpha, 2alpha and a cosine threshold strictly between cos(2alpha) and cos(alpha).

If original selected record r has label 0 and the new record v has label 1, any nondegenerate ridge example yields a changed model. For bias-only sum-squared-loss ridge,

    L(theta;S)=1/2 sum_{i in S}(theta-y_i)^2 + lambda/2 theta^2,

the initial model is zero and the repaired model is 1/(2+lambda). The semantic features defining the graph and the downstream bias-only feature need not be the same representation. Alternatively, using the actual unit vectors as downstream features gives new ridge solution (lambda I+x_r x_r^T+x_v x_v^T)^(-1)x_v, which is nonzero.

### 2.2 Exact deletion-budget candidate buffer

For cumulative deletion budget k, define R_k={v:|B(v)|<=k}. A record can appear in the selected corpus after some request of at most k deletions iff it is in R_k.

Necessity follows from B(v) subseteq F. Sufficiency uses F=B(v), which never includes v. This is an exact characterization, not just an upper bound.

For exact corpus recovery, store each record payload in R_k and its blocker IDs B(v). Payloads outside R_k are unnecessary under this fixed-state, cumulative-budget contract. Blocker metadata for the stored candidates has at most k|R_k| entries. A reverse index from blocker IDs to candidates and a live-blocker count implement exact sequential updates. A candidate becomes selected when its live-blocker count reaches zero, provided it is not itself deleted. Complexity is proportional to reverse-index incidences touched plus records actually output; state changes must use the cumulative F relative to the original graph, not reset k after every request.

This is optimal for exact corpus payload recovery under an opaque-payload access model: assume the graph and IDs are public and each candidate payload contains q independent bits unavailable through the graph or requests. For every candidate v, request B(v) outputs its payload. Any exact summary that answers all requests can therefore be cloned to recover every R_k payload, so its payload-dependent storage is at least q|R_k| bits. This statement is **not** a lower bound for reproducing the downstream model, whose sufficient statistics may compress many records together. It also needs care if deletion requests reveal payloads correlated with the candidate payload.

### 2.3 Group/source deletion

Suppose each document has exactly one source sigma(v), and a request removes all documents from at most k sources. The exact potentially selectable documents are those satisfying

    sigma(v) notin {sigma(u):u in B(v)}
    and |{sigma(u):u in B(v)}|<=k.

The first condition matters: if a blocker comes from the candidate's own source, deleting it would also delete the candidate. The second condition counts sources, not documents. The proof is the same blocker containment argument. More general provenance memberships require a separately defined permitted deletion family.

### 2.4 Random deletion null model

If each raw document is independently deleted with probability p, the exact expected number of additions is

    E|A(F)| = sum_{v:|B(v)|>0} (1-p) p^{|B(v)|}.             (2)

The retention factor 1-p must not be omitted when candidates can themselves be deleted. For a uniformly random request of exactly k raw records from n,

    E|A(F)| = sum_{v:|B(v)|>0}
                 binom(n-|B(v)|-1,k-|B(v)|)/binom(n,k),     (3)

with invalid binomial arguments interpreted as zero. Therefore a blocker-count histogram predicts random-request resurrection without retraining. Dependencies and shared blockers affect variance and adversarial requests.

If only a separate set of source vertices is eligible for deletion and candidates themselves cannot be deleted, weighted activation is sum_v w_v 1[B(v) subseteq F]. For nonnegative w_v this is a supermodular set function: completing a blocker set has increasing returns. Under independent source deletion, covariance of activation indicators is p^{|B(v) union B(w)|}-p^{|B(v)|+|B(w)|}. Do not use this covariance formula when candidates themselves are deletion-eligible without accounting for their retention events and potential mutual blocking.

### 2.5 Worst-case resurrection is computationally hard

Given a graph H=(U,E_H), create priority-earlier vertices U and, for each edge e={a,b}, a later vertex v_e whose only neighbors are a and b. There are no U--U edges or candidate--candidate edges. All U are initially retained; every v_e has exactly two blockers. Deleting F subseteq U activates exactly |E_H[F]| candidates.

Deciding whether at most k deletions can activate at least binom(k,2) candidates solves CLIQUE. Even if all vertices are eligible for deletion, deleting a candidate cannot help activate another vertex and is weakly dominated by deleting an earlier vertex; the same reduction holds. Thus worst-case activation auditing is NP-hard even for a bipartite similarity graph and blocker degree exactly two. This is a standard reduction to a known combinatorial problem, not by itself evidence of a substantial new result.

These graphs are realizable by cosine-threshold embeddings in polynomial dimension. For each u, let z_u be the normalized vector e_u+epsilon sum_{e incident to u}e_e, and let z_{v_e}=e_e. For sufficiently small epsilon, endpoint cosine similarities exceed epsilon/2, nonendpoint similarities are zero, and all earlier--earlier similarities are below epsilon/2. Candidate--candidate similarities are zero. The reduction does not establish hardness in a fixed low embedding dimension.

## 3. Connected-component representatives

For each old component K, let r_K be its minimum vertex and t_K the number of components of G[K\F], with t_K=0 if K\F is empty. Every surviving old representative remains selected: its residual component is a subset of its original component and contains no smaller vertex.

Consequently,

    removals = C_CC(V) intersect F,
    additions in K = t_K - 1[r_K notin F],
    total additions = cc(G-F)-cc(G)+|C_CC(V) intersect F|.   (4)

For an unselected single deleted vertex f in a nonsingleton component,

    additions = cc(K-f)-1 <= deg_K(f)-1.

The bound is attained by a star whose center f is not the earliest vertex. Deleting f produces n-2 additions among n-1 remaining leaves. By contrast, deleting a degree-two articulation point in a long path produces just one addition, but that new representative can be arbitrarily far away from the deleted vertex. A small cardinality bound is not a spatial locality bound.

For a nonempty proper deleted subset F_K of a connected component K, let h=cc(G[F_K]) and let e_cross count edges between F_K and K\F_K. Contract every retained component and every removed component. The resulting bipartite multigraph is connected, so it has at least t_K+h-1 edges. Thus

    t_K <= e_cross-h+1.                                   (5)

This is a graph-fragmentation bound, not a learning theorem.

### 3.1 Exact potential-representative characterization by cuts

For each v, let P_v={u:u<v}. Let rho(v) be the minimum size of a vertex set F excluding v such that G-F contains no path from v to any vertex in P_v\F. Earlier endpoints are allowed to be deleted. Equivalently, this is a vertex-capacitated min cut from v to a supersink connected to all earlier vertices, with unit capacities on all vertices except v and infinite terminal-to-sink edges. Set rho(v)=0 if no earlier vertex is connected to v initially.

Then v can become a CC representative after at most k deletions iff rho(v)<=k.

Proof: v is selected precisely when its remaining component contains no earlier vertex, which is precisely the separation condition. Both directions are exact.

Hence the exact corpus payload candidate buffer is {v:rho(v)<=k}. In a clique, rho(v)=the number of earlier vertices, and only the first k+1 representatives need payload retention. In a star with an early leaf and a later center, all leaves may be potential representatives after one deletion. Thus graph redundancy determines whether a small deletion-ready buffer exists. By the vertex version of Menger's theorem, rho(v) is also the maximum number of v-to-earlier-vertex paths that are vertex-disjoint except at v, with distinct earlier endpoints. This is a characterization built from classical cuts, not a novel graph theorem.

## 4. Greedy MIS: no unselected-only effect, but real cascades

Let a_v be the initial selection bit. With absent vertices assigned a'_v=0, retained vertices satisfy

    a'_v = product_{u in B(v)}(1-a'_u).                    (6)

Here a'_u is the final selection bit, unlike ENS, which checks presence of raw neighbors.

### 4.1 No effect from deleting only unselected vertices

If F intersect C_GMIS(V) is empty, then C_GMIS(V\F)=C_GMIS(V). Proof by induction over priorities: every deleted vertex had selection bit zero, and every retained vertex sees the same earlier selected neighbors as before. This holds for simultaneous batches, not only individual deletions.

### 4.2 Causal cone

Orient edges from earlier to later priority. Every retained vertex whose selection changes is reachable in this DAG from an originally selected deleted vertex. To see this, a changed retained bit requires an earlier neighbor with a changed bit. Tracing backward terminates at a deleted vertex whose original bit was one. Changes along such a witness path alternate between 1-to-0 and 0-to-1.

An exact repair can process this forward cone in priority order and only propagate a status change to later neighbors. A priority queue plus maintained earlier-selected-neighbor counts gives work proportional to affected outgoing incidences, up to scheduling overhead. Dynamic MIS literature already covers closely related update problems.

### 4.3 Worst-case nonlocality at degree two

On an ascending-priority path v_1--v_2--...--v_n, the initial GMIS selects the odd vertices. Deleting selected v_1 makes the exact new GMIS select the even vertices. Every remaining vertex changes status. Thus a single deletion can produce Theta(n) additions/removals, despite maximum degree two, and can affect vertices at distance n-1. Any universal repair method confined to a fixed-radius neighborhood fails on this family. This is also a one-dimensional threshold-graph realization.

## 5. Model-level state lower bounds, carefully scoped

### 5.1 Selected-only state is insufficient

Use the ENS path r--f--v. Hold all IDs, priorities, semantic features, graph, selected r payload, forget f payload, and original model fixed. Let two worlds differ only in the unselected v label (0 versus 1). If saved state contains no information about that label, it is identical in the two worlds. The identical request {f} activates v. The two exact ridge retraining targets are distinct. Therefore no function of that saved state and request can exactly repair both worlds.

If the two target models are deterministic point masses and a randomized repair has the same output distribution in the two worlds, its total-variation distance from at least one target is at least 1/2. For norm approximation, if target separation is Delta, no common output can be within a radius strictly less than Delta/2 of both. These are elementary indistinguishability consequences; they do not prove that every discarded raw payload must be stored.

### 5.2 Many independent singleton requests require many bits even for a scalar learner

Take m disjoint ENS paths r_i--f_i--v_i, with r_i<f_i<v_i in each. All r_i and f_i labels are public zero; v_i has independent label y_i in {0,...,2^q-1}. Semantic topology is fixed independently of labels. Original selected corpus contains only the r_i and the original scalar ridge model is zero.

Deleting unselected f_i activates exactly v_i, and the bias-only sum-squared ridge model is

    theta_i = y_i/(m+1+lambda).

Any saved summary enabling exact repair for all these singleton requests has at least mq payload-dependent bits. Otherwise two label vectors share a summary, and some query f_i requires distinct answers. This survives a deletion request that supplies the full forgotten f_i payload because those payloads are fixed and carry no y_i. It shows why a scalar model can still require linear deletion-ready state. It is an indexing-style lower bound, with its learning interpretation made explicit.

### 5.3 A one-extra-deletion state threshold (IDs-only version)

Let S be b earlier blockers and v_1,...,v_m later candidates, all with B(v_i)=S. An earlier anchor r is adjacent to S but not to the candidates, so r is the only originally selected vertex and every member of S is unselected. Candidate labels y_i have q independent bits. Other labels are public zero.

For cumulative budget b, candidates can activate only together, on deletion S. Their sum Y=sum_i y_i suffices for exact scalar ridge repair; this needs at most ceil(log2(m(2^q-1)+1)) bits. At budget b+1, IDs-only requests S union {v_i} require leave-one-out sums, so all individual labels can be recovered from target answers and mq bits are necessary.

**Access-model caveat:** if the forget request supplies y_i when v_i is deleted, the lower bound at b+1 fails: the sum plus the supplied forgotten label suffices. Do not state this version without the IDs-only assumption.

### 5.4 Robust state threshold with full forget-payload access

The following construction removes that caveat. For each i, create two hidden candidates v_i and w_i carrying the same independent label y_i. Let

    B(v_i)=S,
    B(w_i)=S union {p_i},

where p_i is a private public-zero-label blocker. Let an earliest public-zero anchor r neighbor every member of S and every p_i; no other edges enter r, S, or p_i, and the priorities are r, then S/p_i, then candidates. All S and p_i are originally unselected. The graph and the label-copy relation are public. The only unknown information is y_1,...,y_m.

At cumulative budget b=|S|, no w_i can activate. The v_i can activate only together on request S. Their sum Y is sufficient for all exact counterfactual scalar ridge models, needing only ceil(log2(m(2^q-1)+1)) payload-dependent bits.

At budget b+1, deleting S union {p_i} activates all v_j and precisely w_i. All forgotten payloads are public zeros. With sum-squared ridge,

    theta(S)=(Y)/(m+1+lambda),
    theta(S union {p_i})=(Y+y_i)/(m+2+lambda).

Therefore each y_i can be decoded from these target answers. Every exact summary supporting all requests needs mq bits, even when requests include full forgotten payloads. This gives a sharp sum-compression-to-individual-information threshold as deletion budget increases by one. The initial saved model is identical (zero) for all label vectors. The targets are full counterfactual retraining of this simple learner; no influence approximation is involved.

Proof of the lower bound is injectivity: if two independent label vectors map to the same stored summary, then either their sums differ (query S distinguishes them), or their sums agree and some y_i differs (query S union {p_i} distinguishes them). The same argument applies to uniform deterministic approximate answers with absolute error eta<1/[2(m+2+lambda)] for integer labels: every pair differs in at least one target coordinate by at least 1/(m+2+lambda). For q-bit values normalized to [0,1], divide that separation by 2^q-1. Randomized or distributional lower bounds need their own coding argument and should not be asserted from this proof.

This theorem is a candidate contribution worth checking against deletion-robust summaries and machine-unlearning storage lower bounds. Its distinct feature is that all problematic deletion requests remove originally untrained records, while candidate activation signatures govern the information required for the counterfactual learner.

## 6. General exact-statistic viewpoint

For a fixed scalar sufficient statistic t(v), enumerate allowed deletion requests F and form a matrix M with rows F and columns discarded records, where M_{F,v}=1[v notin F and B(v) subseteq F]. The hidden contribution to a counterfactual statistic is Mt. In an exact real-arithmetic model, the row rank describes the number of independent scalar linear summaries needed to reconstruct all answers; storing a row-space basis is sufficient. For a bit lower bound, choose r linearly independent columns and let their statistics vary independently over a q-bit alphabet while holding other columns fixed. Their map to all answers is injective, so rq bits are necessary. Numerical representation/conditioning must be accounted for separately before turning this algebraic rank into a realistic byte claim.

This matrix viewpoint explains both extremes: common activation signatures can compress to a sum, whereas independent queries force individual information. It also prevents the false inference that all potentially selected raw documents must always be stored to reproduce a model. The root agent is developing a more useful blocker-polynomial representation and truncated Boolean coefficients for this idea.

### 6.1 Exact rank from a forest on blocker sets

This stronger characterization uses document-level deletion variables and IDs-only requests. Let U be any specified set of records whose scalar sufficient statistics are unknown; original selected-record statistics can be excluded from U if they are already available. Remove columns with |B(v)|>k because they are zero on all allowed requests.

On the Boolean Hamming ball |F|<=k, the selection function for an eligible record is

    f_v(x)=x_{B(v)}-x_{B(v) union {v}}  if |B(v)|<k,
    f_v(x)=x_{B(v)}                    if |B(v)|=k,

where x_S=product_{u in S}x_u. Monomials indexed by sets S of size at most k are linearly independent on this ball: their evaluation matrix at all subsets F of size at most k is the invertible inclusion-zeta matrix. Thus rank(M) equals the rank of the displayed monomial coefficient matrix.

Construct a graph whose nodes are the occurring subsets of size at most k, plus a formal sink:

* For each v in U with |B(v)|<k, put an interior edge B(v) -> B(v) union {v}.
* For each v in U with |B(v)|=k, put a boundary edge B(v) -> sink.

The interior graph is a forest. Indeed, every interior edge increases set cardinality, and its head determines v uniquely as the maximum-priority element of that head. Every node has indegree at most one, and a finite DAG with indegree at most one has no undirected cycle. Parallel boundary edges are allowed.

Let e_int be the number of interior edges and let c_hit be the number of connected components of this interior forest that have at least one boundary edge. Include isolated boundary nodes as components. Then

    rank(M) = e_int + c_hit.                              (7)

Proof: the coefficient matrix is the oriented incidence matrix of the graph with the sink row removed. Interior forest edges are independent. The first boundary edge in any interior component increases rank by one; every further boundary edge in that component is dependent, because its two sink-to-component paths yield a cycle relation. Equivalently, reduced incidence rank is the sum of the tree contributions. Boundary edges from distinct interior components each increase rank, although they share the same sink.

This gives an explicit optimal number of real linear summaries. For an unknown scalar t(v), form each nonsink node's weighted divergence

    g_S=sum_{v:B(v)=S}t(v)
          -sum_{v:B(v) union {v}=S, |B(v)|<k}t(v).

The second sum has at most one summand. Store every node divergence in a boundary-touched component, and omit one node divergence in each untouched component, recovering it from the fact that the component's divergences sum to zero. The number of stored real scalars is precisely (7). These values reconstruct the truncated selection polynomial and hence all counterfactual sufficient statistics.

For q-bit nonnegative integer statistics, rank r implies at least rq stored bits in the worst case, by varying r independent columns. A generic upper bound of r*ceil(log2(|U|(2^q-1)+1)) bits follows by storing r independent **rows of M** as bounded nonnegative sums. Divergence storage has signed sums of similar bit length. Thus the scalar-rank optimum is exact, and finite-bit storage is characterized within an additive log|U| bits per scalar. This does not claim optimality of arbitrary nonlinear summaries beyond the displayed bit lower bound.

A useful consequence: every document with |B(v)|<k contributes one independent dimension. Compression among candidate columns occurs only through records lying exactly at the deletion-budget boundary, |B(v)|=k. In particular, if all unknown potentially selected records have |B(v)|<=k-1, their selection functions are linearly independent. For m records with a common blocker set of size b, the rank is one at k=b and m at k=b+1. This recovers the IDs-only budget threshold directly.

**Access-model and source caveats:** if a request supplies forgotten sufficient statistics, the negative self-retention term may be computed from that external input; the above rank is no longer an unconditional storage lower bound. Also, replacing documents by source-deletion variables can merge variables and destroys the unique-head argument. The forest theorem must not be transferred to source-group deletion without a new proof.

Sanity check: exact rational elimination on 280 randomly generated ordered graphs with n=1,...,7 verified (7) for every deletion budget k=0,...,n. This supplements the proof; it is not a substitute for it and is not an experimental publication result.

## 7. What should and should not be sold as the paper

Likely background lemmas: CC fragmentation, Menger separator characterization, MIS causal cone, indexing indistinguishability, and the densest-k-subgraph/CLIQUE reduction. Their derivations are rigorous but their ingredients are classical.

Potentially stronger synthesis: characterize deletion-budget closure for the exact implemented ENS rule; design compressed learner-aware state indexed by activation dependencies; show a sharp storage threshold; and combine exact curation repair with downstream certified updates. This needs comparison against deletion-robust coresets/summaries, source provenance, incremental view maintenance, and unlearning storage lower bounds.

Every theorem about the released SemDeDup pipeline must explicitly state whether clustering and ordering are frozen. For a rerun that refits k-means or centroid order, even pairwise graph edges may remain fixed while the allowed comparison set or order changes. A certificate for frozen-state ENS is not a certificate for such a full rerun. Either define a deletion-ready curation algorithm using exogenous fixed anchors/order, prove retained trajectory stability, or report the frozen-state result as the first stage of a broader repair procedure.
