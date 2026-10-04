# Second structural audit: exact curation, rank, and sequential state

Audited `output/pdf/counterfactual_curation_theory.tex` as supplied on this pass. I found no counterexample to its core activation, polynomial, canonical state, or incidence-rank statements under their stated fixed-state and request-access assumptions. The strongest improvements are an exact one-pass rank-curve algorithm, a complete incidence-rank theorem for whole-source deletion, and compositional/state-dimension monotonicity results.

## 1. Issues and qualifications to address first

1. **Rank-optimal query state is not automatically deletion-clean state.** The rank lower/upper theorem treats the original graph-dependent decoder as fixed side information. A row-space basis, original graph, or original topology-dependent decoder can itself preserve information about removed records. The canonical sparse coefficient map is a different representation with a proved retained-data state identity. Keep the two algorithms and guarantees separate unless the rank decoder is itself reconstructed canonically from retained data. An optimal dimension theorem must not be described as proving an optimal canonical-state algorithm.
2. **The source section is currently weaker than necessary.** Source identification destroys the forest property but does not destroy the incidence representation. Exact source rank remains a connected-components computation, not a generic exponential matrix-rank problem. A full theorem and a counterexample to the forest extension are below.
3. **Source means a partition.** The current source formula requires exactly one source label per record and deletion of the complete source. A record owned by several independently withdrawable entities has a survival factor with multiple `(1-x)` terms. Its columns need no longer have two nonzeros, so the incidence theorem must not be transferred to overlapping ownership without another proof.
4. **Exact dictionary equality matters.** The rank algorithm below can use hashed tuple keys, but it must resolve hash collisions by comparing the actual canonical sets. A collision-prone fingerprint alone gives a randomized failure probability, not the deterministic theorem.
5. **Payload lower-bound language should fix the query access model.** For general allowed deletion families, “no reaccess” alone does not prohibit the request itself from furnishing deleted payloads. The proposition is cleanest with identifier-only requests. For the full document-budget family, it can actually be extended to full forget-payload access by decoding potentially selected records in priority order: request `B(v)` only contains earlier records, which the decoder has already recovered or which were held fixed. For arbitrary source requests this argument fails; see the source diamond below, where deleted payloads can serve as side information.
6. **Released SemDeDup equivalence needs the nonnegative threshold condition.** Its lower triangle and diagonal are padded with zeros before the column maximum. Therefore its rejection test equals “some earlier similarity is strictly greater than tau” when tau>=0. State the conventional cosine threshold range tau in [0,1] and strict `>` comparison. For tau<0, even a singleton has a padded column maximum zero and would be rejected by the matrix code, whereas the blocker rule selects it. Exact threshold ties are retained under the strict rule.
7. **Abstract state is not literal machine state.** A canonical logical coefficient map can be exactly retained-data equivalent while a stable-handle hash table, allocator, iteration order, or counter retains a history-dependent representation. The incremental algorithm may guarantee equality of the designated abstract state modulo handle renaming. Literal canonical serialization requires sorting/ordering keys, writing only the surviving coefficients and allowed metadata, and replacing the prior serialization. Neither theorem by itself establishes erasure of logs, backups, historical outputs, or the complete operating system.

## 2. Assumptions for the structural theorems

Let V be a finite record set with a fixed strict total priority order. Let G be a fixed undirected graph whose retained counterpart is always the induced subgraph. For each v, let B(v) be its earlier neighbors and b_v=|B(v)|. The selector keeps v exactly when no earlier raw record blocks it. The fixed-state assumptions include edges, comparison eligibility, and relative order, not only embeddings.

Requests contain stable record IDs, and the primary query family is every F subseteq V with |F|<=k. Statistics t(v) are arbitrary independently variable scalars when proving optimal query dimension; graph/order and the decoder metadata are fixed side information. The rank result applies over any field. A real linear-sketch lower bound is not an unrestricted nonlinear real-encoding lower bound.

## 3. Horizon-independent reduced blocker signatures

### Definition

For a set J of record IDs, define its reduced signature rho(J) as follows:

1. If J is empty, return the empty set.
2. Let u be the maximum-priority member of J.
3. If B(u)=J minus {u}, replace J by B(u) and repeat.
4. Otherwise return J.

The operation terminates because cardinality strictly decreases. It is a deterministic function of the graph/order and J, independent of the deletion horizon.

### Lemma: rho is exactly the interior-forest root

Construct the untruncated coefficient graph with an edge B(v) -> B(v) union {v} for every record v. Every nonempty head J determines the added record uniquely as max(J). Therefore J has a parent if and only if B(max(J))=J minus {max(J)}; in that case its parent is exactly the set obtained in step 3. The graph is a forest, and rho(J) is its root.

For a node J with |J|<=k, every ancestor edge has a head of size at most |J| and is present in the budget-k interior graph. Consequently its root is unchanged by truncation at any k>=|J|. This is why the signature is horizon-independent.

### Theorem: exact rank from reduced signatures

For every integer k>=0,

    r_k = #{v:b_v<k} + |{rho(B(v)):b_v=k}|.                 (R1)

The second term is zero when no record has blocker count k.

Proof: the interior forest contains exactly the first set of edges. Its independent edge columns span the vectors whose coordinates sum to zero separately on each component. Every boundary column, with b_v=k, is a unit vector at B(v). It increases rank exactly when no previous boundary column touched that component. Two boundary tails are in the same component precisely when their roots rho(B(v)) agree. This proves (R1) over every field.

This sharpens the draft's forest formula by replacing a per-horizon connected-components computation with a horizon-independent canonical signature. It does not change the substantive rank lower bound; it gives an efficient exact structural characterization.

### One-pass algorithm for all horizons through K

Process records in increasing priority. Only records with b_v<=K are needed.

Maintain a dictionary `head_root`, count array `a[j]`, and sets `roots[j]`:

```
for v in increasing priority:
    if |B(v)| > K: continue
    root = head_root.get(B(v), default=B(v))
    a[|B(v)|] += 1
    roots[|B(v)|].add(root)
    if |B(v)| < K:
        head_root[B(v) union {v}] = root

prefix = 0
for k = 0,...,K:
    r[k] = prefix + |roots[k]|
    prefix += a[k]
```

Why the lookup is valid: if B(v) is a previous edge head, its maximum element u is earlier than v, and its unique parent edge was already processed. If it is not a previous head, it has no parent and is its own root. No later edge can create its parent, because that edge would have to belong to max(B(v)). Record-ID sets should be interned so roots can be stored as pointers to canonical keys.

Let L_K=sum_{v:b_v<=K}(b_v+1). Once the eligible blocker lists are available in canonical order, the algorithm uses expected O(L_K+K) dictionary/key work with collision-resolved hashing, and O(L_K+K) metadata storage. If the priority order is not already supplied, sorting has its separate cost. A deterministic balanced-tree dictionary adds the corresponding logarithmic factor. Initial graph construction and finding the eligible blocker sets remain separate costs; this is not a sublinear graph-discovery algorithm.

For unknown statistics only on a subset U of records, the same theorem holds after retaining coefficient edges only for v in U. In the root definition, stripping u is allowed only if u is in U. This models known original statistics or other supplied coordinates without incorrectly charging them as unknown.

### Consequences and sanity examples

Writing a_j=#{v:b_v=j} and d_j=|{rho(B(v)):b_v=j}| gives the entire curve

    r_k = sum_{j<k} a_j + d_k.

In particular r_{k+1}-r_k=a_k+d_{k+1}-d_k>=0, since d_k<=a_k. Rank rises monotonically when the *initial* budget is increased. This is a different statement from the decreasing-horizon deletion result below.

At k=1, a boundary tail {u} reduces to the empty root if u was originally selected; otherwise it stays {u}. Thus

    r_1 = |S| + 1[there is a singleton-blocked record blocked by a selected u]
               + #{unselected u: some v has B(v)={u}}.

The indicator is counted only once across all originally selected blockers because their singleton nodes share the empty-set root. For an ordered path on N>=2 vertices, this gives r_1=N: the earlier-neighbor chain already destroys singleton-query compression. For one initially selected hub with many leaves, r_1=2. These distinguish graphs having similarly small blocker counts but different reduced signatures.

## 4. Whole-source deletion still has an exact incidence theorem

Let source labels form a fixed partition s:V->[m]. Let U(v)={s(u):u in B(v)} be the distinct blocker-source set. If s(v) is in U(v), v contributes zero to every whole-source retained selection query: removing its same-source blocker necessarily removes v.

Discard those zero columns. For every remaining record, s(v) is not in U(v), and its selection polynomial is

    f_v(x)=x_{U(v)}-x_{U(v) union {s(v)}}.

For budget k, drop monomials of degree greater than k, exactly as in the document case. Each nonzero column is still a signed incidence edge:

* U(v) -> U(v) union {s(v)} if |U(v)|<k;
* U(v) -> ground if |U(v)|=k;
* zero if |U(v)|>k.

### Theorem: exact source-query rank

Let Gamma_k^src be this multigraph, p its number of incident non-ground nodes, and c_0 its number of components not containing ground. Then

    r_k^src = p-c_0                                            (S1)

over every field. Equivalently, form the interior graph (which can have cycles), include isolated boundary tails, let r_int be its incidence rank, and let d_partial count its connected components touched by boundary edges. Then

    r_k^src = r_int+d_partial.                                 (S2)

Proof: Boolean monomial evaluation on the source Hamming ball is again an invertible zeta transform. The coefficient matrix is the oriented incidence matrix of Gamma_k^src with the ground row deleted. The standard incidence-rank proof applies. Source grouping can introduce several edges into the same head and hence cycles; the graph need not be a forest, so replacing r_int by its number of edges would be false.

### Minimal counterexample to the source-forest claim

Use four records in order a<b<w<z, with sources

    s(a)=A, s(b)=B, s(w)=B, s(z)=A,

and only graph edges a--w and b--z. The coefficient edges are

    empty -> {A}, empty -> {B}, {A} -> {A,B}, {B} -> {A,B}.

This is a diamond cycle. At the full two-source horizon, the four statistic columns have rank three, with relation

    f_a-f_b+f_w-f_z=0.

The nonempty source-retained sums are a+b, b+w, and a+z; withdrawing both sources produces zero. Thus full-horizon source rank need not equal the number of records. Compression can persist even when every source-deletion request is allowed.

This example also illustrates request-payload side information: if a and b are known, storing w+z suffices to recover either hidden survivor after one source withdrawal when the deleted hidden payload is provided. A source-level corpus-payload lower bound cannot simply reuse the record-order decoder argument.

### All-horizon source rank in near-linear graph work

For each nonzero record column define b_src(v)=|U(v)|, intern the two subset keys, and bucket the edge by b_src(v). Use a disjoint-set union structure (DSU) over these keys, initially isolated, and let R=0 count successful interior unions.

At budget k=0, there are no interior edges. Count distinct DSU roots among boundary tails in bucket 0; this is r_0. Before evaluating budget k>=1, insert every edge from bucket k-1 as an interior edge, incrementing R only if its endpoint components were different. Then count distinct DSU roots of the tails of bucket k. Return

    r_k^src=R+number of distinct roots among bucket-k tails.

Every edge is inserted once and its boundary tail is counted once. With M nonzero columns and total source-key material L_src, all horizons through K cost expected O(L_src+M alpha(M)+K) time and O(L_src+M+K) metadata under exact collision-resolved hashing. This bound includes subset interning, not initial graph/blocker-source discovery. Coordinates associated with zero same-source columns are absent. For K smaller than the maximum source degree, omit edges that never become relevant through K.

This gives a complete exact source-rank algorithm; no exponential source-subset query matrix is formed.

## 5. Canonical state updates compose exactly

Let U_F^k denote the draft's substitution-and-truncation map from horizon k to horizon k-|F|. For disjoint requests F,G with |F|+|G|<=k,

    U_G^(k-|F|) (U_F^k(alpha)) = U_(F union G)^k(alpha).     (C1)

Proof at the coefficient level: substitution x_F=1 followed by x_G=1 equals substitution x_(F union G)=1. If a monomial is removed after the first substitution, its remaining degree is greater than k-|F|. The second substitution lowers degree by at most |G|, so that monomial cannot survive the final cutoff k-|F|-|G|. Conversely, every term surviving the direct final cutoff must have survived the intermediate cutoff. Thus intermediate truncation changes no final retained coefficient. Canonical combination and zero removal produce identical maps.

This is path independence: the final canonical coefficient state depends on the cumulative deleted set and remaining budget, not the order or batching of requests. Repeated IDs should first be removed from the request relative to the current population; the theorem counts each deletion once. The same proof works for source-variable requests.

## 6. The future-query dimension cannot increase during an authorized sequence

Let r_k(V) be the original document-query rank, and let F be a request of size r<=k. Build the retained graph/order on V minus F and use the decremented horizon k-r. Then

    r_(k-r)(V minus F) <= r_k(V).                           (C2)

Proof: pad each retained query row by zeros at deleted coordinates F. For a future request G subseteq V minus F with |G|<=k-r, restriction consistency makes this padded row exactly the original Q_k row for F union G. Hence the padded retained query matrix consists of a subset of the original query rows. Padding does not change rank, so the inequality follows. This proof works over any field and extends directly to whole-source deletion after padding removed record columns by zeros.

The selected corpus can grow while the dimension of all *still permitted future queries* cannot grow. The two observations are compatible because spending deletion budget reduces the remaining query family. The corresponding payload activation sets obey

    E_(k-r)(V minus F) subseteq E_k(V) minus F,

which follows either from the query-family inclusion or from
|B(v)|<=|B(v) minus F|+r<=k. A correct finite-budget corpus buffer therefore never needs a previously ineligible payload during an authorized request sequence.

Canonical sparse-map size also cannot increase under the direct update: each old key maps to at most one new key, and merging/truncation only removes keys. This is a count of vector-valued keys, not a claim that every exact integer bit length or decoder-metadata measure decreases.

Replenishing the horizon is a different contract. For example, a hub plus m leaves at initial horizon one has rank two, but after deleting the hub a fresh horizon-one learner on the edgeless retained leaves has rank m. An initial rank-two summary cannot support that reset for arbitrary unknown scalar statistics when m>2 and requests supply IDs only. Decrementing the horizon is therefore logically necessary for a generic compression guarantee, not merely an implementation choice.

## 7. Numerical and finite checks

The standalone `round2_structure_verification.py` writes `round2_structure_verification.json`. Its reproducible checks passed on 240 random ordered graphs with zero through seven vertices, seed 3032026:

* 2,160 reduced-signature rank comparisons over the rationals, and the same number over F_2, including unknown-coordinate subsets;
* 659 source-DSU rank comparisons over each of those fields;
* 2,742 source-polynomial query checks and 2,742 canonical retained-data source-state rebuilds;
* 8,117 source path-independence checks and 2,455 same-source-blocker zero-selection checks;
* the explicit source diamond, whose rank curve is (1,3,3);
* 531 nonnegative strict-threshold comparisons with the zero-padded upper-triangle SemDeDup rule, plus the negative-threshold singleton counterexample.

It also checks nonincrease of the number of logical coefficient keys. These are small focused additions to the draft's prior verification, not a rerun of its 118,000-case suite. They supplement the proofs and do not imply a natural-language experiment or byte-level state-erasure guarantee.

## 8. Suggested main-text replacement

Replace the source section's final statement that only “general query-rank characterization” remains with the precise source incidence theorem and diamond example. Replace the forest discussion's unspecified rank computation with reduced signatures (R1), and include the all-horizon algorithm in the appendix. Add (C1) and (C2) after canonical state equality, accompanied by the short warning that rank-optimal query sketches and canonical state representations are distinct claims.

These refinements settle the currently stated fixed-curator/source algebra rather than opening another speculative direction. They leave the substantive publication risks unchanged: the novelty of the learning/storage formulation, meaningful natural-text effects, and a measured advantage after including metadata and state-maintenance costs.
