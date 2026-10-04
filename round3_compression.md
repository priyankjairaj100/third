# Third-pass storage closure: a sharp factor-two bound and a canonical rank-optimal baseline

This note supplies two exact algorithmic additions to the current specification. The first is the recommended main-paper addition: the existing fast sparse update is already within a sharp factor two of the best linear statistic-storage dimension. The second supplies a slower rank-optimal baseline whose metadata are a deterministic retained-data object. Neither statement claims total-memory optimality for unknown graph metadata or optimality for constrained ridge moments.

## 1. Sharp factor-two statistic-storage theorem

Fix a current retained dataset, remaining deletion horizon h, and either document deletion or the disjoint-source extension. Form the nonzero-column incidence graph Γ_h of the coefficient operator A_h, with omitted high-degree heads joined to ground. Ignore unused isolated vertices and same-source identically zero columns. Write p for the number of incident nonground key vertices, c_0 for the number of components not containing ground, and r = rank(A_h) = p-c_0. Let M be the number of nonzero s-dimensional coefficient vectors retained by the indexed sparse implementation.

**Theorem.** Over every field,

    M ≤ p ≤ 2r.

Thus the fast implementation retains at most 2sr scalar statistic coordinates, compared with the exact linear minimum sr for s arbitrary independent statistic coordinates. The inequality holds at every stage of every valid cumulative deletion sequence. Its constant two is asymptotically sharp, even for document suppression at a fixed positive cosine threshold and horizon h=3.

**Proof.** A nonzero coefficient key must be a row incident to at least one nonzero column, hence M≤p. Every ungrounded component contains an edge between distinct nonground vertices; it therefore contains at least two vertices. Each grounded component contributes its entire nonground vertex count to rank. Consequently 2c_0≤p, so p≤2(p-c_0)=2r. The rank theorem is field-independent, including characteristic two. After any update, the exact retained-data coefficient matrix is again an incidence matrix of the specified form, so the same argument applies pointwise. ∎

**Geometric sharpness.** Take q≥3 early anchors a_1,...,a_q with unit embeddings e_1,...,e_q. For every unordered pair i<j, append a record v_ij with unit embedding (e_i+e_j)/√2, after all anchors. Fix the strict cosine threshold at τ=3/5 and horizon h=3. Anchor-anchor similarities are zero. An anchor and its incident pair have similarity 1/√2>3/5; other anchor-pair similarities are zero. Distinct pair embeddings have similarity either zero or 1/2<3/5. Hence B(a_i)=∅ and B(v_ij)={a_i,a_j}, independently of the priority among pair records.

Let m=binom(q,2). All coefficient edges are interior. The anchor edges make one component with q+1 nodes. Each pair record makes a separate two-node component: its tail {a_i,a_j} is never an anchor head, and its head contains the unique pair-record ID. Therefore

    p=q+1+2m,    c_0=m+1,    r=q+m.

For sharpness over the reals, take every scalar statistic t(v)=1. The empty coefficient is q; each singleton anchor coefficient is -1; every pair tail is +1; every pair-record head is -1. No key coefficient vanishes, so M=p. The ratio M/r=(q+1+2m)/(q+m) approaches two. Unique sources give the same source-deletion example. Over a finite field, the empty coefficient may vanish if its characteristic divides q; choose q not divisible by the characteristic, or note that losing one key leaves the limiting ratio unchanged. This is a valid fixed-threshold semantic-graph construction in dimension q; no claim that q dimensions are necessary is needed.

**What is and is not compared.** M counts vector slots, or sM scalar statistic coordinates. Key tuples, dictionaries, incidence indices, membership, retained graph metadata, bit lengths, and serialization are separately charged. A particular statistic assignment can yield cancellations and a much smaller M. The minimax linear lower bound allows arbitrary independent statistics; it is not a proof that ridge covariance matrices require sr independent values. The fast algorithm may be preferable to exact rank compression because it needs no surviving graph representation beyond the coefficient keys and its deterministic indices.

**Complete logical storage and work corollaries.** Every live key has length at most h, hence L=Σ|J|≤hM≤2hr. Counting statistic values, represented key tuples, live inverse-index incidences, dictionary/degree entries, and a current membership set, the existing indexed state uses

    O((s+h+1)r + |V_alive|)

logical words, assuming identifiers and statistic elements are word-sized. This bound counts its live metadata rather than treating it as free. It is not a total-space lower bound and does not bound history-dependent allocator capacity. For initial horizon k, Φ_0≤k(k+1)r_k and sM_0≤2sr_k, so the existing cumulative work theorem also gives

    O(Q+(k²+s)r_k)

expected word/arithmetic update work, excluding initial construction, model solves, and serialization. The sharper instance-dependent Φ_0 statement should remain the principal algorithm bound; this corollary connects it to exact query dimension.

## 2. Rank-optimal canonical state with eligible-only metadata

This is a scanning baseline, not an improvement of the indexed algorithm's cumulative work bound. It strengthens the existing abstract factorization observation by making the changing decoder metadata explicit and deletion-clean at the designated abstract-state level.

For current records V and remaining horizon h, retain identifiers/membership and only the blocker lists of activation-eligible records

    E_h={v∈V: |B(v)|≤h}.

The metadata include the eligible record's own ID and its current blocker tuple. They contain no statistic values. A record that becomes ineligible can be dropped permanently.

**Eligibility monotonicity lemma.** If b_v>h and a valid batch F of r≤h records is deleted, then any surviving v has b'_v≥b_v-r>h-r. Thus no currently ineligible record becomes eligible under the decremented horizon. Eligible-only metadata are exactly repairable by deleting removed owners, subtracting F from surviving blocker lists, and dropping newly ineligible records. This equals eligible-only metadata rebuilt from all retained records at the remaining horizon. The same argument applies to source blocker counts, after omitting same-source identically zero records.

For completeness, a record whose own source a lies in its current source blocker set U(v) remains permanently irrelevant: if a is retained then a remains an undeleted blocker, while deleting a deletes the record itself. This justifies discarding such metadata at construction; it cannot reappear after a valid source request. Source budgets count distinct source IDs, not removed record counts. The result assumes each record has one fixed source owner and does not cover overlapping ownership.

Build Γ_h from those current metadata. For each ungrounded connected component C, omit the coefficient row at a deterministic distinguished node o(C), for example the lexicographically least subset key. Retain all other row coefficients; in a grounded component retain every nonground row. There are exactly p-c_0=r stored vectors. Reconstruct an omitted coefficient as

    α_o(C) = - Σ_{J∈C\{o(C)}} α_J.

This is valid because every column in an ungrounded component has total coefficient zero. The retained rows form a row basis, so this is also exactly rank-optimal for arbitrary independent statistic coordinates.

**Canonical rank-optimal repair algorithm.**

1. Reconstruct all coefficient values from the compressed rows and current eligible metadata.
2. Substitute the requested variables by one, merge equal keys, and truncate to the new horizon.
3. Repair the eligible metadata using the eligibility lemma; build its new incidence components.
4. Retain the deterministically chosen new row basis and discard the temporary full map.

The returned metadata, selected row names, and row values equal a fresh retained-data construction. Thus no original graph, original statistic vector, original basis, or old payload-dependent checkpoint is required. Earlier output transcripts and physical allocator traces remain outside the state contract, as in the main specification.

**Complexity.** Let e=|E_h| and D_h=Σ_{v∈E_h}(1+|B(v)|), before a batch. Maintain sorted blocker tuples; deleting variables preserves their order, and inserting the owner into a head costs linear tuple work. Intern subset keys to integer handles before disjoint-set operations. Choose the omitted key by a single lexicographic-minimum pass through each component, without sorting the entire component. During reconstruction, retrieve each stored vector once and then sum its scalar coordinates. With these details, expected dictionary key processing, canonical component-root selection, and direct batch key reduction cost O(Q+D_h); disjoint-set construction adds O(e α(e)). Reconstruction, merging, and recompression use O(sr) statistic arithmetic, since p≤2r and the new rank cannot exceed r. Hence the implemented batch method costs

    O(Q+D_h+e α(e)+sr),

excluding model solves, output serialization, and the bit cost of field arithmetic. The statistic workspace is O(sr), although it may temporarily hold a constant factor more than r vectors. Persistent statistic coordinates are exactly sr (or fewer if zero entries are represented sparsely). Metadata cost O(D_h) can dominate statistic storage. At h=0 with many records, for example, one statistic vector can suffice while owner-level metadata are much larger; that cost is not hidden by the theorem.

The direct update applies a batch once; repeatedly calling it for singletons can rescan all eligible metadata each time. Consequently it does not match the indexed sparse algorithm's sequence-wide bound. The two concrete choices are: (a) sharp factor-two statistic storage with the fast coefficient-only algorithm; (b) exact rank statistic storage with current eligible metadata and linear-size rebuilding. This closes the algorithmic tradeoff with proved, implementable options rather than an unproved promise of simultaneously optimal storage and update time.

## 3. Verification

`round3_compression.py` checks the rank and factor-two formula on all ordered graphs through five vertices, all horizons, and three fields; it checks canonical compressed repairs against fresh retained-data rebuilding through complete small deletion sequences, randomized larger document/source examples, and the geometric sharpness family. Numerical geometry checks use explicit vectors only to corroborate the elementary dot products above. These finite checks are not proofs or natural-language experiments. See `round3_compression_verification.json` for exact executed counts.
