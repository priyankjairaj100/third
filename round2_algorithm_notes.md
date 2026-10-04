# Indexed sequential canonical repair

This note strengthens the full-map scan in the first theory draft. It concerns the same fixed-order, restriction-consistent blocker curator, record-local additive statistics, and a decreasing cumulative deletion horizon. It makes no additional novelty claim about incremental database maintenance.

## Result

Let the initial canonical nonzero coefficient map have $M_0$ keys, statistic dimension $s$, and maximum key degree at most $k_0$. Define

\[
 L_0=\sum_{J:\alpha_J\ne0}|J|,
 \qquad
 \Phi_0=\sum_{J:\alpha_J\ne0}\frac{|J|(|J|+1)}2.
\]

There is an exact indexed implementation with the following properties over an entire sequence consuming at most $k_0$ fresh record deletions:

- At most $L_0$ coefficient-key shrink operations.
- At most $L_0$ removals of identifier-to-key incidences, including removals caused by collisions, cancellations, and horizon pruning.
- At most $M_0$ coefficient-vector additions. Transfers to previously absent keys do not copy or add the statistic vector.
- At most $\Phi_0$ identifier cells inspected to construct shortened sorted-tuple keys.
- No full coefficient-map scan is needed by a deletion operation.

With sparse sorted-tuple keys, expected constant-amortized-time hashing apart from key processing, and unit-cost exact field arithmetic, the total update time is

\[
 O\!\left(Q+\Phi_0+sM_0\right)
 \subseteq O\!\left(Q+k_0L_0+sM_0\right),
\]

where $Q$ counts all submitted identifiers, including duplicates and retries. Initial construction, oracle verification, model decoding, and external serialization are separate costs. Retained storage is

\[
 O(M_0s+L_0+M_0+|V_{\rm current}|)
\]

words, including current membership, keys, inverse incidences, and degree buckets. Both current key count and current incidence count are nonincreasing. A batch can be decomposed into singleton deletions because canonical substitution with horizon decrements composes exactly.

The bound is for exact arithmetic; growing integer or rational bit lengths require the corresponding arithmetic cost. In the implemented integer algorithm, every current coefficient is a sum of a disjoint group of initial coefficient vectors: coefficients never split. Initial integer coefficients of at most $b$ magnitude bits therefore grow to at most $b+\lceil\log_2\max(1,M_0)\rceil+O(1)$ bits. This does not claim constant-cost arbitrary-precision arithmetic.

## Data structures

The logical learner state is the canonical map, current identifier set, and remaining horizon $h$. The implementation adds three live indices:

1. A dictionary from a sorted tuple $J$ to a stable `Entry` object holding $J$ and its nonzero statistic vector.
2. An inverse dictionary $I[u]$ containing references to exactly those live entries whose keys contain $u$.
3. Degree buckets $D[d]$ containing references to exactly those live entries whose keys have degree $d$.

The inverse lists contain stable entry handles, not immutable copies of subset keys. When an entry shrinks, all surviving inverse incidences already refer to the right entry. Only the removed identifier incidence must disappear. This avoids reindexing every surviving element whenever a key changes.

The statistic vector is immutable and can be transferred by reference. Only a collision with an already existing destination key requires $s$-dimensional addition and an exact zero check.

Empty inverse lists and degree buckets are removed immediately. Cancelling an entry removes all of its remaining live references. No deleted-identifier log is required: with a public identifier universe, absence from the current membership set is enough to treat a retry as a no-op. Out-of-universe requests and requests with more fresh identifiers than the remaining horizon are rejected before any mutation.

## One-record deletion

To delete current identifier $u$, with old horizon $h>0$:

1. While $I[u]$ is nonempty, pop an entry with key $J\ni u$.
2. Remove its old key from the coefficient dictionary and its handle from degree bucket $D[|J|]$. The inverse incidence for $u$ has just been removed.
3. Form $K=J\setminus\{u\}$, replacing the entry's key.
4. If no entry at $K$ exists, reinsert the same entry at key $K$ and in bucket $D[|K|]$. Its other inverse incidences remain valid, and its vector is transferred unchanged.
5. Otherwise, merge its vector into the existing entry at $K$. Remove every remaining inverse incidence of the discarded entry. If the sum is zero, remove the destination entry too, including its dictionary, inverse, and degree references.
6. After $I[u]$ is empty, remove every entry still in $D[h]$, together with all its inverse incidences. These are precisely untouched keys violating the new horizon.
7. Remove $u$ from current membership and set $h\leftarrow h-1$.

New keys exclude $u$, so they cannot be processed a second time in this operation. Every touched key loses one element and therefore already has degree at most $h-1$. Thus the final degree-bucket pruning cannot accidentally discard a touched key that should survive. There is no intermediate complete coefficient map or touched-key union.

For a batch $F$, first deduplicate and validate the complete request. Then apply the procedure to each fresh identifier. The final state is independent of processing order in exact arithmetic. An implementation wanting one final model decode should decode only after the whole batch; intermediate algebraic states need not be exposed.

## Correctness and accounting proofs

**Correctness.** The keys incident to $u$ are precisely the monomials changed by setting $x_u=1$. Every such key becomes $J\setminus\{u\}$; collisions are added and zeros removed. Unchanged keys remain unchanged except that degree-$h$ keys are discarded under the new cutoff. These are exactly the coefficients of the substitution polynomial truncated to degree $h-1$. The canonical-state theorem in the main note therefore identifies the resulting map with a fresh retained-data build. Induction establishes batch and sequential correctness.

**Live-index correctness.** Removing $u$ changes no membership relation involving a different identifier and a surviving stable entry. A collision discards one handle and explicitly removes each of its inverse references. A cancellation or horizon pruning does the same for a complete live entry. Reinsertions update degree buckets. Consequently the inverse and degree relations are exactly those reconstructed from the new canonical map; there are no stale live entries or deleted identifiers in them.

**Incidence accounting.** No update ever adds an inverse incidence. Shrinking removes one, and collisions/pruning can only remove more. Hence all incidence removals and all shrink events are bounded by initial $L_0$.

**Vector arithmetic.** Moving an entry to an unoccupied key preserves the number of entries. Merging two entries strictly decreases that number; exact cancellation decreases it by two. No operation increases it. Hence there are at most $M_0$ merges, each requiring $O(s)$ arithmetic. In particular, the vector-arithmetic bound is $O(sM_0)$, not $O(sL_0)$. Temporary vectors allocated for merges are likewise bounded by this accounting.

**Sparse-key processing.** Use potential

\[
 \Phi=\sum_{\text{current keys }J}\frac{|J|(|J|+1)}2.
\]

A degree-$d$ entry moved to a previously absent degree-$d-1$ key decreases this potential by exactly $d$, paying for construction and hashing of its shortened key. If the destination exists, the potential decreases by at least $d$, since no additional destination node is created. Cancellation decreases it further. Pruning a degree-$d$ key decreases the potential by at least $d$, paying for its key/index removal. Constant-degree operations, including removal of a constant coefficient during cancellation, are charged to the merge causing them. Thus key-processing work is $O(\Phi_0)$, and the total bound follows. The implementation measures the exact number of tuple cells inspected during shrink operations, not an inaccurate wall-clock proxy for every hash-table operation.

Expected hashing is a computational-model assumption. A deterministic ordered-dictionary implementation introduces the appropriate logarithmic dictionary factors. Exact integer-bitmask keys offer a different implementation with word work depending on $\lceil N/w\rceil$; neither representation should be called constant-cost independently of key size. No unimplemented persistent-trie performance theorem is needed for the stated result.

## What “canonical state” means here

The equality guarantee concerns the designated abstract state: coefficient values keyed by subsets, current identifiers, remaining horizon, and the induced live incidence relations. Stable entry handles may be renamed. The implementation retains no original coefficient map, original graph, old statistic vectors as checkpoints, historical handle counter, or request log. Instrumentation and original-data oracles in the verification script are held by the external test harness, not by `IndexedState`.

Literal Python heap bytes, object addresses, hash-table capacities, insertion order, and allocator history are **not** guaranteed to match a fresh build. This is not a strong claim about arbitrary full-memory observations, physical secure deletion, or backups. A theorem requiring equality of those observations needs an explicitly canonical memory representation or a separately costed rebuild.

`snapshot()` produces a deterministically ordered serialization using subset keys in place of handles. Its output agrees exactly with a fresh build in the tests. Creating or rewriting a full such snapshot must visit/output its contents and pay sorting costs; it is not charged as a constant-time operation or included in the scan-free update claim. In particular, an external full-state export costs at least the size of the exported values, identifiers, and indices.

## Output-only answers versus state repair

A singleton output-only answer can be obtained as $\alpha_\varnothing+\alpha_{\{u\}}$, with only one vector addition and dictionary lookup. Canonical state repair may also have to merge many keys, remove stale incidences, and prune a newly inadmissible degree bucket. These operations are needed by this persistent representation even when the current model statistic barely changes.

After repair, the current aggregate is simply the new constant coefficient. Decoding ridge weights still requires the specified linear solve; that cost is separate from updating the summary. The total-arithmetic theorem does not count a dense solve after every request as free.

## Rank-optimal sketches can transform, but fast updates do not follow for free

Let $\alpha=At$, and let $B$ be a row basis of $A$, so $A=CB$. A rank-optimal stored value vector is $z=Bt$. Let $R_F$ implement substitution and the reduced horizon. The new coefficient operator, padded by zero columns for forgotten records, is

\[
 A'=R_FA.
\]

Choose a row selector $H'$ whose rows pick a basis of $A'$. The new optimal sketch is obtainable without original statistics:

\[
 z'=H'A't=H'R_FCz.
\]

In particular, rank cannot increase along a valid deletion sequence with a decreasing horizon. This does not contradict the initial-budget phase transition: increasing the initial query budget is a different operation.

The transformation proves existence, not the same fast maintenance bound. Dense application costs $O(sr'r)$, before computing or storing the basis and transformation. Expanding $z$ to all coefficients, applying the indexed algorithm, and recompressing is also valid, but expansion/recompression may require full-map work on each request.

For the incidence matrix in this project, an explicit row basis keeps every nonground node in a component containing ground and omits one fixed node from each component without ground. The omitted coefficient is minus the sum of the others in that component. This reconstructs coefficients without general Gaussian elimination, but changing components and repeatedly reconstructing omitted entries can still be costly. Efficient dynamic rank-optimal storage with equally strong local update bounds is an additional problem; the current implementation prioritizes transparent exact repair and bounded redundant coefficient storage.

## Verification

The standalone standard-library script `algorithm_round2.py` uses exact Python integers and checks both coefficient values and canonicalized live indices against independent full retained-data rebuilds. It covers all 76 ordered graphs on zero through four vertices, all 4,301 horizon-limited ordered deletion sequences, 12,821 intermediate canonical-state comparisons, 12,821 duplicate/retry no-op checks, and 4,301 batch-versus-sequence comparisons.

There are 200 additional random graphs, 880 batch-state comparisons, 1,078 invalid-request atomicity checks, and 200 batch-order comparisons. Six adversarial constructions cover sparse touches with final horizon pruning, nested clique keys, the star storage transition, exact cancellation, and a zero polynomial. These are algorithm checks and operation counts, not NLP evaluations or end-to-end runtime speedup measurements.

Two examples make the accounting concrete:

| Construction | Initial keys | Shrink events | Tuple cells copied | Vector merges | Incidence removals | Full-scan visits for the same singleton sequence |
|---|---:|---:|---:|---:|---:|---:|
| Edgeless, 2,048 records, 128 deletions | 2,049 | 128 | 128 | 128 | 2,048 | 254,144 |
| Ordered clique, 256 records, 128 deletions | 129 | 8,256 | 357,760 | 128 | 8,256 | 8,384 |

The clique attains both $T=L_0$ and the tuple-processing potential $\Phi_0$. It is a useful counterexample to claiming universally constant-time key updates. The edgeless example demonstrates the avoided repeated scans while still explicitly charging 1,920 final-horizon key removals. Complete results are in `round2_algorithm_verification.json`.
