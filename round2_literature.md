# Round 2: adversarial novelty audit of counterfactual-curation theory

Audit date: 3 October 2026. Reviewed the current `output/pdf/counterfactual_curation_theory.tex`, with attention to the finite-horizon statistic polynomial, canonical sequential state, query rank, and scalar-ridge memory transition. This is a qualified literature comparison, not an absence-of-prior-work certificate.

## Main finding

The important missing citations are **PrIU (SIGMOD 2020), F-IVM (SIGMOD 2018), DBToaster (VLDB 2012), and the bounded-deletion sections of Cherapanamjeri et al. (COLT 2025)**. Together these eliminate broad novelty claims about provenance polynomials for model repair, maintaining regression sufficient statistics, higher-order deletion interactions, or deletion-budget-dependent memory. The most defensible remaining candidate is the *specific structural characterization for earlier-neighbor semantic curation*, especially the exact blocker-incidence rank specialization and a matching scalar-learner storage separation caused by increasing the allowed deletion budget by one. These need empirical NLP consequences and careful cost accounting.

## Source-supported novelty table

| Current result | Closest established material | Assessment / surviving distinction |
|---|---|---|
| Activation iff all earlier blockers are removed | The actual SemDeDup implementation; routine Boolean evaluation | A correctness lemma connecting code to the model, not a major new theorem. Important because greedy selected-neighbor dedup is a different algorithm. |
| Statistic polynomial with signed monomials | Provenance for aggregation/difference; first-order provenance with negation; PrIU polynomial annotations of regression updates | Boolean expansion itself is established mathematics. Distinction is the particularly sparse two-monomial form for this selector and budget-specific storage implications. |
| Exact degree-k truncation for at most k deletions | Multilinear Boolean polynomials; higher-order finite differences in DBToaster | The truncation is an elementary support argument. It is useful as a constructive sufficient-state lemma, not independently a convincing theory contribution. |
| Canonical substitution and decrementing horizon | Provenance specialization under homomorphisms; recursively maintained delta views | The exact finite-horizon state contract is a useful specialization. Avoid claiming to invent sequential algebraic maintenance. Explicitly preserve the decrementing rather than replenished horizon. |
| Exact ridge training from counterfactual moments | F-IVM count/vector/covariance rings; PrIU linear-regression provenance | Maintaining these moments and solving the normal equations is standard. Nonmonotone curation determines which moments are needed. |
| General optimal linear sketch dimension = query rank | Kernel containment / rank factorization, basic linear algebra | Background theorem. A finite-field cardinality argument is likewise standard information theory. Do not present as a new lower-bound method. |
| Instance-specific blocker forest rank and frontier-only dependencies | Standard incidence-matrix rank, combined with selector-specific edge structure | The structural specialization was not matched directly in this audit. It is the strongest clean characterization in the current package; proof technology remains elementary. |
| Budget b versus b+1 scalar-ridge memory cliff | COLT 2025 bounded-deletion storage upper/lower bounds; Ghazi et al. 2023 INDEX reductions | “More supported deletions require more memory” and reconstruction-based proofs are known. The narrow candidate is a constant-dimensional, well-defined ridge pipeline with semantic-curation trigger, an adjacent-budget separation, and matching upper bound. |
| Convex signed Newton certificate | Existing certified removal / influence/Newton analysis | Standard certificate extended to the correct signed edit set. Correct normalization matters, but alone is insufficient novelty. |
| Source grouping by Boolean idempotence | Provenance token identification / specialization | Natural extension. A genuinely new source-level rank or geometry theorem would strengthen it; merely identifying variables is standard. |

## Primary sources inspected

### 1. PrIU: direct model-provenance precedent

Yinjun Wu, Val Tannen, Susan B. Davidson. **PrIU: A Provenance-Based Approach for Incrementally Updating Regression Models.** SIGMOD 2020. DOI 10.1145/3318464.3380571.

Full primary: https://arxiv.org/pdf/2002.11791

The introduction explicitly treats regression models as views and training samples as input data. Section 3 annotates vectors/matrices with provenance polynomials. Equation 7 annotates linear-regression updates with per-example moments and an annotated sample count. Deleting examples substitutes zero for their tokens; retained tokens become one. The optimized logistic algorithm uses approximations, with correctness/convergence/error analysis.

This directly anticipates “provenance polynomial + model update.” Its documented input is the training dataset after problematic examples have been identified; it does not in the passages inspected derive a counterfactual nonmonotone semantic selector or optimal bounded-horizon summaries. Cite this distinction, rather than claiming polynomial model repair is new.

Useful tool references: `turn38view1`, `turn41view5`, `turn44view2`. Relevant full-PDF locations: introduction; §3; Eq. 7 and zeroing-out discussion, PDF p. 5.

### 2. F-IVM: regression moments over changing relational training data

Milos Nikolic, Dan Olteanu. **Incremental View Maintenance with Triple Lock Factorization Benefits.** SIGMOD 2018 (author-hosted full version).

Full primary: https://www.cs.ox.ac.uk/dan.olteanu/papers/no-sigmod18.pdf

F-IVM maintains hierarchies of views using task-specific rings and factorization. Section 7.2 explicitly maintains the regression cofactor matrix over joined training data, using a compound aggregate consisting of tuple count, feature sums, and sums of pairwise feature products. Insertions and deletions are signed updates. This is a strong precedent for exact sufficient-statistic maintenance feeding regression training.

Its displayed query language is group-by aggregates over joins. Do not claim it directly implements the earlier-neighbor anti-join selector or the present no-reaccess, bounded-deletion storage result. It does show why “dedup changes training tuples, so update the regression moments” is not by itself a fresh method.

Tool references: `turn38view3`, `turn41view4`, `turn44view0`. Relevant sections: §4 delta rules, §7.2 gradient computation. Official code scope: https://github.com/fdbresearch/FIVM

### 3. DBToaster: higher-order deltas and dynamic state

Yanif Ahmad, Oliver Kennedy, Christoph Koch, Milos Nikolic. **DBToaster: Higher-order Delta Processing for Dynamic, Frequently Fresh Views.** PVLDB 5(10), 2012.

Full primary: https://www.vldb.org/pvldb/vol5/p968_yanifahmad_vldb2012.pdf

DBToaster materializes queries and recursively higher-order finite differences, maintaining each using others. Section 4 proves degree decreases under differencing for the specified fragment without nested aggregates. Section 3 discusses generalized multisets, signed payloads, and expresses difference through nested aggregation; the simple degree theorem should not be extended indiscriminately to these cases.

This anticipates polynomial-style interaction maintenance and state updates. It does not directly state “retain only degree at most a cumulative deletion horizon, then decrement that horizon under substitution.” The latter remains a specialized correctness argument, but its algebraic mechanism is standard.

Tool references: `turn38view2`, `turn44view1`; previous finds `turn33view2`, `turn33view4`.

### 4. Provenance with aggregation and negation

Yael Amsterdamer, Daniel Deutch, Val Tannen. **Provenance for Aggregate Queries.** PODS 2011.

Full primary: https://www.cs.tau.ac.il/~danielde/publications/pods2011b.pdf

Section 5 handles difference via nested aggregation and discusses set-versus-bag semantics. This is more directly relevant to nonmonotone selection than positive semiring provenance alone.

Erich Grädel, Val Tannen. **Semiring Provenance for First-Order Model Checking.** 2017.

Full primary: https://logic.rwth-aachen.de/pub/graedel/ET17.pdf

Sections 2–3 introduce positive/negative token pairs to represent negation and prove specialization commutes with formula evaluation. This establishes a general precedent for provenance of absence. The note’s `(1-x_v) product(x_b)` is a Boolean indicator evaluated in a signed coefficient ring; explain this specific semantics rather than implying ordinary positive natural-number provenance already handles subtraction.

Tool references: `turn44view3`, `turn37view3`, `turn38view5`.

### 5. Ticketed versus central learning-unlearning

Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Ayush Sekhari, Chiyuan Zhang. **Ticketed Learning–Unlearning Schemes.** COLT 2023, PMLR 195:5110–5139. Publisher record: https://proceedings.mlr.press/v195/ghazi23a.html (`turn46view0`).

Full primary: https://proceedings.mlr.press/v195/ghazi23a/ghazi23a.pdf

Definitions 1–2 distinguish central auxiliary memory from per-example tickets. Both give the unlearner the full forgotten examples, not just IDs; ticketed deletion additionally supplies tickets computed from the whole initial dataset. Appendix B.2 proves a central memory lower bound for point functions via one-way INDEX. Thus reconstruction-based memory lower bounds are established in unlearning.

The note’s ID-only rank theorem must retain that contract. The stronger fixed-zero-request ridge construction survives forgotten-payload access. It does **not** establish a central-memory lower bound with unrestricted tickets: tickets for the public blockers can encode the hidden labels. A valid ticketed extension must bound central plus relevant-ticket information.

Tool references: `turn37view1`, `turn41view3`. Relevant locations: Definitions 1–2; Theorem 23 and proof.

### 6. COLT 2025: bounded deletion memory is an explicit prior problem

Yeshwanth Cherapanamjeri, Sumegha Garg, Nived Rajaraman, Ayush Sekhari, Abhishek Shetty. **The Space Complexity of Learning-Unlearning Algorithms.** COLT 2025.

Full primary: https://arxiv.org/pdf/2506.13048

Publisher: https://proceedings.mlr.press/v291/cherapanamjeri25b.html

Section 5 studies known budget k and retains small critical deletion sets. Theorem 5.4 gives central space `O(s_hollow^k k log|Z|)` for realizability testing. Theorem 6.3 gives at least `(1-H(delta))*binom(d,k)-1` central bits for halfspaces with requests of size at most k. Corollary 6.4 gives ticketed complexity `Omega((1-H(delta))*binom(d,k)/d)`. The proof reconstructs a random subset from deletion queries consisting of fixed positively labelled points.

Consequently the current lower-bound method and broad budget-memory theme are known. Differences: their target is realizability testing and hypothesis-class complexity; the note fixes a scalar ridge learner, introduces nonmonotone curation, and seeks an adjacent-budget transition plus per-instance graph rank. Avoid asserting COLT proves exactly the scalar curation result; it does not.

Tool references: `turn38view0`, `turn42view0`, `turn42view1`. The appendix proof uses binomial(d,k), so do not flatten to d^k without qualifications.

## Precise SemDeDup code contract

Official source: https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py

The code computes a pairwise inner-product matrix, zeros its diagonal, applies `torch.triu(..., diagonal=1)`, and takes the column maximum. Removal uses the strict test `M > 1-eps`. Hence, for normalized embeddings and **tau = 1-eps >= 0**, the retained rule is exactly “no earlier raw record has similarity strictly greater than tau.” A selected neighbor is not required. Lower-triangle padding is zero, so for negative tau the mathematical equivalence fails: padded zeros themselves pass the strict threshold. Singleton clusters are specially retained. The usual tau in [0,1] is safe.

Priority order may be reversed or shuffled by options. Graph/order theory requires a fixed retained restriction of whichever order is chosen. Re-fitting centroids, clusters, or corpus-dependent priorities is a different oracle. Tool refs: `turn37view2`, `turn38view6`.

## Recommended positioning and next theory target

Suggested restrained sentence: “Building on provenance-based incremental model maintenance and storage-complexity formulations of unlearning, we characterize counterfactual training information for an earlier-neighbor semantic-curation rule. The resulting storage requirement depends on which omitted records can become eligible within a cumulative deletion budget.”

The report should add the four central references and lower the prominence of elementary expansion/rank lemmas. The highest-value next results are: (i) realistic source-level or geometric restrictions giving nontrivial compression bounds, (ii) a randomized/approximate storage lower bound at an accuracy scale relevant to NLP rather than inverse corpus size, or (iii) measured evidence that naturally occurring blocker structure permits useful exact summaries while deletion-induced additions change task behavior. None follows merely from the present provenance polynomial.

Fresh searches included 2025–2026 curation/dedup/provenance/unlearning terms and data-selection papers. Nearby results include UPCORE, GRACE (arXiv:2608.28361), and forget-set curation; their advertised targets are selecting data *for unlearning*, not repairing the selection that originally generated the training corpus. This is a scope distinction, not an exhaustive novelty claim.
