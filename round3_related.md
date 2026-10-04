# Third-pass adversarial prior-art audit

Audit date: 3 October 2026. Scope: the particular claims in `output/pdf/counterfactual_curation_theory.tex` version 2. This is a targeted primary-source check, not an exhaustive novelty certificate. No main TeX edits made.

## Executive judgment

The narrow candidate contribution remains defensible as a *specialization to finite-horizon ordered suppression*: its blocker-specific rank profile, deletion-clean canonical coefficient state, efficient maintenance contract, and adjacent-budget constant-accuracy ridge obstruction. This audit did not locate a paper proving that exact package.

The following are definitely not standalone novelty claims: dynamic antijoin maintenance; nonmonotone query updates; polynomial provenance; unlearning as history independence; exact ridge add/delete updates or Woodbury updates; an INDEX/entropy decoding argument; or budget-dependent memory lower bounds in unlearning. The version-2 note largely says this already, but needs three closer citation families described below.

The strongest reviewer objection is: “This is a known antijoin followed by a known incremental ridge learner.” The answer must be the *information/access contract*: which information can be discarded before unknown deletion requests, which finite-horizon query family remains reconstructible, and how to remove obsolete information from the summary itself. Deletion-induced admissions and a faster Gram update are insufficient by themselves.

## 1. DBSP is the closest missing query-maintenance reference

**Source.** Mihai Budiu, Tej Chajed, Frank McSherry, Leonid Ryzhyk, Val Tannen. *DBSP: Automatic Incremental View Maintenance for Rich Query Languages*. PVLDB 16(7):1601–1614, 2023. DOI 10.14778/3587136.3587137.

- Archival PDF: https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf
- Longer earlier text, with explicit antijoin section 7.5: https://arxiv.org/pdf/2203.16684
- Retrieved references: `turn51view0`, `turn59view4`.

**What it establishes.** A compositional transformation incrementalizes rich relational queries, including negation, aggregation, and recursion; the archival paper reports machine-checked core theory. The longer version explicitly reduces antijoin to join and set difference (§7.5). Its general language already expresses our selection rule.

**Our inference/application.** With retained relation R and fixed blocker-edge relation E(u,v), selected records are `R ANTIJOIN projection_v(E JOIN_u R)`. Removing a retained blocker can produce positive output changes. Thus “deletion inserts examples” is an ordinary nonmonotone-query phenomenon in database terms. Our earlier-neighbor count buffer is a specialized incremental antijoin baseline.

**Correction.** Add DBSP beside DBToaster/F-IVM and require a strong count/index or database baseline. The claimed advance is a bounded-future summary that can discard payload information and satisfy its own canonical repair contract, not discovery or generic maintenance of nonmonotonicity. No evidence here that DBSP supplies our rank profile or constant-accuracy learner lower bound.

Suggested concise table row: “DBSP / DBToaster / F-IVM: compositional nonmonotone query maintenance, higher-order deltas and sufficient-statistic updates; distinguish the finite-horizon information requirement and canonical summary repair.”

## 2. Exact continual ridge unlearning already has a nearly matching downstream layer

**Source.** Yijun Quan, Wentai Wu, Giovanni Montana. *Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models*. arXiv:2603.12977v3 (14 June 2026); abstract states acceptance at ECML-PKDD 2026.

- https://arxiv.org/abs/2603.12977
- https://arxiv.org/html/2603.12977v3
- Read detailed equations in v1, unchanged central structure in v3: https://arxiv.org/html/2603.12977v1
- Retrieved references: `turn57view0`, `turn57view1`, `turn63view2`.

**What it establishes.** Frozen features, additive Gram and feature-label moments, arbitrary streams of add/delete messages, exact-arithmetic retraining equivalence, and order/client-partition invariance. Variants use SPD solves or Sherman–Morrison–Woodbury updates, with numerical reset discussion. Their displayed objective uses summed loss and fixed ridge coefficient.

**Difference.** Clients supply statistics for actual additions/deletions. It does not answer how a raw deletion *determines and recovers* newly admitted, formerly excluded retained examples without rereading their payloads. Our average-loss normalization adds a count statistic but is routine, not novelty. Our contribution lies upstream and in the future-query state, not the ridge solution or its inverse update.

**Correction.** Cite directly in exact ridge recovery and in the literature table. “Exact frozen-head unlearning” and “signed add/delete ridge updates” must be supporting machinery. Preserve separate costs for recovering the changed corpus/statistics and solving the model.

**Additional direct predecessor.** ACU, arXiv:2505.12239, gives analytic continual least-squares unlearning. The original v1 title/authors are *ACU: Analytic Continual Unlearning for Efficient and Exact Forgetting with Privacy Preservation*, Jianheng Tang et al. (18 May 2025). Latest v2 changed to *Towards Efficient and Exact Forgetting Services in Pre-Trained-Model-based Continual Learning*, Yajiang Huang et al. (5 June 2026). Cite an exact version to avoid mixing bibliographic metadata:

- https://arxiv.org/abs/2505.12239v1
- https://arxiv.org/abs/2505.12239v2
- https://arxiv.org/html/2505.12239v1
- Retrieved: `turn61view0`, `turn61view1`.

No need to make ACU a separate long survey subsection; Quan et al. plus one supporting citation is enough.

## 3. Graph-object exact unlearning: related, but not our selection problem

**Source.** Aditya Gaur and Charu Sharma. *Closed-Form Node Classification with Exact Graph Unlearning*, arXiv:2605.25662v1, 25 May 2026.

- https://arxiv.org/html/2605.25662v1
- Retrieved: `turn57view3`, `turn59view0`, `turn63view3`.

**What it studies.** Graph feature propagation followed by ridge and/or kernel-ridge heads. Proposition 1 handles label, feature, edge, node, and subgraph modifications by re-solving. Theorem 1 states local affected-row sufficient-statistic updates for the shallow ridge components; deeper components use global re-solving.

**Difference.** Its graph changes training representations. Our fixed similarity graph controls which fixed-feature records enter training; no propagation-feature rederivation is required. It does not provide our bounded-budget information characterization or canonical blocker-summary state.

**Correction.** Do not claim that dependency-aware local ridge repair is new. Do not copy its strong floating-point “byte-identical” wording: exact-arithmetic equality and finite-precision tolerance/certification should stay separate in our own note. This audit did not independently validate their theorem or empirical claims; cite as related stated results, not proof machinery for ours.

## 4. History independence is a required conceptual citation

**Source.** Aloni Cohen, Adam Smith, Marika Swanberg, Prashant Nalini Vasudevan. *Control, Confidentiality, and the Right to be Forgotten*. ACM CCS 2023. DOI 10.1145/3576915.3616585.

- Author PDF: https://aloni.net/wp-content/uploads/2023/10/Control-Confidentiality-and-RTBF-CCS-23-FINAL.pdf
- Retrieved: `turn51view1`, `turn52view1`, `turn52view3`, `turn63view0`.

**What it establishes.** Section 3 defines adaptive history independence and relates it to deletion and existing unlearning definitions. It distinguishes logical from physical state and conditions on an adaptive interaction view. Section 3.4 explicitly specializes to updatable machine learning.

**Correction.** Our substitution/composition theorem establishes canonical *logical* state for our particular summary; it should not be marketed as inventing history-independent unlearning. A physical-state implementation needs a separate representation theorem, and adaptive model-only Gaussian output comparisons do not establish a full transcript guarantee. The version-2 scope caveats are correct and should remain.

**Efficient physical-state precedent.** Guy E. Blelloch and Daniel Golovin. *Strongly History-Independent Hashing with Applications*. FOCS 2007.

- https://www.cs.cmu.edu/~guyb/papers/BG07.pdf
- https://research.google/pubs/strongly-history-independent-hashing-with-applications/
- Retrieved: `turn63view1`, `turn61search2`.

This develops strongly history-independent hashing with linear space and constant expected update time under its model. Therefore full sorted serialization is an available export method, not a universal lower bound on implementing physical history independence. Retrofitting our stable handles/inverse sets requires a composition and allocation argument; do not assert that replacing one hash map is sufficient. Keeping physical-state claims outside this paper is sensible.

## 5. Bounded-budget memory obstruction: existing broad idea, narrower new instance

**Source.** Yeshwanth Cherapanamjeri, Sumegha Garg, Nived Rajaraman, Ayush Sekhari, Abhishek Shetty. *The Space Complexity of Learning-Unlearning Algorithms*. COLT 2025.

- https://arxiv.org/html/2506.13048v1
- Retrieved: `turn51view3`, `turn52view2`.

**Specific overlap.** Sections 5–6 study bounded deletion explicitly. Theorem 6.3 has a randomized central-memory lower bound `(1-H(delta))*binom(d,k)-1` for halfspaces with requests of size at most k. Corollary 6.4 extends to ticketed schemes with the appropriate normalization. Their proof reconstructs hidden information through deletion queries.

**Distinction.** Our candidate lower bound concerns a prescribed strongly convex ridge minimizer, a fixed cosine curator, bounded binary labels, constant model-accuracy tolerance, and a separation between adjacent budgets for hidden-label information. Neither budget dependence nor random-access decoding itself is new. The core note already cites this appropriately. The free-public-metadata and all-label-dependent-state accounting must stay in the theorem.

## 6. Other hits that should not expand the main paper

*Natively Unlearnable Large Language Models*, Ghosal, Maini, Raghunathan, arXiv:2606.13873 (https://arxiv.org/abs/2606.13873; `turn49search6`), uses source-specific neural parameter sinks and semantic-overlap evaluation. It is not a finite-horizon semantic-curation repair paper. Source withdrawal alone remains far too broad a novelty claim.

*Data Selection for Transfer Unlearning*, Sepahvand, Dumoulin, Triantafillou, Dziugaite, arXiv:2405.10425 (https://arxiv.org/abs/2405.10425; `turn56view2`), uses static-data selection as a substitute for revocable target data under a relaxed transfer-unlearning definition. It reinforces that data selection and unlearning are already paired; it does not supply our exact curator-rerun target.

## Recommended finalized claim hierarchy

1. **Main mathematical result:** exact characterization of the finite-budget counterfactual statistic family for ordered suppression, with a structural rank computation. Clearly delimit arbitrary scalar-statistic linear sketches versus actual ridge summaries or nonlinear finite-bit encoders.
2. **Main algorithm result:** a canonical coefficient-state update with an explicit cumulative-work bound, exact arithmetic contract, and measured storage including metadata. Do not imply it is rank optimal if it is not.
3. **Main lower bound:** the bounded-label, fixed-threshold, constant-accuracy, adjacent-budget ridge construction. State that it concerns hidden-label information and no rereads/tickets, and cite the standard INDEX/entropy technique.
4. **Corollary/application:** source-collapsed incidence structure and exact ridge recovery. Source rank and numerical linear algebra are established tools applied to this selection family.
5. **Supporting boundary modules:** curation-refit mismatch, convex residual certification, and physically history-independent implementations. Do not add them to the main contribution count.

Suggested one-sentence contribution: “We characterize the information needed to reproduce a fixed semantic-suppression curator followed by additive-statistic learning after a bounded number of raw-record deletions, give a canonical summary repair algorithm, and show that one additional permissible deletion can force linear hidden-label memory even for constant-accuracy ridge recovery.”

Final status: no directly matching complete specialization found in this targeted audit; a mathematical proof audit and natural-language utility evidence remain logically separate from novelty review.

## Late-breaking September 2026 conceptual precedent

Haim Kaplan, Refael Kohen, Yishay Mansour, Kobbi Nissim, Uri Stemmer. *Machine Unlearning as Private Retroactive Algorithms*, arXiv:2609.05329v1 (4 September 2026).

- https://arxiv.org/html/2609.05329v1
- https://arxiv.org/pdf/2609.05329
- Retrieved full text: `turn68view0`, `turn68view1`.

The paper separates counterfactual maintenance from privacy of a release sequence, defines suffix-distribution retroactivity, and combines it with continual differential privacy. Its linear-query, clustering, and histogram constructions do not supply our blocker rank/profile or finite-deletion-budget memory tradeoff. Definition 2.9 explicitly separates marginal approximate guarantees from joint suffix guarantees.

Add this citation to the contract discussion. Our result is exact counterfactual maintenance within a prescribed abstract state, not a guarantee of transcript privacy. The compared logical state includes the remaining deletion horizon: histories with identical retained records but different remaining horizons are not equivalent under our theorem. This is a scope condition, not a new privacy definition.
