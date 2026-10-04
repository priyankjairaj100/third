# Phase 9: adversarial novelty audit

Audit cutoff and retrieval date: **4 October 2026**. Target: ACL 2027. This is a targeted comparison with retrieved primary literature, not an exhaustive novelty certificate or a prediction of acceptance. It supplements the frozen theory and the three earlier literature audits; it changes none of them. The accompanying [source registry](sources/novelty_sources.json) records versions, retrieval limits, and local input hashes.

## Decision

The most defensible main contribution is **a realizable, model-output information requirement for finite-horizon counterfactual semantic curation**. The strengthened candidate asks how much hidden retained feature/response information must survive preprocessing so that later raw deletions can reproduce the prescribed ridge head. Its distinguishing requirements are a shared cosine geometry for curation and learning, a fixed graph across the hard family, nonleaking forgotten payloads, and an adjacent deletion-budget separation. The new private-feature construction is more substantial than multiplying an unconstrained moment-rank lower bound by the nominal number of ridge statistics.

This is a narrow candidate contribution. The general ideas of pipeline-aware deletion, deletion-induced additions, representative retention, provenance-polynomial updates, exact ridge maintenance, history independence, and deletion-budget memory lower bounds all have direct precedents. The mathematical tools for the proposed strengthening are classical. The existing primary semantic study remains unstarted; no retrieved paper comparison establishes the practical prevalence or NLP value of our phenomenon.

## Closest comparisons and what they remove from the novelty claim

| Prior work | Verified overlap | Defensible distinction to establish |
|---|---|---|
| [Cao and Yang, 2015](https://www.cs.columbia.edu/~junfeng/papers/unlearning-sp15.pdf) | Whole-lineage unlearning and maintainable summation forms, including preparation stages. | A particular nonmonotone curator's future information requirement; not the first full-pipeline target. |
| [ARCANE, IJCAI 2022, §3.2](https://www.ijcai.org/proceedings/2022/0556.pdf) | Representative selection reduces retraining. Its accounting treats deletion outside the chosen subset as requiring no retraining. | We prescribe rerunning the curator on retained raw data. This is a different target contract; do not declare ARCANE incorrect under its own contract. |
| [DBSP, PVLDB 2023](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf) | Compositional incremental maintenance supports negation, aggregation, and nonmonotone queries. | A bounded-future summary with restricted payload access and explicit information accounting. Generic update propagation is already solved. |
| [PrIU, SIGMOD 2020](https://arxiv.org/pdf/2002.11791), [F-IVM, SIGMOD 2018](https://www.cs.ox.ac.uk/dan.olteanu/papers/no-sigmod18.pdf), [DBToaster, 2012](https://www.vldb.org/pvldb/vol5/p968_yanifahmad_vldb2012.pdf) | Polynomial provenance for regression updates, regression moment aggregates, and higher-order deltas. | The sparse selector-specific state and its finite-horizon recoverability contract; not provenance or signed moments themselves. |
| [Quan et al., v3, 14 June 2026](https://arxiv.org/html/2603.12977v3) | Exact continual ridge-head maintenance from additive Gram/cross moments with arbitrary add/delete messages. | Derive and recover admissions caused by raw withdrawal when their retained payloads cannot be reread. Count-dependent regularization is routine. |
| [Cherapanamjeri et al., COLT 2025](https://proceedings.mlr.press/v291/cherapanamjeri25b.html), [full version](https://arxiv.org/html/2506.13048v1) | Explicit bounded-deletion space complexity and entropy/reconstruction lower bounds, including realizability testing. | A prescribed ridge minimizer under a realizable semantic-curation trigger, with dimension-dependent hidden payload and adjacent budgets. The broad storage obstruction is prior. |
| [Ghazi et al., COLT 2023](https://proceedings.mlr.press/v195/ghazi23a.html) | Central versus ticketed unlearning distinguishes who holds deletion information. | State whether tickets, public features, request payloads, or external caches convey the hidden variables. They cannot silently be free. |
| [Mirzasoleiman et al., 2017](https://proceedings.mlr.press/v70/mirzasoleiman17a.html), [Kazemi et al., 2018](https://proceedings.mlr.press/v80/kazemi18a.html), [Zhang et al., 2022](https://arxiv.org/abs/2203.01241), [Dütting et al., 2025](https://www.jmlr.org/papers/v26/23-1219.html) | Small deletion-robust summaries, backup items, and memory dependence on deletion budget are established. | Approximate high-value solutions differ from exact recovery of a designated curator and learner. “Keep excluded backups” alone is weak novelty. |
| [Deletion-robust k-center, CIKM 2024](https://researchportal.helsinki.fi/en/publications/coresets-for-deletion-robust-k-center-clustering/), [dynamic consistent submodular maximization, 3 June 2026](https://arxiv.org/html/2606.04946v1) | Geometric robust coresets and dynamic solutions with controlled replacement/recourse. | Our output is a specified counterfactual head, not any sufficiently good clustering or objective solution. |
| [Cohen et al., CCS 2023](https://aloni.net/wp-content/uploads/2023/10/Control-Confidentiality-and-RTBF-CCS-23-FINAL.pdf), [Kaplan et al., 4 September 2026](https://arxiv.org/html/2609.05329v1) | Deletion, history independence, and retroactive maintenance have established conceptual treatments; release sequences require separate privacy semantics. | Byte-canonical logical state is a scoped engineering guarantee. It is not physical erasure, transcript privacy, or a new definition of unlearning. |

The precise database reduction is our application of known machinery: for live relation `R` and fixed blocker relation `E(u,v)`, the chosen corpus is `R ANTIJOIN projection_v(E JOIN_u R)`. Deleting the last live blocker creates a positive output delta. A count/index implementation is therefore an essential strong baseline, not an ablation that can be omitted to manufacture speedup.

## Audit of the existing theory stack

| Component | Novelty assessment | Safe role in the paper |
|---|---|---|
| Activation iff every earlier raw blocker is deleted | Immediate Boolean characterization of the stated rule. | Correctness and implementation alignment. |
| Two-monomial statistic polynomial; horizon truncation; variable substitution | Elementary polynomial/provenance operations. | A useful specialization with explicit finite-horizon and no-reaccess semantics. |
| Sparse indexed repair and amortized accounting | Standard indexing/potential tools; the particular state and accounting are application-specific. | Algorithmic theorem with preprocessing, solver, keys, serialization, and bit costs separated. |
| Query rank through the subset-zeta transform | Standard invertible-transform/rank argument. | Reduction supporting a curator-specific characterization. |
| Blocker-incidence forest, reduced frontier signatures, full rank curve | The strongest existing structural specialization. No directly matching treatment was located in inspected sources. | State and prove the exact selector assumptions. Distinguish record deletion from the more general source case. |
| At most twice the rank in coefficient keys | A meaningful representation bound under the stated incidence model. | Statistic-coordinate comparison only; not total bytes or arbitrary constrained ridge degrees of freedom. |
| Scalar-label adjacent-budget ridge construction | Standard information argument applied to a specific realized pipeline. | Motivating lower bound, strengthened by the new private-feature result. |
| Joint Gram/cross factorization, rational pivot chart, convex release certificate | Linear algebra, canonical coordinates, and convex residual bounds are established tools. | Necessary implementation/theory support, not three additional headline innovations. |

The full-pipeline target remains valuable, but the title “Unlearning What Was Never Trained” should introduce a concrete problem rather than imply that unseen raw records or nonmonotonicity were previously unrecognized.

## New theorem candidates: where the additional substance lies

### Private features and actual model outputs

The Phase 9 construction under separate proof review varies hidden unit feature directions and positive bounded response vectors inside strict graph-margin neighborhoods. Hard requests expose actual rank-one ridge heads while all deleted blocker payloads stay fixed. In the current formulation, with fixed anchor `a`,

\[
C=aa^\top+2\lambda I,\qquad
W_i=C^{-1}w_i\beta_i^\top,
\]

and responses are chosen so this is exactly the prescribed average-loss ridge solution. A sphere patch contributes `d−1` continuous parameters and the response factor contributes `q`; the intended exact-state obstruction is `m(d−1+q)` continuous coordinates. A separated finite packing is intended to yield an `Ω(m(d+q))` bit obstruction at a sufficiently small fixed Frobenius error and fixed regularization.

This closes a real logical gap: positive-semidefinite Gram matrices and rank-one cross moments cannot be treated as independently free coordinates. The output map must actually identify the constructed degrees of freedom. The earlier scalar-label result did not itself imply feature-dimension dependence.

The following are **proof obligations, not literature claims**: nonempty strict-margin patches; a common graph and common public information; injectivity of the head map including sign/scale resolution; explicit error constants and their dependence on regularization; applicability to every required request; and an admissible constant state for the smaller horizon. The continuous dimension statement requires a continuous encoder on a fixed-dimensional state branch. It is not an impossibility for unrestricted discontinuous real encodings. The finite-bit theorem must be stated separately and include all hidden-information-bearing state and side channels.

[Hatcher, Theorem 2B.3](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf) supplies classical invariance of domain. [Scarlett and Cevher, Theorem 1 and §5.2](https://arxiv.org/pdf/1901.00555) supply the standard Fano/packing route. Neither tool is new here. The potential contribution is the simultaneously realizable curation-and-ridge family and its access/budget separation. This audit does not independently certify its proof; use the final Phase 9 proof review for that status.

### Public-feature label rank is a secondary result

For public fixed features and private labels, the query family can induce a linear observable with columns `a_v ⊗ z_v`. Invertible per-query ridge systems preserve the observable's kernel. A row-space sketch therefore gives the usual exact linear sufficient-state characterization. This is a useful specialization, not a new principle of observability.

The generic rank machinery has a **direct precedent**: [Naderializadeh, El Gamal and Avestimehr, v2, 22 April 2015](https://arxiv.org/pdf/1501.07544), Theorem 1 and §III, analyze concatenated matrices with independently row-scaled blocks via matroid union. Our transpose has the form

\[
[\operatorname{diag}(z_{:1})A^\top\ \cdots\
\operatorname{diag}(z_{:d})A^\top].
\]

After replacing `Aᵀ` by a column basis, this is that established matrix family. **Do not claim a new generic Khatri–Rao rank theorem or a new matroid-union formula.** A blocker-forest/frontier simplification or exact ridge-head interpretation may still be useful. Generic independent continuous features are not automatically actual encoder outputs, nor automatically compatible with a graph induced by those same features. Public-feature memory savings also cannot be promoted to total-memory savings when the feature cache is needed at execution.

## Faithfulness to semantic deduplication

[SemDeDup](https://arxiv.org/abs/2303.09540) was first posted on 16 March 2023, with the inspected v3 dated 22 March. The locally vendored official commit `6b4194511202c29b0e1ac8c730996777449ea2a4` computes the strict upper triangle of the within-cluster similarity matrix and its columnwise maximum, then removes entries satisfying `M > 1−eps`. Earlier **raw** neighbors can suppress a later record even when those neighbors are themselves excluded. This supports the chosen fixed-rule model; it does not support a greedy-MIS or connected-component interpretation. Zero padding and singleton handling also matter at unusual thresholds.

The main theorem freezes representation, threshold, partition eligibility, and relative priority. Refitting clusters or corpus-dependent order may violate restriction consistency. [D4](https://arxiv.org/abs/2308.12284), first posted 23 August 2023, explicitly reclusters after deduplication. Thus global fixed-graph repair, frozen official-curator replay, and full corpus-fitted rerun must remain separately named targets. Passing the first does not prove the last.

[Forget Set Curation, 14 August 2026](https://arxiv.org/html/2608.14855v1) maps a suppression request to a forget set; our problem reruns the original training curator after an already specified raw withdrawal. These are related upstream decisions but different mappings. [Natively Unlearnable LLMs, 11 June 2026](https://arxiv.org/abs/2606.13873) concerns source-specific neural parameter allocation, not the specified curator's admission state. Neither permits broad “first curation-aware/source-aware unlearning” language.

## Recommended paper claim and evidence hierarchy

Use three connected contributions: **(1)** an explicitly specified counterfactual curator target and realizable model-output storage obstruction; **(2)** a finite-horizon repair state with the selector-specific structural characterization, supporting algorithms, and honest access accounting; **(3)** a natural-language study that measures when admissions alter retained information and predictions, including structural zeros and failures.

A defensible prospective claim is: “For a fixed ordered semantic-suppression curator, we characterize bounded-future repair state and construct shared-geometry ridge pipelines in which increasing the allowed deletion budget by one forces retention of hidden feature and response information, even though that information was absent from the original trained head.” Add quantitative claims only after the reviewed theorem fixes their precise model.

The paper should not claim total-byte optimality from coordinate rank, universal speedup from algebraic updates, privacy from retraining equivalence, full SemDeDup refit from a frozen graph, or meaningful semantic admissions from lexical engineering checks. The genuine semantic/source and human-quality evidence remains essential for ACL relevance.

## Search limits and remaining novelty risks

Searches used both engines, favoring the stronger engine for authoritative and recent coverage, followed by primary full-text retrieval where possible. Query families covered pipeline unlearning, representative deletion, database/provenance maintenance, deletion-robust summaries, bounded-deletion memory, exact ridge heads, generic row-scaled matrix rank, and 2025–2026 semantic/LLM curation. Some longer exact-term searches returned irrelevant single-word results; direct known-source retrieval and the second engine partly repaired this, but not exhaustively.

Three unresolved leads must remain visible: Leonardo Dominici's 2025 thesis on end-to-end unlearning through preparation pipelines and feature stores (author/lab metadata found, thesis not obtained); [SSRN `7520178`](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7520178), *Predictive State Is Not Explanatory State: An Exponential Memory Separation Between Forecasting and Counterfactual Auditing* (written 24 September and posted 26 September 2026; primary cached abstract verified, full proof unavailable); and arXiv `2608.28361`, *GRACE: Gradient-guided Coreset Selection for LLM Unlearning* (direct primary retrieval failed). The SSRN abstract claims a binary-history family with logarithmic predictive bits versus linear counterfactual bits, including an approximate-recovery separation. This is a concrete conceptual precedent against a general prediction-versus-audit memory claim. Its exact query and success semantics still require the inaccessible full text. The other unresolved retrievals support no theorem-level comparison. The k-center comparison uses the institutional publication abstract; its linked author PDF failed.

Before submission, refresh forward citations and these inaccessible leads, compare the final theorem statement rather than an earlier draft, and correct bibliography versions. In particular, the COLT 2025 space-complexity paper is `cherapanamjeri25b.html`; the `25a` page is unrelated. No matched treatment of the complete narrowly stated private-feature construction was located in the inspected material. That qualified finding does not establish priority or uniqueness.
