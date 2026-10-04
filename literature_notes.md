# Literature audit: counterfactual curation repair

Audit date: 3 October 2026. This is an internal research note. Facts below come from primary papers/code except where explicitly marked as inference or an unverified lead. Search failure is not evidence of novelty.

## Most consequential implementation fact

The released Meta SemDeDup implementation does **not** compute one representative per connected component and does **not** implement greedy maximal-independent-set selection. In `semdedup.py`, its `semdedup` function computes all pairwise cosine similarities, keeps the strict upper triangle, and takes its columnwise maximum. A document is removed if this maximum exceeds `1-eps`. Every earlier document can therefore block a later one, including earlier documents that are themselves removed. `advance_semdedup.py` uses the same rule. Default ordering favors hard examples; the code also supports reversing for easy examples and random shuffling.

Primary code:
- https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py — tool refs turn7view2 and turn9view2; core rule web lines 677–708, removal predicate 811–816.
- https://github.com/facebookresearch/SemDeDup/blob/main/advance_semdedup.py — turn7view4; analogous rule web lines 1048–1057.
- https://github.com/facebookresearch/SemDeDup/ — turn3view0; workflow explicitly computes embeddings, k-means, sorted clusters, then deduplicates.

Our inference: with fixed embeddings, block assignment and priority, the selector is `S={v: P(v)=empty}`, where `P(v)` contains earlier raw neighbors above threshold. After deletion F, a retained excluded v enters iff all P(v) lie in F. Surviving selected documents cannot become excluded in this fixed-state model. This statement does **not** cover rerun k-means or centroid-distance ordering. On the priority path a-b-c with no a-c edge, original selection is {a}; deleting excluded b produces {a,c}. Greedy MIS instead selects {a,c} originally and is insensitive to deletion of b.

## Primary literature and what it rules out

1. **Cao & Yang. Towards Making Systems Forget with Machine Unlearning. IEEE S&P 2015.**
   - https://www.cs.columbia.edu/~junfeng/papers/unlearning-sp15.pdf (turn17view1; turn19view6)
   - Explicitly treats derived data lineage and every learning stage, including feature selection and modeling. Their Zozzle example updates data-dependent features as well as the classifier. Their method transforms computations to maintainable sums where possible.
   - Consequence: do not claim the first pipeline-aware unlearning formulation or discovery that preprocessing must be repaired. Specific non-monotone corpus membership, recoverability, and NLP consequences are the defensible target.

2. **Yan et al. ARCANE: An Efficient Architecture for Exact Machine Unlearning. IJCAI 2022.**
   - https://www.ijcai.org/proceedings/2022/0556.pdf (turn10view2)
   - Section 3.2 selects an informative representative subset using joint entropy; lines 311–324 explicitly split deletion into selected-sample requests requiring retraining and unselected-sample requests requiring none.
   - Consequence: representative selection plus unlearning is prior art. Its stated frozen-subset protocol differs from counterfactual rerunning of a data-dependent curator. Describe the target difference carefully rather than declaring a general flaw in ARCANE.

3. **Abbas et al. SemDeDup: Data-efficient learning at web-scale through semantic deduplication. 2023.**
   - https://arxiv.org/abs/2303.09540 (turn0academia38); official code above provides the audited decision rule.
   - Do not rely on secondary summaries claiming a centroid-nearest representative or retained-neighbor greedy algorithm. The actual released tensor operation is unambiguous.

4. **Tirumala et al. D4: Improving LLM Pretraining via Document De-Duplication and Diversification. NeurIPS Datasets and Benchmarks 2023.**
   - https://papers.nips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf (turn12view1; turn13view0)
   - Applies semantic deduplication and subsequent diversification through embedding-space clustering. Section 4.4.2 directly ablates reclustering between SemDeDup and SSL Prototypes and reports worsened performance without it.
   - Consequence: a frozen-cluster experiment is a deliberate restricted counterfactual, not a full D4 repair. A full-pipeline theorem must certify or repair clustering and ordering as well.

5. **Mirzasoleiman, Karbasi & Krause. Deletion-Robust Submodular Maximization: Data Summarization with the Right to be Forgotten. ICML 2017.**
   - https://proceedings.mlr.press/v70/mirzasoleiman17a.html (turn26view3)
   - Streaming summaries remain useful after selected elements are deleted, with constant-factor approximation.
   - Consequence: backup representatives and deletion-resilient summarization are established.

6. **Kazemi, Zadimoghaddam & Karbasi. Scalable Deletion-Robust Submodular Maximization: Data Summarization with Privacy and Fairness Constraints. ICML 2018.**
   - https://proceedings.mlr.press/v80/kazemi18a.html (turn26view4)
   - Memory-efficient centralized, streaming and distributed methods with approximation guarantees after adversarial deletions.
   - Consequence: distinguish reproducing the output of one specified curator from obtaining any high-objective-value summary.

7. **Dütting et al. Deletion Robust Non-Monotone Submodular Maximization over Matroids. JMLR 26(66), 2025.**
   - https://www.jmlr.org/papers/v26/23-1219.html (turn26view2)
   - Small summaries preserve a high-value independent set after adversarial deletion. Bounds depend on matroid rank k, deletion budget d, and precision epsilon. The centralized nonmonotone result has O((k+d) epsilon^-2 log(k/epsilon)) summary size.
   - Consequence: a generic compact backup-set approximation theorem would face mature direct prior art.

8. **Behnezhad et al. Fully Dynamic Maximal Independent Set with Polylogarithmic Update Time. FOCS 2019.**
   - https://arxiv.org/abs/1909.03478 (turn11view0)
   - Maintains lexicographically first MIS under a random order, with O(log^2 Delta log^2 n) expected time per edge update, plus a high-probability worst-case variant.
   - Consequence: generic dynamic greedy-MIS repair/cascade results are not novel. Also its random-order guarantees should not be transferred to centroid-priority semantic selection.

9. **Blelloch et al. Parallel Batch-Dynamic Maximal Independent Set. arXiv 2604.07515, April 2026.**
   - https://arxiv.org/abs/2604.07515 (turn13view2)
   - Maintains random-order lexicographically first MIS for batches of edge changes using O(b log^3 n) expected work and polylogarithmic depth. Its abstract emphasizes that batch influence is not simply the union of single-update influence.
   - Consequence: any batch cascade claim for an MIS variant must compare to this paper. It does not directly solve the earlier-raw-neighbor SemDeDup rule.

10. **Ginart et al. Making AI Forget You: Data Deletion in Machine Learning. NeurIPS 2019.**
    - https://papers.neurips.cc/paper/2019/file/cb79f8fa58b91d3af6c9c991f63962d3-Paper.pdf (turn26view1)
    - Q-k-means quantizes centroids, saves trajectory metadata, and checks whether a deletion would change a quantized centroid. If yes or an initialization seed is deleted, it retrains; otherwise it updates metadata. Also develops divide-and-conquer k-means.
    - Consequence: margin/stability certificates plus fallback for clustering are prior art. A new full-curation result must compose this with exact selection and downstream training, or handle a genuinely different regime.

11. **Ghazi et al. Ticketed Learning–Unlearning Schemes. ALT 2023.**
    - https://proceedings.mlr.press/v195/ghazi23a.html (turn26view0)
    - Separates centrally stored state and tickets distributed to users and later supplied during deletion.
    - Consequence: declare whether deletion requests carry raw records, identifiers only, or tickets; otherwise memory lower bounds can be vacuous or misstated.

12. **Cherapanamjeri et al. The Space Complexity of Learning-Unlearning Algorithms. COLT 2025.**
    - https://proceedings.mlr.press/v291/cherapanamjeri25b.html (turn24search1)
    - https://arxiv.org/abs/2506.13048 (turn24view0)
    - For realizability testing, proves linear storage can be necessary even when VC and Littlestone dimensions are constant, and lower bounds using eluder dimension. Separates central and ticketed-memory models.
    - Consequence: do not claim the first memory lower bound for unlearning. A curator-specific tight characterization or storage/repair frontier is still a distinct target.

13. **Suriyakumar & Wilson. Algorithms that Approximate Data Removal: New Results and Limitations. NeurIPS 2022.**
    - https://proceedings.neurips.cc/paper_files/paper/2022/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html (turn16view1)
    - Explicitly proves failure of then-existing approximate unlearning algorithms in settings with common cross-validation model selection.
    - Consequence: broad adaptive-preprocessing/model-selection blind-spot claims are not new.

14. **Cohen, Kohen, Nissim & Stemmer. Protecting the Undeleted in Machine Unlearning. FORC 2026.**
    - https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2026.17 (turn23view0)
    - Shows that exact retraining-style deletion can expose retained data in some tasks and proposes a definition protecting the undeleted.
    - Consequence: counterfactual equivalence is an algorithmic guarantee, not a blanket privacy guarantee. Relevant if publishing selection manifests or sequential deletion transcripts.

15. **Lee et al. Deduplicating Training Data Makes Language Models Better. ACL 2022.**
    - https://aclanthology.org/2022.acl-long.577/ (turn0search17)
    - Establishes language-model deduplication motivation and supplies practical lexical deduplication code.
    - Consequence: include lexical/exact deduplication baselines and explain what semantic selection changes.

## Recent nearby leads, with limits of verification

- **What to Forget in Unlearning? Forget Set Curation for Language Models**, Jha et al., arXiv 2608.14855. Paper abstract replicated at https://huggingface.co/papers/2608.14855 (turn20view3); direct arXiv full text retrieval failed. Studies mapping suppression requests to forget sets and CleanSlate verbatim suppression benchmark. Its upstream task is different from retraining-corpus curation, but “first unlearning with curation” would be indefensible. Treat as abstract-level checked only.
- **A General Framework for Dynamic Consistent Submodular Maximization**, arXiv 2606.04946 (turn17academia37). Search found primary record, but direct full-text retrieval failed. Must inspect before strong novelty claims about recourse under insertions/deletions.
- **Natively Unlearnable LLMs**, arXiv 2606.13873, found via a secondary summary mentioning semantic deduplication to identify repeated facts. Primary retrieval failed; do not cite its detailed claims yet.
- **GRACE: Gradient-guided Coreset Selection for LLM Unlearning**, arXiv 2608.28361 (turn24academia51). Abstract indicates selection of forget/retain subsets in gradient space. Distinct from counterfactual recuration; inspect before broad coreset novelty claims.
- **End-to-end Machine Unlearning Through Data Preparation Pipelines and Feature Stores**, Dominici MSc thesis 2025: title found in author announcement; author's primary webpage explicitly describes ongoing provenance/deletion-propagation work in 2026: https://leodom01.github.io/ (turn11view3). No thesis text retrieved, so not evidence of a theorem matching this proposal.

## Research-positioning judgment (our inference)

The useful opening is precise: raw-document withdrawal changes the output of a deterministic semantic curator, including documents that never reached downstream training; characterize exactly which excluded payloads must remain recoverable and certify the induced learner change. The observed SemDeDup rule supplies a particularly clean, faithful model with conjunctive deletion triggers. The difficulty is avoiding a paper that is only anti-join maintenance plus standard ERM perturbation. Stronger candidates are a tight graph/budget-dependent recoverability result, a full-state versus frozen-state stability boundary, and a convincing NLP phenomenon where activation changes predictions beyond near-identical replacement.

Do not promise absence of matching prior work. The search audited several strong adjacent lines, but did not exhaust literature or inspect every recent full text.
