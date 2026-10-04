# Curator and representation protocol for the empirical program

This memo specifies design choices, not experimental results. All datasets must be naturally occurring; generated paraphrases, injected duplicates, relabeled constructions, and synthetic graph fixtures are deferred. The paper's claim is counterfactual fidelity to a declared learning pipeline, with useful NLP consequences tested separately.

## 1. Three targets must never be pooled

**Primary, externally anchored suppression.** Let immutable configuration P contain a pinned public encoder, independently calibrated centroids, threshold, and deterministic priority rule. For every document x, cluster assignment and centroid distance are functions of x and P. Assign to its nearest fixed centroid; sort within each cluster by decreasing centroid distance, then stable record ID; suppress if any earlier raw neighbor has cosine similarity strictly greater than tau. This resembles the released SemDeDup `hard` policy while satisfying restriction consistency. Call it externally anchored semantic suppression, not complete standard SemDeDup refitting.

The immutable calibration corpus is outside the deletion universe and must be explicitly identified and licensed. Use sources and records disjoint from the deletable study corpus, its validation/test sets, and requested source groups. A held-out research-training calibration partition is acceptable as declared permanent reference data, but is not magically public or deletion-free. Record the excluded scope. Do not fit centroids, PCA, IDF, thresholds, class weights, or feature normalizers on the deletable corpus and silently retain them under an end-to-end guarantee. A fixed seed alone does not make a shuffle restriction-consistent: priority is an identifier-keyed total order.

**Secondary, standard full refit.** On a manageable natural corpus slice, rerun clustering, priorities, suppression, and model training on each retained raw corpus using one pinned implementation/configuration. This is a separate oracle. Compare original-corpus-fitted but subsequently frozen curation against that oracle only as a mismatch diagnostic. It is not the theorem's counterfactual learner. Count centroid fitting and graph rebuilding costs. Repeat same-data refits to measure numerical/randomness instability before attributing differences to deletion. Reclustered partitions are compared through selected record IDs, not arbitrary cluster numbers.

**Diagnostic, frozen selection.** Train on originally selected surviving documents. Its discrepancy against the appropriate curation-rerun oracle identifies the contribution of admissions. It is not a faithful unlearning baseline under the paper's target and must be labeled as such.

Primary oracle construction must bypass our polynomial/blocker update code: recompute similarities and the original suppression rule on retained documents. Matching two implementations sharing the same erroneous blocker cache would prove little.

## 2. Implementation faithfulness requires a source-level contract

The released Meta SemDeDup `semdedup.py` computes a columnwise maximum of the strict upper triangle, then removes `M > 1-eps`. Earlier rejected examples still suppress later ones. A greedy maximal-independent-set implementation is different. Pin source commit, comparator, numeric dtype, embedding normalization, all tie rules, and actual within-cluster order. [S1]

Current NeMo Curator documentation describes clustering and within-cluster comparison and exposes `hard`, `easy`, and random policies; its shared duplicate-identification documentation specifies `>= 1-eps`. It also changed its default embedder across releases. Therefore do not use unspecified latest defaults or assume interchangeability with Meta. NeMo is an engineering reference/large-corpus implementation after a differential audit, not an unexamined oracle. [S2,S3]

For the exact main target, compute every within-cluster threshold edge in blocked matrix multiplication; do not cap neighbor count or treat approximate nearest-neighbor retrieval as exact. If approximate retrieval is studied at scale, declare it a separate curator and independently audit missed threshold edges. If bounded-horizon preprocessing short-circuits a count after k+1 blockers, distinguish that sufficient eligibility computation from a complete graph used for full rank curves and audits.

Use stable supplied IDs plus provenance. Do not replace distinct records with one content hash: duplicate instances and different sources must retain separate identities. Normalize text with a pinned, record-local rule. Any prior exact-dedup stage is itself part of the pipeline, not an undocumented preprocessing step that already discards the phenomenon.

## 3. A compact representation design

Recommend **multilingual-e5-base** as the single primary representation (768 dimensions, MIT license), with **all-mpnet-base-v2** as English-only robustness (also 768 dimensions, Apache-2.0). This provides different training families without dimension-confounded memory comparisons. The multilingual E5 card prescribes a `query: ` prefix for non-retrieval applications, average pooling, L2 normalization, and at most 512 tokens. MPNet is intended for sentences/short paragraphs and defaults to truncation at 384 word pieces. Pin revision SHAs, tokenizer revisions, inference library versions, pooling, prefix, normalization, precision, and device. [S4,S5]

Core labeled tasks should use short documents when possible. On full news/web documents, report the truncated fraction and tokens lost. Prefer a predeclared deterministic, non-overlapping, whole-document chunking policy with token-weighted pooling of chunk embeddings then final L2 normalization; use each model's supported window including special tokens. Treat this explicitly as a document representation design, and audit head-only versus pooled full-text curation on the same fixed natural slice. Do not infer full-document semantics from an undocumented first-512-token truncation. No corpus-fitted pooling or PCA is permitted in the core fixed contract.

Keep the experiment matrix restrained: the primary representation covers all core datasets; the second covers two English datasets and one full-refit audit. On one fixed English dataset, run the 2-by-2 curation-encoder/learner-encoder cross: E5/E5, E5/MPNet, MPNet/E5, MPNet/MPNet. This tests whether the NLP effect depends on sharing an embedding space between deduplication and prediction. This cross is a robustness check, not four new headline benchmarks.

## 4. Prespecify thresholds without choosing favorable deletions

Use a calibration-only retention target, e.g. 90% as the primary operating point and 80%/95% as sensitivity, with exact tie behavior and attainable-retention intervals. These numerical targets are proposed choices to lock before running final data, not properties of SemDeDup. Calibrate separately by encoder on the immutable reference corpus, then freeze the numeric thresholds for every deletion run. Record actual retained fractions in final corpora; do not retune to equal final-corpus retention after observing results.

Do not choose thresholds, cluster counts, or priority policies using activation frequency, compression ratios, model discrepancies, or deletion outcomes. A blinded calibration human audit of suppressed pairs should establish that the selected operating point is recognizably semantic deduplication rather than topical pruning. If the primary target fails a preregistered duplicate-quality gate, report that failure and use a previously declared conservative operating point, not a newly searched threshold. Record the rule and all failed candidates. Since even high cosine pairs can differ in negation, numbers, entities, or labels, similarity alone is not a quality check.

Use a fixed external cluster count selected for memory feasibility and quality on calibration data. Do not retune k-means cluster count with retained N after deletion. Audit cross-cluster high-similarity misses on one exact all-pairs natural subset to expose the partition approximation. The target remains the declared within-cluster curator, so these misses are quality limitations rather than algebraic correctness failures.

## 5. Negative controls establish the mechanism

1. **No curation:** deleting previously excluded records is inapplicable; ordinary ridge removal provides the implementation/performance floor.
2. **Exact normalized-text deduplication with stable representative:** deleting only unselected instances cannot admit new examples. Selected representative deletion can substitute a duplicate, usually with identical task features, while label conflicts are reported.
3. **Greedy independent-set curation using the same graph and order:** deleting only currently unselected records cannot change its selected set. This is a specific mechanism control. For general selected deletions, greedy changes can cascade and the blocker theorem does not apply. Do not benchmark our method against its oracle as though they implement one task.
4. **Label-blind random subset of the same initial size:** compare downstream task utility and cost to see whether semantic pruning has a reason to exist, not as an unlearning oracle.

Main claims require natural excluded-record or excluded-source admissions in raw-neighbor suppression, along with task-relevant changes. Counts alone are weak evidence: exact copies can change membership while leaving the optimum unchanged.

## 6. Task definition and normalization details

Use gold labels already present in the task datasets. Never use an LLM to invent missing ground truth for the primary learner benchmark. For multi-class tasks train one-vs-all or matrix ridge on fixed one-hot targets. State whether an intercept is absent or penalized; an unpenalized intercept requires a corresponding separate theorem/solver derivation. Prefer a fixed penalized intercept feature for the main exact contract.

Pin lambda using an independent calibration/validation contract once, with a small specified grid, and hold it fixed across deletions. A validation search after every request defines a different pipeline. Use average-loss ridge consistently: the system is `M + lambda*n*I`. Changing n changes every diagonal entry; do not claim a rank-one system update after an admission. For class imbalance, primary training uses uniform example weights and reports macro-F1, balanced accuracy, per-class recall, and ordinary accuracy. If class weights are included, they must be fixed externally; recomputing from surviving labels changes all retained statistics and is outside the simple implementation.

Report oracle fidelity independently of task benefit: relative parameter error, normalized moment error, test logit RMS, disagreement, retained-objective excess, and the task metric versus fresh retained-data retraining. Recovery need not improve ordinary accuracy after every request. Query selection may use metadata or graph structure but must not be tuned on test labels. Test examples close to activated records may be reported as a prespecified slice with the full-test result alongside it.

## 7. The main memory risk should be visible from day one

For C labels and d downstream features, packed ridge statistics have `s=d(d+1)/2+d*C+1` values per coefficient key. This can be much larger than storing d-dimensional embeddings plus labels for activation-eligible records. Any total-memory claim must beat the strong eligible-payload/count baseline at actual byte precision, including keys, inverse lists, identifiers, cached encoder weights, and transient peaks. Coordinate-rank optimality does not imply byte-optimality for constrained ridge moments.

A fixed public projection to d=128 can be a declared compact-head setting; full 768-dimensional heads must appear in a prespecified audit and payload baselines use the exact same projected features. Never fit projection/PCA on deletable data or silently reduce our model dimension only. Plot model utility versus bytes across the prespecified two dimensions; report cases where summary storage loses. Curation can still use full 768-dimensional embeddings in both cases. A compressed-head speedup alone does not establish a practical advantage of the theorem.

## 8. Human interpretation and provenance

Adjudicate a prespecified stratified probability sample of natural admitted-document/former-blocker pairs, blinded to method, model change, and deletion-selection regime. Categories: substantively same information; new task-relevant fact or distinction; contradiction/negation/entity/number difference; topical similarity only; uncertain. Separately compute existing gold-label disagreement; do not use that as the entire semantic definition. Sample across similarity, source, language, and deletion regime and weight estimates back to their sampling population. Publish counts, agreement, adjudication procedure, permitted example IDs/excerpts, and uncertainty. Examples in the paper must follow the sampling rule, not merely showcase dramatic cases.

Whole-source withdrawal is meaningful only where source identifiers correspond to documented ownership/collection units; domain is not automatically copyright ownership and reviewer/author ID is not automatically consent. A partition-source model needs one declared source per record. Multisource ownership is outside the present theorem. Source-level calibration must not include the supposedly withdrawn source as immutable context.

## Primary sources checked

[S1] Meta SemDeDup released source, `semdedup.py`, strict-upper-triangle maximum and strict comparator; hard/easy/random selection paths: https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py . Web ref `turn84view0`.

[S2] NVIDIA NeMo Curator text semantic-dedup documentation, current page reports v1.3.0 (26.07), clustering workflow, parameters, file-based fitting subset semantics and changed embedder default: https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup . Web refs `turn91search2`, `turn100view2`. Historical v25.09 defaults: https://docs.nvidia.com/nemo/curator/v25.09/curate-text/process-data/deduplication/semdedup . Ref `turn91search5`.

[S3] NVIDIA shared semantic deduplication workflow documentation describes `>=` comparator: https://docs.nvidia.com/nemo/curator/curate-video/process-data/dedup . Refs `turn91search1`, `turn91search4`. This is a documented discrepancy to verify against the pinned text implementation, not proof that every release uses identical code.

[S4] Multilingual E5 model card: https://huggingface.co/intfloat/multilingual-e5-base . Refs `turn84view2`, `turn88view2`. Technical report: https://arxiv.org/abs/2402.05672 . Ref `turn82academia37`.

[S5] MPNet sentence embedding model card: https://huggingface.co/sentence-transformers/all-mpnet-base-v2 . Refs `turn84view3`, `turn88view0`, `turn88view1`.
