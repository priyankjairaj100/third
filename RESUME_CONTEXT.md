# Resume this research project

Last substantive checkpoint: **4 October 2026, Phase 3 empirical preparation complete**.
Repository: **https://github.com/priyankjairaj100/third**.

This is a project handoff reconstructed from the available conversation and the saved research files. It is not a verbatim export of all chats. It deliberately contains only this project's context.

## Paste this into a new chat

> Continue the ACL 2027 counterfactual semantic-curation unlearning project at https://github.com/priyankjairaj100/third. Read README.md, RESUME_CONTEXT.md, THEORY_AND_RESULTS_HANDOFF.md and empirical_execution/CURRENT_STATUS.json, then inspect the relevant code and result files before acting. Preserve the distinction between exact theory, natural-preview engineering evidence, and the unstarted primary semantic study. All work must be carried out in this workspace; do not start paid or external compute without my instruction. Continue the highest-value executable work, keep me informed, and save all project progress back to this repository with an updated handoff. Do not fabricate missing corpus provenance, semantic embeddings, or human ratings.

## User objective and preferences

- The chosen research direction is the **third** proposal from an unlearning research agenda: semantic data curation can make deletion require adding previously excluded surviving examples. It is separate from the supervision-source and quantization projects.
- Working title: **Unlearning What Was Never Trained: Counterfactual Repair of Semantic Data Curation**. Publication goal is ACL 2027; an early user message contained the typo “ACL 207.”
- The user requested several deep theory/algorithm passes, a strong natural-data empirical program, then implementation and memory-algorithm improvements. They prefer completing substantive work over repeatedly proposing the next step.
- No synthetic empirical datasets. Small algebraic/unit-test fixtures are allowed only as clearly labeled software tests, never empirical evidence.
- Everything is to be done here. No remote jobs were launched. No one was contacted for annotation. Historical local hardware availability in unrelated projects is not authorization to use it here.
- The latest instruction authorizes pushing **all project files and full resumable context** to this exact repository, so loss of a chat should not lose the work.
- Do not promise that a paper is “reviewer proof.” State actual strengths, failures, proof assumptions, missing inputs and limits candidly.

## Problem and central distinction

For raw corpus D, curator C and learner A, the target after deleting F is:

    A(C(D minus F))

It need not equal deleting F from the previously selected set and updating that learner. Curation can admit surviving documents that never trained the original model. The initial motivating connected-component example is illustrative; it is not a claim about SemDeDup's actual implementation.

The main fixed-curator contract eventually became **global earlier-raw-neighbor suppression** with frozen representations and priorities. It is not greedy suppression against selected neighbors and not connected-component representative selection. Full corpus-fitted SemDeDup refitting is a separate experimental branch. The source theory and handoff specify the finite deletion horizon, output/access model, eligibility and memory bounds.

## Where we actually are

| Stage | Status and scope |
| --- | --- |
| Theory rounds 1–3 | Saved sources, proofs/qualifications and verification scripts; read later qualifications before citing early claims |
| Initial natural-data execution | Civil100 and News90 connector previews; lexical engineering only |
| Memory repair | Joint feature-span/response-span compression, baseline comparison and byte accounting implemented and audited |
| Strict mode | Exact rational canonical state, history comparisons, residual certificate and bytes-only release implemented |
| Phase 2 | 896 development checkpoints on reused Civil100, with held-out diagnostic labels and independent result audit |
| Phase 3 | Source panels, leakage guards, shared numerical scorer, blinded calibration workflow, exact quality bound and local CLI implemented/audited |
| Primary semantic study | **Not started.** No E5/MPNet embeddings, no original source-rich corpus snapshot and no completed human ratings |
| Paper-ready confirmatory evidence | **Absent.** Do not turn development checks into semantic or source-withdrawal claims |

## Current authoritative files

1. `empirical_execution/CURRENT_STATUS.json` is current. `empirical_execution/STATUS.json` describes an earlier phase.
2. `output/empirical_program/counterfactual_curation_empirical_protocol.tex` is the narrative empirical authority. `study_design.json` omits some narrative nuances.
3. `empirical_execution/phase3/PREPARATION_AMENDMENT.txt` fixes prospective operational choices and records corrections.
4. `empirical_execution/canonical_theory_addendum.tex`, `certified_ridge_contract.txt`, `exact_chart_algorithm_notes.txt`, and `README_strict_mode.txt` describe the strongest implemented numerical guarantee.
5. `empirical_execution/memory_theory_addendum.tex` and `README_memory_repair.txt` qualify the memory story. Low coordinate dimension is not total-byte optimality.
6. Phase-specific `independent_review.txt` and JSON audits establish what was independently checked versus implementation self-checks.

The newest complete user-facing report is `output/pdf/counterfactual_curation_empirical_preparation.pdf`. Previous reports and ZIPs are retained as historical deliverables, not overwritten.

## Facts a new assistant must preserve

### Data and semantic-model boundary

- Civil preview: 100 actual comments with original toxicity labels. File SHA256: `83de928bcece5e074ae669285630571db405fecdd2c48136e37c64ce321c7cc9`.
- It lacks authentic native article/publication/parent/date fields. Source withdrawal is unavailable on this preview; unknown singletons must not substitute for genuine sources in a source arm.
- News preview: 90 actual articles across five observed hosts, with failed pagination recorded. No task labels were invented. It is neither a complete source population nor verified registrable-domain grouping.
- Connector `*_raw.json` means untouched connector response, **not original corpus bytes**.
- Lexical hashing at dimension 768 is still lexical hashing. It is not an E5 model just because the dimensions match.
- No source-rich original corpus, model/tokenizer binaries or pinned semantic embedding cache were present at the last substantive checkpoint. Supported connectors exposed metadata/previews but no usable byte-import route. Direct model/corpus download hosts were outside the supplied network allowlist. Reassess actual capabilities in a future session; do not bypass access restrictions.
- News bodies are omitted from the public backup. Their hashes/acquisition metadata and results are preserved. Some News-specific checks therefore cannot replay from this repository alone.

### Strict mode

- 140 exact checkpoints and 10 history comparisons passed; maximum reported rigorous radius approximately 4.3356e-14. Full deletion gives the zero target.
- Native d=768 strict checkpoint was about 1.514 MB; the 11.053 MB Python-object estimate is not measured peak RSS.
- The exact target uses frozen stored FP32/FP64 values interpreted exactly; it is not legacy BLAS summation-byte equivalence.
- History claims cover the specified canonical abstract state. They do not establish physical erasure, transcript privacy or observational equivalence of all runtime behavior.

### Phase 2 development result

- One fixed reused-Civil split: 82 train / 18 diagnostic evaluation; lexical curator dimension128, threshold0.6; learner dimensions64 and768; lambda0.01.
- Initial selection55, exclusions27, graph edges195. R64/U32/A16 request trajectories, four checkpoints, two dimensions =896 checkpoint evaluations;658 had zero admissions.
- All 12 sampled strict audits passed, maximum radius about4.5091e-14. Independent result audit verified896 rows,8064 metric fields and336 bootstrap intervals.
- There were119 cross-split lexical pairs above the curator threshold. The source/semantic leakage guard was not complete. This is not an untouched-test result.
- Random-deletion effects were small and uncertainty intervals crossed zero. Graph-adversarial requests had larger effects. Initial curation was worse than uncured/hash/mean baselines on this diagnostic split. Retain negative and zero effects.
- A-arm early checkpoints contained repeated deletion sets; repeated copies must not be presented as distinct interventions.

### Phase 3 numerical and sampling amendment

- Calibration, semantic guard and graph construction use the shared `panels.graph_normalize` and `reference_cosines`: frozen FP32 -> FP64, ordered sums of squares, normalization, ordered coordinate dot products, strict `score > tau`.
- This gives pair/context invariance within the locked runtime. Surviving-edge predicates remain the same under deletion, so a fresh retained graph is the induced original graph for fixed vectors/priorities/threshold.
- Primary ID priority is the narrative `SHA256(priority-v1|seed|record-id)`, with ID tie-breaking. Do not silently substitute the earlier engineering core's different salt.
- Exact-text and native duplicate-link precedence is test > calibration > train. Civil parent guard only removes a training child whose observed parent is held out. Semantic E5 guard removes training matches only; MPNet reuses the same retained population.
- Source split is70/15/15; primary/replication pools80/20; whole surviving groups cross panel boundaries in full. Hash serialization and exact quantile comparisons are now explicit.
- Threshold selection:20 bins, up to30 SRS pairs per bin, exact inclusion weights, six categories, three independent human judgments; >=2 positive judgments makes a positive pair. `uncertain` is nonpositive.
- Select the least-strict grid threshold with weighted ratio precision>=0.95 and nonzero support. If none qualifies,0.995 is a failed diagnostic value, not a passed main threshold.
- Independent validation samples up to200 from ALL above-threshold pairs, allows selection overlap and requires fresh blinded assignments. One-sided95% exact finite-population lower bound must be>=0.90; no post-validation retuning.
- Statistical coverage is conditional on fixed/potential majority-protocol outcomes and appropriate independent sampling/rater assignment. It does not cover latent semantic truth or unspecified annotator variation.
- Hashes and completion declarations verify consistency, not authenticity. `confirmatory_study_ready` stays false in the preparation modules.
- All241 engineering sample pairs /723 assignments remain blank. They are lexical plumbing examples, not the main semantic annotation pack.

## Immediate continuation plan

1. Read the handoff and inspect the current checkout. Verify `tools/verify_backup.py` before changing frozen evidence. Establish what assets and runtimes are actually available now.
2. If the source-rich corpus/model assets remain missing, say so and request those concrete files. Do not repeatedly rerun the same lexical pilot as purported new empirical progress.
3. Implement and validate original-corpus adapters when authentic inputs arrive. Civil is the first prepared path; Stack date/tag/link extraction and News title/body/domain/date rules remain unfinished. The legacy News branch of intake v2 is not a finished protocol-compliant News parser.
4. Follow `empirical_execution/phase3/README.txt`: prepare source partitions/fixed guards; bind a pinned E5 cache; generate the actual calibration-only selection pack; collect genuine ratings; freeze threshold; generate independent validation pack; collect fresh ratings; evaluate gate; apply training-only semantic guard; form whole-source panels.
5. Complete label vocabulary/lambda selection, source universes, request manifests, memory/thread/timeout and other pre-execution locks from the original empirical protocol. Then run the primary structure/task/systems program with the prespecified strong baselines and failures retained.
6. Preserve all raw signed effects, zero admissions, failures and costs. Never claim a full-model speedup or practical advantage solely from a small lexical preview or a coordinate count.
7. Update `CURRENT_STATUS.json`, this context file and the evidence-specific documentation. Commit and push the next substantive checkpoint to this repository.

## Practical environment and replay

Original local workspace was `/workspace/scratch/35d4d4d8ba2c`; use repository-relative paths in a new environment. Python3.12.14, NumPy2.3.5, SciPy1.17.0 and scikit-learn1.8.0 were installed. No torch/transformers/sentence-transformers/datasets/pyarrow/onnxruntime or GPU was available for this project's last run. Those are checkpoint facts, not assumptions about every future environment.

`empirical_execution/ccu/native/README.txt` explains the GMP helper and Python fallback. Frozen checks may bind full source-file hashes; modifications require a new version/audit instead of quietly reusing old manifests. Many checks write output JSON, so use an isolated checkout for reproduction.

The backup intentionally excludes caches, temporary PDF render/extraction folders and disallowed News-body redistribution. `BACKUP_MANIFEST.json` records the actual backed-up file set. No API keys, tokens or unrelated personal memories are required to resume.
