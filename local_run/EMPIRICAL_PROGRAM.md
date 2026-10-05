# The empirical program to run locally

Date: 5 October 2026. Target: ACL 2027.

This is the complete execution handoff for the existing prospective study.
It does not add a scientific allocation or report new measurements.
At handoff, **zero primary semantic jobs** and **zero genuine human judgments** exist.
The saved theory and lexical engineering checks do not count as primary empirical evidence.

Start with `START_HERE.md` in this directory.
Use `INPUTS_AND_PREPARATION.md` for acquisition and preparation.
Use `EXECUTION.md` for commands and their actual boundaries.
Keep private inputs and outputs under the ignored `local_workspace/` directory.
Never move a reviewed input root afterward.

## 1. What the experiment must establish

The target after deleting raw records F is `A(C(D minus F))`.
It includes surviving records newly admitted by curation.
Deleting only originally selected records generally defines a different target.

| Question | Evidence required | Necessary comparison |
|---|---|---|
| Do admissions occur naturally? | Frequency, number and distribution, including zeros | Uniform records and genuine sources on unchanged natural text |
| Do admissions matter for NLP? | Prediction changes, signed held-out loss changes and blinded text judgments | Correct recuration versus frozen-selection retraining |
| Does repair recover the target? | Every applicable membership, moment, count, horizon and head check | Independent selection and fresh retained-state reconstruction |
| Does the algorithm offer value? | Complete lifecycle time, physical memory and failure rates | Optimized compact eligible-payload baseline B-E |
| Does the fixed-curator scope matter? | Selected-set and prediction discrepancies after refitting | Separate frozen-fitted and full-refit SemDeDup branch |

A favorable result is not a condition for retaining an experiment.
Do not tune thresholds, populations or requests until our method wins.
Zero effects, unfavorable memory and slower repair remain results.

## 2. Authority, registry and count reconciliation

Read the narrative first:
`output/empirical_program/counterfactual_curation_empirical_protocol.tex`.
Apply the explicit Phase 3 preparation and Phase 4 News calendar amendments.
Apply the later Phase 6 recipe book for its resolved allocations and policies.
Phase 9 supplies the current assembly and evidence interfaces.

`EXPERIMENT_MATRIX.json` binds all these files by SHA256.
It includes all 43 group definitions, 21 recipes, method lists and allocations.
The expanded executable job list remains:
`empirical_execution/phase6/results/recipes_release/jobs.jsonl.gz`.
Its matching `registry.json` and `recipes.json` are authoritative.

| Accounting item | Jobs |
|---|---:|
| Original Phase 5 experiment rows, preserved | 16,912 |
| Original unresolved placeholders, replaced | 21 |
| Explicit Phase 6 extension jobs, added | 4,420 |
| **Current complete registry** | **21,332** |

The earlier 16,933 total included the 21 placeholders.
Do not add those placeholders again.
A job usually represents one paired trajectory containing several method workers.
Some jobs represent repetitions, human stages or analyses.
A job is not one head release, one method call or one independent replicate.

The design was not submitted as an external preregistration.
Hashes and a complete prospective registry do not make it one.
Every planned row remains visible, including blocked, failed, empty and unavailable rows.
Actual horizons and request IDs appear only after authentic populations are frozen.

## 3. Acquire exactly these natural datasets

| Dataset | Records and task | Source unit | Required role |
|---|---|---|---|
| Civil Comments, TFDS `CivilComments` 1.2.4 | Own comment text; original toxicity fraction | Publication-qualified article ID | Core regression, structure and article withdrawal |
| Ask Ubuntu, corrected company-issued April 2024 dump | Question title and body; original top-20 tag indicators | Site-qualified account ID | Core multilabel task, structure and contributor withdrawal |
| CC-News extracted release | Original article title/body/URL/domain/date; no task labels | Registrable domain | Core structure and linguistic audit |
| English Language & Usage, matching Stack archive policy | Question title/body and original tags | Site-qualified account ID | Prespecified same-platform replication |
| WCEP-100 | Article instances within complete event groups | Record; domain only with verified URLs | Event-enriched structural replication |

Civil must include genuine article/publication/parent/date fields.
The old Civil100 preview lacks this source evidence.
Civil article withdrawal is not author withdrawal.
Keep crowd toxicity fractions for training.
The secondary binary endpoint uses toxicity at 0.5.

For Stack, use `Posts.xml` and `PostLinks.xml`.
Keep `PostTypeId=1` questions created from 2018-05-02 through 2023-12-31.
Do not concatenate answers or substitute the old duplicate-retrieval benchmark.
Missing and community owners become separate unknown-source singleton records.
They stay in the dataset but never become genuine source-sampling units.
Accounts are contributor proxies because questions can have collaborative edits.

For CC-News, freeze the actual extracted population and all acquisition hashes.
Only a documented original CC-NEWS WARC window is a predeclared replacement.
A replacement defines a different extraction population and needs its own manifest.
WCEP without verified source URLs remains available for record-only analysis.
Its unavailable source and matched-source jobs stay in the ledger.

Do not replace a dataset because redundancy or effect size is weak.
Do not publish News bodies, connector copies or unnecessary personal metadata.
Retain record IDs, permitted metadata, hashes and acquisition instructions.
Record licensing and access limits for every release.

## 4. Freeze parsing, populations and leakage guards

Use the existing adapters and replay acceptance.
Normalize text with NFC and whitespace collapse.
Preserve case, punctuation, negation, numbers, quotations and code.
Unique record IDs remain distinct even when text is identical.

Civil uses its own text only.
Stack uses title plus parsed body.
Remove only the frozen markup and platform-boilerplate rules.
News uses title plus body under the documented repeated-title rule.
Log empty or unparseable records without outcome-dependent filtering.

Source hashes assign **70% training, 15% calibration and 15% test**.
The same assignment applies to explicitly identified unknown singletons.
They do not become authentic provenance groups.

Apply these fixed guards before any confirmation state:

1. Exact normalized-text and native duplicate-link precedence is test, then calibration, then train.
2. Civil removes a training child with an observed held-out parent.
3. Missing parent identifiers never become shared parents.
4. At the locked E5 threshold, remove training-side semantic matches to held-out records.
5. Keep the resulting population identical across the E5/MPNet cross.
6. Report the MPNet cross-split audit without changing that population.

News has a separate cohort rule.
Use calibration-assigned domains from 2017 for calibration.
Use train-assigned domains from 2018–2019 for analysis.
Reserve test domains and log other excluded domain/year combinations.
Do not replace these cohorts with generic random splitting.

Reserve 20% of eligible training source groups for replication.
The remaining 80% supply nested primary targets of **10k, 25k, 100k and 200k records**.
Use independent fixed hash order and include each boundary group completely.
Never cut or skip a large source to meet a target.
Report actual counts, overshoot, shortage and source-size quantiles.
The second 10k panel uses the disjoint replication pool.

Full-refit panels use separate whole sources outside the largest primary and replication panels.
Their soft target is 5k.
The fixed selection salt is `ccu-v1-refit-panel`.
A shortage is not permission to reuse sources.

WCEP includes whole events in fixed hash order up to 25k articles.
The News calendar panel uses post-guard counts and verified complete months.
Choose the earliest qualifying complete month from January 2018 onward.
If none qualifies, use the complete chronological prefix, stopping at the first coverage gap.
Never bridge a gap or select a later favorable window.

Freeze a genuinely disjoint development population before examining effects.
The frozen protocol does not give one universal numerical pilot quota.
Development counts must support the requested resource profiles and precision analysis.
Record the chosen scope prospectively instead of presenting an invented quota as registered.
Keep every development record and source outside the confirmation populations it qualifies.

## 5. Produce the actual semantic representations

Primary encoder: `intfloat/multilingual-e5-base`.
Robustness encoder: `sentence-transformers/all-mpnet-base-v2`.
Both use 768 output coordinates.
Pin complete model/tokenizer revisions, file hashes and runtime versions.
Use the asset manifests in the preparation guide.

For E5, use `query: `.
For MPNet, do not add that prefix.
Split each document into nonoverlapping chunks of at most 256 content tokens.
Reserve space for prefixes and special tokens.
Encode every chunk.
Pool with content-token-count weights, then normalize the document vector.
Report token lengths, chunk counts and first-window truncation fractions.

Freeze the stored FP32 cache and replay every row through the accepted encoder process.
A matching cache hash alone does not establish genuine model execution.
Never substitute lexical hashing or another encoder under an E5/MPNet label.

Graph scoring promotes stored FP32 coordinates to FP64.
Use the shared ordered normalization and pairwise scoring implementation.
The graph predicate is strict `score > threshold`.
Use the actual frozen scorer, including its documented absence of score clipping.
Priority is `SHA256(priority-v1|seed|record-id)`, with ID ties.
Seed 0 is primary; seeds 1 and 2 are separate sensitivities.

The primary curator suppresses against every earlier **raw** neighbor.
An excluded earlier record can still suppress another record.
It is neither selected-neighbor greedy suppression nor connected-component deduplication.
Use an exhaustive graph; no hidden ANN or top-k truncation is allowed.
Large exact-graph failures remain scale failures.

Learner vectors are the same stored FP32 vectors, without method-specific renormalization.
Primary dimension is 768, with no intercept.
Projection dimensions 32/64/128/256 are separate learner ablations.
They keep the curation graph unchanged and use the fixed public projection seed.

## 6. Collect genuine calibration evidence before effects

There are five required corpus/curator calibration pairs:

- Civil/E5 and Civil/MPNet.
- Ask Ubuntu/E5.
- CC-News/E5.
- English Language & Usage/E5.

WCEP inherits the frozen CC-News/E5 threshold.
Ask Ubuntu does not require an MPNet curator under this bounded program.

For each required pair:

1. Enumerate the actual source-disjoint calibration pair population.
2. Sample up to 30 pairs in each of 20 equal-width score bins over [-1, 1].
3. Census bins containing fewer than 30 pairs; retain inclusion weights.
4. Obtain three independent blinded judgments for every selected pair.
5. Only `exact_copy` and `substantially_same_meaning` are positive categories.
6. At least two positive judgments make a positive pair; `uncertain` is nonpositive.
7. Select the least strict grid threshold with weighted precision at least 0.95 and nonzero support.
8. The grid is 0.50 through 0.99 in 0.01 steps, plus 0.995.
9. Independently sample up to 200 pairs from **all** above-threshold calibration pairs.
10. Allow sample overlap but obtain fresh blinded assignments.
11. Require the one-sided 95% exact finite-population precision lower bound to reach 0.90.

A failed selection keeps 0.995 only as a failed diagnostic value.
It is not a successful threshold.
An empty positive population cannot pass precision validation.
Do not retune after independent validation.
Any development-only sample enlargement requires a recorded pre-confirmation amendment.

Civil/E5 threshold sensitivities use separate 200-pair validation samples at clipped threshold ±0.02.
There are at most 12,000 judgments across the five selection/validation pairs.
These two sensitivity audits add at most 1,200 judgments.
These maxima assume full pair populations; shortages remain visible.
Neither count includes the separate admission audits below.

A local LLM can prepare forms and validate files.
It must not fabricate independent human responses, provenance reviews or collection testimony.
Document actual annotator qualifications, consent, compensation, skips and independent assignment.

## 7. Freeze task settings on calibration only

Use average-loss ridge with all coordinates penalized:

`(sum(z z^T) + lambda * current_selected_count * I) * head = sum(z y^T)`.

Empty selection gives the zero head.
The current count includes admissions and removals.
Do not retain the original count in the regularization term.

Choose lambda from `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`.
Use five source-group calibration folds and unweighted squared loss.
Exact loss ties choose the larger lambda.
At least five genuine source groups are required.
Each learner representation receives its own calibration lock.
No deletion outcome or test label can choose lambda.

Choose Stack's 20 most frequent tags on calibration only.
Break tag-count ties lexicographically.
Keep questions with no chosen tag.
Use the existing out-of-fold F1 threshold policy for each output.
Exact F1 ties choose the strictest threshold.
Zero-positive outputs use a never-positive rule.
Logistic thresholds use logistic probabilities, not ridge scores.

| Task | Primary metric | Secondary metrics |
|---|---|---|
| Civil | MSE against original toxicity fractions | MAE, AUROC and AUPRC at the fixed binary endpoint |
| Stack | Mean squared loss over records and outputs | Micro-F1, macro-F1, mean average precision, per-tag counts |
| News/WCEP | No invented supervised target | Structural and human interpretation outcomes |

Do not clip ridge scores for primary MSE.
Before deletion, compare curated ridge, uncurated ridge, same-size hash ridge and converged curated logistic.
These are utility references, not alternative unlearning targets.
Keep poor initial utility visible.

## 8. Requests and service contracts

| Arm | Request law | Initial horizon | Main checkpoints |
|---|---|---:|---|
| R | Uniform permutation of all record IDs | min(128, N) | 1, 8, 32, 128 |
| S | Uniform permutation of genuine source IDs | min(8, genuine-source count) | 1, 2, 4, 8 |
| U | Uniform permutation of initially excluded record IDs | Record horizon, clipped to that pool | 1, 8, 32, 128 |
| A | Label-blind blocker search on real records | Record horizon | 1, 8, 32, 128 |

Clip and deduplicate checkpoints to the actual feasible horizon.
Retain empty pools and empty retained targets explicitly.
For A, sample up to 1,024 eligible excluded candidates per fixed seed.
Choose the blocker set producing most admissions, breaking ties by stable ID.
Append the fixed random record order to finish the trajectory.
Charge this search separately.
An absent candidate is not permission to choose from model effects.

Every S path has a separate record-volume-matched R path.
Use one independent record permutation at the S path's cumulative raw deletion counts.
Build its own sufficient initial horizon.
Its methods match the parent S method list.
Do not count matched controls as extra primary R replication.
Repeated audit/timing references do not create new matched paths.

Every trajectory resets once to its original state.
Do not reset between its checkpoints.
Remaining budget subtracts new unique deleted units.
Retries do not replenish or consume additional budget.
Check batches, reverse order, resume, duplicates, invalid IDs and exhaustion atomically.
Fresh state comparisons use the same **remaining** horizon.

Primary requests contain identifiers only.
Repair workers cannot reread undeclared raw records or feature arrays.
The common output is the head only.
Oracle-selected IDs remain isolated audit outputs.
Count any declared retained payload, graph, cache, metadata and workspace.

## 9. Run the complete A–G matrix

Counts below are per named corpus and panel, before clipping.
“First” means stable run-ID order, never favorable outcomes or completion order.
Each S allocation additionally carries its matched-volume R allocation.

| Block | Population/configuration | Primary trajectory allocation | Workers or outputs | Registry jobs |
|---|---|---|---|---:|
| A | Civil, Ask Ubuntu, CC-News; each nested 10k/25k/100k/200k panel | 256 R, 256 S, 128 U, 32 A | Independent structure oracle | 11,136 |
| B | First Civil and Ask Ubuntu 10k panels; native dimension | 256 R, 256 S, 128 U, 32 A | O-G and B-F relevance | 1,856 |
| C | Same labeled 10k panels | First 64 R, 64 S, 32 U, 16 A | All eight main methods; timing repeats below | 640 |
| D | Second 10k panels for all three core corpora; English 10k; WCEP whole-event 25k | 64 R, 64 S where genuine, 32 U | O-G/B-F for labeled tasks; structure for News/WCEP | 1,120 |
| E | Separate Civil and Ask Ubuntu 5k panels | 32 R, 32 S, 16 A | Full-refit, frozen-fitted and frozen-selection branch | 240 |
| F | First Civil and Ask Ubuntu 10k panels | 32 R, 32 S | Fresh logistic optimum, eligible retraining, certified convex repair | 192 |
| G | Fixed Civil 10k and fixed News calendar window | 32 shared R and 32 shared S per alternative | Listed boundary workers | 1,728 |
| **A–G subtotal** | **43 groups** | | | **16,912** |

B has 2,688 unclipped releases per corpus per method before matched controls.
C has 704 before matched controls and timing repeats.
Reuse identical O-G/B-F outcomes across blocks without claiming independent replication.

### Main method list

| ID | Role and access |
|---|---|
| O-G | Fresh retained-data global curation, fresh moments and common solver |
| O-T | Receives correct selected IDs; rebuilds moments and head |
| B-E | Compact eligible features/labels, blocker incidence, buckets and signed batched updates |
| B-A | Same antijoin machinery with all current feature/label payload |
| P-I | Proposed indexed finite-horizon coefficient summary |
| P-S | Full-scan substitution version of the summary |
| P-R | Rank-coordinate summary with its actual decoding and scan costs |
| B-F | Originally selected survivors only; explicitly wrong-target diagnostic |

B-E must receive compact incidence structures and batched numerical operations.
Do not force it to retain per-record outer products.
Match allowed compaction and tombstone policies across comparable methods.
A win against O-G alone does not establish strong incremental superiority.

### Correctness and repetitions within C

Check all core releases against independently reconstructed selection and target moments.
Check B-E's selected and eligible membership.
Summary methods are not assumed to decode selected IDs.
On the first 16 R and 16 S paths, rebuild the full designated state at every checkpoint.
Compare every designated key and moment coordinate at the remaining horizon.

On the first eight R and eight S paths, run five fresh-process timing repetitions.
These 80 timing jobs per labeled corpus are included in C's total.
Keep repetitions nested within their original trajectory for inference.
Use one CPU thread and seed-locked method order.
OS page-cache state is uncontrolled and must be reported.
Fresh process is not proof of uncached disk.

### E: full-refit boundary

Use official pinned SemDeDup with 71 clusters, 25 iterations and spherical clustering.
Use hard policy, one thread and seed 0.
The epsilon is one minus the actual calibrated threshold.
Qualify these defaults on disjoint development before confirmation.
A retained corpus below the cluster count yields explicit failure, not adaptive cluster reduction.

Each branch trains its own original head.
Compare selected IDs and predictions, not arbitrary cluster labels.
For the first two R and two S paths, evaluate every individual service step.
Also retain five no-deletion fits with the same seed/hardware.
Retain three no-deletion fits with alternative seeds 1, 2 and 3.
These eight repeats per corpus are included in E's total.
Charge clustering, ordering, suppression and downstream learning.

### F: convex extension

Civil uses fractional-label logistic loss.
Ask Ubuntu uses independent binary logistic outputs.
Do not reuse ridge moments as if they encoded logistic training.
Charge retained gradient/Hessian queries and feature/label reads.
Keep the exact verifier's parameter-radius gate at 1e-8.
The default verification cap is 2,000,000 `n*d*q` coordinates.
Native larger caps require the bound observed development qualification.
A cap refusal is not an observed timeout.
Pointwise verifier qualification does not qualify optimizer or full-trajectory runtime.

### G: all 18 named alternatives

Civil has 16 named configurations:

- Four learner projections: 32, 64, 128 and 256.
- Four curator/learner cells: E5/E5, E5/MPNet, MPNet/E5 and MPNet/MPNet.
- Two priority seeds: 1 and 2.
- Two threshold variants: clipped primary threshold minus/plus 0.02.
- Two lambda factors: 0.1 and 10.
- FP32 state.
- Common zero-start CG.

Projection, lambda, FP32 and CG variants include all eight main methods.
Encoder, priority and threshold variants use O-G/B-F.
Reuse the E5/E5 native cell appropriately without independent-replication claims.
Every FP32 method must identify its actual state dtype.
FP32 results do not inherit exact FP64 arithmetic guarantees.
CG starts at zero, has no preconditioner and allows at most `10*d` iterations per output.
It uses the same numerical residual gate.

News adds two configurations: ANN versus exhaustive graph, and calendar window versus source panels.
The pinned ANN policy uses eight tables, ten bits and seed 202710041.
Its audit uses sample size 200 and seed 202710042.
Include nonretrieved pairs in the miss audit.
Report edge recall, complete-blocker rate and selection effects.
High edge recall is not exactness.
Do not create a full factorial across these factors.

## 10. Run all 4,420 extension jobs

| Extension | Exact frozen allocation | Jobs |
|---|---|---:|
| Source mass-PPS | Each of 12 structure panels: 64 weighted source paths plus 64 matched R paths | 1,536 |
| One-percent horizon | Each structure panel: 64 R, 32 U, 16 A; fresh horizon `ceil(N/100)` | 1,344 |
| Excluded-blocker stress | Each structure panel: 32 A paths restricted to initially excluded blockers | 384 |
| News chronology | One complete-day stream, cumulative record budget 128 | 1 |
| Fresh graph audits | Each core 10k: 8 R, 4 S, 2 U, 2 A; every registered checkpoint | 48 |
| Cached O-G tier | Each labeled 10k: first 8 R and 8 S, five repeats | 160 |
| Initial utility references | One complete four-reference panel per labeled 10k | 2 |
| Negative controls | Core 10k: 32 own-excluded paths each for exact-text and greedy; labeled 10k: 64 no-curation R | 320 |
| Count normalization diagnostic | Civil R000, final feasible checkpoint | 1 |
| Valid graph envelopes | Each core 10k: 32 R, 32 S and matched R | 288 |
| Every-release persistence | Each labeled 10k: first 8 R and 8 S, five repeats, all eight methods | 160 |
| Warm service | Same timing subset and repetition count, all eight methods | 160 |
| Human threshold stages | Five selection/validation pairs plus two Civil sensitivity validations | 12 |
| Human admission analysis | One complete balanced pair dossier | 1 |
| Human context analyses | Separate former-neighborhood and nearest-survivor dossiers | 2 |
| Statistical analysis | Complete bound outcome/evidence dossier | 1 |
| **Total** | | **4,420** |

PPS samples remaining genuine sources proportionally to their original record mass.
Use exact integer tickets; do not pool this law with uniform S.
The one-percent checkpoints are clipped, deduplicated values from `[1,8,32,128,K]`.
It requires a separately constructed state, not horizon extension after requests.

Chronology stops before the next whole day exceeds 128 records.
It neither splits nor skips that day.
One stream supports descriptive findings, not hundreds of independent replicates.

Negative curation controls delete from their own initially excluded sets.
An empty excluded pool remains unavailable, not a successful invariant test.
The normalization diagnostic stays at R000 even if its count effect is zero.
Do not search for a more favorable request.

Graph envelopes require uniformly valid bounds for the actual scorer and independent interval audit.
ANN recall or observed maximum drift cannot replace that guarantee.
Retain failed or uninstantiated status if the assumptions fail.

Primary persistence is final-checkpoint only.
The sensitivity persists every release, including serialization, writes, fsync and compaction.
The frozen compaction fraction is 0.5 and batch size is 1,024 rows.
Warm-service repetitions still construct a fresh state before requests.

Five policy-only recipes add no jobs.
They cover method scope, matched controls, numerical release, dependency accounting and optional Civil official splits.
The official-split sensitivity is explicitly inactive.
Do not silently activate it.

## 11. Run genuine admission and contextual audits

Use accepted natural admission frames after actual experiments.
Select unique pairs probabilistically with their full eligible frame and inclusion weights.
Fix one uniformly chosen former blocker per admitted record.
Preserve request multiplicity without counting repeated records as independent judgments.

| Corpus | Admission pairs from R/S | Admission pairs from U/A | Naturally excluded controls |
|---|---:|---:|---:|
| Civil | 67 | 67 | 67 |
| Ask Ubuntu | 67 | 66 | 67 |
| CC-News | 67 | 66 | 66 |
| **Maximum** | **201** | **199** | **200** |

Within these budgets, preserve the frozen similarity-band and label-relation strata.
The sampler can select fewer pairs when strata lack records.
Do not transfer a shortfall to another corpus or favorable stratum.
Three independent fluent annotators judge every pair.
This is at most 1,800 pair assignments.

Ask separately about meaning, consequential distinctions and relevance to the original task.
For News, judge factual assertions; do not invent class labels.
Blind annotators to method, request arm, labels and model effects.
Keep disagreements, skips and uncertainty.

Two separate audits each sample up to 100 admitted records.
Each has corpus quotas Civil 34, Ask Ubuntu 33 and CC-News 33.
One displays the complete former-blocker neighborhood subject to the declared context cap.
The other displays the nearest surviving selected text at that checkpoint.
Each uses three independent judgments, totaling at most 600 contextual assignments.
The target-text limit is 8,000 characters; the context limit is 20,000.
Log truncation and do not call a truncated neighborhood complete.

Pair and contextual interval families each receive error probability 1/40.
Their combined simultaneous coverage is at least 95% under the stated sampling assumptions.
Missing corpora leave balanced population targets undefined.
Missing judgments remain bounded missingness, not inferred positive outcomes.
Illustrations follow fixed hashes within declared categories, never maximum model effects.

## 12. Measure correctness, resources and statistics

Use common FP64 Cholesky as the primary decoder.
Require normalized residual `eta <= 1e-10`.
Report absolute residual, conditioning and moment checks.
Residual verification alone is not a rigorous accumulated-error certificate.
Call results numerically certified only when the required error budget actually exists.

Never silently clamp eigenvalues, zero small coefficients or rebuild from unavailable retained rows.
Never relax tolerances after seeing failures.
The separate exact rational service has its own stronger target and costs.
Do not transfer its guarantees to unrelated floating runs.

Before dense allocation, report:

- Eligible-record count, structural rank, coefficient-key count and blocker/source incidences.
- Packed FP64 coefficient forecasts and actual compact FP32 payload costs.
- Metadata, index, cache, serialization and solver workspace.
- Shared acquisition, encoding and graph preparation.
- Construction, every request, solve, output, persistence and compaction time.
- Persistent bytes, process RSS, sampled process-tree RSS and accelerator memory.
- Bytes read/written, access class, timeout, OOM and all other failure statuses.

The original 256-GiB host and 128-GiB per-method cap are planning figures only.
Use actual local measurements and a common prospectively locked limit.
Do not change only our method's dimension or panel when it exceeds capacity.
Keep forecast infeasibility separate from observed OOM.
The measured resource qualifier sets scoped operational limits, not worst-case memory guarantees.

For relevance, save raw prediction RMS discrepancy `D_F`.
Also report `D_F/s_0`, signed normalized loss difference `Z_F`, its absolute value and both denominators.
A zero denominator gives an undefined normalized value.
Report exceedances at `|Z_F| > 0.01` and `D_F/s_0 > 0.01`.
These are fixed reporting scales, not universal linguistic importance thresholds.
Negative signed effects are possible and must remain visible.

For structural prevalence, report probability of any admission and admission count distributions.
Report admissions per deleted record and, for S, per deleted source.
Check Monte Carlo means against the exact blocker-based finite-population expectation.
That verifies the sampler on its graph; it is not independent natural-data confirmation.

Use 10,000 seed-locked trajectory bootstrap resamples; seed 20271004.
Preserve all checkpoints and paired methods within each resampled trajectory.
Timing repeats stay nested within their trajectory.
The primary inference conditions on the fixed test set.
The separately labeled crossed analysis also resamples whole genuine test sources.
Recompute nonlinear metrics and normalization denominators in each replicate.
Never treat request-by-example cells as independent.

Use exact binomial intervals for appropriate activation probabilities.
For directional significance, the four Civil/AskUbuntu × R/S loss contrasts form one Holm family at 0.05.
The two equal-R/S-weighted P-I versus B-E lifecycle contrasts form another family.
Equal weighting is fixed, not an asserted deployment frequency.
Do not derive a significance test from a percentile interval alone.

The frozen implementation accepts justified directional p-values but does not define their test algorithm.
Before effects are inspected, lock the exact null, directional estimand, test, assumptions and executable analysis code.
This remains an explicit analysis-lock item; the handoff does not invent a test.
If no justified prospective test is locked, retain null p-values and report effect estimates and intervals.
The Holm implementation keeps missing tests in their original family with conservative effective p-value 1.
A local LLM must not choose a favorable test after reading the results.

Use disjoint development effects for the precision analysis.
A nonsignificant result is not equivalence.
Missing planned trajectories make unconditional estimands unavailable under the frozen binder.
Do not replace them with a favorable complete-case analysis.

## 13. Execution order and stop/fix rules

| Stage | Work | Advance when | If it fails |
|---|---|---|---|
| 0: authentic inputs | Acquire, parse, replay sources, encode and replay caches | Real population and representation bindings are accepted | Keep the arm blocked; do not substitute previews |
| 1: semantic/task locks | Genuine selection and fresh validation; guards; vocabulary/lambda | Required quality and source gates pass | Preserve failed quality; no effect-based threshold search |
| 2: development | Disjoint correctness pilot, optimized B-E, native bytes, measured resources and precision | Correctness holds and declared scoped limits are justified | Fix bugs with version history; record infeasible cells |
| 3: confirmation | Freeze code, scopes, roots, requests, settings and execute all registered families | Every planned cell has an explicit outcome | Preserve failed/blocked cells; rerun affected comparisons after explicit fixes |
| 4: interpretation | Genuine admission/context audits and complete bound statistics | Authentic judgments and complete evidence bindings exist | Retain missingness or unidentified estimands |
| 5: reporting | Six figures, three tables, claim ledger and reproducibility package | Every aggregate maps to source run IDs | Do not fill missing plots with invented positive results |

A development pilot is diagnostic and never relabeled confirmation.
Small lexical fixtures cannot qualify native semantic resource profiles.
A successful pilot need not favor P-I.
Correctness and valid measurement are advancement gates; empirical superiority is an outcome.

The current assembly keeps all 21,332 jobs in every finalized ledger.
`completed_dispatch` means complete accounting, not complete successful experiments.
Automatic crash recovery and cross-invocation result pooling are not implemented.
Do not concatenate partial ledgers and call them one accepted confirmation run.
Use new output directories and preserve each interrupted attempt.

Authentic human collection and accountable evidence review remain real workflow steps.
No script can manufacture their testimony.
The local LLM must report missing evidence rather than inserting approval fields.

## 14. Scientific deliverables and final claim decisions

Produce six figures:

1. Natural selection/exclusion/admission flow and R/S distributions, including zero mass.
2. Size and budget curves for eligibility, rank, keys, admissions and source collapse.
3. Prediction discrepancy, signed loss changes and weighted linguistic judgments.
4. Persistent/peak memory versus complete lifecycle time against B-E, with infeasible cells.
5. Correctness, residuals, state failures and cumulative cost through the actual horizon.
6. Full-refit mismatch, representation cross and convex results.

Produce three tables:

1. Provenance, actual panels, source units, guards, tasks and curation quality.
2. Method access/output/state contracts, complete costs and failures.
3. Corpus-level claim ledger with estimates, intervals, boundaries and failed gates.

Return the complete result manifest and the bound final ledger.
Include request manifests, accepted settings, saved predictions, metrics, resource receipts and failure histories.
Include the scripts and environment required to regenerate every aggregate.
Keep restricted source text and private responses out of the public repository.
Follow the result-return instructions in `START_HERE.md`.

| Actual finding | Supported conclusion |
|---|---|
| R/S admissions, meaningful distinctions and prediction consequences | Natural counterfactual curation matters for those populations |
| Effects mainly under U/A | Natural-data stress vulnerability; no common-case prevalence claim |
| Membership churn but equivalent text and tiny task changes | Structural effect with limited demonstrated NLP significance |
| P-I faster but larger than B-E | Measured latency–memory tradeoff |
| P-I beats only full rebuilding | Exact repair, without strong incremental-efficiency superiority |
| Native summaries exceed capacity | Native limitation; projected variants remain different learners |
| Full refitting differs substantially | Fixed-curator contribution with a measured applicability boundary |
| Correctness failures persist | No demonstrated exact implementation for failed configurations |
| No material natural effect at valid thresholds | Negative scope result; intended strong empirical claim remains unsupported |

Do not promise novelty or acceptance from completion counts.
The program succeeds scientifically by determining which of these conclusions the evidence supports.
