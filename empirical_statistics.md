# Statistical protocol for counterfactual semantic curation

This is a proposed preregistration, not an experimental result. It is designed for natural corpora only. Synthetic examples, injected duplicates, generated paraphrases, label corruption, and planted graph motifs are outside the confirmatory program. A topology-targeted request on an unchanged natural corpus is allowed, but is a stress test, not an estimate of typical deployment behavior.

## 1. Separate the claims and their populations

There are five distinct questions. They must not share one undifferentiated “unlearning score.”

| Claim | Primary estimand | What constitutes evidence | What it does not show |
|---|---|---|---|
| Natural activation | Probability of any admission; mean additions per deleted record; distribution of absolute additions, under a declared request distribution | Census structural quantities and paired request outcomes, including zeros | Population prevalence from a deliberately chosen high-amplification request |
| Task relevance | Frozen-selection versus fixed-curator-oracle prediction discrepancy and held-out loss difference, under the same requests | Fixed independent test distribution, task loss and decision disagreement, gradient change, and blind text audit | Training-set churn alone, or a statistically significant tiny parameter difference |
| Exactness | Selected IDs, designated state, moments and model agree with the correctly scoped oracle | Every-case invariant checks and numerical residual certificates | Failure to reject a test-score difference; privacy of historical outputs |
| Algorithm value | Total bytes, total work and latency at equal recovery contract, horizon and accuracy | Paired end-to-end measurements against strong maintenance baselines | Faster query time with expensive state repair excluded, or value-coordinate counts called total memory |
| Scope robustness | Fixed-curator versus full-refit target discrepancy; validity/tightness of graph-envelope certificates | Both oracles and independently checked envelopes | Exactness for a refitted pipeline from an exact fixed-graph experiment |

Datasets are fixed benchmark populations. With three corpora we cannot justify a universal corpus-population confidence interval. Report per-corpus effects and a clearly labeled fixed-benchmark macro average. Requests sampled independently from a fixed corpus can legitimately be independent Monte Carlo replicates even when they reuse documents; the resulting inference is conditional on that corpus and curator. It is not inference to unseen corpora.

## 2. Lock the experimental object before confirmatory outcomes

Create an immutable manifest containing corpus versions, licenses, source mapping and missing-source policy; raw record hashes; train/dev/test split rules; fixed curation encoder and revision; downstream encoder and revision; normalization; partition; priority and tie rule; strict similarity comparator; threshold rule; labels and label mapping; average-risk regularization; empty-set convention; arithmetic precision; horizon; request seeds; and expected output files.

Use a real-data development panel that is disjoint by provenance/duplicate group from the confirmatory panel. It is for pipeline debugging, approximate cost measurements, threshold calibration and variance estimation. It cannot be reused as confirmatory evidence. Lock the natural corpora before observing activation prevalence. If a corpus has few duplicates or no effects, it remains in the results. Replacement is allowed only for predeclared access, license, missing-field, or corrupted-data failures, with a logged reason independent of the result.

Freeze thresholds using the development panel by a documented deployment criterion (for example a chosen semantic-duplicate precision target), or use a published default plus a preregistered sensitivity grid. Do not pick each corpus's threshold to maximize admissions, model discrepancy or method advantage. A retention-rate calibration is permissible only if it is the declared curator contract and the resulting threshold is frozen before requests; a retained-data threshold refit is a different oracle.

Treat tuning as part of the specified learner. Choose regularization on training/development data before deletion and then keep it fixed for the primary contract. If a hyperparameter is refit after deletion, rerun the full tuner in that oracle and label that study separately. Never call fixed-hyperparameter matching “full pipeline” matching when the nominal pipeline retunes.

## 3. Immutable holdout and leakage rules

Prefer the established task test split. Before constructing the deletion-sensitive training corpus, apply one fixed, separately specified train-versus-dev/test duplicate guard; remove train-side exact and certified near duplicates of held-out texts. This guard is part of data setup and is never rerun as a function of a deletion request. Do not use test labels to set a deduplication threshold, source mapping, priority, request sampler or hyperparameter. If creating splits, split at available provenance/duplicate-group level before model selection, and record whether groups are source, event, conversation, product, author or URL family. A single enormous connected component is a reason to use a different defensible leakage grouping, not to split it silently.

The held-out test texts, labels, slice definitions, and metric code are immutable. Test documents never enter deletion requests. Full-refit oracles refit on retained training data only. Base-encoder pretraining contamination cannot be certified absent merely by these steps; state the encoder's provenance and keep claims about the comparison conditional on the fixed public encoder.

Use one fixed test distribution for the main model-discrepancy analysis. For multiclass ridge use logit-vector RMS discrepancy and mean squared task loss; also report macro-F1/accuracy. Do not treat ridge outputs as calibrated probabilities. For logistic heads use cross-entropy and probability discrepancy. Prediction disagreement is interpretable but can hide changes away from the decision boundary, so it is secondary. Report original pre-deletion utility and retained-data-oracle utility to show that the chosen head is a credible task learner.

## 4. Request populations: never pool them without declared weights

The primary stochastic populations should be (R) uniform records and (S) uniform native sources. If native source metadata are missing, the source arm is unavailable; inferred semantic clusters cannot silently stand in for publishers or owners. Keep these distinct from:

* (U) uniform initially unselected records. This isolates the headline phenomenon without claiming it is common in all requests. State whether eligibility is initial or current; initial is simpler and reproducible.
* (W) source selection proportional to initial source record mass. This approximates a different workload from source-uniform selection. Its effect cannot be inferred from S.
* (A) label-blind topology-targeted requests, constructed using blocker sets and a fixed search budget. These establish stress behavior on real data, not frequency. Report search runtime and all seeds, including unsuccessful searches. No test labels or held-out gradients enter the search.

For R at fixed size k, draw a uniform k-subset without replacement. For sequential R, draw one uniform random permutation and use its prefixes. Thus each checkpoint has the same uniform marginal as its one-shot counterpart, while intermediate states remain genuinely sequential. For source deletion, do the same over source IDs, remove every owned record, and report both number of sources and raw/selected records removed. Equal source count is not equal deleted record mass; include a mass-matched record-control arm if attributing differences to correlation.

For U, take a random permutation of the initially unselected pool. The request distribution is conditional on exclusion and must be labeled that way. If using a dynamic policy that selects currently excluded records, log its transition rule; it no longer has the simple uniform-subset marginal.

Keep a separate natural chronological/source-event stream if metadata support one. It is one observed workload, not hundreds of independent requests for inference. Report descriptive results, and use multiple genuinely separate streams only when available.

### Exact stochastic prediction

For uniform k-record deletion from N records, a candidate with b_v>0 blockers activates with probability

    choose(N-b_v-1, k-b_v) / choose(N,k),

with invalid binomial arguments interpreted as zero. Therefore expected additions are the sum of this quantity over initially excluded records. The Bernoulli approximation (1-p)p^b is not exact for fixed-size requests. The same formula applies to uniform k-source deletion with N replaced by source count and b_v by distinct blocker-source count, provided the candidate's own source is not one of its blocker sources. Such same-source-blocked candidates have probability zero.

For a deletion-eligible pool E, candidates with B(v) not contained in E can never activate; the probability additionally depends on whether the candidate itself belongs to E. In U the initially excluded candidate belongs to E, so substitute |E| for N after checking B(v) subset E.

Compare observed Monte Carlo means against these parameter-free predictions. This verifies the sampler/implementation and shows prevalence under the declared population. It is not an independent validation of theoretical generalization, because the same graph defines prediction and outcome.

## 5. Horizons, resets, and fair comparisons

Choose a small budget grid in advance from operational scales, not from observed rank transitions: record budgets {1, 10, 100, 0.1% N, 1% N}, deduplicated and clipped to feasible values; source budgets {1, 2, 5, 1% S}, likewise. Reporting the full structural rank curve is cheap and descriptive, but does not authorize choosing only favorable operating points for downstream models. The grid may be revised once from a cost-only development pilot before confirmatory outcomes.

In one-shot tests, rebuild every method from the same initial corpus and horizon for each request. In sequential tests, all methods see the same cumulative sequence; remaining horizon is k minus cumulative deletions. Do not reset the horizon after each request. The freshly built state oracle uses the remaining horizon, not the original budget. A budget-exhaustion rebuild is a separate, fully costed deployment policy. Record and source horizons are different units.

Primary sequential summary: predeclared final-budget outcome plus integrated error/work over the checkpoints, computed once per trajectory. Checkpoint curves are useful, but checkpoints are not independent replicates. A query-only cached answer is not a state-repaired method; both state repair and final model release must be measured.

## 6. Concrete sample sizes and economical execution

Use the following starting allocation per primary corpus/primary curator, then lock after a development-only cost/power pilot:

* Structural census: all records and sources; all rank horizons if feasible; no sampling confidence interval for a complete finite-corpus census.
* Main stochastic trajectories: 256 independent permutations for R and 256 for S; all predeclared checkpoints. This gives worst-case Monte Carlo standard error <= 3.125 percentage points for a binary activation event. If zero events occur, its exact one-sided 95% upper bound is about 1.16%, rather than “no risk.”
* Conditional U: 128 independent trajectories. It is a diagnostic arm, not combined with the primary prevalence estimate.
* Adversarial A: 32 fixed search initializations on each eligible corpus/configuration. Report the complete distribution and attained maximum, never a p-value treating the maximum as random prevalence.
* Full-refit and expensive alternative-head audit: a preselected random 32 of the primary trajectories per population, shared across all methods, plus 16 topology-targeted trajectories. The subset is chosen before any repair errors or task effects are seen. Report it as a scoped robustness audit, with wider intervals; it cannot support a rare-failure claim.
* Threshold/encoder/order sensitivity: 32 shared primary stochastic trajectories per fixed alternative configuration, plus the full structural census. Full primary replication is unnecessary for every sensitivity arm unless its outcome becomes a main claim; that would require a new preregistered replication, not promotion of the same exploratory results.

These are Monte Carlo numbers, not promises of task-level statistical power. With paired trajectory differences of SD sigma, detecting an effect Delta has approximate required n = ((z_(1-alpha/2)+z_(power)) sigma / Delta)^2. At alpha=.05 and 80% power, n=256 detects approximately 0.175 SD before multiplicity; with six Bonferroni-planned contrasts, approximately 0.22 SD. Heavy tails warrant a development-panel simulation or conservative interval rather than this normal approximation.

Precommit a precision or power target on one scalar primary estimand, choose a largest affordable n, and lock n before confirmatory outcomes. If it is unaffordable, reduce the number of secondary configurations rather than silently weaken the oracle or report a subset chosen for favorable effects. A blinded development variance estimate can set n; an observed test effect must not. No “continue collecting until significant.” An optional sequential design would require a separately specified valid confidence sequence and stopping rule, not repeated conventional 95% intervals.

Three corpora, two request populations and 256 trajectories is 1,536 complete primary sequences. At five checkpoints it is 7,680 model releases per method, with shared graphs, moments, requests and held-out embeddings. Deterministic ridge needs no cosmetic five training seeds. Put replication into the actual random source: requests, curator fitting/order sensitivity, and hardware blocks. For a stochastic neural extension, pair at least five training seeds across oracle and repair, retain every run, and do not pool seeds as if they were independent corpora.

## 7. Inference and multiple comparisons

Pair methods on identical initial state, request/trajectory, test set and workload timing block. For scalar trajectory outcomes, give mean, median, 90th/95th percentile and paired differences or ratios with 95% intervals. Zero-heavy admissions require both event probability and magnitude conditional on activation; do not show only the latter. Runtime should include arithmetic mean total work/cost, not only median microbenchmark latency; log-ratios can summarize multiplicative comparisons, with failures handled separately.

Use a request/trajectory bootstrap for inference over the declared request distribution conditional on corpus and test set. To add uncertainty over held-out examples, use a crossed bootstrap: resample entire trajectories and independent held-out provenance/duplicate groups, preserving all methods together. A simple bootstrap over request-by-test-example cells creates pseudoreplication. Within a trajectory, keep every checkpoint together. Repeated admissions of the same document do not create independent human-audit examples.

For small numbers of fitted-curator seeds, report each seed and conditional request intervals. Do not rely on a three-seed hierarchical bootstrap to claim robust population inference. Dataset-level pooling is descriptive unless many independent datasets are available. Recompute macro-F1 within each bootstrap replicate rather than averaging per-example “F1.”

Predeclare at most six headline directional comparisons across the fixed benchmark macro average, and control familywise error with Holm at .05 if making significance claims. Per-corpus effects are still all reported. Prefer estimation and simultaneous bands for budget curves to dozens of isolated stars. If there is no defensible smallest meaningful NLP effect, publish the effect-size intervals without declaring a binary practical-significance gate.

Equivalence requires its own positive evidence. For approximate heads, choose a utility margin in task units with a deployment justification before outcomes, then require the full confidence interval inside that margin (or a valid TOST procedure with multiplicity accounted for). A non-significant difference from the oracle is not equivalence. For exact ridge, use the stronger numerical criterion below instead of statistical equivalence of task metrics.

## 8. Exactness, numerical certification and failures

At every primary checkpoint compare selected IDs to an independently implemented fixed-curator rerun, and compare canonical abstract-state serialization with a fresh remaining-horizon rebuild on an audit subset selected before outcomes. Check all changed IDs and all moment coordinates; sample-based state comparison does not establish state equality. If full state comparison is too expensive at scale, verify a deterministically selected bounded panel and label the remainder as checksum plus model tests, not complete verification.

The floating-point model oracle is not mathematical truth. Record ridge residual norms and condition estimates for both oracle and repair. For A=M+lambda*n*I, each residual gives a parameter-distance bound relative to its exact moment system, using lambda*n when n>0. Bound/measure moment accumulation error separately, or recompute failed cases in higher precision. For two solves of the same system, the sum of their certified solution-error bounds is a principled discrepancy tolerance. Testing every output against an arbitrary fixed 1e-6 can hide a broken algorithm on badly conditioned problems.

Use exact integer ID/count/state-key checks; deterministic zero-edge tie tests and finite precision sensitivity on real embeddings; high-precision moment/solve adjudication for numerical failures. Keep numerical fidelity distinct from floating-point bitwise history independence, which is not the theoretical claim.

All errors, invalid requests, memory exhaustion, solver nonconvergence, certificate failures, exhausted horizons and fallback invocations remain in the denominator. Predeclare recovery policy and charge it. On failure, report successful-run latency and failure rate separately, plus total cost under the prescribed fallback; never simply omit failures. A timeout is a censored runtime/lower bound, not a measured completion time. Never stop a baseline earlier because it appears slower.

## 9. Human audit of natural admissions

Use 400 unique admitted-document/former-blocker pairs across the primary corpora and request populations, selected with recorded inclusion probabilities, plus 200 matched naturally excluded control pairs. If fewer than 400 unique admissions exist, audit the complete set and report scarcity; do not manufacture pairs. Stratify the audit across cosine bands, request populations and label-disagreement status, then use inverse-inclusion weighting for any estimate about the full admission population. A balanced case study without weights must be labeled qualitative.

Two independent fluent annotators per pair; blind them to method, request population, model change and the paper's desired direction. Give enough context and native task definitions. Use separate questions: (i) exact duplicate / same meaning / overlapping information / merely related / unrelated; (ii) task label consistent / meaningfully different / ambiguous; (iii) does the admitted text contain a new task-relevant fact or distinction, yes/no/uncertain. These are separate constructs; avoid a single undefined “usefulness” ordinal score. Report category counts, disagreement and Krippendorff's alpha where suitable, with third-rater adjudication retained as a separate view. Release deidentified annotation instructions and example rationales where licensing permits.

Cluster uncertainty by unique admitted document or provenance group, not by repeated pair exposures. Annotators assess semantic/task distinctness, not legal ownership, deletion compliance or model memorization. Do not call labels wrong merely because they differ on semantically similar text; ambiguous/temporally changing examples deserve separate reporting.

## 10. Paper-proof reporting rules

* The full effect distribution and all zero-effect datasets survive into the paper/appendix. No after-the-fact “representative” request selection.
* Show naturally occurring excluded-only withdrawals as cases and as a separate sampler, with random-record/source prevalence beside them.
* Report both fixed-curator and full-refit oracles by name. Use the same encoder, threshold, order, normalization and solver criterion within each contract.
* Show total bytes including keys, graph metadata, identifier membership, embeddings/raw payloads, solver workspace and physical allocated peak memory; generic value-coordinate rank is not total-memory optimality.
* Equality of scoped state and task fidelity do not imply transcript privacy, fact suppression or forgetting retained paraphrases.
* Refitted-cluster random-seed variance must not be confused with deletion-caused change. Couple initial/refit random seeds and also report the no-deletion rerun variation, if the oracle is stochastic.
* Report construction amortization and the break-even request count against both full rebuild and blocker-count maintenance. If break-even exceeds the permitted cumulative horizon, there is no operational advantage in that regime.
* Label all post-lock analyses exploratory. Corrections of implementation bugs require a versioned rerun of every affected method/corpus cell, rather than changing only disappointing cells.

The study cannot be made reviewer-proof, and no design can promise useful effects. It can be made resistant to predictable objections by locking the populations, oracles, comparison budgets and analysis before results. The aim is a decisive outcome even if some findings are negative.

## Review of final concrete allocation (parent's narrowed program)

The final proposed allocation—256 record and 256 source structural trajectories, 128 initially excluded-only, and 32 topology-targeted trajectories per core corpus—is sound. Record horizon 128 with checkpoints 1, 8, 32, 128 and source horizon 8 with checkpoints 1, 2, 4, 8 are clearer than a broad budget factorial. Clip to corpus/source size, remove duplicate checkpoints, and explicitly retain the prescribed empty-target convention if total deletion is possible. Do not silently remove total-source-withdrawal cases. The structure-only 1%-of-records horizon is a separately built initial state; it is not a reset of the primary horizon.

A fixed, outcome-blind full-method panel of 64 R + 64 S + 32 U + 16 stress trajectories is defensible for costly state/latency audits. It is weak for ruling out small model effects: at n=64, zero events gives a one-sided 95% upper bound of 4.57%; approximate paired 80%-power sensitivity is 0.35 SD unadjusted, or about 0.44 SD with six-comparison planning. If affordable, run the two decisive task-relevance models (fixed-curator oracle and frozen-selection baseline) on all 256 R/256 S trajectories, while retaining the 64/64 subset for expensive full-method comparison. This improves scientific power without multiplying every baseline run. If using 64 throughout, call the task analysis an estimation study and publish its full uncertainty; negative results cannot establish rare failures or negligible effects.

For fixed-test normalized RMS prediction discrepancy, define the multiclass convention explicitly:

    sqrt(mean_{i,c} (f_frozen(i,c)-f_oracle(i,c))^2)
    / sqrt(mean_c Var_i f_predelete(i,c)).

The denominator is frozen across requests. Report raw RMS and the denominator as well. If it is zero, report the normalized quantity as undefined rather than introducing a posthoc floor. For scalar predictions this reduces to the ordinary pre-deletion score SD. State population versus sample variance convention, which only changes a small scale factor.

Use signed normalized task change as

    [MSE(f_frozen)-MSE(f_oracle)] / MSE(f_predelete).

Report the raw numerator and denominator; if the denominator is zero the relative statistic is undefined. Absolute task change is secondary. A negative signed value means that failing to reproduce retraining happened to improve that held-out score; it does not repair the fidelity failure. For a test-group bootstrap, recompute these denominators within each bootstrap replicate. For request-only Monte Carlo intervals conditional on the fixed test set, hold them fixed.

The proposed 1% relative-MSE reporting threshold is a predeclared engineering margin, not universal practical significance. A two-sided interval inside [-.01,.01] establishes only equivalence of the signed mean under its assumptions, and can conceal opposite-signed per-request changes. If the claim is negligible per-request impact, use the mean absolute change or a tail/exceedance estimand and its appropriate upper confidence bound. An interval containing zero proves neither equivalence nor absence of task-relevant impact.

## Primary methodological sources checked

These support the statistical/reproducibility principles, not the original curation-specific formulas or the proposed sample allocations:

* Dror, Baumer, Shlomov and Reichart (ACL 2018), *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*. https://aclanthology.org/P18-1128/ (root-openable source ref turn90view0).
* Reimers and Gurevych (EMNLP 2017), *Reporting Score Distributions Makes a Difference*. https://aclanthology.org/D17-1035/ (turn90view1).
* Dodge et al. (EMNLP-IJCNLP 2019), *Show Your Work: Improved Reporting of Experimental Results*. https://aclanthology.org/D19-1224/ (turn90view2).
