# Counterfactual curation: theory and results handoff

Snapshot: 4 October 2026. Intended venue: ACL 2027. Working paper name: **Unlearning What Was Never Trained**. This is the third research direction in the originating research agenda; it is not the supervision-source project or the calibration-quantization project.

## Read this first

The project has a developed fixed-curator theory, locally implemented repair algorithms, completed natural-text engineering experiments, a strict exact-state/certified-release implementation, and an audited preparation workflow. **The primary semantic empirical study has not started.** There are no actual human ratings, no selected primary semantic threshold, no passed primary semantic-quality gate, and no pinned E5/MPNet embeddings in the available workspace.

`empirical_execution/CURRENT_STATUS.json` is the machine-readable current status. Older `empirical_execution/STATUS.json` and phase-specific reports are historical, not competing current status. Read `empirical_execution/phase10/CHECKPOINT.md` and its `REMAINING_TASKS.md` for the current inventory. Phase 9 retains current experiment assembly. Earlier phase-specific TODO files are historical. The frozen phase-3 workflow remains a dependency. This handoff is a checkpoint, not a background-run promise.

The user has repeatedly requested that all computation be done here, that synthetic empirical datasets be deferred, that weaknesses be solved algorithmically rather than defended, and that actual work continue instead of receiving another plan. The new repository instruction is to preserve all project files and enough context to resume in another chat. Do not substitute generated labels, fake provenance, lexical hashing described as semantic embeddings, or additional repetitions of the small preview for the missing primary study.

## Authority and project evolution

Read sources in this order when continuing:

1. This handoff and `empirical_execution/CURRENT_STATUS.json` for state and boundaries.
2. `output/pdf/counterfactual_curation_theory.tex` and its PDF for the integrated theory after three passes. Earlier root-level `round2_*`, `round3_*`, `curation_theory_notes.md`, and `learning_theory_notes.md` preserve derivation, review, and verification history. The repository backup also preserves the unique first/second theory drafts originally at `tmp/pdfs/round1_theory.tex` and `tmp/pdfs/round2_theory.tex` under `archive/theory_history/`, and the original report build/package scripts under `tools/legacy_build/`. Those are historical sources, not replacements for the integrated theory.
3. `empirical_execution/memory_theory_addendum.tex` for the later feature-dimension memory repair; it supplements the integrated theory.
4. `empirical_execution/canonical_theory_addendum.tex` for the later exact rational state and rigorous floating-release certificate; it does not retroactively make the previous floating implementation canonical.
5. `output/empirical_program/counterfactual_curation_empirical_protocol.tex` for the empirical design. Its narrative resolves ambiguities in `study_design.json`. `claim_ledger.csv` records claim requirements.
6. `empirical_execution/phase3/PREPARATION_AMENDMENT.txt` for prospective implementation choices and corrections before any confirmatory study. Historical empirical outputs remain untouched.
7. Phase-specific README files and raw result JSON/JSONL for actual evidence. PDFs summarize; they are not substitutes for the registries.

Evolution:

- Initial idea: deleting a raw document can change curation and add retained documents that never trained the original model. The initial component-representative example was only illustrative.
- Theory refinement: settled on **earlier-raw-neighbor suppression**, with a fixed graph and ID-derived order. This is consequentially different from greedy independent-set selection and from connected-component representatives.
- Three theory passes: finite-horizon statistic polynomial, sequential canonical repair, indexed update accounting, exact query rank, source extension, constant-accuracy randomized information lower bound, factor-two statistic-space comparison, rank-sensitive learning and conditional full-refit certificates.
- First local engineering: real Civil Comments and CC-News previews, lexical features only. Dense polynomial moments were much larger than the eligible-payload comparator. This motivated a genuine memory revision.
- Memory repair: joint-span coefficient factorization removed the persistent quadratic feature-dimension expansion. Strong compact-payload baseline remained smaller; no claim of beating it was made.
- Strict qualification: unique rational pivot chart removed floating/history-dependent representation choices; exact residual arithmetic certified numerical head releases. This is a separate exact-aggregation target.
- Phase 2: prediction/loss consequences on a fixed 82/18 development split, with unfavorable curation utility and structural zeros fully preserved.
- Phase 3: source-disjoint preparation, corrected leakage guards, common pair-local finite-precision scoring, calibration/validation sampling, exact quality-bound arithmetic, offline CLI and input bindings. Actual scientific inputs remain missing.
- Phase 4: original-schema adapters, offline encoder integration, calibration-only task selection, paired request manifests, final fixed-input comparison, partial dossier inspector and fresh-process measurement harness. News/calendar and per-encoder calibration software, the scoped convex pilot/certificate, and blank admission packs are also completed in their separately recorded scopes. None starts the primary semantic study or replaces missing genuine assets/ratings.

## Core model and theorem contracts

For raw records `D`, curator `C`, and prescribed learner `A`, the deletion target is

\[
\theta_F=A(C(D\setminus F)),
\]

not generally `A(C(D) \ F)`. Corpus recovery, model recovery, and state recovery are distinct contracts. The main implementation releases a model head; independent oracles provide selected IDs for evaluation. It does not promise recovered document payloads.

### Fixed earlier-raw-neighbor curator

With a strict total order and threshold graph, let `B(v)` be all earlier **raw** neighbors of record `v`. A neighbor can suppress `v` even if that neighbor is itself excluded. Under restriction consistency,

\[
S_F=\{v\notin F:B(v)\subseteq F\},\qquad
P_F=\{v\notin F:0<|B(v)|,\ B(v)\subseteq F\}.
\]

Originally selected surviving records remain selected. Singleton deletion of `u` admits exactly the records with blocker set `{u}`. There is no recursive selected-status cascade under this rule. A three-record path `a < b < c` can select only `a`, then admit `c` after excluded `b` is removed. A different rule needs separate analysis.

Restriction consistency fixes embeddings, edge eligibility, threshold, relative priorities, source ownership, and record-local downstream statistics. A clustering or encoder fitted to deletable data and then refitted does not satisfy this assumption automatically. A seed alone does not preserve relative order of a shuffled shorter input. Use stable ID-derived priorities.

At cumulative horizon `k`, the activation-eligible set is `E_k={v:|B(v)|<=k}`. Its payload can be retained as a legitimate exact baseline. Under a decrementing horizon, a currently ineligible record can never become eligible. One deletion can admit a linear number of documents, but this is a worst-case fact, not observed natural-data prevalence. The blocker histogram gives exact expected admissions under specified uniform request frames; variance/stress behavior also depend on overlap. Worst-case activation optimization is NP-hard in a stated graph family, not established as fixed-low-dimensional cosine hardness.

### Counterfactual statistic polynomial and sequential repair

For a fixed record-local statistic `t(v)` of dimension `s`, with deletion indicators `x`,

\[
T(x)=\sum_v t(v)(1-x_v)\prod_{u\in B(v)}x_u
=\sum_J\alpha_Jx_J.
\]

Each record contributes to at most two keys. Keeping keys of degree at most `k` answers every request with at most `k` deletions. To repair state for a batch `F`, set its variables to one, merge equal reduced keys, remove zero coefficients, and truncate at `k-|F|`. This equals fresh retained-data construction **at the same remaining horizon**. Batch, singleton, and reverse-order updates compose exactly in exact arithmetic. Future query rank cannot increase along an authorized cumulative sequence.

There is no budget replenishment. Unknown/out-of-universe IDs and over-budget requests are validated before mutation. Historical implementations have distinct retry policies; follow each API's documented strict service contract rather than assuming all reject/no-op rules are interchangeable.

For fixed features `z_v` and targets `y_v`, ridge uses `t(v)=(z_v z_v^T,z_v y_v^T,1)`. For average loss and fixed positive lambda,

\[
W_*=(G+\lambda n I)^{-1}H.
\]

Empty selected data has prescribed zero head. Counts and the `lambda*n` normalization are essential. A fixed-size regularizer added to an unnormalized sum would define a different learner.

### Indexed algorithm and rank characterization

The indexed sparse map maintains stable entry handles, inverse ID-to-entry incidences and degree buckets. For initial nonzero key count `M0`, incidence count `L0`, and

\[
\Phi_0=\sum_J |J|(|J|+1)/2,
\]

sequence-wide work is `O(Q + Phi0 + s*M0)` under the stated expected hashing and unit-cost field-arithmetic model. There are at most `L0` key shrinks/incidence removals and `M0` coefficient-vector additions. Construction, model decoding, serialization, graph/encoder work, and arbitrary-precision bit costs are separate.

The selection-query operator and coefficient operator have the same rank `r_k`, by an invertible subset-zeta transform. For arbitrary independent scalar statistics, exact linear statistic storage needs and can use `r_k` coordinates; for `s` independent coordinates, `s*r_k`. Incidence-graph rank is `p-c0`, where `p` counts incident nonground nodes and `c0` ungrounded components. For record deletion, reduced blocker signatures compute the full rank curve without enumerating queries. For disjoint-source deletion the incidence theorem survives but the forest shortcut can fail.

The fast map has `M<=p<=2r`, a sharp factor-two statistic-coordinate bound. Total logical state is `O((s+h+1)r + N_alive)` words under word-sized identifier/statistic assumptions. This is **not** total-byte optimality, an arbitrary-real nonlinear encoding lower bound, or a proof of `s*r` independent degrees of freedom for constrained ridge moments. Decoder metadata and keys are charged separately. The slower rank-optimal scanning baseline explicitly repairs its eligible-record metadata.

### Information lower bounds and their limitations

The activation-frontier family has `m` independent hidden binary labels behind `b+1` blockers. At horizon `b`, zero hidden-label-dependent information suffices; at `b+1`, scalar ridge error below `1/[4(1+lambda)]` with per-query failure at most `delta<1/2` needs

\[
B\ge m[1-h_2(\delta)].
\]

All hidden-label-dependent models, caches, private coins and external lookup information count toward `B`. Forgotten payloads in the hard requests are label-independent, so giving them does not invalidate the bound. Public graph metadata are free only in this information comparison, not in system memory accounting. The constant-threshold cosine construction uses `O(log m)` dimensions, not a fixed dimension independent of population size. The same vectors can be learner features, and a fixed public test distribution witnesses prediction-fidelity separation.

The tight normalized squared-error information curve is `R(D)=1-h2((1-sqrt(1-4D))/2)` for `0<=D<=1/4`. A public randomized query summary attains `m R(D)+O(log m)` bits asymptotically. This coding upper bound is not a practical canonical deletion-clean codec, and its fixed-query expectation does not cover requests chosen after seeing the realized code/state. The earlier aggregate-to-individual example intrinsically needs accuracy of order `1/m`; do not describe it as the constant-accuracy result.

### Source withdrawal

Each record has one fixed source owner `s(v)`; ownership is a disjoint partition. Let `U(v)` be its distinct blocker sources. It survives and is selected iff `s(v)` is not withdrawn and every source in `U(v)` is withdrawn. If `s(v)` belongs to `U(v)`, it can never activate while surviving. The source polynomial is `(1-x_s(v))*product_{a in U(v)}x_a`, using Boolean idempotence. Budgets count sources, not raw records. The algorithm/theory covers this case; authentic source-withdrawal model evidence has not been produced on the source-free Civil fixture. Unknown-source singletons are service units, not evidence of genuine author/publisher withdrawal.

### Supporting convex and full-refit modules

For strongly convex target `F`, a candidate step with certified gradient error `a`, Hessian error `e`, linear residual `xi`, and Hessian-Lipschitz constant `rho` has target gradient bound

\[
r=a+\xi+e\|s\|+(\rho/2)\|s\|^2,
\quad \|w+s-w_*\|\le r/\lambda.
\]

Ordinary floating residuals alone are not rigorous certificates. Fitted original-data anchors/bases are not automatically deletion-clean, and nonlinear derivative summaries cannot generally be reanchored without retained payload access.

For a true refitted curator versus a fixed proxy, a proved task-gradient mismatch `c_w` gives `(r+c_w)/lambda` parameter error. Fixed lower/upper blocker envelopes can bound selection mismatch and give a sharp count-based TV bound `1-a/max(p,a+c)` with `a=|I|`, `p=|P|`, `c=|O\P|`. This is conditional on valid fixed envelopes and fixed losses. It does not prove arbitrary SemDeDup reclustering/encoder refitting stable, efficient, or exactly repaired. The Gaussian distributional extension requires the prescribed randomized reference learner and independent output smoothing; it is not a deterministic-head or transcript-privacy theorem.

## Memory repair and strict numerical qualification

### Fast joint-span implementation

`empirical_execution/ccu/joint_span_summary.py` represents a signed coefficient by `G=UAU^T`, `H=UB`, count, or packed dense moments if smaller. The basis must span the **joint range of G and H**: signed Gram cancellation can leave a nonzero cross moment. Intrinsic ranks sum to at most `2 E_h`, yielding `O(E_h(d+c))` numeric entries in exact arithmetic. The floating implementation uses a safe initial-eligibility envelope, retains rank residuals, and does not silently demote dense entries.

Its canonical projector orientation removes basis-rotation ambiguity in exact arithmetic but does not make floating state bytes history-independent. Dense/compact modes and numerical rank/pivot choices can depend on history. Array bytes need not decrease monotonically. Its allocation cap is neither process peak RSS nor lifecycle memory. No separate feature/target table is retained, but factors/moments can reveal record information; this is not privacy.

`ccu/compact_payload.py` is the mandatory stronger eligible-payload comparator: no persistent ambient Gram/kernel or historical head, with primal/dual release as appropriate. Do not compare only against a weaker payload-plus-dense-Gram baseline.

### Exact canonical chart

`ccu/exact_canonical.py` interprets frozen finite FP32 features and FP64 labels exactly as dyadic rationals. For each coefficient, compute the unique RREF chart of `range([G H])`, with pivot rows `P`, implicit identity rows `U[P,:]=I`, free rows, packed `A=G[P,P]`, `B=H[P,:]`, and exact count. Fractions are normalized and keys/IDs are sorted. There is no historical dense/compact mode bit.

Per coefficient with rank `r`, stored rational entries are `(d-r)r + r(r+1)/2 + cr + 1`, and total is at most `2 E_h(d+c+1)`. This counts rational entries, not fixed eight-byte words. Arbitrary-precision numerator/denominator growth, temporary exact arithmetic, indices, serialization copies, and staged old/new maps must be charged. The bit bound in `canonical_theory_addendum.tex` includes input exponents and minor growth.

Exact repairs serialize identically to fresh retained construction at the same horizon. The currently implemented strict reference **sorts/scans all live keys** and stages the next map; it does not inherit the touched-key-only runtime of the indexed floating algorithm. Charge `O(K(h+1)log(K+1))` metadata work plus exact collisions. The GMP helper accelerates exact algebra; Python `Fraction` is the portable fallback. Its source and Linux LP64 restriction are in `ccu/native/README.txt`.

This exact mode was rebuilt once from authorized input rows. It cannot recover discarded exact aggregation information from a rounded old checkpoint. Its target is fresh exact aggregation of frozen binary inputs, not byte equality to legacy BLAS summation/Cholesky.

### Rigorous floating-head release

`ccu/certified_ridge.py` and `ccu/strict_service.py` check exact PSD premises and an exact rational residual for any finite candidate `W_hat`. With `alpha=lambda*n`,

\[
\|\widehat W-W_*\|_F^2\le
\|G\widehat W+\alpha\widehat W-H\|_F^2/\alpha^2.
\]

Upward radius rounding uses integer-square-root comparisons; ordinary nearest-rounded `sqrt` is insufficient. Target and candidate are hash-bound. Failure to meet the exact tolerance gives no release. Empty data gives exact zero. The nonorthogonal rational chart's reduced equation is `(A(U^T U)+alpha I)Z=B`, `W=UZ`; do not reuse the orthonormal fast-mode equation `(A+alpha I)Z=B`.

This certifies numerical fidelity, not semantic quality, encoder unlearning, privacy, physical memory erasure, backups, historical outputs, or arbitrary hardware-independent head bytes.

## Actual experiments and results

All data experiments below used existing natural text and original labels where available. Small graph/algebra fixtures are software verification, not synthetic empirical corpora. All execution was local. No remote training or inference job was used.

### Data available

- `empirical_execution/data/civil_comments_engineering_preview.jsonl`: 100 real Civil Comments preview rows, seven original toxicity fields; no authentic article/publication/parent/date/native record IDs. Row provenance IDs support record deletion only.
- `empirical_execution/data/cc_news_engineering_preview.jsonl`: 90 locally acquired article previews across five observed hosts; ten requested rows were unavailable. No labels. Hosts are not verified registrable domains/ownership. News article bodies were excluded from distributable ZIPs; do not silently reintroduce them into a public repository.
- `*_raw.json` files are untouched connector responses, **not original corpus bytes**. See `DATA_ACCESS_NOTE.txt`, acquisition manifests, and `phase3/acquisition_route_check.json`.
- No source-rich original corpus snapshot, semantic cache, E5/MPNet weights/tokenizer, or human responses exist in the recorded input set.

### First engineering run, v1.1

Evidence: `empirical_execution/README.txt`, `results/execution_summary.json`, per-dimension registries, `results/structure_audit.json`.

- Fixed lexical word/bigram curator dimension 128, threshold 0.6; 100 Civil records, 65 initially selected, 83 horizon-eight eligible. Learner dimensions 64/128/768 are lexical hashing, not semantic encoders.
- 324 checkpoints over 81 trajectories passed. Maximum head absolute difference `6.661338147750939e-16`; maximum moment difference `8.881784197001252e-16`. These were ordinary FP64 diagnostics.
- Eight complete coefficient-map comparisons; four fresh-process aggregate-state replay checkpoints with a scoped Python audit hook. No OS isolation/erasure claim follows.
- Nine structural configurations: thresholds 0.35/0.60/0.85 with Civil record deletion and News record/host deletion. Additional audit: 96 source-subset comparisons, 18 exact expectation comparisons, 2,112 structural checkpoints, and 640 negative-control executions covering 453 distinct cases.
- Dense coefficient numerical arrays at d=768 were 234,483,480 bytes versus 4,980,376 for the then-current payload-plus-moment baseline. This exposed the memory problem; the run did not show storage savings.

### Memory revision

Evidence: `README_memory_repair.txt`, `memory_results/summary.json`, `memory_results/independent_audit.json`, `memory_results/process_peaks.json`.

Initial **owned numerical array bytes**, excluding other memory:

| d | outputs | Dense | Gram-factor | Joint-span | Compact payload |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 1,698,840 | 119,141 | 70,512 | 21,912 |
| 128 | 1 | 6,640,920 | 270,949 | 172,144 | 43,160 |
| 768 | 1 | 234,483,480 | 1,628,326 | 1,039,232 | 255,640 |
| 128 | 7 | 7,249,176 | 879,205 | 183,136 | 47,144 |
| 768 | 7 | 238,133,016 | 5,277,862 | 1,047,200 | 259,624 |

544 release checkpoints passed for all four methods; the joint reduced decoder was also exercised 544 times. Maximum joint common-head discrepancy was `1.1102230246251565e-15`, reduced-head discrepancy `1.1657341758564144e-15`. Another 100 full-deletion releases ended at exactly zero. Twelve rejection checks, three order/batching checks, and 20 complete coefficient audits per compressed method passed.

Single fresh-process peak observations after construction/four releases: dense 356.71 MiB, Gram-factor 135.42 MiB, joint/reduced 121.28 MiB, compact payload 118.39 MiB. These include runtime/input memory and are not repeated performance estimates. The joint method solves the dense expansion problem but **does not beat compact-payload retention** on this fixture. Timings are not lifecycle speedup evidence.

### Strict exact-state qualification

Evidence: `README_strict_mode.txt`, `canonical_results/summary.json`, independent exact-chart/certificate/loader audits.

- 40 regular plus 100 full-deletion releases matched fresh retained exact-state bytes.
- Ten history comparisons covered batch/singleton/reverse order/fresh/reload equality.
- All 140 certificate bounds met exact tolerance `1/10^10`; maximum rigorous radius below `4.34e-14`. Final state/head/bound were exactly empty/zero/zero.
- Independent checks included 243 rational-solution checks, 1,984 upward-square-root checks, 66 ambient RREF comparisons, three large-integer cases, six malformed-loader cases under `python -O`, 20 native/Python algebra fixtures, and 50 certificate-module checks.
- d=768, one-output initial exact checkpoint: 1,514,487 serialized bytes; rational integer payload 184,525 bytes; recursive Python live estimate 11,053,348 bytes; largest numerator/denominator 94/92 bits. These are different layers, not peak RSS.
- Initial exact build 1.869 seconds; median batch repair 0.919, certificate 0.638, fresh exact build 1.577 over 12 checkpoints. Unreplicated, incomplete timer scopes: **not end-to-end speedup**.

### Phase 2: utility development

Evidence: `phase2/README.txt`, `phase2/results/{summary,geometry,utility_context,distinct_interventions,analysis_d768,strict_audits,independent_result_audit}.json`, saved predictions and labels.

The reused Civil100 fixture was split by normalized exact-text-group hash into 82 training and 18 diagnostic evaluation records. No outcome-driven split retry. All records were already used/visible in engineering, so this is not untouched test data. Native source guards and calibrated semantic guards are absent; there are 119 train/evaluation lexical pairs above 0.6.

Fixed training graph: 55 initially selected, 27 excluded, 195 edges. Learner dimensions 64 and 768, lambda 0.01, no intercept/clipping, original fractional toxicity targets. 64 random-record R, 32 initially excluded-record U, and 16 graph-stress A paths, each at 1/2/4/8 cumulative deletions, produced 896 rows. There is no authentic source arm.

At eight deletions, d=768:

| Arm | Paths with admissions | Mean admissions | Relative MSE effect | Descriptive 95% interval |
|---|---:|---:|---:|---:|
| R | 19/64 | 0.34375 | 0.09993% | [-0.04319%, 0.26377%] |
| U | 17/32 | 0.90625 | 0.12681% | [-0.12929%, 0.45036%] |
| A | 16/16 | 6.0625 | 4.45329% | [4.27438%, 4.62364%] |

Effect is `(MSE_frozen_selection - MSE_oracle)/MSE_predelete_curated`. B-F retrains on surviving initially selected records; it is not an unchanged original head. All zeros/negative effects remain. There were 658 zero-admission rows. Bootstrap intervals are conditional descriptive trajectory intervals, not population/confirmatory significance. A paths have only one distinct deletion set at early checkpoints and 15 at checkpoint eight; repeated prefixes are not independent mechanisms.

Unfavorable context is essential: d=768 predeletion curated MSE 0.02364 exceeded uncurated 0.02093, matched-size hash selection 0.01919, and mean prediction 0.02029. This preview does not show useful semantic curation. Twelve strict audited phase-2 checkpoints matched exact state and passed certificates; their maximum radius was `4.5091e-14`. This does not certify every floating row. Independent saved-result audit checked 896 rows, 8,064 metrics and 336 intervals; independent dual predictions differed by at most `1.138e-15`.

### Phase 3: preparation completed, semantic study blocked

Evidence: `phase3/README.txt`, `PREPARATION_AMENDMENT.txt`, `independent_review.txt`, `independent_*_audit.json`, and `results/`.

- Source hash splitting, leakage guards and whole-source panel construction are implemented.
- One scorer now defines calibration, training leakage guard, and fresh graph reconstruction. FP32 inputs are promoted to FP64; norms and dot products are accumulated coordinate-by-coordinate. Strict score `> tau`; no score clipping. This prevents batch-shape/deletion-context changes to pair predicates within the locked runtime. It is not a real-arithmetic cosine certificate or arbitrary-hardware reproducibility guarantee.
- `reference_graph.py` uses narrative priority SHA256(`priority-v1|seed|record-id`), with stable ID tie-break. Historical `ccu.core` graph construction remains for old replays.
- Blinded packs, independent validation sampling and exact hypergeometric bounds are implemented. Actual human ratings remain zero.
- The natural lexical engineering pack has 4,950 frame pairs, 241 sampled pairs and **723 blank assignments**. It is not the primary E5 annotation pack and yields no selected semantic threshold.
- Recorded checks: panel 48, calibration 31, intake regression 69, graph 10 (10,000 score entries), CLI nine success/refusal cases. Independent arithmetic audit checked 2,900 bound cells and 5,200 coverage cells; worst noncoverage was exactly 0.05. Independent scalar scoring checked 4,950 pairs at each of dimensions 128 and 768, multiple tile sizes, retained rebuilds and reversed input order. Independent guard audit checked 27 endpoint cases plus missing/malformed source IDs.

### Phase 4: executable integration, primary study still unstarted

Evidence: `phase4/README.md`, `TODO.json`, each module's README, and
`results/execution_engineering_final/`. The earlier `execution_engineering/`
directory predates the correction to shared request streams and is historical;
do not pool its outputs with the final run.

The core adds original-schema Civil/Stack/News adapters; local E5/MPNet all-chunk
encoding and asset/runtime binding; calibration-only tag/lambda/decision-threshold
selection; R/S/U/A and matched-record request manifests; a six-method fixed-input
engineering runner; a partial dossier inspector; and a fresh-process resource
measurement harness. Real archive acceptance and the real transformer backend
remain unexecuted. Code completion does not supply authentic original sources,
semantic caches, human ratings, primary configurations or the full method system.

The final replay uses the same natural Civil100 lexical fixture. Curator d=128,
threshold0.6; learner d=64/768, lambda0.01, no intercept, current-count shift.
Four R, four U and two A trajectories at checkpoints1/2/4/8 produce **80 rows**,
with **zero failures and 54 zero-admission rows**. Maximum same-target head
discrepancy is `8.326672684688674e-16`; moment discrepancy
`4.440892098500626e-15`. Independent dual ridge checks **480 saved heads**,
including the wrong-target B-F head against its declared target, with maximum
discrepancy `8.881784197001252e-16`. This is numerical agreement on these inputs,
not a rigorous certificate for every floating operation or new semantic evidence.
Compact eligible payload remains smaller than joint-span state on this fixture.

New prospective task conventions are explicit: hash-sort whole sources and assign
round-robin five-fold CV, pool every validation record/output equally, choose
larger lambda only on exact computed loss ties, and maximize per-tag OOF F1 using
exact count ratios and strictest thresholds. Zero-positive tags predict no
positives. Unknown-source and all-zero-target records remain; at least five genuine
groups are separately required for primary source-CV. Test tags never select these
settings; a separate evaluation API can apply the frozen vocabulary. Projected
FP32 features are not silently renormalized. The available preview lacks genuine
groups, so no actual task configuration was calibrated.

R/S/matched-R random streams are graph-independent across curator sensitivities,
while each graph is sealed separately. U/A and graph stress use explicitly
graph-dependent populations. The source service may represent unknown singleton
units, but genuine-source sampling never relabels them as real sources.

The systems harness's 50 checks exercise short software commands, distinct fresh
processes, default five repeats, failure retention, hashes, elapsed time and wait4
RSS. This is not the registered systems experiment. It cannot certify access
isolation, fair method implementations, undeclared I/O, absence of competing jobs,
or summed simultaneous process-tree peak memory. All method code/imported project
files must be explicitly pinned as inputs; interpreter/argv hashes alone do not
bind them. Its measurements remain separate from the multi-method engineering
runner, whose shared arrays do not establish isolated method capability parity.

News/calendar and the encoder bridge are implemented, with 42 original-adapter
checks and 18 calendar/source checks. `NEWS_CALENDAR_AMENDMENT.txt` fixes the
2017 calibration versus 2018–2019 analysis convention and calendar-completeness
requirements. The local encoder passes 63 software checks; versioned E5/MPNet
calibration passes 46. Actual original archive acceptance, transformer execution,
pinned semantic vectors, complete-month evidence and human ratings remain absent.
See `EMBEDDINGS_README.txt`; the software tests do not instantiate real encoders.

The separately named single-output fractional-logistic extension completes 12
natural-preview lexical checkpoints, with 25 rigorous optimizer-error certificates
including the initial fit. Maximum certified parameter radius is
`6.729862803437307e-13`; six rows have zero admissions. The fixed release tolerance
remains `1e-8`. The initial pilot had a warm-solver objective-resolution failure at
A000/checkpoint4: its certificate radius `1.3940385712210056e-8` failed. That run,
source snapshots, certificate and failure diagnosis remain in
`results/convex_engineering/`; the new candidate-generation safeguard and final
run are separately frozen in `results/convex_engineering_release_v2/`, including
the explicitly saved initial head and `convex_checks.json`. The intermediate
`results/convex_engineering_final/` is retained as history. No tolerance
relaxation or hidden cold fallback was used. Extra trial-gradient work is charged.
The rigorous residual certificate, rather than the numerical convergence flag,
governs release. `CONVEX_README.txt` states its exact stored-value target and
arithmetic/resource limits. This does not establish canonical state bytes, privacy,
no-reaccess, source-service support, useful semantic effects or systems speedup.
The prescribed Stack multioutput extension is still unimplemented.

The admission-pair software uses the natural Civil100 lexical records and a
declared frame to select 24 unique pairs, producing 72 **blank** assignments.
No person supplied a rating; missing AskUbuntu/News quotas stay unfilled. The
complete former-blocker and nearest surviving selected-text contextual audits
of 100 admissions each remain pending. Pair packs alone do not complete these
audits or establish that admissions add useful new information. Instructions,
probability rules and limits are in `HUMAN_ADMISSION_AUDIT_README.txt`.

The existing fixed-curator, information, memory and strict-ridge theorem stack is
unchanged by this separately scoped convex optimizer certificate.
Optimized primary B-E/B-A/P-S/P-R integrations, native source services, no-reaccess
workers, persistence/compaction, the true SemDeDup refit branch and other registered
boundary studies remain unfinished. `study_lock.py` intentionally leaves
`execution_allowed=False`; the engineering runner also refuses a primary-role run.

### Phase 5: algorithms, enforced access and experiment components

This section supersedes Phase 4's implementation-pending inventory; the older
results and qualifications remain historical. Read `phase5/COMPLETION_LEDGER.md`
for exact current closure and remaining acceptance rather than reimplementing
completed modules.

- Packed CSR eligible/all-current payload uses integer incidences, eligibility
  buckets, signed batched BLAS, atomic cumulative budgets, source expansion,
  physical stale-byte accounting and fixed compaction. Its8,817 checks include
  8,704 exhaustive small algebra configurations and80 natural Civil states.
- Summary services now include genuine dense indexed P-I, full-scan P-S and
  structural incidence-rank P-R. Joint-span and compact eligible payload remain
  separately named stronger comparators; they are not relabeled P-R.
- Separate construction/repair processes implement all nine methods plus O-G/O-T
  greater-access oracles. After reviewed own-state initialization, seccomp TSYNC
  denies all new opens, network/process inspection and listed bypass syscalls;
  input FDs are closed. Independent kernel tests include already existing native
  threads. This is a trusted-code no-reaccess contract, not physical erasure or
  arbitrary-native-adversary security.
- The natural Civil100 v1 matrix executes 216 jobs and 864 releases, including 648 zero
  admission method/checkpoint rows. Independent audit checks 1,080 heads with
  maximum absolute discrepancy 9.43689570931383e-16. The initial/final snapshots
  are regenerated; exact bytes/hash accounting and all heads/reports are saved.
  Detailed 3,888 files are losslessly archived, with restore/check tooling.
- The negative memory finding persists: at d=768 compact eligible payload is
  about 0.29 MB serialized initially, joint-span about 1.15 MB, dense P-I about 232 MB.
  Rank compression also has significant build workspace. Shared-host timing
  cannot establish a publication speedup; no native 10k feasibility is claimed.
- A separately versioned decoder/worker path adds normalized residual gates,
  conditioning and common zero-start CG. Floating residual diagnostics do not
  certify moment error; the exact-rational strict service remains distinct.
- Multioutput convex training uses a sum of per-output row-average logistic
  losses plus Frobenius regularization. Joint bounds combine exact scalar bound
  squares. The natural seven-target Civil check has 13 joint/91 scalar certificates;
  maximum joint radius 1.070102464743579e-9 below 1e-8. The separate three-arm
  single-toxicity driver checks 8 rows/24 releases/30 certificates; actual S inputs
  stay missing. Twenty-output/source tests are algebra, not Stack experiments.
- The official SemDeDup adapter pins Meta commit
  6b4194511202c29b0e1ac8c730996777449ea2a4 and its license. Separate NumPy
  reference 24-checkpoint/72-head results verify branch orchestration but are
  explicitly not Faiss-equivalent. Real official backend acceptance is pending.
- Boundary code adds true approximate LSH, complete comparison and probability
  sampling of nonretrieved pairs, FP32 state, conditioning and conservative
  graph/model envelopes. LSH recovers 101/332 edges, with 0/33 complete nonempty
  blocker sets and 9 initial selection mismatches. Keep this failure visible.
  The natural interval fixture has no uncertain pairs at its threshold; that
  does not establish an informative semantic-envelope regime in real models.
- WCEP original-event parsing and whole-event panels, chronological News
  deletion, local E5/MPNet encoding lineage and inherited News calibration are
  executable components. No WCEP, original complete News or real transformer
  output has been acquired or executed.
- Complete former-blocker and nearest-surviving-selected context packs provide
  60 blank slots; the original pair pack provides 72. Human-response analysis
  preserves original three-rater outcomes, exact inclusion weights, missingness,
  disagreement and separate adjudication. Human responses remain 0.
- Statistics preserve every planned trajectory, zeros, failures and signed
  losses; use 10,000 whole-path resamples and optional whole-test-source resampling;
  recompute nonlinear F1 and normalizers. The A–G registry contains 43 groups and
  16,933 planned jobs with 21 explicit unresolved recipe/input obligations. A
  manifest does not replace original provenance, quality judgments or primary
  activation acceptance.

The 216-job append journal originally lost 3 rows while each per-job artifact was
intact. The original 213-row journal, recovered 216-row ledger and exact reconciliation
record are preserved. No experimental outcome was rerun or silently imputed.

## Locked preparation rules and why they matter

The high-level protocol and phase-3 amendment are the sources; do not infer rules from a simplified JSON summary alone.

1. Text normalization: NFC and whitespace collapse; preserve case, punctuation, negation and code. Preserve original fields unchanged. Normalized original text must match normalized prepared text; raw-byte equality is not required after prescribed normalization.
2. Split by native source or explicitly unknown singleton, 70/15/15 train/calibration/test. Hash exact bytes `UTF8(salt + NUL + source_unit_id)` using unsigned SHA256 and exact integer quantile tests. Native identity/provenance still requires independent review.
3. Exact text and native duplicate-link guards have precedence **test > calibration > train**. Remove lower-precedence endpoints against the originally assigned population.
4. Civil parent guard is directed and narrower: remove a **training child whose observed parent is calibration/test**. Do not remove reverse or calibration/test parent directions automatically. Missing parent links remain unresolved, not evidence of independence. Null/empty/0/-1 forms are no-parent sentinels.
5. Stack nonpositive community owners become explicit unknown singletons. They are not authentic shared withdrawal sources.
6. E5 semantic filtering removes only training matches against calibration/test, after threshold lock. Calibration/test remain immutable. Use the same post-E5 population for MPNet; no secondary-encoder refiltering.
7. Primary/replication pools are whole-source 80/20 assignments. Panels follow fixed hash order and soft 10k/25k/100k/200k sizes; include whole boundary groups and disclose overshoot/shortfall.
8. Selection: 20 score bins over [-1,1], SRS up to 30 per bin, exact inclusion weights, maximum 600 pairs. Six categories remain available. Only `exact_copy` and `substantially_same_meaning` count positive; `overlapping_information`, `merely_related`, `unrelated`, `uncertain` do not. Three distinct independent blinded humans per pair; at least two positives makes the pair positive.
9. Choose the least strict grid threshold 0.50,...,0.99,0.995 whose weighted count ratio is at least 0.95 with nonzero support. Support is reported; this is a development selection rule, not proof. If none qualifies, 0.995 is only a failed diagnostic.
10. Fresh validation is independent SRS up to 200 from all above-threshold pairs, or census. Selection overlap is allowed but needs fresh blinded judgments. Invert exact hypergeometric upper tails; lower count is least `K` with `P_K(X>=x)>0.05`. Pass only if `10L>=9N` and selection qualified. Empty pair populations have undefined precision. No threshold retuning after validation.
11. This coverage is conditional on fixed operational majority outcomes under SRS. It does not include latent truth or unmodeled rater variability. Attestation fields/seals are consistency checks, not signatures or human authentication. `confirmatory_study_ready` remains false even after the mechanical gate.

The correction from phase-2 intake is versioned in `phase3/intake_v2.py`. Do not overwrite old phase-2 code/results to make them appear retrospectively compliant.

## Remaining empirical program

The planned primary labeled corpora are source-rich Civil Comments and question-level AskUbuntu from the declared corrected April 2024 Stack snapshot. Civil uses original toxicity fractions; AskUbuntu uses 20 calibration-selected tags, preserving zero-target questions, with owner-account source proxies. CC-News provides structural/human evidence, English Stack Exchange within-platform replication, and WCEP event-enriched replication. Native source metadata and archive/parser/date/link completeness matter; current previews cannot instantiate these populations.

Primary encoder is `intfloat/multilingual-e5-base`; secondary is `sentence-transformers/all-mpnet-base-v2`. Native dimension 768, data-independent projection ablations 32/64/128/256. The declared document representation uses all nonoverlapping 256-content-token chunks, content-token-weighted pooling and normalization; E5 prefix is `query: `. Pin actual model/tokenizer revisions and runtime. The learner consumes the frozen FP32 vector promoted exactly without additional normalization; graph scoring uses the defined normalization.

Primary methods include independent fixed-graph rebuild O-G, fresh moments on the correct selected set O-T, eligible payload B-E, all-current payload B-A (greater capability disclosed), indexed summary P-I, scan P-S, rank-coordinate P-R, wrong-target frozen-selection diagnostic B-F, and separate actual refitted-curator O-R. Later joint-span/compact-payload/strict variants must be integrated prospectively and fairly. Shared solver/stopping rules, input geometry, request paths and storage accounting are mandatory. A speedup over full retraining alone does not establish advantage over a strong incremental baseline.

Planned record horizon is 128 at checkpoints 1/8/32/128, source horizon eight at 1/2/4/8; clipping and feasibility are explicit. R/S/U/A requests and source-size-matched record controls are frozen without outcome inspection. Structural/task allocations are larger than full-method allocations; see `study_design.json` for exact counts. No optional stopping. Include all structural zeros, failures, empty targets and signed adverse effects. Native source uncertainty is separate from trajectory randomness. Replication, original-SemDeDup frozen-versus-refitted branch, convex extension, human admission audit, and repeated isolated systems timing are still required by the full program, not fulfilled by the current pilot.

The nominal protocol proposes a 256 GiB host, 128 GiB method caps and an accelerator; these are proposed resources, **not available measured hardware or authorization to launch cloud jobs**. Current local resources are much smaller. Exhaustive scoring requires O(N²d) arithmetic even though tiles use O(Bd+B²) temporary storage; threshold-edge storage is O(N+m), quadratic in the worst case. The CLI gathers calibration FP32 vectors in O(N_cal*d) storage. No scalability or memory feasibility at 10k–200k has been demonstrated.

## Exact next actions

1. Preserve this repository snapshot and current result hashes. Start a new run directory for any fresh empirical execution; do not overwrite original registries or annotation manifests. Re-read permissions/capabilities in a new chat: prior network or runtime restrictions may change, but never assume missing assets now exist.
2. Acquire authentic source-rich original corpus files by a permitted route into this workspace, or receive them as attachments. The checked HF connector exposed metadata/schema/preview rows and public URLs but not an available byte-import operation. Its Civil export has text plus seven labels, no required article/publication IDs. No supported import was found; do not bypass restrictions or relabel metadata as data.
3. Run acceptance checks on the implemented `phase4/adapters.py` using authentic archives: Civil native IDs/parents/publication/article completeness; Stack question/date/tag/owner/duplicate links; News title-plus-body, dates and pinned-PSL domains. Code exists, but original-archive acceptance does not. The frozen phase-3 News body-only verifier is not the newer title/body rule. Check the News/calendar extension's final state and coverage evidence before constructing calendar windows.
4. Run local phase-3 `prepare` on normalized authenticated records with original fields and native duplicate links. Review source/missing-source counts and fixed guards before embeddings/panels, preserving the versioned Phase 4 adapter provenance.
5. Supply pinned E5/MPNet FP32 N-by-768 vectors with ordered row IDs and derivation evidence, or local model/tokenizer assets and runtime for `phase4/embeddings.py`. The adapter is implemented; its actual transformer backend remains unexecuted. The recorded environment lacks the required torch/transformers stack. No remote compute was used or authorized by the study instructions.
6. Bind cache bytes, normalized text/IDs, full prepared rows, preparation audit, encoder revision and scope using phase-3 and the final versioned per-encoder wrapper instructions. Generate selection assignments. Have genuine independent blinded humans complete them; do not fill blank labels or flip attestations computationally.
7. Lock threshold, generate independent validation assignments, collect fresh human judgments, run exact quality gate. A failed gate stays failed; no retrospective retuning. Human collection/provenance remains outside what JSON can prove.
8. Apply training-only E5 semantic guard and freeze whole-source primary/replication panels. Execute the implemented calibration-only vocabulary/lambda/threshold and request selectors on real inputs. Freeze code/runtime hashes, resource caps/timeouts, compaction/persistence, full-refit policies, annotation protocol and costs. These locks are empirical outputs, not values that may be invented from tested code.
9. Use the completed Phase 5 method/analysis components and current completion ledger; do not restart the obsolete Phase 4 implementation list. Resolve the remaining recipe, instrumentation and authentic-backend acceptance obligations, then exercise the complete primary activation chain. Run structural audit and native memory feasibility before the prescribed task/source/replication/full-refit program. The decisive unknowns remain meaningful natural additions, task consequences and measured total cost versus compact eligible payload.

## Reproduction entry points

Recorded versions: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0. See `empirical_execution/requirements.txt`. Reruns need no network once dependencies and the shipped Civil fixture are present. Preserve recorded results first; many historical runners replace their result directories.

From repository root, light phase-3 verification:

```bash
python3 empirical_execution/phase3/check_reference_graph.py
python3 empirical_execution/phase3/check_workflow.py
python3 empirical_execution/phase3/check_intake_v2.py
python3 empirical_execution/phase3/audit_finite_population_independent.py
python3 empirical_execution/phase3/audit_scoring_and_pack_independent.py
python3 empirical_execution/phase3/audit_guards_independent.py
```

`check_calibration.py` verifies the existing pack and refuses differing replacement; code changes require preserving its old code/frame hashes. `check_panels.py` also needs the local News text fixture that portable archives intentionally omit. Its recorded audit remains valid historical evidence, not automatically a replay result after code changes.

For current Phase 4 software verification, read `phase4/README.md` first. Examples:

```bash
python3 -m empirical_execution.phase4.check_model_selection
python3 -m empirical_execution.phase4.check_systems
```

These checks use explicitly marked transient algebra/command fixtures; they do
not create synthetic empirical datasets or execute the primary study. Preserve
their recorded audit JSON or rerun in a separate checkout. The authoritative final
engineering comparison registry is `phase4/results/execution_engineering_final/`.

Historical Civil-only pilot from `empirical_execution/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_engineering_pilot.py --skip-structure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_memory_repair.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_canonical_validation.py
```

Phase-2 replay from root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 empirical_execution/phase2/run_development.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 empirical_execution/phase2/audit_development_results.py
```

The exact helper is platform-specific; use its provided source/compile instructions or `backend='python'` fallback. `run_preparation.py --help` and `phase3/README.txt` specify the production `prepare`, `selection`, `threshold`, `validation`, `quality`, and `guard` sequence. These refuse inconsistent inputs or absent genuine responses rather than silently advancing.

## Deliverable map and resumption warning

Readable PDFs are in `output/pdf/`: theory, empirical protocol, execution_01, memory_repair, strict_mode, empirical_phase2, and empirical_preparation. Their corresponding ZIPs preserve stage-specific source/results. `output/counterfactual_curation_reproducibility.zip` is earlier theory/reproducibility material, not the complete latest state. Use the repository's live source plus explicit current status for continuation; do not resume only from an old archive.

There is not yet a finished ACL submission backed by the promised semantic study. The algorithmic qualifications addressed above are real progress, but “reviewer-proof,” superiority to strong incremental baselines, broad learned-curator exactness, and semantic task value are not established facts. A new chat should resume from `phase4/TODO.json`: acquire/accept authentic inputs, execute the phase-3 plus versioned phase-4 preparation workflows, and finish the remaining primary integrations while preserving theorem scope and all unfavorable results.


## Phase 6: complete routes and stronger verification

This section supersedes earlier software gaps.
The Phase 6 completion ledger gives the final counts and review bindings.
The primary semantic study remains unstarted.

The main fixed-curator theorem is unchanged.
Selection still uses every earlier raw neighbor, including previously excluded records.
The finite horizon and complete source partition remain mandatory.
No algorithmic result establishes minimum total bytes for the compressed summaries.
Compact eligible payload remains the key comparison.

The dispatcher now joins the registered branches and methods.
The recipe book resolves the 21 prior design obligations prospectively.
Its expanded registry preserves the original experiment jobs.
It distinguishes unavailable inputs from implementation choices.
These decisions are not an externally submitted preregistration.

Acceptance replays source parsing, preparation, caches, guards, panels, and original labels.
The registered panel contract determines the target population.
The registered request contract determines the cumulative horizons.
External evidence remains necessary for archive identity, coverage, permissions, and independent human work.
Positive control tests use explicit mocks where the real backend is unavailable.
They do not establish authentic corpus or transformer acceptance.

The worker layer now includes B-F, a cached graph tier, and all eight named FP32 variants.
It also joins common CG, fresh process, warm service, and every-release panels.
Memory reports distinguish logical state, process observations, and unmeasured physical traffic.
Process sampling does not guarantee capture of every transient peak.
The no-reaccess boundary remains a trusted-code contract.
It does not establish physical erasure or general adversarial security.

The exact convex verifier gains a dyadic integer backend.
It preserves the original sigmoid intervals and every rational gradient endpoint.
It therefore preserves the existing strong-convexity certificate.
Tests include empty targets, subnormal values, and genuine Civil labels.
Forty separate process runs compare sparse and dense lexical representations.
The dense representations use a fixed orthogonal transformation.
They are not semantic model outputs.
The native primary problem has not run.
Work caps and resource forecasts remain explicit.
A larger cap must be locked from suitable development measurements before confirmation.

Logistic decisions now use separate source-fold calibration probabilities.
They retain the registered ridge regularization choice.
They do not reuse ridge score thresholds on logistic probabilities.
Human category intervals now invert the actual finite population sampling distribution.
The analysis handles overlapping sampling routes and arbitrary missing outcomes.
Simultaneous coverage uses the declared Bonferroni allocation.
Its target is the fixed protocol outcome, not latent truth or new-rater variation.
All actual human responses remain absent.

The remaining scientific questions require genuine primary inputs.
They concern meaningful additions, task effects, source behavior, memory, runtime, and full-refit scope.
Repeated engineering checks cannot answer those questions.
The next productive action is original input intake, followed by accepted local execution.


## Phase 7 input and execution follow-up

The user resumed after the saved Phase 6 checkpoint.
No theory statement or frozen Phase 6 implementation changed.
The new dispatcher CLI passes an explicit resource policy unchanged.
The calibration CLI replays accepted inputs before creating blank forms.
A resource preflight evaluates existing audit and convex work limits.
It does not turn cap counts into measured runtime or memory.

The native pair audit exceeds the default limit for 10,000 records at dimension 768.
A larger limit needs genuine development qualification.
Original archive schemas also require explicit export lineage.
The acquisition map records official locations and published hashes where available.
No original archive, encoder execution, or human response was acquired.
See phase7/CHECKPOINT.md for the current continuation order.

## Phase 8 preparation and cost qualifications

No theory statement or frozen Phase 3–7 source changed.
Pinned original archive exports now preserve raw values and full derivation lineage.
Independent replay hashes regenerated bytes without writing a second full corpus.
The exporter still requires storage for the original archive and complete first export.
Downstream records, semantic caches, model assets, and snapshots add separate costs.
Full-corpus peak memory and runtime remain unmeasured.

The new dossier stages separate calibration, unsigned candidates, development attachment, and supplied final review.
They cannot invent source provenance, labels, human outcomes, or external testimony.
Stable paths are part of the signed input meaning.
The current finalization scope is one explicit family.
The current resource policy scope is native core audit counts.
Other routes remain in the Phase 8 remaining-task ledger.

An observed work envelope is an operational cap-selection rule.
It is not a theorem bounding runtime or peak memory on another graph.
Graph structure, symbolic overlap, conditioning, serialization, and host behavior still affect cost.
Unknown future symbolic-key counts remain unknown.
Neither matching coordinate counts nor larger configured limits prove practical native feasibility.
The main semantic study, human collection, and paper-ready empirical conclusions remain absent.


## Phase 9: realizable feature-scale memory

Read `empirical_execution/phase9/THEORY_EXTENSION.md` and its independent `THEORY_REVIEW.md`.
The LaTeX fragment is `empirical_execution/phase9/theory_extension.tex`.
The novelty audit and contribution-position note define the intended paper claims.
Earlier theory files remain frozen.

The new hard family closes the arbitrary-statistics versus realizable-ridge gap.
It varies hidden unit features inside strict-margin caps while keeping the complete cosine graph fixed.
The curator and learner use those same features.
Forgotten blockers have fixed public payloads.
Every initial trained model is zero.
All targets through deletion budget b are zero.
At budget b+1, a probe selects exactly the anchor and one hidden retained candidate.

With C = aa-transpose + 2 lambda I, its head is C-inverse w beta-transpose.
Positive response factors make the unit feature and response coordinates identifiable.
This gives the exact continuous query-summary minimum m(d-1+q).
All actual response coordinates are bounded.
A separate finite packing yields a randomized information lower bound at an explicit positive error scale.
The proof uses classical topology, packing, and Fano arguments.
The new content is the simultaneously realizable geometry, outputs, and access/budget construction.

The binary-label corollary fixes every candidate label to one and every blocker label to zero.
Only candidate features remain private.
Normalizing C times the resulting head still recovers each feature direction.
The exact continuous minimum becomes m(d-1), with a separate finite-bit packing bound.

The coordinate result requires a continuous deterministic encoder into a fixed-dimensional Euclidean state.
An unrestricted discontinuous real encoding can invalidate such a coordinate claim.
The finite-bit result counts all candidate-dependent state and permits independent public randomness.
Its approximation scale depends on regularization and can be small.
No simultaneous-success event across all probes is assumed.
The adjacent-budget lower bound already applies to possible first requests of a sequential service.
Its matching chart upper bound has only the initial query-summary contract.

For fixed b, this construction matches O(E(d+q)) numerical scaling in a worst-case coordinate sense.
It does not prove optimal total bytes, rational bit complexity, update time, or workspace.
It does not establish a fixed-FP32 packing or natural encoder realizability.
The original compact eligible-payload comparison remains unfavorable to larger summaries.

The experimental preparation update preserves numerical targets and earlier result bytes.
No primary semantic experiment or human collection occurred.
Read the Phase 9 checkpoint for software review evidence and remaining requirements.


## Phase 10 finite-format and sequential qualification

The new proof closes two implementation-model gaps from Phase 9.
It keeps earlier sources and conclusions unchanged within their original scopes.
Read the Phase 10 theory, scorer transfer, and independent review before citing it.

Public signature blocks and private sign blocks occupy orthogonal feature coordinates.
All coordinates are exact dyadic FP32 values.
A uniform arithmetic proof transfers every edge decision to the frozen FP64 scorer.
It covers all private signs, not merely tested fixtures.
The graph, labels, and forgotten blocker payloads are public and fixed.
Targets through budget b vanish; designated budget-b+1 requests reveal individual private blocks.
The resulting exact finite-state lower bound needs no continuity assumption.

The initial private-information minimum is mD bits.
The conditional minimum after deletion F is e_F D bits.
The canonical codec keeps only future-eligible blocks under the decreasing cumulative horizon.
Fresh retained initialization and sequential repair produce identical authoritative bytes under the public construction contract.
The byte format includes extra metadata and padding.
Public construction storage, outputs, and workspace remain separately charged.
The code is a service for this finite family, not a general compressor for semantic corpora.

The approximation theorem uses classical rate-distortion reasoning at an explicit head-error scale.
Its initial-summary upper bound does not supply an efficient sequential approximate service.
The certified-release corollary bridges exact targets and sufficiently accurate released heads.
It does not certify an arbitrary floating solver automatically.
The output contract remains essential, including signed-zero and rounded-output distinctions.

The literature follow-up sharpens the claim using primary manuscripts and author code.
The broad memory separation already has a close counterfactual-auditing precedent.
GRACE selects forget/retain coresets for behavioral unlearning.
It does not rerun the original training curator under our raw-deletion target.
Full proof access for the recent auditing paper and the Dominici thesis remains unresolved.
No exhaustive novelty guarantee is made.

The semantic study, genuine human audits, paper figures, and empirical prose remain pending.
The missing assets are unchanged.
The existing compact-payload byte advantage remains visible and unchanged.
