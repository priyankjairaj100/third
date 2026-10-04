# Systems and algorithm evaluation design — natural corpora only

This is a design specification, not a report of experiments. It is based on theory specification version 3. All corpus experiments use existing, naturally occurring documents and their real labels/source fields. No planted chains, generated paraphrases, randomized labels, or artificial duplicate replication belongs in the empirical evidence. Existing proof-verification scripts remain software checks, not dataset experiments.

## The claims the systems experiment should adjudicate

1. At the same fixed-curator target and the same cumulative horizon, does the indexed coefficient method improve the total storage/latency frontier over an optimized eligible-payload antijoin plus exact moment updates?
2. Do observed work and storage follow the structural quantities the theorems identify: eligible-record count, rank, coefficient count, key incidences, merges, and release-time solves?
3. Does its advantage survive physical metadata, float precision, solver cost, construction, serialization, and realistic source/request distributions?
4. When the curator is actually refitted, how large is target mismatch, and when is the conditional envelope certificate useful?

Do not set up the experiment to guarantee that the coefficient method wins. A result that it loses at native embedding dimension but wins for count statistics or very low-rank blocker structure is meaningful and must change the paper's claim.

## 1. Pin the target before naming a baseline

| Target | Oracle | Legitimate comparison |
|---|---|---|
| F: fixed public encoder, partition, total order, threshold | Rerun suppression on retained records using the specified fixed objects; rebuild moments and solve the same objective | Primary exactness and speed/storage comparison |
| C: encoder fixed, partition/order originally fitted once to the deletable corpus and then frozen by policy | Rerun suppression conditional on that frozen state | A separate operational target; do not call it full-pipeline erasure |
| R: encoder fixed, clustering/priorities refitted on retained data | Execute the pinned original clustering/order/suppression algorithm on retained embeddings, then rebuild moments and solve | Full-refit fidelity audit and conditional bounds; the exact F theorem does not apply |
| S: retain original selected set and remove only selected forgotten records | Relearn on C(D) minus F | Wrong-target diagnostic, never the main speedup denominator |

For F, the fair rebuild oracle is permitted to reuse retained fixed features and the fixed graph if that state is charged. Do not force repeated embedding or pairwise-similarity computation on the oracle while caching it for our method. Also show the uncached initial raw-text pipeline cost once. Under R, reuse of fixed retained embeddings is legitimate; rerunning a frozen encoder on unchanged text is avoidable work. Clustering/refitting costs remain chargeable.

The headline core should use an external calibration partition/order or a genuinely corpus-independent rule. A centroid-distance ordering fitted to D is not made restriction-consistent merely by pinning its seed. Describe any external calibration data, its disjointness, retained bytes, and the policy that it is outside the deletion universe. If a learned projection/PCA is fitted to D, either refit it and change the target or remove it from the core. A public seeded projection is a valid, explicitly declared alternate learner.

## 2. Mandatory baseline ledger

| ID | Method and retained state | Reason it is required |
|---|---|---|
| O-F | Fixed-target fresh rebuild from retained fixed embeddings/blockers; reconstruct selected IDs, BLAS moments, common decoder | Correctness oracle and realistic rebuild cost |
| O-R | Full curator refit, with the same retained fixed embeddings and same downstream optimizer | Measures scope gap; not the same-target speed comparator |
| B-S | Original-selection deletion-only ridge update | Exposes target error when F contains initially excluded records; always labeled wrong target |
| B-E | Eligible-payload incremental antijoin: store z,y for E_k, blocker lists, inverse lists, live blocker counts, degree/eligibility buckets, selected membership, and current moments; update moments on admissions/removals | Strongest essential baseline; straightforward data maintenance already solves the curation dynamics |
| B-A | All-retained-feature antijoin plus current moments | More general service baseline with larger supported horizon; separate capability label |
| P-scan | Sparse coefficient map with whole-map substitution and canonicalization | Isolates the effect of indexing |
| P-index | Proposed indexed coefficient implementation, same precision and decoder | Main proposed method |
| P-rank | Canonical rank-optimal coordinate baseline with eligible-record structural metadata and explicitly charged reconstruction | Tests whether saved statistic bytes justify larger repair/reconstruction work |

B-E must receive the same horizon k. After a cumulative request of r new identifiers, its eligible-payload state is pruned against remaining horizon k-r; charge degree-bucket/inverse-index work and freeing/compaction policy. It must not retain all N rows and be called the optimal-payload baseline. Conversely, P-index cannot silently keep z/y in a global array accessible during repair. Both may receive full forgotten payloads in an explicitly separate access-model ablation, but core IDs-only repair is simpler and stronger. Publish two distinct access strata if both are evaluated: IDs-only, and IDs-plus-full-forgotten-payload. The latter changes the side information and may lower the needed summary; the arbitrary-statistic IDs-only rank theorem cannot be applied unchanged. Give every method the same request payload within each stratum and account for its transmission/read cost.

Implement B-E well: CSR or compact integer incidence arrays, counter decrements, vectorized batches of additions/removals, and BLAS moment updates. Use the same systems language and numeric backend for P-index and B-E. The current dictionary proof implementation is not sufficient evidence for systems superiority. An optional SQL/streaming-engine implementation of the antijoin is useful corroboration, but a deliberately slow generic SQL baseline cannot replace B-E. Attribution to incremental-view maintenance and existing signed ridge updates is essential.

## 3. Native-dimensional storage is the central feasibility risk

For C response columns and d features, pack the symmetric moment matrix. Each coefficient needs

    s = d(d+1)/2 + d C + 1

numeric coordinates. At FP64, for C=4:

| d | Bytes/coefficient excluding metadata | 100,000 coefficient vectors |
|---|---:|---:|
| 64 | 18,696 | 1.87 GB |
| 128 | 70,152 | 7.02 GB |
| 384 | 603,656 | 60.37 GB |
| 768 | 2,386,952 | 238.70 GB |

These are decimal GB and illustrative arithmetic, not measured results. Recompute for the actual class count and representation. With equal per-number precision, statistic storage beats eligible feature storage only at roughly M/|E_k| < 2/d, before metadata; FP64 moments against FP32 features require still more compression. The factor-two-to-rank theorem alone does not imply a practical memory win.

There is a particularly strong existing-theory screening bound: for k>=1, every originally selected record has b_v=0<k, hence r_k>=|S|. Thus a rank-optimal linear summary for s arbitrary independent statistic coordinates already needs at least |S|s coordinates. This is a floor for that representation/information model, not a newly established lower bound for every nonlinear or structured PSD-ridge encoding. It is nevertheless decisive for our concrete dense P-rank/P-index designs. At 50% initial retention and d=768, individual FP32 rows are about 3 KB while individual FP64 coefficient vectors are about 2.4 MB; normal retention rates make a blanket space-saving claim implausible. Directly audit actual M and r rather than hiding this behind the word compression.

Before allocating summaries, compute the blocker/rank/count curves and a byte forecast at the native encoder dimension first. Report all infeasible settings as infeasible under the predeclared memory cap. Do not drop them from figures or switch to an unreported lower dimension. The stage-0 pass/fail gate is literal: if the native-byte forecast cannot fit the predeclared cap or no practical byte advantage exists over B-E, do not promise an overall memory-efficient native-feature method. Continue only with an honest time/memory Pareto claim, smaller natural audit sets, or a low-dimensional alternate-learner ablation. A public fixed projection to 128 may be such an ablation; report its task-utility loss and compare all methods on identical features. Include d=32/64/128/256/native as a one-factor sensitivity on one corpus, not a giant factorial. Do not retrain/test-tune projections to favor compression. Negative native-dimensional feasibility is a result, not permission to rewrite the primary target after seeing outcomes.

Keep curation feature dimension d_c and learner feature dimension d separate in manifests. In the learner-dimension ablation, hold the curation graph fixed; otherwise changing d changes both the problem and the storage cost. If a separate curation-encoder robustness experiment is run, it is a separate intervention with a new exact graph and newly computed structural quantities.

Persistent memory table must include values, packed keys, incidence indices, degree buckets, live membership, source mappings, model/decoder state, public calibration/projection artifacts, allocator slack/tombstones, and all retained features/text. Report logical live bytes, serialized compressed/uncompressed bytes, process private/resident memory, and peak RSS separately. Count any retained original graph or original coefficient copies. Include temporary copies and decompression/compaction peaks. Peak GPU allocated/reserved memory is separate from host memory. A total-bit optimality claim is unsupported.

## 4. Solver parity and numerical correctness

Use average-loss ridge throughout the exact core:

    (M_t + lambda n_t I) Theta_t = H_t.

Do not omit the lambda times count shift when documents enter. Sum-loss ridge is a separate ablation with a different target. Fix intercept regularization conventions; the simplest primary target regularizes all coordinates, including an appended constant if used. The theorem's lambda-strong-convexity statement does not automatically cover an unregularized intercept.

Primary fairness uses a common decoder with common stopping tolerance and precision for every maintenance method. Report maintenance-only and maintenance-plus-solve times. Two decoder modes: FP64 Cholesky with residual verification for moderate d, and zero-start CG using current moments, a declared residual criterion, and the same tolerance across methods. Common decoder removes algorithm-specific solver engineering from the comparison. A separate best-engineered end-to-end comparison may let methods choose among correct Cholesky/CG/update variants using validation timings; publish the chosen policy and charge all state. Woodbury is valid only for the actual system change, which includes the count shift.

Measure moment/count error, normalized linear-system residual, parameter error, prediction error, held-out objective discrepancy, and selected-ID equality. FP64 is the primary arithmetic. FP32 is a separately labeled approximation frontier. Near-cancellation in signed coefficient sums should be audited by long sequences, varying lambda, and independent rebuilding; use compensated sums/rebuild fallback only if its work and retained access are charged. Do not remove a coefficient using an arbitrary near-zero threshold while retaining the exactness label.

There is no universal bitwise equality claim for floating-point sums in different orders. Before test runs, set numeric acceptance through backward error/residual and an explicit error certificate. Example initial release gate: zero selected-ID discrepancies, exact integer count equality, normalized solve residual <=1e-10 in FP64; parameter comparison additionally scaled by conditioning. If a configuration cannot meet the declared bound, retain it as a numeric failure instead of loosening tolerance after seeing results. Exact integer/rational proof tests remain available as unit checks, but not as synthetic empirical experiments.

## 5. Exact graph construction and approximate-search control

The core must evaluate all within-partition threshold pairs, in tiled exhaustive matrix multiplication or exact normalized inner-product range search, for its declared graph. It may shard by a fixed public partition; this is the target, not an unmeasured approximation to a global graph. Strict threshold > tau, normalization precision, ties, priority tie-breaking, chunking, and index/ID mapping must be pinned. A top-K nearest-neighbor list is not a threshold-neighbor oracle: dense neighborhoods can have more than K blockers.

Approximate candidate retrieval is a secondary engineering experiment. Missed blockers create false admissions, while retained-data rebuilding on the approximate graph merely validates the wrong graph. On audit subsets, exhaustively compute all pairs and report edge recall/precision, per-record blocker completeness, eligibility errors, selected-set mismatch, and model discrepancy. Estimate false negatives with a probability sample of pairs/candidates that includes ANN misses; inspecting only retrieved pairs cannot establish recall. Even 99.9% edge recall is not a universal exactness certificate. Extrapolation to the entire large corpus is statistical, not a graph-validity theorem.

Fixed graph envelopes are a separate certified mode only when upper/lower edge bounds follow from deterministic score/partition/order bounds. Empirical ANN recall is not such an envelope. Arbitrary reclustering/order changes cannot be smuggled into an edge-only fixed-order certificate.

## 6. Workloads and trajectory semantics

Preserve natural documents, labels, and source ownership. Workload selection may be adversarial on this fixed natural corpus and is labeled as such. Include uniform-record requests, initially excluded-only requests, observed natural whole-source withdrawals, and graph-only high-activation requests. No request family is chosen using held-out task loss. A high-activation heuristic is not an optimal adversary. Report realistic and targeted cases separately, never average them together.

All methods replay the same precommitted request IDs, order, batches, releases, and original state. Cumulative horizons decrease; a reset requires rebuilding from retained data and its cost. Count fresh unique deleted IDs rather than requests. Source budgets count disjoint sources; report both sources and records affected. For source/document comparisons, include record-volume-matched workloads because one publisher can contain thousands of records. Source collapse uses distinct blocker-source identities and removes same-source impossible activations correctly.

At minimum, run a long singleton trajectory and a batched trajectory with identical final deleted IDs. Compare end states after identical cumulative prefixes, plus a reversed/permuted order and retries. Test out-of-budget and invalid-ID handling before benchmark runs; requests are atomic and retries do not consume horizon. Use real IDs for these tests. Probe budget exhaustion with one real extra deletion to verify the documented refusal/rebuild policy. Include checkpoint save/load/resume and canonical serialization equality modulo declared numeric tolerance. Do not equate logical canonical state with erasure from allocator memory, logs, or backups.

Useful main horizons are 32 and 512 records, with a one-factor horizon curve through all structurally available k; natural-source horizons 1 and 8 are a separate panel. The eventual corpus sizes may require smaller source universes: declare exclusions before outcomes. For a larger corpus, also include a 1% cumulative-record budget to reveal whether an apparent advantage survives beyond tiny-k settings. Structural curves are cheap enough to display the whole support, while dense learner construction need only instantiate selected horizons.

## 7. Timing, I/O, and break-even accounting

Record four separate cost stages: (i) encoder and graph preparation; (ii) method-specific state construction and initial fit; (iii) request validation, repair, solve, and required output; (iv) persistence/compaction/export. Report elapsed wall time and cumulative totals, not only kernels. Cold-process and warmed-service results are separate; state startup/load latency belongs in cold results. Avoid undocumented OS cache eviction requiring privileged commands: use fresh processes and disclose cache policy. If a true uncached-disk test is unavailable, label it accordingly.

Use paired hardware runs with randomized method order, pinned CPU thread count, GPU/model IDs, RAM, BLAS version, precision, and software commits. Ensure no competing jobs. Warm up numeric libraries for warm results. Explicitly synchronize asynchronous accelerator work at timing boundaries or use the official benchmark timer; report GPU kernel time separately from wall time. Use at least five independent fresh-process timing repetitions of representative precommitted trajectories; bootstrap uncertainty over independent trajectories, not individual correlated updates. Do not repeatedly execute a destructive update on already-mutated state as a timing repetition.

For method A versus B, plot measured cumulative cost

    setup_A + sum_{t<=q} release_cost_A(t)

against the same quantity for B. Break-even is the first q where the proposed cumulative cost is lower and remains lower over the observed remaining trajectory, if it exists. Include load/export policy identically. No extrapolated break-even beyond the supported cumulative horizon is a measured result. Report speedup against B-E and O-F separately. A method slower than B-E must not headline a speedup over only a slow rebuild.

No-hidden-reread audit: isolate the repair service from raw text/feature files after construction; use an explicit capability interface for allowed forgotten-payload access; deny or log all retained-record lookups. Track retained bytes read, forgotten bytes read, metadata bytes read, and fallback calls. A common in-process corpus array retained by the benchmark is a leak even if our function usually does not inspect it. Oracle worker and evaluation process may retain data but must be inaccessible from the repair worker. In particular, evaluation labels/test data cannot be used to select repair decisions.

Instrument incidence touches, key moves, key merges, expired entries, statistic additions, active M/L/Phi, output admissions/removals, solver iterations, and peak bytes per request. Plot cumulative observed touches against initial L and merges against initial M. Operation-count assertions validate the implementation contract; wall-time regressions against these counts test practical predictiveness. State serialization is not in the indexed theoretical update bound, so show its cost explicitly if the service contract requires a checkpoint every release.

## 8. Minimal informative design instead of a huge factorial

Use three nested experiment blocks.

**Block A: natural-structure audit.** Every selected corpus and every preregistered curation threshold/order configuration; calculate blocker histogram, E_k, rank, coefficient count, source-collapse structure, estimated bytes, and activation workload outcomes before dense moments. No model training is needed. Include nested raw-document samples of 20k/100k/500k if the corpus has that scale; preserve source/date grouping for uncertainty and report sampling effects. The graph changes under subsampling, so each size is an independently specified natural subcorpus, not a claim that its structure represents the full graph.

**Block B: decisive exact systems comparison.** Two labeled natural corpora, two curation operating points, two record horizons; all mandatory same-target methods at the same feature dimension. All workload families reuse the same states. Three fixed priority seeds may assess order sensitivity, with one public priority fixed as the primary setting. On a provenance-rich corpus, add the separate source-deletion panel. Replay multiple independent trajectories; the statistics lead should power their count. Use five fresh-process timing repetitions on representative frozen trajectories, not five unpaired dataset variants.

**Block C: one-factor boundary tests.** On one sufficiently large natural corpus: feature dimension; cumulative budget; batching/release cadence; approximate neighbor retrieval; FP32/FP64; count-normalized ridge vs explicitly different sum-loss target; full refit vs fixed target; persistence policy. Keep these out of the main full factorial. Run the full-refit oracle at predeclared endpoints and high-risk prefixes; at least one smaller complete trajectory must be fully oracled to rule out endpoint-only failures.

Resource planning is driven by measured graph size and the byte formula, not a promised number of GPU-hours. One 24–48 GB GPU for batched embedding/exact tiled similarities and one CPU host with 128 GB RAM are a reasonable initial configuration; publish this as a proposed resource envelope, not a guarantee. Cap dense method states at a predeclared 64 GB host allocation on that machine so workspace and oracles remain available. Store sparse graph audit artifacts on disk. Do not claim web-scale dense ridge summaries from a corpus audit alone. The largest natural corpus may support only structural/count-statistic audits; mark that limitation.

## 9. Main figures/tables that make objections answerable

1. Target table: selection equality, moment residual, prediction discrepancy for all methods; F and R targets never pooled.
2. Total persistent bytes versus p95 release latency Pareto plots, with exactness constraints and both native/practical dimensions; mark OOM/failure regions.
3. Cumulative end-to-end cost and break-even versus consumed horizon, against B-E and O-F.
4. Structure-to-cost panel: measured r/E, M/r, incidences, activated additions, and observed bytes/work. The factor-two result is not substituted for actual bytes.
5. Source panel with natural record-volume distribution and source-budget versus record-volume-matched comparisons.
6. Refitting-gap/certificate panel: actual mismatch, certificate, and certificate-computation cost, including vacuous/inapplicable cases.
7. Supplement ledger: every persistent object, raw/feature read count, solver settings, build cost, all failed configurations, and complete request manifests.

## 10. Non-negotiable release gates

- Oracle implementation independent of the proposed coefficient code: direct retained selection plus independently rebuilt moments.
- B-E implemented and profiled before any systems claim; same solver, cache policy, and horizon.
- Native-dimensional byte audit completed; no surviving hidden retained-row access in P-index/P-rank.
- Integer selection/count equality at every release of smaller full trajectories; floating certificates checked and failures retained.
- Strict similarity/order semantics validated against the pinned released suppression code on actual corpus shards.
- All-state claims restricted to designated logical state; serialization and numeric tolerance explicitly defined.
- Fullrefit mismatch never relabeled exact unlearning; freeze the interpretation before test results.
- All unsuccessful/infeasible runs appear in the registry; each headline result includes its strongest relevant baseline.

## Primary sources checked for this design

- Official SemDeDup implementation: https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py (web ref turn95view2). This establishes the concrete implementation to pin, not a proof that arbitrary fixed priorities reproduce fitted centroid priorities.
- Quan, Wu, Montana, Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models, arXiv:2603.12977v3: https://arxiv.org/abs/2603.12977 (turn95view0). Add/delete sufficient-statistic ridge updates are established prior work; our baseline must include them.
- Official FAISS index documentation: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes (turn95view1). It distinguishes exhaustive flat search from nonexhaustive HNSW/IVF; exact postverification of retrieved candidates does not remove missed-candidate errors.
- Official PyTorch benchmark utilities: https://docs.pytorch.org/docs/stable/benchmark_utils (turn81search3) and tutorial https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html (turn81search1). Use documented warmup/thread/synchronization behavior and pin the actual installed version.
