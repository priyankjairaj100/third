# Secondary search and numerical boundary studies

This module completes the offline boundary-study software contract. The authoritative natural-data check is `results/boundary_engineering_release_v2/`. It reuses the existing 100 Civil comments with lexical features. It is **not** the registered semantic News-window experiment, a new independent dataset, a source-withdrawal result, or a speed/memory benchmark. Actual News input, calibrated semantic threshold and authentic encoder cache remain required.

## Approximate candidate graph

`boundary.lsh_candidates` is an independently implemented random-hyperplane LSH index: eight tables of ten bits, seeded NumPy PCG64 normal planes, nonnegative signs assigned one, all same-bucket pairs unioned. The complete configuration, versions and plane/signature/candidate hashes are recorded. There is no top-k cap. Candidate pairs are rescored by the frozen Phase 3 scorer and only strict threshold edges remain. This is explicitly approximate and never replaces the exhaustive primary graph.

`audit_ann` checks the supplied exhaustive graph against all scored unordered pairs. It reports edge recall and complete-blocker rate over all records, separately over records with at least one true blocker, and selected-set/head discrepancies. Empty-edge recall is undefined. High average recall would not establish exactness.

The retrieved stratum is a census. A streaming reservoir is an SRS without replacement over **every nonretrieved unordered pair**, with inclusion probability `min(sample_size, population_size)/population_size`. The default sample is 200 and the seed is fixed. The saved sample includes scores and edge indicators, a Horvitz–Thompson missed-edge count estimate, and a conservative 95% interval from two exact hypergeometric inversions with alpha 1/40 each. The complement-empty case has zero missed edges and no defined per-pair inclusion probability. An audit sample size of zero is rejected. Pair-frame enumeration and exhaustive validation are evaluative costs: this implementation makes no ANN scalability claim. Candidate sets and buckets can use quadratic Python storage.

## FP32 state and conditioning

`run_frontiers` accepts the already locked request manifest; it does not regenerate requests after seeing boundary results. Every checkpoint is run with the identical feature geometry at `0.1`, `1`, and `10` times the frozen lambda, including all zeros and failures. The FP32 variant stores dense Gram/cross moments, starts with ordered FP32 accumulation and applies signed removals/admissions. Source-precision targets are explicitly rounded for each FP32 contribution. The solve also uses FP32. Its target comparator is fresh FP64 moments/solve with the original FP64 labels. Empty selection uses the specified zero head while any cancellation in state moments is still reported.

This is a separately named **FP32 signed-moment payload variant**. It is not an FP32 implementation of the polynomial summary or a theorem-backed exact-arithmetic release. The evaluator owns raw rows for comparisons; there is no no-reaccess or deployment-memory claim. Reported state bytes count the dense numeric arrays, with shared feature/target bytes separately disclosed; graphs, Python storage and peak RSS are not included in these array counts. Those costs must be integrated with the isolated systems harness before making a practical memory comparison.

The natural integration uses one seeded, data-independent Gaussian projection of the stored lexical 128-dimensional vectors to 32 coordinates. It stores the projection, binds its hash, performs no postprojection normalization, and reuses the complete Phase 4 final request manifest. Choosing this small dimension is an explicitly scoped engineering check, not completion of the registered native semantic dimension and projection grid.

## Uniform graph envelope and model radius

`score_intervals` processes every unordered pair, including nonretrieved pairs. It propagates lower/upper endpoints through the same coordinate-ordered sum of squares, square root, division, products and coordinate-ordered dot sum used by the actual fixed scorer. Each operation is evaluated in binary64 and enlarged by `nextafter` toward each infinity. Products/divisions consider all endpoint corners. The only clamp is the mathematically known nonnegative sum of squares before sqrt; scores are never clipped.

The certificate assumes IEEE binary64 round-to-nearest elementary operations, gradual underflow, correctly rounded sqrt and no operation contraction. These are explicit arithmetic/runtime assumptions, not conclusions of a finite test suite. Under them, rounded endpoint monotonicity and outward enlargement enclose both the real operation and its floating-point result. Induction therefore encloses the **actual fixed-order finite-precision score** and also the exact cosine of the stored inputs. This does not certify encoder error or semantic truth. Inputs with unresolved/nonpositive normalization intervals are rejected.

A lower edge has `lo > tau`; an upper edge has `hi > tau`. Therefore `E_lower ⊆ E_reference ⊆ E_upper`. With identical fixed priorities and any retained subset, global earlier-raw-neighbor suppression gives

`selected(E_upper) ⊆ selected(E_reference) ⊆ selected(E_lower)`.

This is a uniform finite graph statement, not an estimate from observed maximum drift or ANN recall. The code checks all-pair score containment and each evaluated selection sandwich. Boundary unit fixtures place tau at the computed score and include subnormal FP32 values, maximum finite FP32 values, negative/cancelling coordinates, and all deletion subsets of small algebraic arrays.

`model_envelope_bound` gives an exact-rational conservative radius for every possible selected set between lower and upper selection bounds. For `B ⊆ S ⊆ T`, fixed released matrix `W`, and row contributions

`r_v = x_v (y_v - x_vᵀW)ᵀ - lambda W`,

it computes exact rational `r_B = sum_(v in B) r_v` over stored FP32/FP64 values and a radius

`(||r_B||_entrywise1 + sum_(v in T\B) ||r_v||_entrywise1) / (lambda max(1, |B|))`

for nonempty sets, using `lambda |S|` strong convexity and the triangle inequality. If an empty set is possible it also takes the maximum with `||W||_entrywise1`; if only the empty set is possible this is the radius. Entrywise L1 dominates Frobenius norm. The rational numerator/denominator is the certificate; the displayed float is rounded upward. This bound may be very loose. Tests enumerate all admissible subsets of small algebraic fixtures; a bound is not presumed practically useful merely because it is valid.

No annotation is fabricated and no synthetic empirical corpus is created. All mathematical fixtures are software verification inputs. Frozen Phase 3 and Phase 4 code/results remain unchanged.

## Reproduction

From repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m empirical_execution.phase5.check_boundary --output-dir /path/to/new/output
```

The runner refuses to overwrite a nonempty result directory. `boundary_engineering/` was a development execution while the source was being revised, and its source-hash association is not authoritative; `boundary_engineering_release/` is the first frozen candidate before standalone-API validation hardening; only the final `boundary_engineering_release_v2/` directory is a frozen evidence checkpoint.
