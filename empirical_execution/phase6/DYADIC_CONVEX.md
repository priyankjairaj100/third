# Exact dyadic verification

`dyadic_convex.py` reduces the cost of exact gradient accumulation.
It preserves the original sigmoid enclosure and optimization bounds.
It also preserves the default verification budget.
The original verifier remains available without changes.

The backend name is `exact_dyadic_integer_v1`.
The scalar and multioutput APIs match the original certificate arguments.
Their schemas identify the new backend.
Their bound fields keep the original meanings.

## Exact construction

Every finite stored FP32 or FP64 value is a dyadic rational.
Represent the weights with a common denominator:

\[
w_j = W_j 2^{-e_w}.
\]

Represent each feature row with its own common denominator:

\[
x_{ij}=X_{ij}2^{-e_i}.
\]

The exact logit becomes one integer dot product:

\[
t_i=2^{-(e_i+e_w)}\sum_j X_{ij}W_j.
\]

The original verifier receives that exact logit.
Its unchanged sigmoid routine returns the same rational interval \([a_i,b_i]\).
The routine already rounds outward onto a fixed dyadic grid.
Thus both endpoints and the original target admit a common binary denominator.

Let the exact residual bounds be

\[
a_i-y_i=L_i2^{-r_i},
\qquad
b_i-y_i=U_i2^{-r_i}.
\]

For a nonnegative feature, its gradient bounds use \(X_{ij}L_i\) and \(X_{ij}U_i\).
For a negative feature, the endpoint order reverses.
These are exactly the original interval products.

The accumulator stores integer numerators for

\[
n\lambda w_j+\sum_i x_{ij}[a_i-y_i,b_i-y_i].
\]

Its common binary exponent increases when a new row requires more precision.
An increase shifts every existing integer numerator by the same exact power of two.
This operation preserves each represented rational value.
After the last row, divide once by \(n2^E\).
For an empty target, use the exact regularizer gradient without division by zero.

This proves coordinate endpoint equality by induction over rows.
The initial regularizer bounds agree exactly.
Every subsequent interval product and interval addition agrees exactly.
Therefore both final gradient endpoint vectors agree with the original verifier.

The squared norm bound also uses one integer sum:

\[
B^2=\frac{\sum_j\max(|A_j|,|B_j|)^2}{n^2 2^{2E}}.
\]

Here \(A_j,B_j\) are the accumulated endpoint numerators.
This equals the original sum of squared rational endpoint magnitudes.
The same strong-convexity inequalities therefore apply:

\[
\lVert w-w^*\rVert^2\le B^2/\lambda^2,
\qquad
f(w)-f(w^*)\le B^2/(2\lambda).
\]

The multioutput verifier sums the same scalar bounds.
Its Frobenius bound and release decision therefore remain unchanged.

## Cost and limits

The implementation removes per-coordinate rational gradient arithmetic.
It uses integer products, additions, and exact binary shifts instead.
It retains one rational logit and the original sigmoid calculation per output and row.
Sigmoid verification can still dominate total cost.
The backend does not change the number of retained data queries.
It does not remove the required retained payload.

The default budget remains 2,000,000 output-row-coordinate evaluations.
For \(n\) records, \(d\) coordinates, and \(q\) outputs, this count is \(ndq\).
The complete multioutput budget is checked before verification starts.
The original Taylor, range-reduction, and precision limits remain unchanged.
A larger coordinate budget needs an explicit prospective resource policy.
A faster implementation does not silently increase that budget.

The integer working state scales with the feature dimension.
The code retains one row of converted integer features.
It does not construct a full rational feature matrix.
Integer bit lengths still depend on input exponents and accumulated sums.
The corrected metric is a descriptive shallow byte sum by reference.
Shared integer objects can count more than once.
This metric is neither a lower bound nor an upper bound on total memory.
It excludes conversion temporaries, sigmoid work, inputs, and output serialization.
Separate fresh-process measurements report process peak RSS.
The process peak includes imports and setup costs.
Process setup can include inherited startup memory before the new executable loads.
Equal reported peaks do not establish equal verifier workspace sizes.

Native \(10,000\times768\times20\) verification requires 153.6 million coordinate evaluations.
This release does not claim that run occurred.
Development throughput can inform a resource proposal.
It cannot guarantee runtime on different logits, sparsity, dimensions, outputs, or hardware.

## Checks and measured scope

`check_dyadic_convex.py` checks exact equality against the original verifier.
The algebra cases include negative features, fractional targets, empty targets, and subnormal values.
The natural inputs are the existing 100 Civil comments and their original targets.
The feature vectors use the existing lexical construction.
They are not semantic encoder outputs.

The first release uses two configurations:

| Dimension | Original outputs | Coordinates |
| --- | --- | --- |
| 64 | Seven Civil target fractions | 44,800 |
| 768 | Toxicity fraction | 76,800 |

Each backend runs five times in a fresh child process.
A fixed seed sets the complete execution order before measurements.
Every run uses the same candidate and original sigmoid code.
The result directory retains all 20 runs.
The raw outputs preserve certificate times, process wall times, and process peak RSS.

All exact gradient endpoints and certificate bounds agree.
All candidates pass the unchanged release tolerance.
The source-bound checks reside in `results/dyadic_convex_release/checks.json`.

The sparse lexical measurements give these median certificate times:

| Dimension | Outputs | Original verifier | Dyadic verifier |
| --- | --- | --- | --- |
| 64 | 7 | 0.876 seconds | 0.450 seconds |
| 768 | 1 | 0.728 seconds | 0.124 seconds |

Lexical sparsity affects these costs.
Those measured differences do not establish native semantic throughput or complete repair speedup.

`check_dyadic_dense.py` adds a separate prospective dense arithmetic check.
It rotates the same natural lexical vectors with a fixed orthogonal matrix.
It preserves all original records and all seven original target fractions.
It casts the rotated vectors to FP32 without renormalization.
This is a feature-coordinate check, not a new corpus or semantic model.
Its complete result directory is `results/dyadic_dense_release/`.
Its final `checks.json` records the actual outcome and measurements.
The independent audit records its own source hashes and exact comparisons.

The dense seven-output release passes all ten fresh-process runs.
It checks 537,600 coordinates per run.
Median certificate times are 7.692 seconds for the original verifier and 1.581 seconds for the dyadic verifier.
Every exact gradient endpoint and certificate bound agrees.

`check_dyadic_dense_scalar.py` checks the actual Civil single-output contract separately.
It takes the original toxicity column and corresponding head from the same dense candidate.
It introduces no new records or labels.
All ten fresh-process runs pass at 76,800 coordinates per run.
Median certificate times are 1.077 seconds and 0.166 seconds, respectively.
Its results reside in `results/dyadic_dense_scalar_release/`.

Together, the releases contain 40 fresh-process runs.
These runs qualify development cost measurements only.
Genuine Stack20 development targets remain unavailable.
The seven-output Civil result does not qualify a twenty-output Stack resource lock.
The optional backend must enter dispatch through a prospective development policy.
Its integration check belongs to the dispatcher and recipe reports.
This verifier report alone does not establish that integration passed.

## Memory metadata erratum

Independent review found an incorrect label in the original byte metadata.
The code summed each integer reference separately.
Copied lists can share integer objects, including zeros.
Thus `tracked_integer_list_bytes_lower_bound` was not a valid lower-bound label.

The corrected field is `tracked_integer_list_shallow_byte_sum`.
Its value follows the same calculation.
Its description states that shared objects can count repeatedly.
No arithmetic, sigmoid interval, certificate bound, or release decision changed.

The exact original source remains at `history/dyadic_metadata_v1/dyadic_convex.py`.
Its SHA256 is `eaf003048efdf9e59975f5a1e2ba26e646a362b8fc68862c1aab0447448e13f1`.
All 40 timing reports remain unchanged and bind that historical source.
They did not run the corrected source version.
Their elapsed times remain historical measurements.
Their byte field must receive the corrected descriptive interpretation.

`check_dyadic_metadata_erratum.py` checks the corrected meaning with an explicit shared-object example.
It also checks unchanged exact endpoints on the genuine dense Civil candidate and an empty target.
It verifies every historical report against the preserved source hash.
The source-bound result is `results/dyadic_metadata_erratum.json`.
This focused check adds no new timing claim.
No timing benchmark was repeated for this metadata correction.

Run the two releases with:

```bash
PYTHONPATH=.:empirical_execution OPENBLAS_NUM_THREADS=1 python3 empirical_execution/phase6/check_dyadic_convex.py --output-dir NEW_SPARSE_RELEASE
PYTHONPATH=.:empirical_execution OPENBLAS_NUM_THREADS=1 python3 empirical_execution/phase6/check_dyadic_dense.py --output-dir NEW_DENSE_RELEASE
```

Both commands refuse an existing release directory.
Neither command supplies missing corpora, model assets, or human ratings.
