# Fixed task metrics

`task_metrics.py` computes metrics from aligned, immutable labels and predictions.
It fits nothing, clips no ridge scores and removes no zero-target question.

- Civil: fractional-label MSE/MAE; AUROC and noninterpolated average precision for
  the fixed event toxicity >= 0.5.
- Stack: mean squared error over rows and outputs, sample-averaged sum of squared
  errors (both normalization conventions explicit), micro/macro F1 from locked
  per-tag thresholds, and micro/macro average precision with all tags retained.
- Fixed-test effects reuse the signed loss and raw/normalized RMS implementation
  in `statistics.py`; zero denominators remain undefined.

Average precision groups tied scores before evaluating threshold precision. It
is not trapezoidal interpolation of the precision–recall curve. These definitions
follow the [scikit-learn metric documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).
We checked the independent implementation against installed scikit-learn 1.8.0.
Positive infinity is the calibration-only never-positive threshold; decisions
use `>=`. Prospective completion conventions: zero-positive AP is zero and
zero-denominator F1 is zero; one-class AUROC is undefined. Per-tag class weights
and the count of zero-positive tags accompany macro scores, so these cases
cannot silently disappear. Optional row weights encode whole-source bootstrap
multiplicity; tests verify equality with explicit repeated rows.

Run `python3 empirical_execution/phase5/check_task_metrics.py`. Seventeen checks
cover tie handling, weighting, zero labels, undefined quantities, negative
effects and invalid inputs. The additional 896 saved Civil prediction vectors
are historical lexical development outputs; their metrics are compatibility
checks and do not establish a new semantic or source-disjoint experiment.
