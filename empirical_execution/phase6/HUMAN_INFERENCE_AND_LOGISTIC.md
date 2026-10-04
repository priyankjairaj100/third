# Human intervals and logistic decisions

This release closes two software gaps from Phase 5.
It does not supply human responses or a semantic task configuration.
Frozen Phase 4 and Phase 5 code remains unchanged.

## Human interval target

The unit is one original three-rater outcome for one sampled pair or context.
Three responses do not create three independent observations.
Repeated admission roles do not create independent observations.

The target population is the complete frozen frame in the annotation manifest.
The analysis conditions on the chosen former blocker and context occurrence.
It also conditions on fixed potential outcomes for the original three-rater protocol.
Coverage does not describe latent semantic truth.
Coverage does not describe a new population of raters.
Hashes and collector declarations do not authenticate human origin.

The category outcome is binary for each reported category.
For example, the outcome equals one when the original majority selects `exact_copy`.
The outcome equals zero for every other complete original-majority category.
`uncertain` and `no_majority` remain explicit categories.
A skipped or incomplete unit has an unknown outcome.
Two matching responses do not replace the original complete three-rater outcome.

The full-context target treats truncated contexts as unknown.
The displayed-context target uses valid judgments about the displayed context.
An unavailable comparison remains unknown in both targets.
No unit gets replaced after a skip, truncation, or missing response.

## Exact construction

Let stratum \(h\) have frame \(U_h\), size \(N_h\), and sample size \(n_h\).
Its sample follows the original simple random sampling without replacement design.
Pair frames can overlap between admission roles.
Context frames remain disjoint by corpus and context kind.

Assign each unit to one stratum that contains it.
Choose the highest inclusion probability \(n_h/N_h\).
Break ties with the original manifest stratum index.
This rule uses no response values.

For a requested domain \(D\), let \(D_h\) contain units assigned to stratum \(h\).
These sets partition \(D\).
Define the transformed stratum outcome as

\[
Z_{hi}=\mathbf 1\{i\in D_h\}Y_i,
\qquad
K_h=\sum_{i\in U_h}Z_{hi}.
\]

Every stratum unit outside \(D_h\) has a known zero outcome.
Under the recorded sampling design, its complete sample total obeys

\[
X_h\sim\operatorname{Hypergeometric}(N_h,K_h,n_h).
\]

The implementation evaluates tails with integer binomial coefficients and exact rational arithmetic.
It accepts a candidate population count when both exact tails exceed their allocated error thresholds.
Equality remains accepted, which gives conservative coverage at discrete boundaries.
Monotonicity permits binary search over possible population counts.

Suppose the owned sample has \(s_h\) observed positives and \(m_h\) unknown outcomes.
The complete sample total lies between \(s_h\) and \(s_h+m_h\).
The algorithm takes the union of count intervals across all those completions.
Monotonicity gives its endpoints from those two extreme sample totals.
This operation allows arbitrary outcome-dependent missingness.
It does not assume missingness at random.

The full union sample can contain additional observed units assigned to another sampling route.
Their values impose deterministic constraints on each owned total.
If those observations contain \(a_h\) positives and \(b_h\) negatives, then

\[
a_h\le K_h\le |D_h|-b_h.
\]

Intersect the randomization interval with these constraints.
An empty intersection remains an explicit empty confidence set.
The code does not replace that event with a convenient interval.

Sum the owned count bounds and divide by \(|D|\).
The result targets finite-frame category prevalence.
Phase 5 still provides the separate HT and Hajek point estimates.
The interval need not equal an interval around either estimate.

## Coverage proof

For complete outcomes, exact tail inversion covers each \(K_h\) with its allocated probability.
The missing-outcome union contains the interval for the actual complete sample total.
The deterministic constraints always contain the actual \(K_h\).
Their intersection therefore preserves the same coverage event.
On the intersection of all stratum events, summing bounds covers the domain total.

The report divides its error budget among all nonempty category endpoints.
Each endpoint divides its budget among its nonempty owned strata.
The union bound gives simultaneous coverage of at least \(1-\alpha\).
This proof needs no independence between stratum coverage events.
It also needs no independence between overlapping report domains or category indicators.

The design assumption remains essential.
Each original stratum sample must follow its declared probability design.
Potential outcomes must not change because another unit entered the sample.
The frame and sampling policy must precede response inspection.
Fixed seeds implement reproducibility; observed hashes alone cannot establish the randomization assumptions.

Each report names its complete simultaneous family.
The saved pair and context reports each use \(\alpha=1/40\).
Together, their families have coverage of at least 95% by another union bound.
Equal-corpus intervals average the three corpus bounds on that same coverage event.
If a corpus frame is absent, the equal-corpus target remains undefined.
The analysis never transfers its weight to another corpus.

The construction is deliberately conservative.
Disjoint ownership can discard some randomization information from overlapping routes.
Known union-sample outcomes still tighten deterministic count constraints.
No claim of shortest intervals or optimal efficiency is made.

## Observed acceptance

Run from the repository root:

```bash
PYTHONPATH=.:empirical_execution OPENBLAS_NUM_THREADS=1 python3 empirical_execution/phase6/check_human_inference.py
```

The local suite passes 15 checks and 284 exact coverage cases.
The minimum coverage is \(20/21\) at a nominal 90% level.
These are finite algebraic checks, not an empirical dataset.
Independent review evaluates additional exact cases in its separate report.

The actual pair pack still has 72 blank assignments.
The actual context pack still has 60 blank assignments.
Every nonempty category frame has interval \([0,1]\).
Every absent frame has an undefined prevalence.
Completed human responses remain zero.

The saved outputs are `results/human_inference_blank/pair_intervals.json` and `context_intervals.json`.
The source-bound check report is `results/human_inference_checks.json`.

For supplied genuine responses, use:

```bash
PYTHONPATH=.:empirical_execution python3 empirical_execution/phase6/human_inference.py \
  --manifest INPUT_MANIFEST.json --responses INPUT_RESPONSES.json \
  --alpha 1/40 --output NEW_INTERVAL_REPORT.json
```

The command refuses to overwrite an existing output.
It uses the unchanged Phase 5 response validator.
It does not collect, complete, or send assignments.

## Logistic decision rule

`logistic_selection.py` preserves the existing utility policy.
The logistic learner uses the same frozen ridge lambda.
It does not tune another lambda.
It preserves the selected 20-tag vocabulary and five source-group folds.
Unknown singleton records and zero-target records remain in calibration.
At least five genuine source groups remain necessary.

The selector first recomputes the ridge lock from the supplied calibration inputs.
It rejects a lock that differs from that result.
This checks the complete data binding before any logistic decision selection.
It does not authenticate source metadata or feature provenance.

The selector then fits the logistic learner on each four-fold training partition.
It predicts probabilities for that fold's held-out calibration partition.
It never uses test labels or ridge predictions for logistic thresholds.
All calibration rows receive one out-of-fold probability per tag.

For each tag, choose the threshold that maximizes exact count-based F1.
Use the comparison `probability >= threshold`.
Choose the largest threshold when exact rational F1 values tie.
Use the never-positive rule when the calibration target has no positives.
Threshold selection uses every score tie as one block.

The objective is

\[
\sum_c\frac{1}{n}\sum_i
\bigl[\log(1+e^{x_i^\top w_c})-y_{ic}x_i^\top w_c\bigr]
+\frac{\lambda}{2}\lVert W\rVert_F^2.
\]

The learner has no intercept.
Every coordinate receives the regularization penalty.
The frozen numerical gate requires each output gradient norm to reach \(10^{-11}\).
The iteration limit is 100, with at most 40 backtracks per iteration.
A failed output causes selection to fail.
This configuration gate is not a rigorous optimizer-error certificate.
The main convex release retains its separate certificate requirement.

Out-of-fold values are selection diagnostics.
They are not unbiased estimates of performance after threshold selection.
The selected rules apply unchanged to held-out evaluation probabilities.
The logistic application API rejects ridge threshold locks and probabilities outside \([0,1]\).

Public APIs are:

```python
select_logistic_decision_thresholds(
    fp32_features, calibration_records, provenance,
    ridge_lambda_lock, vocabulary_lock,
)
apply_logistic_thresholds(probabilities, thresholds_lock)
```

Run the checks with:

```bash
PYTHONPATH=.:empirical_execution OPENBLAS_NUM_THREADS=1 python3 empirical_execution/phase6/check_logistic_selection.py
```

All 14 checks pass.
The independent BFGS probability discrepancy is approximately \(3.29224\times10^{-10}\).
Those fits use labeled algebraic fixtures only.
The actual Civil preview correctly fails the genuine-source requirement.
No empirical logistic task configuration was selected.
The source-bound result is `results/logistic_selection_checks.json`.
