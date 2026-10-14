THIRD — pure model-only statistics core, 14 October 2026

This new package implements the numerical estimators in the sealed LLM-only v3
protocol. It is not an ingestion CLI, execution lock, source-admission route or
inference runner. It reads no experiment outputs automatically, creates no labels,
draws no entropy and has no human annotation dependency. Software-test examples
are algebraic fixtures, not research observations.

selection(rows, populations, observed)
- Requires all min(30, N_bin) selected units from each of exactly 20 bins, exact
  reduced rational inclusion probabilities and weights, canonical finite raw
  score hex, and the frozen saturated decimal-bin convention.
- observations have only opaque_pair_id, model_id, status and category. They must
  already be validated, canonical, attempt-zero, correctly staged and bound to
  immutable raw outputs by a separate future ingestion layer. Unknown/duplicate
  pair/model records and invalid status/category combinations are rejected.
- Absent pair/model observations become explicit missing_output records. Every
  selected unit keeps its entire weight. A positive requires both fixed models
  to produce valid exact_copy or substantially_same_meaning categories.
- Emits every strict raw_score > float(tau) grid result, using exact rational
  numerator/denominator and the least threshold meeting 19/20 with nonzero
  support. No success produces no selected threshold; 0.995 is diagnostic only.
- The optional optimistic missing/uncertain coding is a sensitivity diagnostic
  with a changed coding convention. It is not a confidence bound, alternative
  primary gate, semantic-truth estimate or permission to relabel observations.

validation(rows, population, observed, selection_rows, selection_populations,
           selection_observed, selection_result, bound)
- Reconstructs the complete selection result and demands exact equality before
  consuming its passing threshold. A changed ledger, nonleast threshold, stale
  response set or failed selection cannot become validation.
- Requires exactly min(200, N_above) unique sampled pairs and strict above-tau
  raw scores. The separate sampler and ingestion layer must establish the actual
  full above-threshold population, independent uniform-draw sampling, exact pair
  membership, frame identity, canonical orientation, seeds and fresh calls.
- Uses only the pinned phase3 finite_population_lower_bound numerical kernel,
  with alpha=1/20. No human-only collector, majority vote, quality_gate or legacy
  acceptance wrapper is invoked. frozen_bound(repo) verifies both scorer files
  and imports the numeric helper through an inert namespace package.
- Numeric success requires lower proportion >=9/10. Empty population/sample is
  undefined and unavailable. Any overlapping model status or valid six-way
  category change makes model validation unavailable even if the numeric gate
  passes. Reasons are deliberately absent from this comparison. Zero overlap
  explicitly means zero replay evidence, not proven global stability.

Only the fixed Qwen2.5-7B and Mistral-Nemo IDs are allowed. The exact model revision,
prompt, raw schema validation, token/grammar constraints, runtime, stage/call ID,
budget, full source and sample provenance are prerequisites enforced outside this
pure core. A caller cannot treat returned booleans as end-to-end scientific
admission. Model-assessed development is distinct from semantic validity or human
agreement, both unestablished. Existing human protocols stay historical/unmet.

Tests cover incomplete/unavailable observations, both positive categories,
strict boundaries, least-threshold choice, exact unequal weights, malformed
designs, forged/stale results, overlap changes and empty/census validation.
An independent combinatorial enumeration verifies the inherited finite-population
coverage convention for all N <= 12, all nonempty n, and all success counts K.
No model calls, source dataset outputs or real labels are involved.

Use an existing Python with NumPy; no installation is needed:
  python -B -m unittest discover -s local_run/llm_annotation_statistics_v1_20261014 -p test_statistics.py -v
