# Phase 4: executable preparation and integration

Checkpoint: 4 October 2026. The primary semantic study remains **unstarted**.
This phase completes further offline preparation and validation software and
replays the existing natural Civil100 lexical fixture. It does not create new
semantic evidence, authentic source metadata, human judgments or a primary
threshold. No external compute was launched.

The full outstanding work is recorded in [TODO.json](TODO.json). Status values
distinguish completed software, completed engineering runs, unavailable inputs,
and still-unimplemented parts of the registered research program. A green unit
test does not move an empirical dependency to completed.

## Completed core

| Component | Implementation and contract | Verified scope |
| --- | --- | --- |
| Original-schema parsing | `adapters.py`, `ADAPTERS_README.txt` | Civil, Stack Posts/PostLinks, extracted News; native IDs, dates, fractional labels, preserved tags, local text rules, pinned PSL; actual archive acceptance still pending |
| Offline semantic encoding | `embeddings.py`, `EMBEDDINGS_README.txt` | Complete 256-content-token chunks, record-local batching, weighted normalized pooling, local asset/runtime hashes; actual transformer execution still pending |
| Task configuration | `model_selection.py`, `MODEL_SELECTION_README.txt` | Calibration-only tag vocabulary, five source-group folds, six lambdas, frozen tag decision thresholds; unknown singletons retained |
| Requests | `requests.py`, `REQUESTS_README.txt` | R/S/U/A, count-matched R, excluded-blocker stress; graph-independent paired R/S random streams |
| Fixed-input comparison | `execution.py`, `EXECUTION_README.txt` | Six explicitly labeled methods, fresh graph and moment oracles, shared ridge solve, all failures and zero outcomes retained |
| Dossier inspection | `study_lock.py` | Partial artifact/hash and calibration consistency review; always refuses primary execution |
| Independent audit | `audit_phase4_independent.py`, `independent_review.txt` | Separate scalar graph, dual ridge, CV, threshold and metadata checks; exact final scope in its JSON |

Further extensions have their own modules and README files. Their current
verification state is in `TODO.json` and `results/`; none overrides the full
protocol with a readiness claim.

The extensions include `news_calendar.py` (source/year cohorts and complete-month
policy), `calibrate_cache.py` (both encoders and an E5-only guard),
`requests_extensions.py` (mass-proportional sources and a separate 1% horizon),
`convex.py` (fractional-label logistic optimization with an exact residual bound),
`systems.py` (fresh-process measurement), and `admission_audit.py` (blinded blank
human-assignment preparation). Each has separate scope and dependency limits.

## What the new comparison actually establishes

Authoritative results are `results/execution_engineering_final/`. The earlier
`results/execution_engineering/` directory is historical: its request seed policy
preceded the shared-path correction. Do not pool these runs.

The final run uses the same 100 natural Civil preview texts and original toxicity
fractions. The curator uses lexical features at dimension 128, threshold 0.6.
Learners use lexical dimensions 64 and 768, lambda 0.01, no intercept, and the
current-count regularization shift. Four R, four U and two graph-stress A paths,
each with checkpoints 1/2/4/8, give **80 checkpoints** across the two dimensions.
There are **zero failures** and **54 zero-admission rows**, all retained. Maximum
same-target head discrepancy is `8.326672684688674e-16`; maximum moment absolute
discrepancy is `4.440892098500626e-15`.

An independent dual-ridge reconstruction checks all **480 saved heads**, including
the deliberately wrong-target B-F heads against their own declared target. Its
maximum discrepancy is `8.881784197001252e-16`. This verifies implementation
agreement within these inputs and numerical tolerances; it is not the strict
roundoff certificate from the earlier exact-state experiment.

The compact payload comparator remains smaller than joint-span state on this
fixture. This run does not establish total-memory advantage, speedup, semantic
quality, natural-source prevalence, generalization or useful NLP effects.

## Prospective completions and boundaries

These conventions were filled before a primary run and are explicit additions
to details not fully specified by the narrative protocol:

- Source CV uses stable source-hash order and round-robin five-fold allocation.
  Unknown singletons remain in vocabulary, targets and CV; they cannot substitute
  for the separate requirement of at least five genuine source groups.
- Tag thresholds maximize per-tag calibration OOF F1 using exact count ratios,
  with strictest ties and a never-positive rule for zero-positive tags. These OOF
  values select settings and are not unbiased post-selection performance estimates.
- Source-service identifiers include unknown singleton units, while the S
  sampling frame contains genuine native sources only. Preview host proxies
  cannot be relabeled as genuine source withdrawals.
- R, S and matched-R seeds exclude graph identity so curator sensitivities use
  shared random paths. Graph identity remains sealed separately. Graph-dependent
  U/A frames and stress decisions are declared as such.
- Native normalized features and fixed-projection features have distinct
  contracts. Projections are not silently renormalized. Learner lambda is selected
  separately for each declared representation.
- The new News verifier checks title-plus-body construction. The frozen phase-3
  intake branch expected body alone and must not be used to claim this News rule
  was validated. Historical phase-3 sources and results remain unchanged.
- The admission interpretation sampler first chooses one uniform former blocker
  per eligible record and then samples its frozen unique-pair frame. Inclusion
  probabilities condition on this first-stage realization and the logged request
  union; they do not cover all possible deletion requests or all blocker pairs.
  The 2017/2018–2019 News cohort policy and each sampler's exact quotas/bands are
  prospective completions documented in their dedicated files.

## Execution sequence with genuine inputs

1. Parse authentic local original files, verify adapter outputs and inspect
   archive coverage, unknown sources, dates and unresolved duplicate/parent links.
2. Use frozen phase-3 preparation for source partitions and fixed leakage guards.
   Apply the separately declared News year/source policy to that branch.
3. Supply a byte-pinned local model/tokenizer/runtime manifest and run the offline
   encoder. A supplied cache also needs ordered record/text/preparation bindings
   and externally reviewed derivation evidence.
4. Generate the blinded calibration pack for the declared encoder. Collect genuine
   independent ratings; select the threshold once; generate fresh validation
   assignments and run the exact quality gate. No automated replacement for
   missing human responses is provided.
5. Apply the E5 training-only guard and freeze whole-source panels. MPNet uses
   the same post-E5 population and its own threshold; it never refilters training.
6. Freeze calibration-only task settings, source/record request manifests,
   supported horizons, methods, runtime, resources and persistence policies.
7. Finish and independently audit the registered systems/full-refit integrations
   listed in TODO before primary execution. Run structural feasibility first.
   Preserve infeasible cases and failures under the common locked limits.

This is a dependency order, not a declaration that steps 1–7 have been executed.
The dossier helper checks some supplied artifacts but does not reconstruct
end-to-end lineage. Even matching artifacts leave `execution_allowed=False`.
The engineering comparison similarly refuses a primary-role request even if a
caller supplies a purported primary lock.

## Reproduction

Run from the repository root with the existing Python dependencies:

```bash
python3 empirical_execution/phase4/check_adapters.py
python3 empirical_execution/phase4/check_embeddings.py
python3 empirical_execution/phase4/check_model_selection.py
python3 empirical_execution/phase4/check_requests.py
python3 empirical_execution/phase4/check_study_lock.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 empirical_execution/phase4/check_execution.py --output-dir /absolute/new/run
python3 empirical_execution/phase4/audit_phase4_independent.py
```

Some checks write audit JSON and timing records. Preserve the historical files or
use a separate checkout. Unit fixtures are explicitly software-only parser and
algebra cases; they are not synthetic research datasets. News body checks are
conditional on the omitted local input and are not portable publication assets.

The 16.46 MB request trace is preserved losslessly as
`results/requests_natural_engineering.json.gz`; `large_result_archive.json`
records both compressed and original byte checksums. Run
`python3 tools/restore_large_results.py` to restore and verify the exact original
JSON. This avoids the publication connector's per-request size limit without
discarding any trace rows. The uncompressed generated copy alone is git-ignored.

## Unavailable evidence

`results/access_recheck.json` records the local asset/runtime inspection. Original
source-rich archives, authentic pinned semantic model/cache assets and genuine
human responses remain absent. The environment lacks the transformer stack used
by the real encoding adapter. No one was contacted or paid for annotations, and
no outside training or cloud job was launched. The current network restrictions
were not bypassed.

The next scientifically decisive milestone is an authentic semantic structural
audit. More correctness replays of Civil100 cannot establish the missing natural
effect or make the full research program complete.
