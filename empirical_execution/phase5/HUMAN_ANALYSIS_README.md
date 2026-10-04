# Human admission and context analysis

`human_analysis.py` analyzes the Phase 4 pair-audit and Phase 5 contextual-audit schemas. It validates supplied responses; it never fills blanks, dispatches assignments, recruits or contacts anyone. The current actual packs contain **zero completed human responses**: 72 expected pair assignments and 60 expected contextual assignments remain blank.

```bash
PYTHONPATH=empirical_execution python -m phase5.human_analysis \
  --manifest path/to/private_sampling_manifest.json \
  --responses path/to/collected_responses.json \
  --output path/to/new_analysis.json
```

An optional `--adjudication` accepts the separate format described below. Output refuses overwrite. The programmatic API is `analyze_human_audit(manifest, bundle, adjudication=None, software_fixture_only=False)`.

## Collection validation and raw observations

The exact frozen manifest, complete finite frame, SRS stratum sizes, stratum samples and deduplicated inclusion probabilities are checked. Pair overlap uses its actual union probability `1 - product_h(1 - n_h/N_h)` once per sampled pair. Former-blocker/context choice probabilities are not silently multiplied into a different population. Context inclusion is `n/N`. Repeated appearances of an admission at several request checkpoints do not create independent human units.

Every supplied response must identify a registered assignment and its correct pair/context unit. Unknown IDs, duplicate assignments/response IDs, malformed categorical answers, invalid distinction types and repeated raters on one unit are rejected. Three distinct fluent humans are required by the protocol; the software verifies distinct declared opaque IDs but cannot authenticate people or fluency. Real completed bundles require collector and human-only/independence/blinding declarations. These declarations remain declarations; schema acceptance is not evidence of consent, independence, blinding or human authorship.

Absent assignments are retained as missing. Untouched template blanks remain blank. A recorded skip has `human_completed=true`, a rater/response ID, a timezone-aware completion timestamp and a reason, with substantive fields empty; it is counted as a completed human interaction and an unavailable substantive rating. Partial answers must be resolved in collection before ingestion instead of being silently interpreted. Contexts with no comparator can only remain blank or receive a recorded skip, never an automatic novelty answer. No skipped item is replaced or imputed.

The prospective outcome convention requires all three original substantive ratings. Each categorical majority needs at least two equal categories; a three-way split is explicitly `no_majority`. `uncertain` is retained as its own category, not treated as a positive semantic judgment. With only two substantive answers, even agreement does not become a completed three-rater unit. Distinction-type indicators use at least two of the three raters. This defines reported operational annotation outcomes; none is a certificate of latent semantic truth.

Original response bundles, per-unit tallies, disagreements and majority outcomes are preserved verbatim/deep-copied in the local analysis output. Annotator and collector IDs should be opaque, not personal names. A future public release must review permitted annotation fields; the private raw collection bundle need not be published. Actual blank-check outputs contain no filled judgments or human identities.

## Estimation and missingness

Domains stay separate by corpus and declared population: unique admissions, R/S, U/A and observational controls for pair audits; former neighborhood and nearest surviving selected for context audits. A sampled pair that belongs to two roles can contribute to each declared domain, with its one union inclusion weight and one shared annotation outcome. Those domain estimates are correlated, not independent replications.

For a domain of size `N`, inverse inclusion weight `w=1/pi` and binary category indicator `Y`, the module reports the Horvitz–Thompson estimate `sum_sample w*Y/N` and the Hájek ratio `sum_sample w*Y/sum_sample w`. The HT estimate is design-unbiased under complete fixed potential outcomes and the recorded design; the Hájek ratio is bounded/self-normalized but generally not exactly design-unbiased. The two are not interchangeable. HT values need not sum to one or lie within the unit interval on a realized unequal-probability sample; no favorable clipping is performed.

If any sampled domain outcome is unavailable, full-outcome point estimates are null. The explicitly named observed-only Hájek description is conditional on response and is not the domain truth. The following distinct quantities remain visible:

- Weight-adjusted completion, skip, uncertainty and context-truncation rates; zeros are preserved.
- HT and Hájek **missing-outcome extreme estimates**, assigning unavailable sampled outcomes zero or one. They are missingness sensitivity calculations, not design-based confidence intervals or deterministic population bounds.
- **Finite-frame bounded-outcome identification bounds**, using only observed units and allowing every other frame unit—including unsampled units—to be zero or one. These are conservative deterministic partial-information bounds for the stated operational fixed outcomes, not sampling confidence intervals or bounds on latent semantic truth.

No sampling confidence interval is invented. The overlapping pair-sample design would require its joint-inclusion covariance or a justified survey-design resampling scheme before adding such intervals. Inverse inclusion weights correct the recorded sampling design; they do not correct voluntary skip bias or rater uncertainty. Empty frames have undefined prevalence, never a favorable zero.

Three-corpus balanced outputs use exactly one third per prescribed corpus. A missing corpus, empty domain or unavailable required outcome leaves the balanced estimate undefined; remaining corpora are not silently reweighted. The finite frame remains conditional on the declared checkpoint union and realized blocker/context choices, not all deletion requests, all possible blocker pairs or all corpora.

## Agreement and context scope

Weighted pairwise agreement, design-weighted Fleiss-style kappa and uncertain-rating fraction are reported among units with three substantive ratings, clearly labeled as complete-unit descriptions. Rational arithmetic computes weight/tally/chance quantities before final floating conversion; kappa is undefined when its chance-agreement denominator is exactly zero. Missingness rates accompany the agreement description. Sampling weights do not make agreement representative under outcome-dependent missingness.

Context outputs contain two scopes. `complete_original_context` treats truncated targets/neighborhoods and unavailable comparators as missing for the full question. `displayed_context_only` reports judgments about the actually shown text and still excludes unavailable comparators. These scopes do not turn the former-blocker neighborhood into the entire original training corpus, or the nearest surviving text into all surviving information. No global novelty claim follows.

## Separate adjudication

Adjudication is optional and subsequent. Its bundle contains:

```json
{
  "manifest_sha256": "frozen manifest hash",
  "response_bundle_sha256": "hash of the complete preserved response bundle",
  "policy": {
    "policy_id": "declared policy identifier",
    "policy_text": "the actual adjudication procedure",
    "fixed_before_adjudication": true
  },
  "decisions": [
    {
      "item_id": "registered pair or context unit ID",
      "adjudicator_id": "opaque declared adjudicator ID",
      "reason": "recorded rationale",
      "completed_at": "timezone-aware ISO time after all three original ratings",
      "answers": {
        "meaning_relation": "category for pair audit; use distinct_information for context audit",
        "consequential_distinction": "category",
        "distinction_types": [],
        "task_relevance": "category"
      }
    }
  ]
}
```

This is a schema illustration, not a response. The validator requires all three original substantive judgments before adjudication, a later recorded timestamp, a bound original response hash, valid answers and a policy/rationale. It does not authenticate chronology or adjudicator identity. Original majority and agreement remain unchanged. A separately labeled supplementary summary replaces only declared resolved-unit outcomes with adjudication and retains original majority elsewhere; its mixing rule is explicit, and it is never relabeled independent majority evidence.

## Illustrative IDs

`lock_illustration_policy` records the corpus/domain/context/question/category list, seed and per-category count without reading responses. Actual preoutcome timing requires external evidence. `select_illustrative_ids` then chooses the first eligible original-majority units by the frozen stable hash rule, retaining shortfalls. It never ranks task-loss effects, performs post hoc category search, exports publisher text, or invents an interpretation. The output contains permitted-review IDs only; excerpts require a separate permission-aware release. Original complete-context and comparator gates still apply. Repeated IDs across categories are not independent evidence.

## Checks

```bash
PYTHONPATH=empirical_execution python empirical_execution/phase5/check_human_analysis.py
```

Filled answers appear only in in-memory, explicitly tagged schema/algebra fixtures and are never counted or written as human observations. Tests check unequal/overlapping inclusion weights, finite-frame versus weighted missingness quantities, missing-corpus handling, distinct raters, partial/skip preservation, category ties, agreement, adjudication immutability/timing, probability tampering and unavailable comparators. The genuine local blank packs are analyzed without modification; all semantic estimates remain unavailable. `results/human_analysis_checks.json` records checks; `results/human_analysis_blank/` records the actual missingness outputs.
