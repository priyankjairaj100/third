# Source and cache acceptance

`acceptance.py` verifies derivation from actual local bytes. It does not authenticate the publisher of those bytes.
It does not authenticate human collection or data permissions.
No original corpus or actual transformer cache has passed this workflow here.

The API returns two separate states:

- `machine_acceptance_passed` reports whether every required replay passed.
- `execution_allowed` remains false in this module.

The dispatcher must also check external evidence, human gates, development choices, and the complete study contract.
A matching digest or an approval field cannot establish those facts.

## Before threshold selection

`accept_calibration_inputs(dataset_id, encoder, dossier, base_dir=..., design=...)` accepts inputs for a blank calibration pack.
The dossier contains `source_acceptance`, `records`, `features`, and `provenance`.
This route replays the original parser, fixed preparation, and complete encoder cache.
It then compares the complete calibration partition.
It requires no threshold, semantic guard, selected panel, or human responses.
Thus threshold preparation has no dependency on its future threshold.
WCEP cannot use this route.
WCEP must inherit the accepted News threshold.
External provenance review remains separate.

## Source replay

The adapter reads the original supplied files again.
It compares every derived record, exclusion, original label, and duplicate link.
It compares the parser settings and input checksums.
It checks the original files again after replay.
It never edits the original files or frozen output files.

The preparation replay derives source identifiers from original metadata.
It independently recomputes the source split hash.
It replays exact-text, duplicate-link, and parent guards.
It checks the complete prepared population, including unknown sources.
It never promotes an unknown source into the source sampling frame.

The Civil100 preview can pass parser replay.
Its missing native metadata prevents primary acceptance.
The test report identifies this preview as reused engineering evidence.

## Full encoder replay

Each encoder uses a fresh process.
This includes repeated acceptance calls for the same encoder.
The process checks pinned model files, tokenizer files, runtime versions, and the complete CPU execution contract.
It then runs the actual local transformer on every prepared record.
It compares each FP32 vector exactly.
It recomputes every token and chunk log.
It recomputes summary counts, truncation counts, model bytes, and cache bytes.
It checks the complete ordered record identifier file.

A sample cannot pass this acceptance test.
A changed runtime or changed FP32 value causes refusal.
The workflow has no tolerance fallback.
It never downloads a model or starts external compute.

Full replay requires model memory, record metadata, token buffers, mapped caches, and temporary output files.
The workflow makes no constant-memory claim.
Its cost belongs to acceptance, not repair performance.

## Guard and panel replay

The E5 guard must use the exact accepted E5 threshold and quality dossier.
The dispatcher must recompute those human gates from actual response files.
The module checks the complete calibration frame against the replayed cache.
It then recomputes every training-versus-held-out pair required by the guard.
MPNet uses the same resulting population.

The registry determines panel size and pool.
The caller cannot substitute a smaller panel.
Primary panels use the primary pool.
The second 10,000-record panel uses the replication pool.
Each panel includes its entire boundary source.
The untouched evaluation population remains complete and ordered.

News source panels use the complete prepared analysis population.
Only `complete-calendar-window` applies the calendar rule.
That branch keeps the entire selected window.
It does not apply a second source-prefix truncation.
Its coverage report still requires substantive external review.

The separate refit panel excludes sources from the largest registered primary panel.
That primary target is 200,000 records.
It also excludes sources from the second 10,000-record panel.
It orders remaining training sources with `ccu-v1-refit-panel`.
It keeps complete sources until the soft 5,000-record target is met.
Insufficient remaining records cause refusal.
The workflow never fills that deficit from excluded sources.
Calibration and test sources never enter this panel.

WCEP uses the full registered 25,000-record event prefix.
It keeps every article instance in the boundary event.
It recursively replays the supplied CC-News acceptance bundle.
It requires matching encoder bytes, revisions, runtime, and threshold dossiers.
It never selects a threshold from WCEP events.
Its source frame remains record-only.

## Bundle interface

Call:

```python
report = accept_source_cache(group, bundle, design, base_dir="/local/bundle")
```

The dispatcher uses the normal Phase 5 task bundle.
Add this field:

```json
{
  "source_acceptance": {
    "dataset_id": "civil_comments",
    "adapter": {
      "directory": "adapter",
      "inputs": {"records": "original/export.jsonl"},
      "kwargs": {"input_format": "jsonl", "input_schema": "tfds_1_2_4"}
    },
    "prepared": {
      "records": "prepared/records.jsonl",
      "audit": "prepared/audit.json",
      "target": 10000
    },
    "guarded": {
      "records": "guarded/records.jsonl",
      "audit": "guarded/audit.json",
      "panel_ids": "panels/panel_ids.json",
      "selection_lock": "e5/selection_lock.json",
      "quality_report": "e5/quality_report.json",
      "target": 10000,
      "pool": "primary",
      "block_size": 256
    },
    "caches": {
      "e5": {
        "directory": "e5/cache",
        "assets": "e5/assets.json",
        "model_dir": "e5/model",
        "tokenizer_dir": "e5/tokenizer"
      }
    },
    "external_evidence": []
  }
}
```

Use `posts` and `links` input names for Stack.
Stack also needs `vocabulary_lock` in `source_acceptance`.
News needs `coverage` and the parser's pinned PSL arguments.
The calendar branch uses pool `calendar`.
Its panel identifier file contains `{"calendar": [...]}`.
Refit uses pool `separate_refit` and target `5000`.
Its panel file contains `separate_refit`, `calibration`, and `test` lists.

WCEP uses named split paths in `adapter.inputs`.
It needs `panel.directory` and `inherited_news_group` in `source_acceptance`.
It also needs the complete `inherited_news_bundle` in the main bundle.
Its curator dossiers contain the actual inherited News calibration records and vectors.

All curator and learner arrays must match the replayed native caches.
The bundle must include complete `train_records` in the accepted panel order.
Every text, date, label, and native field must match the replayed records.
The bundle must include complete `evaluation_source_ids`, including an empty list for unlabeled branches.
The later dispatcher applies registered projections.
Stack targets use a vocabulary selected only from calibration rows.
Every learner, projection, lambda lock, and decision rule must use that exact vocabulary.
Test tags only produce fixed evaluation targets.
Unlabeled corpora cannot receive invented task targets.

Native learner provenance adds `feature_contract: native_normalized_fp32` to the cache calibration provenance.
Projection provenance instead uses `fixed_projection_fp32` and the exact projection lineage.
Its source checksum names the complete source `.npy` file.
Its matrix checksum uses `model_selection._array_hash`.
Its dimensions identify the native and projected feature counts.

`replay_binding_sha256` binds the exact source, cache, panel, target, evidence, and code reports.
The dispatcher must bind its external review to these exact inputs.
Evidence bindings carry no automatic authenticity approval.

## Tests and limits

Run:

```bash
python3 empirical_execution/phase6/check_acceptance.py --out NEW_RESULT.json
```

The checks replay all 100 available Civil comments.
They also test tampering, missing assets, source changes, panel rules, and abstract source membership.
Abstract source fixtures test software only.
They do not provide empirical source-withdrawal evidence.
An explicit mock tests the complete encoder replay loop on three existing comments.
Its temporary vector files use invalid production manifests.
The actual production loader refuses those files.
Another explicit dependency mock tests the successful machine acceptance path.
It also tests changed targets, source identifiers, and registry sizes.
That test does not discharge its replaced parser, encoder, panel, or human gates.
It never saves the resulting acceptance report.
No test retains human judgments or a successful semantic cache.

The actual E5, MPNet, original Stack, original News, and original WCEP acceptance paths remain unexecuted.
Their authentic inputs are absent.
Full replay proves derivation only when those inputs are supplied and the replay actually passes.
