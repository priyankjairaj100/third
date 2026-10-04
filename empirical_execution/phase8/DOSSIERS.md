# Staged local dossier construction

`dossiers.py` assembles inputs for the unchanged Phase 6 dispatcher.
It creates no source identities, labels, human responses, external reviews, or development observations.
Missing requirements produce a blocked receipt.
They do not become empty approved values.

Four commands keep calibration, candidate construction, resource qualification, and external review separate.
The final map covers one explicit request family.
It does not certify the entire study or every group sharing that family.

## 1. Assemble a prethreshold calibration dossier

Prepare an actual source specification first.
All paths below resolve against the source specification's directory.
The assembler stores canonical absolute paths for subsequent stages.

```json
{
  "dataset_id": "civil_comments",
  "archive_export": {
    "directory": "civil_export",
    "archive_path": "civil_comments_v1.2.zip"
  },
  "adapter": {
    "directory": "civil_adapter",
    "inputs": {"records": "civil_export/records.jsonl"},
    "kwargs": {"input_format": "jsonl", "input_schema": "tfds_1_2_4"}
  },
  "prepared": {
    "records": "civil_prepared/records.jsonl",
    "audit": "civil_prepared/audit.json",
    "target": 10000
  },
  "caches": {
    "e5": {
      "directory": "civil_e5",
      "assets": "civil_e5/assets.json",
      "model_dir": "e5_model",
      "tokenizer_dir": "e5_tokenizer"
    }
  }
}
```

This is a schema example, not an accepted dossier.
The referenced archives, exports, preparation, cache, and model files must actually exist.
Civil and News require the Phase 8 original-archive exporter and its exact replay.
Stack and WCEP retain their original parser routes in frozen acceptance.
News also needs its supplied public-suffix list and complete calendar-coverage evidence.

```bash
python3 empirical_execution/phase8/dossiers.py calibration \
  --source-spec local_inputs/civil_source.json --encoder e5 \
  --out local_outputs/civil_calibration_e5
```

The command derives the complete calibration partition from the prepared cache.
It calls `accept_calibration_inputs`, including full local encoder replay.
It writes exactly four top-level fields in `dossier.json`:
`source_acceptance`, `records`, `features`, and `provenance`.
Arrays and record files use hash-bound descriptors compatible with the frozen loader.
No threshold, guard, panel, or human response is required at this stage.
WCEP cannot select its own threshold; it inherits News calibration.

The resulting dossier can feed the Phase 7 blank-pack command:

```bash
python3 empirical_execution/phase7/prepare_calibration.py \
  --dataset civil_comments --encoder e5 \
  --dossier local_outputs/civil_calibration_e5/dossier.json \
  --out local_outputs/civil_selection_e5
```

Blank assignments do not establish genuine collection or semantic quality.
Actual selection and fresh-validation responses must come from the declared human protocol.

## 2. Build an unsigned candidate

Use an explicit list of registered groups from one request family.
Human file paths resolve against this configuration's directory.

```json
{
  "group_ids": ["C_full_methods/civil_comments/primary-10000/native"],
  "source_acceptance": "civil_source.json",
  "calibrations": {
    "e5": {
      "selection_manifest": "selection/private_sampling_manifest.json",
      "selection_responses": "selection/actual_responses.json",
      "validation_manifest": "validation/private_sampling_manifest.json",
      "validation_responses": "validation/actual_responses.json"
    }
  }
}
```

```bash
python3 empirical_execution/phase8/dossiers.py candidate \
  --config local_inputs/civil_candidate.json \
  --out local_outputs/civil_primary10000_candidate
```

The assembler reconstructs selection, threshold, validation, and quality decisions from the supplied human files.
It applies the single E5 guard and the registered whole-source panel contract.
MPNet uses that same retained population.
Original official Civil split metadata remain lineage; the primary split remains source-disjoint.
It never promotes unknown sources to genuine source-withdrawal units.

Original toxicity fractions or original question tags supply targets.
Vocabulary, ridge regularization, and decision thresholds use calibration records only.
Evaluation tags only receive an already fixed vocabulary.
Source-fold selection still requires five genuine groups.
Their absence remains a failure.

Requested projection groups receive the frozen seeded projection and separately recomputed calibration choices.
Convex Stack groups require the separate logistic decision selector.
Threshold-sensitivity groups require supplied `sensitivity_quality` files and their own passed response gate.
Full-refit, ANN, separate-refit panels, calendar windows, and WCEP retain their frozen branch contracts.
WCEP configuration additionally supplies an existing `inherited_news_bundle` and its registered `inherited_news_group`.

The command builds requests with the fixed study design and the family's largest registered allocation.
It then runs complete source/cache acceptance.
`candidate.json` is still unsigned and unapproved.
`groups.json` records the exact proposed scope.
`review_request.json` supplies digests for review, never a reviewer attestation.

## 3. Attach actual development and policy evidence

Obtain an actual resource qualification through `resource_policy.py`.
The development dossier and preliminary development-replay binding must be supplied externally.
The assembler does not create them.

Keep attachment inputs beneath the candidate root.
Their development report and evidence paths must resolve against that stable root.
The attachment has these fields:

- `development`: the complete original development dossier.
- `external_evidence_review`: the supplied preliminary development-replay binding.
- `resource_qualification`: one qualified receipt for a single group.
- `development_audits` and `target_profiles`: the exact inputs that reproduce that receipt.

For several groups, supply `resource_qualifications`, keyed by every covered group ID.
Every receipt must reproduce and declare the same execution policy.

```bash
python3 empirical_execution/phase8/dossiers.py attach-development \
  --candidate local_outputs/civil_primary10000_candidate/candidate.json \
  --attachment local_outputs/civil_primary10000_candidate/attachment.json
```

The real resource verifier reruns its checks.
A checksum or qualified-looking status alone cannot pass.
The command writes `review_candidate.json` and `final_review_request.json` at the existing root.
This resolves the review-digest dependency: the complete policy and observations exist before final review.
Resource receipts exclude their own derived fields and final-review testimony from their canonical input scope.

The current resource qualifier covers the declared native C-core scope only.
Other profile families remain explicitly unqualified.
Constructing their unsigned candidates does not establish resource readiness.

## 4. Apply a supplied complete external review

The reviewer must bind the exact `reviewed_bundle_sha256` and `replay_binding_sha256` values supplied in `final_review_request.json`, the development replay digest, and the actual evidence files.
The hash of the request file itself is not the reviewed bundle digest.
The review uses the frozen `ccu-external-evidence-review-1` contract.
The assembler neither signs it nor fills accepted decisions.

```bash
python3 empirical_execution/phase8/dossiers.py finalize \
  --candidate local_outputs/civil_primary10000_candidate/review_candidate.json \
  --review local_outputs/civil_primary10000_candidate/external_review.json
```

Finalization repeats original-archive replay, resource verification, and every selected group's frozen acceptance gates.
It writes `bundle.final.json` and a one-family `bundles.json` only after those checks pass.
The dispatcher still rechecks acceptance and retains every registered job, including missing families and failed cells.
No study execution starts in these commands.

Do not relocate or rewrite reviewed files.
Literal source paths and loaded array fingerprints belong to the reviewed object.
Relative development and evidence paths depend on its stable candidate root.
Naively combining separately signed roots changes that interpretation.
Shared-root multi-family assembly remains separate software work.

## Storage, memory, and failure scope

Every stage refuses an existing output directory.
Each stage writes a private-output `.gitignore` guard.
This prevents accidental ordinary Git addition; it is not a security boundary.
Candidate files can contain corpus text, original labels, and supplied human judgments.
Never publish them as ordinary test reports.

Full caches remain mapped in their original order.
Selected train/evaluation rows, calibration subsets, and projections require new arrays.
Prepared and guarded metadata, graphs, request manifests, encoder replay, and operating-system caches incur additional costs.
The implementation makes no constant-memory or native-scale feasibility claim.
Receipts record persisted stage-directory bytes, separately from unknown peak resident memory.
These are shared-preparation costs, not repair performance.

Failures retain a blocked receipt and mark partial outputs unaccepted.
Later stages require a successful matching candidate-stage receipt.
No failed preparation can be advanced merely by pointing at a leftover file.

The focused checks use actual Civil preview text and labels.
Positive source/cache and task-choice boundaries are explicit temporary software mocks.
The checks create no genuine semantic cache, native source population, human judgment, or primary result.
The failed first staging check and its exact source snapshot remain preserved under `results/` and `history/dossier_attempt1/`.
