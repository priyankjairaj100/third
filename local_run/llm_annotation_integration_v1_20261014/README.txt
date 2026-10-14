Third: LLM-only annotation integration, version 1, 2026-10-14

This package supplies activation admission, independent output auditing, and
selection-statistics ingestion. Writing or testing this package does not activate
inference. No production activation, entropy, model call, human annotation, or
threshold outcome was created during development. Its helper tests contain only
small, explicitly synthetic judgments.

Authority and dependency separation
  expectations.json pins the existing component, statistics, protocol and request
  preparation packages. It also pins the actual completed request result, its
  independent 198-request tokenizer audit, and that audit's source bytes. The
  actual executing repository and separately staged protocol package remain
  different roots. Historical evidence snapshots do not claim checkout equality.

  The final resource plan supplies the as-yet separately sealed runner, integration
  and operational control manifests. This avoids circular manifest hashes. A
  producer/consumer qualification receipt must bind every package and the exact
  request result before activation. The receipt contains zero model calls: actual
  request rendering/tokens and software fixtures are tested on CPU. The first
  GPU generation by the new runner is its first counted planned call, not an
  unreported warm-up or an earlier claimed GPU qualification.

  An externally hash-pinned admission JSON lists paths and bindings for all seven
  packages, the request directory/review/audit source, selection result/review,
  original sampling lock, resource plan and producer/consumer report. Package
  members and all prepared outputs are rehashed initially and finally. Receipt
  checks do not establish publisher authenticity or semantic accuracy.

Activation APIs
  inspect_admission(path, expected_binding) performs read-only admission checks.
  read_activation(admission_path, admission_binding, activation_dir,
      activation_binding, component_contract=None) returns (Activation, state).
  state contains activation_contract, resource_plan, admission, watch_files and
  the full initial/final Watch. A runner must pass its already verified component
  contract module to preserve the Activation class identity.
  create_activation(admission_path, admission_binding, new_output_dir) creates
  only an exclusive durable intent.json followed by activation.json. Existing
  activation or family output paths refuse. There are no entropy calls, retries,
  resume semantics, seed draws, submissions, or implicit activation at import.
  CLI defaults to read-only inspection; --create and --out are explicit.

  The exact resource-plan schema is validated in activation.py. It binds seven
  package manifests, actual input/audit/lock, complete 90+9 requests per family,
  fixed distinct private output slots, four CPU cores, one A10080GB, Qwen48GiB or
  Mistral64GiB host RAM, bounded time/output/console limits, and an unchanged8GiB
  quota reserve. Parent-reviewed prospective allocations may be stricter than
  the validator's maxima. Runtime failure stops that family; every remaining
  request is not_attempted. Invalid JSON and overflow stay in the denominator.

Output audit
  audit.py independently reconstructs v3 call IDs and pair/model seeds, input
  text joins and hashes, rendered prompts, every input token ID, schedule,
  canonical/swapped phase and complete planned denominator. It binds exact
  per-call intents/completions/artifacts and the full output inventory.

  Full raw output decode is checked using the actual pinned tokenizer. An
  independent previously reviewed helper reconstructs the complete standalone
  vocabulary inventory and raw-prefix consistency. The LMFE0.11.3 tokenizer
  table is rebuilt directly without importing its torch integration. The exact
  pinned guard and native token filter are replayed as a separate audit pass.
  Raw JSON uses separate strict duplicate/nonfinite rejection, JSON Schema and
  Python decoded reason.split() <=80 checks. Valid statuses require all checks;
  invalid_json must reproduce at least one format/grammar failure. Statuses are
  never repaired, relabeled, or regenerated. Saved finite-candidate counters
  are checked for consistency; full logits are absent, so score optimality and
  unconstrained greedy equivalence are not independently claimed.

  Runtime/Flash-only and generation settings must match the qualified family
  receipts. Qwen's unchanged loader omits huggingface-hub from its runtime
  package map; its existing component preflight separately checks that version.
  The two exact observed package maps are pinned without inventing fields.

  A recorded runtime_error followed by not_attempted calls may pass provenance
  audit while remaining unavailable scientifically. Incomplete/killed runs,
  missing_output, and artifact-integrity faults are quarantined and cannot enter
  ingestion through this successful-run auditor. They require explicit later
  reconciliation; this is not permission to drop them or rerun their calls.
  Model-weight hashes and before/after equality are bound runner attestations;
  this audit rechecks metadata and tokenizer payloads, not all model weights.

Statistics ingestion
  ingest.py requires exactly two audit reports and their raw external bindings,
  binds the execution results and response files, and checks that audited
  observations equal raw completed responses. Only canonical selection rows
  enter the pinned statistics core. The original sealed sample and its exact
  bin populations/weights are retained. Invalid, runtime-error, not-attempted
  and uncertain outcomes cannot disappear from the denominator. Swaps are
  reported separately and never increase estimator sample size.
  The result is model-defined selection evidence only. Human validity,
  publisher authenticity and confirmatory readiness remain false. It does not
  draw validation pairs, execute validation, modify the original analysis lock,
  or perform post-outcome retuning.

Development tests
  python -B -m unittest discover -s THIS_DIRECTORY -p test_integration.py -v
  Staged roots: THIRD_INFERENCE_REPO (or THIRD_INTEGRATION_REPO),
  THIRD_INTEGRATION_COMPONENTS, THIRD_INTEGRATION_PROTOCOL,
  THIRD_INTEGRATION_STATISTICS. There are16 tests and no dependency skips.
  Tests require the existing jsonschema and LMFE0.11.3 environment, no torch,
  model weights, network or real labels. A tiny synthetic vocabulary includes
  the full ASCII alphabet needed by LMFE's first-character alphabet convention;
  its development fixture failures were corrected before the passing run.
