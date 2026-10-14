Third: prospective LLM-only inference components, 2026-10-14

STATUS: NONLAUNCHING SOFTWARE COMPONENTS. No activation writer, production
runner, Slurm wrapper, model call, entropy draw or label artifact is supplied.
component_check.py only checks local source pins and prints remaining gates.
It cannot submit jobs or load model weights. This package must not be described
as a qualified end-to-end inference runner.

Scope
  contract.py validates prospective activation metadata, prepared request rows,
  identities, context census, and the new model-only response envelope.
  kernel.py imports each family's exact qualified loader, renderer, schema
  validator and guard by pinned source identity. It implements the single-call
  generation sequence with new v3 identities, preserving BF16/Flash-only greedy
  batch-one controls, 32768 total tokens, 512 output tokens and no retries.
  ledger.py provides exclusive private call intents/completions and a complete
  planned denominator after an ordinary caught process failure.

These are library components for a subsequently reviewed controller. Directly
constructing Python objects is not an authorization mechanism. The activation
reader checks raw bytes, exact intent/inventory, internal seal and an externally
supplied exact contract; it does not certify that the bound review contents or
resource plan are scientifically sufficient. Its schema must be adopted by the
future exclusive activation writer and jointly tested before activation.

Qualified kernels, unchanged
  Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28:
    local_run/model_annotation_runtime_v5_20261011/run_pilot.py
  Mistral-Nemo-Instruct-2407 04d8a90549d23fc6bd7f642064003592df51e9b3:
    local_run/model_annotation_runtime_mistral_v1_20261011/run_pilot.py
  The shared bounded_reason.py bytes, LMFE 0.11.3, standalone-valid token
  restriction, exact raw-prefix guard and <=80 decoded reason words are retained.
  This excludes valid split-codepoint tokens and literal U+FFFD; it does not prove
  semantic accuracy or unrestricted-token decoding equivalence. Greedy score
  ranking is runtime-attested, not independently established by token replay.
  The family loader's original engineering runtime receipt is kept unchanged
  and scoped to the reused component. It is not a legacy scientific admission.

Identity and API
  The v3 protocol package is an explicit separately bound root. The executing
  repo may remain at its older frozen revision. Historical protocol evidence,
  staged protocol code and executing frozen model code are distinct roles.
  Request input_row_sha256 uses the qualified pilot's ensure_ascii=True compact
  sorted JSON. Rendered prompt serialization remains the qualified renderer's
  ensure_ascii=False JSON with its original spacing and key order.
  model_seed comes from protocol.model_seed(PLAN raw SHA, frame result raw SHA,
  opaque pair ID, model ID, model revision); stage/orientation are excluded.
  call_id uses the v3 call-id domain over raw activation SHA, stage, opaque pair
  ID, model ID/revision, orientation and attempt zero. Sampling lock alone cannot
  activate inference. Pre-activation call IDs and activation hashes stay null.

  validate_prepared_rows(shared, tokens, texts, alias, protocol) -> pilot rows
  read_prepared_bundle(folder, activation, alias, protocol) -> content/bindings
  resolve_calls(tokens, request_manifest_binding, activation, protocol) -> calls
  schedule(tokens) -> positions sorted full tokens descending, selection index
      ascending, canonical first on a tied pair (original indices are retained)
  validate_envelope(response, planned_call, strict_judgment_callback) -> response
  FamilyKernel(repo, alias, protocol_package).load(activation, model_path, weight_manifest,
      empty_runtime_directory) -> inherited runtime receipt
  execute_call(kernel, call, prepared_token_row, full_pilot_row, CallWriter)
      -> validated new response
  Ledger(new_directory, byte_cap, one_model_selection_phase_calls, judgment_validator)
      .begin / .record / .finish -> immutable artifacts and full planned ledger

Response schema: third-llm-model-response-v3
  Shared identities: stage, opaque_pair_id, model_id, model_revision, orientation,
  attempt, call_id, model_seed, request_index, selection_index, request_manifest,
  request_row_sha256, input_token_ids_sha256, activation_lock_sha256.
  Other keys: annotator_type=model, human_completed=false, status, judgment,
  raw_response_sha256, runtime_manifest_sha256, artifacts, error.
  Exact statuses: valid, refusal, invalid_json, context_overflow, runtime_error,
  missing_output, not_attempted. Only valid carries a judgment. No refusal
  classifier is implemented; unparseable refusal prose remains invalid_json.
  Generated valid/invalid/refusal require actual raw/token/guard/metrics bindings.
  Missing artifacts never become a valid judgment or disappear from denominator.

Persistence and accounting
  Exclusive output root and call directory; durable intent before generation;
  one immutable completion. No restart/resume/retry path. Existing paths refuse.
  Private files use 0600/directories0700 and a configured aggregate byte cap.
  Requests and texts are referenced, not copied into every call artifact.
  A durable call intent without completion becomes missing_output after an
  ordinary caught fatal condition. Never-started calls become not_attempted.
  A killed process has no fabricated completion: a separate auditor must
  reconcile its missing records against planned_calls.json.
  File/model/code SHA and HF symlink targets are checked initially and via
  FamilyKernel.recheck(); the future controller MUST run this final check and
  bind its result before publishing any successful inference result. The low-
  level Ledger completion alone is not an accepted inference result.
  Integrity exceptions from artifact writes propagate; they are not model votes.
  Peak RSS, GPU allocation/reservation, generation/guard timing are separate.

Remaining required gates before any production model call
  1. Actual prepared request bundle and independent text/token audit.
  2. Final exclusive activation writer, substance checks of every bound review,
     and exact source/history/resource plan; no new sampling entropy.
  3. Full controller orchestration of per-model fresh selection-phase processes (canonical and swap diagnostics),
     initial/final file and runtime checks, watchdog/resource limits, scheduling,
     killed-process reconciliation and final audit-qualified publication.
  4. Independent output auditor reconstructing raw tokens, vocabulary, grammar,
     call/request identities, initial/final bindings and complete denominators.
  5. Joint producer/consumer test with new model-only statistics and activation.
  6. Actual pinned-tokenizer/torch and GPU integration qualification of THIS
     orchestration. Historical pilots are not execution tests for new code.

Software checks (no inference)
  THIRD_INFERENCE_REPO=/path/to/Three python -m unittest discover \
    -s /path/to/this/package -p 'test_*.py' -v
  python component_check.py --repo /path/to/Three \
    --protocol-package /path/to/staged/model-annotation-v3-20261014

All examples/fixture judgments are tiny labeled software tests. No real labels,
threshold choice, validation sampling or human annotation has occurred here.
