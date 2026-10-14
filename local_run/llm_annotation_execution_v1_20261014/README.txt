Third: activated selection execution, version 1 (2026-10-14)

This package composes the sealed inference components and independent admission
package into a concrete one-family runner. Its creation, tests and CPU qualifier
do not activate inference, generate entropy, submit jobs or make model calls.
The external controller must supply the reviewed admission, resource plan,
activation and raw package/file hashes. Historical packages remain unchanged.

run_model.py
  Required path arguments: --repo --workspace --protocol-package
  --component-package --integration-package --requests-dir --admission
  --activation-dir --model-dir --weights-manifest --output-dir.
  Required hashes: --package-sha256 --admission-sha256 --activation-sha256
  --request-result-sha256. Required family: --model-alias qwen|mistral.
  The output must equal that family's fixed activated slot and must not exist.
  The runner verifies source/package origins and exact prepared request bindings,
  then resolves new call IDs from the reviewed activation. Canonical selection
  and diagnostic swaps are distinct stages in the same selection phase.
  The executing frozen repository and separately staged protocol stay distinct;
  historical protocol evidence is never an overlay or source-authenticity claim.

  The original qualified Qwen-v5 and Mistral-v1 loaders, prompt, tokenization,
  standalone-token guard and final judgment checks run through sealed components.
  Every model weight is hashed initially and at the final input recheck. A fresh
  process loads one model: Linux/pinned Python and libraries, BF16, one A10080GB,
  forced SDPA Flash only, no fallback, batch one, greedy, 512 output tokens and
  32768 total tokens. Full inputs are checked against their prepared IDs/masks.
  Zero retries and zero human annotation calls. The narrower standalone-valid
  output vocabulary and 80-word bound remain part of this model-defined target.

  Requests run longest first with original request indices preserved. Every
  planned request has one final ledger status. Invalid JSON and overflow retain
  the denominator. The first runtime_error stops that family; remaining requests
  become not_attempted. Artifact or input integrity faults are distinct failures;
  an existing intent without completion is missing_output, never a made-up label.
  A killed/incomplete process has no accepted complete result. There is no resume,
  overwrite or alternate-output retry. A nonzero exit needs external review.

  Private root output: planned_calls.json, run_intent.json, input_bindings.json,
  runtime/{generation_config.json, token_eligibility.json,
  effective_constraint_config.json, runtime_manifest.json}, calls/<call_id>/,
  responses.json, ledger.json, final_bindings.json, result.json and failure.json
  if needed. Each call starts with an exclusive intent and keeps its raw response,
  output token IDs, guard-step ledger and metrics when generation produced them.
  Every file except result itself is byte/hash-bound in result.files. Directories
  are private and writes are exclusive. Four MiB is reserved for final metadata;
  the activated total artifact cap is enforced, including inherited runtime files.
  If the cap or filesystem prevents completion, execution fails closed.

  result schema: third-llm-annotation-execution-v1. passed/completed are false
  after a runtime-error stop. passed_execution_integrity may still be true when
  this ordinary failure and its unavailable remainder were recorded correctly.
  The independent output auditor, rather than that flag alone, admits artifacts.

qualify.py
  Required path arguments: --repo --workspace --protocol-package
  --component-package --integration-package --requests-dir --resource-plan
  --output-dir --qwen-model-dir --qwen-weights-manifest --mistral-model-dir
  --mistral-weights-manifest --control-package --statistics-package
  --engineering-replay-manifest.
  Required hashes: --package-sha256 --request-result-sha256
  --resource-plan-sha256 --engineering-replay-sha256.
  Requires a real CPU Slurm allocation, exact pinned interpreter/libraries and
  no visible GPU. It creates no activation, new production call IDs or labels.
  The output is exclusive, private and capped at 16 MiB (console is external).

  CPU qualification runs five source-bound suites: components 30, integration 16,
  execution 14, controller 8, statistics 14; all 82 must pass with zero skips.
  run_statistics_tests.py verifies/imports the sealed statistics sources and
  changes only the test module's REPO variable to the explicit executing repo.
  This test-only relocation avoids its historical parents[2] staging assumption;
  no production function, test assertion or sealed byte is rewritten.

  It reconstructs every actual prepared request from full text, verifies both
  qualified tokenizer identities, and replays all 198 input-token sequences,
  masks, rendered-prompt/input-row hashes and schedules. Tokenizer/config bytes
  are hashed; weight inventory and sizes are checked but weight contents are not
  read or loaded. Exact model load/weight checks remain in activated execution.
  It then independently replays fixed call-0000 from each already exposed pilot:
  nine pinned artifacts per family, unchanged raw UTF-8 bytes, output IDs, strict
  judgment, character/guard/full-prefix transitions, guard metrics and runtime
  identities. No old response bodies are copied into qualifier results.

  Reports: qualification.json (third-llm-inference-producer-consumer-validation-v1),
  software_suites.json and private test logs, token_replay.json,
  historical_engineering_replay.json, input_bindings.json; failure.json on error.
  Initial/final bound inputs are rehashed. The report explicitly says that new
  GPU orchestration has not been tested. The first future GPU generation must be
  an activated, recorded planned attempt, not an unrecorded smoke call.

Scope
  Synthetic control fixtures and historical output replays establish software
  compatibility only. They do not establish semantic accuracy, human equivalence,
  publisher authenticity, held-out validation success or paper readiness.
  Raw model logits and greedy optimality are runtime-attested, not independently
  replayed. Actual output token/prefix/schema checks are independently replayable.
  Model-only statistics are admitted separately from audited complete artifacts.
