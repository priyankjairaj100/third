# Explicit-policy dispatcher CLI

Phase 7 adds a CLI around the unchanged Phase 6 dispatcher.
The old CLI could not forward a development-locked policy.
Its Python API already supported that policy.
This wrapper forwards an explicit JSON policy unchanged and preserves omission as `policy=None`, which invokes the legacy Phase 6 default.

```bash
python3 empirical_execution/phase7/run_dispatch.py --help
python3 empirical_execution/phase7/run_dispatch.py \
  --registry empirical_execution/phase6/results/recipes_release/registry.json \
  --jobs empirical_execution/phase6/results/recipes_release/jobs.jsonl.gz \
  --bundles LOCAL_INPUT/bundles.json \
  --policy LOCAL_INPUT/execution_policy.json \
  --mode accepted_primary_inputs \
  --out NEW_OUTPUT_DIRECTORY
```

This command requires actual accepted input dossiers.
It does not create them or approve provenance, human collection, or development choices.
Primary bundles must contain the identical `execution_policy`; all Phase 6 acceptance and complete-registry gates remain in force.
Every bound file must reside within the bundle's root and match its checksum.
Existing output paths are refused before dispatch.

The policy requires integer `memory_bytes` (at least 64 MiB), positive integer `cpu_seconds` and `threads`, and finite positive `wall_seconds`.
Optional fields are `policy_role`, `convex_verifier`, `convex_certificate_coordinates`, `state_audit_max_pair_coordinates`, `state_audit_max_coefficient_coordinates`, `state_audit_absolute_tolerance`, and `state_audit_relative_tolerance`.
Unknown fields, duplicate JSON keys, booleans as numbers, negative work caps, and nonfinite numbers are refused.
No values are coerced or inserted.
The verifier is either `fraction_reference` or `exact_dyadic_integer_v1`.
Optional audit tolerances may tighten the Phase 6 defaults but may not exceed absolute `1e-11` or relative `1e-10`.
Work caps may change prospectively; this wrapper does not establish their native-scale feasibility.
The common ridge residual and convex certificate tolerances are unchanged.

The output adds `cli_invocation.json`, binding the exact top-level input files, policy file and forwarded value, wrapper source, and dispatcher source.
Each top-level file is captured once; parsing and hashing use those same bytes, including the original compressed job file bytes.
Later changes to those paths do not change the loaded snapshot or its receipt.
The unchanged dispatcher supplies its complete input lock, acceptance reports, job ledger, and execution summary.
Omission of a policy retains the legacy development envelope and is not an automatic primary resource qualification.

Focused reproduction, using a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 empirical_execution/phase7/check_dispatch_cli.py \
  --out empirical_execution/phase7/results/dispatch_cli_new_run
```

The checks forward all 21,332 registered jobs through a transient mock without executing them.
They check policy forwarding, omission, malformed-policy refusal, output collisions, and exact schedule retention.
One real Civil20 lexical job runs one O-T worker with three releases and an explicit nondefault policy.
It reuses existing texts and original labels; it is a CLI smoke check, not a benchmark or primary semantic result.
All Phase 6 sources and reports remain unchanged.

The initial wrapper passed 27 checks and the one real job in `results/dispatch_cli_attempt1`.
Independent review then identified a receipt race: the initial wrapper parsed files and separately reread them for hashes.
The corrected wrapper captures each file once and has a routing-only regression that changes all four paths after capture.
Original wrapper/checker bytes remain in `history/dispatch_cli_v1/`; the initial result is preserved.
The focused final check reuses that exact natural evidence, under its historical source hash, without a second worker run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 empirical_execution/phase7/check_dispatch_cli.py \
  --out empirical_execution/phase7/results/dispatch_cli_final \
  --reuse-natural-result empirical_execution/phase7/results/dispatch_cli_attempt1
```

The final report distinguishes current-wrapper routing tests from the historical natural execution.
It does not claim that the natural job was rerun under the corrected wrapper.
