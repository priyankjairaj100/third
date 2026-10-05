# Local execution and accounting

Read `START_HERE.md`, `INPUTS_AND_PREPARATION.md`, and `EMPIRICAL_PROGRAM.md` in this directory first.
The user runs empirical work locally from this checkpoint onward.
The assistant receives the resulting reports and checks their scientific interpretation.

This guide runs existing, frozen entry points.
It does not claim that authentic inputs and reviews already exist.
There is no complete unattended command for the whole study today.
Human responses, evidence reviews, and several branch manifests still require real preparation.
The local LLM can author manifests from actual artifacts.
It must not invent the evidence those manifests describe.

## 1. Prepare one stable private workspace

Use a local Linux environment for the measured campaign.
Read the runtime limits in `INPUTS_AND_PREPARATION.md` before installing packages.
The launcher uses the Python interpreter that invokes it.
Activate the intended environment first.

Run these commands from the repository root:

```bash
python tools/verify_backup.py
python local_run/run_local.py init
python local_run/run_local.py routes
```

The first command verifies the published files.
The second creates directories and a private-output Git guard.
It does not download data, install dependencies, or create scientific evidence.

| Private directory | Contents |
| --- | --- |
| `local_workspace/inputs` | Original acquisitions and authored source/configuration manifests |
| `local_workspace/models` | Pinned model and tokenizer files |
| `local_workspace/prepared` | Original-schema exports, source partitions, caches, registry |
| `local_workspace/calibration` | Blank packs, actual responses, thresholds, validation |
| `local_workspace/development` | Source-disjoint development data and observed profiling |
| `local_workspace/candidates` | Stable per-owner candidate roots and their review evidence |
| `local_workspace/plans` | Immutable complete study routing plans |
| `local_workspace/runs` | Immutable engineering, confirmatory, and analysis attempts |
| `local_workspace/reviews` | Supplied review evidence before binding into owner roots |
| `local_workspace/launch_specs` | Explicit launcher specifications |
| `local_workspace/runner_logs` | Captured subprocess commands, output, errors, and receipts |
| `local_workspace/reports` | Private summaries and publication-screened return candidates |

Choose the workspace location before preparing candidates.
Phase 9 binds each candidate's canonical absolute root.
Moving a reviewed candidate changes its contract, even if its file bytes remain unchanged.
Do not upload the private workspace to GitHub.

## 2. Use the thin launcher

Each `templates/run_*.template.json` file describes one allowlisted command.
Copy the needed template into `local_workspace/launch_specs`.
Edit paths, run identifiers, and dependency bindings using actual files.
All paths in launcher specifications resolve from the repository root.
Paths inside scientific dossiers retain their original frozen resolution rules.

For example, generate the unchanged registry:

```bash
cp local_run/templates/run_registry.template.json local_workspace/launch_specs/registry_v1.json
python local_run/run_local.py plan --spec local_workspace/launch_specs/registry_v1.json
python local_run/run_local.py run --spec local_workspace/launch_specs/registry_v1.json
```

This writes `recipes.json`, `registry.json`, and `jobs.jsonl.gz` beneath `local_workspace/prepared/registry_v1`.
It does not execute any study job.
Inspect those files before selecting complete registered scopes.

`plan` validates top-level paths, dependency conditions, and new output locations.
It prints the exact argument list without executing the route.
`run` calls that argument list with `shell=False`.
It preserves stdout, stderr, the interpreter identity, Git commit, and input hashes.
It refuses reused run identifiers and existing output stages.
It refuses local changes inside frozen Phase 3–10 and study-design paths.

The launcher does not authenticate testimony or expand scientific defaults.
It does not replace any underlying input gate.
Its `evidence_role` field is a declaration, not an acceptance certificate.
The frozen executor still checks nested assets and scientific evidence.

Declare real dependencies explicitly:

```json
{
  "path": "local_workspace/plans/confirmation_v1/receipt.json",
  "sha256": "REPLACE_WITH_ACTUAL_RECEIPT_SHA256",
  "pointer": "/status",
  "equals": "routing_plan_ready"
}
```

Compute the digest from file bytes with `hashlib.sha256` or `sha256sum`.
The pointer identifies the exact JSON field that must match.
Use an empty dependency list only when there are no earlier launcher dependencies.
Scientific dependencies remain enforced by the frozen route even when that list is empty.

## 3. Preparation and genuine calibration

Follow `INPUTS_AND_PREPARATION.md` for downloads, original exports, source adapters, partitioning, and CPU FP32 encoding.
Keep original source lineage and full cache replay available.
Then use these templates:

| Launcher route | Frozen entry point | Input that must already exist |
| --- | --- | --- |
| `calibration-dossier` | `phase8/dossiers.py calibration` | Original source specification and complete encoder cache |
| `calibration-pack` | `phase7/prepare_calibration.py` | Accepted prethreshold calibration dossier |
| `candidate` | `phase8/dossiers.py candidate` | Explicit group scope, real selection responses, and fresh validation responses |

For each filled launch specification:

```bash
python local_run/run_local.py plan --spec local_workspace/launch_specs/REPLACE_WITH_FILLED_SPEC.json
python local_run/run_local.py run --spec local_workspace/launch_specs/REPLACE_WITH_FILLED_SPEC.json
```

This placeholder command is a usage pattern, not a runnable scientific configuration.
Use the actual filenames you created.

The calibration-pack route creates blank selection assignments only.
It does not conduct human collection, choose a threshold, or provide fresh validation responses.
Use the frozen human workflow described in `EMPIRICAL_PROGRAM.md` and the calibration preparation instructions.
WCEP inherits News calibration; it must not select a new threshold using WCEP outcomes.

The unsigned candidate route performs source/cache replay and reconstructs task choices.
Its `candidate.json` remains unapproved.
Keep every failed receipt and any rejected partial outputs.

## 4. Development and resource evidence

Keep development sources disjoint from confirmation and full-refit pools.
Specify the development population, resource caps, and intended profiles before observing method effects.
Read the development gate in `EMPIRICAL_PROGRAM.md`.
There is no registered numerical pilot quota that the launcher may silently supply.

For diagnostic engineering, `run_engineering_dispatch.template.json` calls the frozen Phase 7 wrapper.
It always passes `--mode natural_text_engineering` and an explicit policy file.
The bundle must satisfy that route's engineering contract.
The complete canonical registry and job file still remain intact.
These runs do not pass primary acceptance and cannot become confirmation later.
Do not reuse the old Civil100 lexical preview as fresh semantic development.

The primary resource path is:

1. Build the actual Phase 8 candidate with the intended registered groups.
2. Supply the real source-disjoint development dossier and preliminary replay review.
3. Run `prepare-resource` with an attachment containing `scope`, `development`, and `external_evidence_review`.
4. Compile the exact profiles from the resulting `resource_candidate.phase9.json`.
5. Freeze observation identifiers and collect matching actual observations.
6. Run `qualify-resource` on the complete bound evidence.
7. Attach the reproduced qualification before final review.

The scope contains exactly `key`, `group_ids`, and `extension_recipes`.
Use owner keys and group names derived from the unchanged registry.
Resource qualification supports complete registered scopes, not a caller-chosen favorable trajectory subset.

The profile compiler is an API, not a standalone observation-producing CLI:

```python
import json
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "empirical_execution"))
from phase8 import dossiers
from phase9.resource_profiles import compile_profiles

root = repo / "local_workspace/candidates/civil_primary10000_v1"
bundle = dossiers.load_value(root / "resource_candidate.phase9.json", base=root)
key = bundle["phase9_scope"]["key"]
manifest, context = compile_profiles(key, bundle, base_dir=root)
with (root / "resource_profiles.json").open("x") as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
    stream.write("\n")
```

Use this only after actual accepted inputs exist.
Its default `require_acceptance=True` must remain enabled.
Do not serialize the internal replay context as an alleged evidence receipt.

The local LLM must author the observation plan and request descriptors against these profiles.
Read `phase9/RESOURCE_PROFILES.md` and its receiver implementation before authoring them.
The plan uses `actual_disjoint_development` and binds the manifest and resource-lock hashes.
It fixes the ordered observation identifiers before execution.

`measure-resource` supports the existing graph and pointwise convex-verifier producer.
It does not fit the convex model or measure optimizer trajectories.
Ridge worker and FP64 audit observations require the matching frozen worker/audit orchestration.
The collector must retain its actual configurations, full checkpoint schedule, and raw measurements.
Read the receiver's `worker_observation` requirements before writing that local collection adapter.
No existing generic launcher produces all such observations automatically.
Keep any new adapter versioned and review its bindings before confirmation.

`qualify-resource` reads a request with exactly `owner_key`, `bundle`, and `evidence`.
The bundle descriptor uses `{path, sha256, kind: "json"}`.
The receiver replays the real evidence before exporting a policy.
An exported policy qualifies its stated observed scope only.
It is not a guarantee of future runtime, peak memory, or optimizer feasibility.

Do not raise caps because a method lost or failed on confirmation.
Record a blocked profile if accepted matching development observations are unavailable.

## 5. Review a complete candidate at its original root

Use these launch templates after resource evidence exists:

| Route | Reads | Writes |
| --- | --- | --- |
| `prepare-resource` | `candidate.json` and development attachment | `resource_candidate.phase9.json` |
| `prepare-review` | Resource candidate and qualification attachment | `review_candidate.phase9.json`, `final_review_request.phase9.json` |
| `finalize-review` | Review candidate and supplied actual review | `bundle.phase9.final.json` after all checks pass |

All three stages write within the original candidate root.
Each stage refuses an existing stage directory.
Their attachments and evidence paths must retain the same interpretation throughout.

The final reviewer must examine the actual artifacts and supply the required evidence bindings.
The request-file hash is not the reviewed-bundle hash.
The review binds `reviewed_bundle_sha256`, `replay_binding_sha256`, the development replay, and real evidence files.
An LLM can help inspect files and draft a review record.
It cannot claim an independent human review or historical execution it did not observe.
Use the actual reviewer identity and role required by the frozen contract.
If the required review is unavailable, leave the candidate blocked.

## 6. Assemble and execute complete confirmation accounting

Read `phase9/STUDY_ASSEMBLY.md` before writing `study_owners_v1.json`.
Each owner entry explicitly supplies:

- The canonical `key` and stable `root`.
- A `{path, sha256}` binding to its final bundle.
- Its complete selected `group_ids` and `extension_recipes`.
- Task-resource `request` and `receipt` bindings at that same root.

The selected scope cannot exceed the reviewed scope.
Never rewrite signed paths while merging owners.
Human owners use their human stage contracts and genuine evidence.
Do not invent a model-resource receipt for them.
Human dependencies route to the corresponding task family's accepted root.

Fill and run `run_assemble.template.json`, then inspect its receipt:

```bash
python local_run/run_local.py run --spec local_workspace/launch_specs/assemble_v1.json
python local_run/run_local.py inspect --directory local_workspace/plans/confirmation_v1
```

Assembly derives all 21,332 canonical jobs and their owners.
There are 43 core groups and 28 distinct input keys.
Unselected or missing jobs remain visible.
`routing_plan_ready` means only that routing was assembled.
It does not mean that inputs were approved or every scope was selected.
Review the owner counts and explicit missing scope before launching.

Bind the plan receipt in the execute launch specification.
Then execute the reviewed configuration:

```bash
python local_run/run_local.py run --spec local_workspace/launch_specs/execute_v1.json
python local_run/run_local.py inspect --directory local_workspace/runs/confirmation_v1
```

These filenames assume you copied and filled the matching templates.
No command edits the registry or silently selects a pilot subset.
Scope choices must follow the prospective scientific plan.

Watch `local_workspace/runner_logs/<run_id>/stdout.txt` and `stderr.txt` during execution.
The wrapper captures output rather than printing large raw scientific objects into the terminal.
Do not launch overlapping campaigns against the same mutable inputs.

The authoritative outcome is the complete final `jobs.json` bound by `summary.json`.
Per-job files and the append journal remain provisional until final accounting succeeds.
`completed_dispatch` can include blocked, failed, unavailable, and structural-zero rows.
It does not mean all experiments passed or the method won.
Input or source changes can revoke primary flags in the final ledger.

The launcher reports subprocess exit status separately from scientific acceptance.
Some frozen commands return exit code zero with a blocked scientific receipt.
Always inspect the underlying receipt, its gates, and the final ledger.

## 7. Human admission audits and final statistics

Actual semantic admissions require the registered human admission and context audits.
Use the frozen graph serialization from `phase9/graph_dossier.py`.
It preserves the signed graph fingerprint and provides no judgments or review testimony.
Read `phase9/GRAPH_DOSSIER.md` and the human protocol before authoring the human owner dossiers.

Final statistics require a previously finalized complete confirmation ledger.
Create the statistics dossier after that ledger exists.
Bind `phase9_dispatch_evidence` to its unchanged `summary.json`.
Each `job_outcomes` descriptor must reference the same complete `jobs.json` and its recorded digest.
Bind predictions, test-source identifiers, missingness, thresholds, and actual review as required by the frozen statistical route.

Build a new analysis-only owner configuration and a new plan.
Set both launcher roles to `analysis` and use fresh output directories.
Its unselected model rows are accounting placeholders, not new model measurements.
The earlier confirmation ledger remains the empirical evidence.
Never substitute provisional job files for that ledger.

## 8. Resume safely and report results

The coordinator has no automatic crash resume or cross-invocation pooling.
Do not append recovered rows to a signed final ledger.
After a crash, preserve the entire attempt and logs.
Classify its outputs as provisional if no valid final ledger exists.
Use a new attempt identifier and output directory for any rerun.
Record its relationship to the previous attempt.

Keep the same accepted input roots only if their bytes and bindings remain unchanged.
Never merge favorable results across attempts.
A future resumable executor requires explicit versioning and review before scientific use.

Use the results-return instructions in this directory to report outcomes.
Return complete denominators, failures, zeros, unavailable arms, and qualification scope.
Return file hashes and machine/runtime details with measured costs.
Do not publish News article bodies, raw human responses, or other private workspace contents.

## Automation boundary

| Work | Current implementation |
| --- | --- |
| Run fixed preparation and dispatch commands | Frozen CLIs, wrapped by `run_local.py` |
| Obtain licensed originals and pinned model files | Local acquisition; see preparation instructions |
| Generate blank selection assignments | Automated after accepted input replay |
| Supply real selection, validation, and admission judgments | Genuine human collection required |
| Prepare unsigned candidates and task choices | Frozen staged assembler |
| Compile resource profiles | Frozen API; use actual accepted dossiers |
| Produce graph/pointwise-verifier measurements | Frozen Phase 9 producer |
| Produce all ridge-worker/audit development observations | Actual orchestration adapter and manifests still require local preparation |
| Attest provenance and complete evidence review | Accountable supplied review required |
| Run complete immutable job accounting | Frozen Phase 9 coordinator |
| Build final statistics dossier | Local manifest construction from the finalized ledger |
| Restart a crashed invocation automatically | Not implemented |

These limitations are execution tasks, not successful experiments.
Do not bypass them to produce an apparently complete report.
