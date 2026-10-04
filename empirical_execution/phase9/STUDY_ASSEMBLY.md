# Study assembly across stable input roots

`study_assembly.py` adds a single complete outcome ledger for the frozen
21,332-job registry. Each input family retains its original root. Reviewed
relative paths, literal bundle fields, and loaded arrays are not relocated or
rewritten. Core groups and extension recipes have explicit, exclusive owners.

This is an input and execution interface. It supplies no corpus, semantic
cache, source identity, human response, external review, or development
measurement. The authentic primary study remains unstarted.

## Why a new coordinator is needed

The frozen Phase 6 dispatcher accepts one input root and a complete registry.
Running it once for every family would risk widening a reviewed group scope.
Rewriting relative paths would change the reviewed object.

The Phase 9 coordinator validates the complete canonical registry, derives
ownership, and calls the unchanged Phase 6 acceptance and branch functions.
It preserves group input gates, fixed request schedules, threshold validation,
shared package charges, independent head/state checks, full-refit orchestration,
and extension acceptance. The numerical and service implementations remain
unchanged. Each selected job is attempted once per invocation; unselected,
missing, failed, and zero cells remain in the same ordered ledger.

Ownership is derived from the compiler:

| Registered job | Input owner key |
| --- | --- |
| A–G group | Its exact `request_family` |
| Ordinary H recipe | `corpus/panel` |
| Threshold human recipe | `human_threshold/corpus/encoder` |
| Main admission audit | `human_admission` |
| Context audit | `human_context` |
| Final statistics | `statistics` |

There are 43 core groups and 28 distinct input keys. Each key has at most one
stable root in a plan. Group lists and extension recipe lists are explicit.
A selected recipe expands to all its canonical jobs for that owner; there is
no outcome-dependent trajectory sampling. A family does not acquire every
other group merely because its bundle exists.

## Prepare a broader reviewed task bundle

These stages extend the unsigned Phase 8 candidate route without modifying
Phase 8. Start with an actual Phase 8 `candidate.json` and its successful
preparation receipt. Keep the same private candidate root throughout.

First, supply an attachment containing:

- `scope`: `key`, `group_ids`, and `extension_recipes`.
- `development`: the actual source-disjoint development dossier.
- `external_evidence_review`: the externally supplied preliminary development
  replay binding required by resource qualification.

The selected group IDs must be present in the original candidate. The new
`phase9_scope` is inserted before resource qualification, so its exact content
belongs to the qualification fingerprint.
The same stage inserts `phase9_stable_root`, the canonical absolute root.
The final review binds that field as well as the original relative strings.

```bash
python3 empirical_execution/phase9/study_assembly.py prepare-resource \
  --candidate local_outputs/civil/candidate.json \
  --attachment local_outputs/civil/development_attachment.phase9.json
```

This creates `resource_candidate.phase9.json` at the same root. Use
`resource_profiles.py` to qualify its exact registered scope from actual
bound observations. The qualification is not an execution approval.

Next, supply a hash-bound attachment with `qualification` and `evidence`:

```bash
python3 empirical_execution/phase9/study_assembly.py prepare-review \
  --candidate local_outputs/civil/resource_candidate.phase9.json \
  --attachment local_outputs/civil/qualification_attachment.phase9.json
```

The real resource verifier must reproduce the receipt. Source/cache and
original-archive replay must also pass. The command attaches the qualification
and its evidence inside the bundle, writes `review_candidate.phase9.json`, and
creates `final_review_request.phase9.json`.

The final external reviewer must bind the request's
`reviewed_bundle_sha256` and `replay_binding_sha256` values, the immutable
development replay digest, and the actual evidence files. The request-file
hash is not the reviewed-bundle digest. No command creates this testimony.

```bash
python3 empirical_execution/phase9/study_assembly.py finalize-review \
  --candidate local_outputs/civil/review_candidate.phase9.json \
  --review local_outputs/civil/external_review.phase9.json
```

Finalization replays the qualification and each selected core group's frozen
input gates, then writes `bundle.phase9.final.json`. Its approval scope is
those groups only. Each extension must still pass its own frozen gate during
execution. Review preparation currently requires at least one selected core
group as its source/cache replay reference; an extension-only execution can
use a reviewed bundle with a wider group scope.

## Assemble multiple families

An assembly configuration contains an `owners` list. Each entry has:

- `key`: one canonical input key.
- `root`: the stable root, relative to the configuration or absolute.
- `bundle`: exactly `{path, sha256}`, referencing the final JSON object inside
  that root. Nested JSON/JSONL/NPY descriptors retain the same root.
- `group_ids` and `extension_recipes`: explicit selected scope.
- For task families, `resource.request` and `resource.receipt`: each exactly
  `{path, sha256}`, rooted in the same directory. Their loaded values must equal
  the evidence and qualification attached inside the reviewed bundle.

The key inside the final bundle's `phase9_scope` must match, and the plan's
selected scope cannot exceed it. Resource verification must cover the actual
selected canonical job IDs and reproduce the exact bound execution policy.
Human and statistics dossiers use their own frozen stage review contracts;
they do not receive a fabricated model-resource receipt.
They must also include their canonical absolute `phase9_stable_root` before
the supplied stage review is signed. Execution refuses a different root even
if the files were copied without changing their bytes.

```bash
python3 empirical_execution/phase9/study_assembly.py assemble \
  --config local_inputs/study_owners.json --out local_outputs/study_plan

python3 empirical_execution/phase9/study_assembly.py execute \
  --plan local_outputs/study_plan/plan.json --out local_outputs/study_dispatch
```

Assembly can retain missing inputs as explicit blocked owner entries. It never
labels a plan accepted. Execution rechecks loaded input fingerprints, source
versions, ownership, resource evidence, and the frozen primary gates.
Every plan includes all 21,332 jobs, even when no owner is available.

Different families can retain different resource policies justified by their
own frozen development locks. The coordinator never silently replaces them
with a common maximum. Such differences remain visible and must be considered
in any cross-family comparison. A resource count qualification remains an
operational work limit, not a runtime or peak-memory guarantee.

## Human dependencies and statistics

A human corpus frame stores its graph using
`phase9.graph_dossier.graph_payload(actual_graph)`. This produces canonical
JSON containing the original IDs, source ownership, priority, blocker CSR,
and exact threshold. The loader supplies the frozen graph interface through
`CanonicalGraph`, while preserving the exact signed JSON fingerprint.
`reviewed_dossier_sha256` computes the same input digest as the frozen stage
review gate; it supplies no approval or collector declaration.
Plain arbitrary graph dictionaries and unserializable Python graph objects
are not accepted as substitutes. The frozen human executor still compares
the hydrated graph with the graph rebuilt from accepted source/cache inputs.

A human stage uses its own root for its review evidence. Read-only acceptance
of a required task family is routed to that task family's original root.
Missing, substituted, or resource-blocked dependencies cause an explicit
failed/blocked job. Dependency checks do not run the task group's experiments.

Per-job output files and the append journal are provisional. The authoritative
result is the final complete `jobs.json` plus its hashed `summary.json`.
If a bound input or source changes, the coordinator revokes primary flags in
that final ledger.
`completed_dispatch` means the coordinator finished accounting for the whole
schedule with unchanged inputs. It may contain blocked, failed, unavailable,
or zero cells; it does not mean every experiment succeeded or every estimand
is identified.

A statistics dossier must additionally provide `phase9_dispatch_evidence`,
an exact `{path, sha256}` binding to a successful, unchanged Phase 9
`summary.json`. Every `job_outcomes` descriptor must reference that run's same
complete final `jobs.json`, using the ledger hash recorded in the summary.
Standalone provisional outcome files and revoked summaries are refused.
The unchanged statistics binder still checks all its source, prediction,
threshold, missingness, and review requirements.

Statistical analysis necessarily uses an already finalized dispatch. The
coordinator does not fabricate a future analysis dossier or edit a signed
dossier after model outcomes arrive. A later analysis-only invocation retains
unselected tasks as unowned; it does not present those rows as a rerun or as
new measurements. The earlier complete dispatch remains the bound empirical
evidence. Automatic crash resume or cross-invocation result pooling is not
implemented. Existing output directories are refused, so a failed invocation
cannot silently overwrite its history.

## Cost, privacy, and testing scope

All new stage and dispatch roots carry a private-output `.gitignore` guard.
This prevents accidental ordinary Git addition; it is not a security boundary.
Inputs and downstream outputs may contain protected corpus text and human
judgments. Only body-free test reports belong in the public checkpoint.

Multiple loaded bundles, metadata, graphs, and decoder caches can coexist.
The coordinator does not claim constant memory or native-scale feasibility.
Per-owner loading/resource replay time and total coordinator wall time are
recorded separately from the frozen repair-service charges. Preparation,
oracle verification, and serialization costs are not relabeled as repair time.

Focused checks use metadata-only routing controls. They explicitly mock
unavailable acceptance, qualifications, and branch execution. They create no
corpus records, source identities, responses, or workers. An unmocked empty
dispatch preserves all planned jobs and executes none. Positive routing flags
in these controls are software evidence only. Historical attempts and their
source snapshots remain separate from the final source-bound report.
