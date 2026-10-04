# Prospective native audit resource policy

This module selects operational audit caps from accepted, disjoint development observations.
It does not create measurements, run benchmarks, or activate the primary study.
Current genuine native observations remain absent.
The saved real-input check therefore remains blocked and exports no policy.

## Implemented scope

Version 1 covers exact canonical `C_full_methods` native primary-10,000 groups.
It supports their FP64 E5 curator and E5 learner with the same accepted revision.
The solver, cold-process setting, batch size, and compaction setting must match the frozen core configuration.
It requires all eight full-state method profiles for each available deletion unit.
It also requires B-E, P-I, P-S, and P-R light profiles.
Record and genuine-source units require separate observations.
Unavailable genuine-source profiles remain explicitly unavailable.
They never receive fabricated singleton substitutes.

Caller-selected subsets cannot establish complete group coverage.
Altered groups cannot borrow this qualification through a matching corpus or panel name.
Other groups and configurations are refused.

This selector handles state audits only.
It leaves the convex verification cap at 2,000,000 coordinates.
It neither selects a convex backend nor qualifies existing nondefault convex settings.
The separate Phase 6 dense-verifier selector remains available.
A source-bound convex qualification adapter and genuine matching observations remain unfinished.
Twenty-output native Stack qualification requires actual twenty-output development evidence.

Additional implementations remain necessary for mixed encoders, projections, larger panels, alternative solvers, and process-panel variants.
Those implementations also require matching real measurements.
The current module does not establish a universal study resource policy.

## Acceptance and evidence order

1. Build an actual source/cache candidate using the Phase 8 dossier workflow.
2. Supply the complete disjoint-development dossier required by frozen `development_acceptance`.
3. Supply its externally reviewed preliminary development replay binding.
4. Bind a prospective observation plan and every actual audit observation, including failures.
5. Run this qualifier to derive the candidate common policy.
6. Attach the full qualification receipt through the dossier assembler.
7. Obtain the complete external review of that concrete candidate.
8. Recompute resource qualification and every frozen acceptance gate before dispatch.

The preliminary review supplies `development_source_replay_sha256`.
The module never invents this value or reviewer testimony.
It first derives only the four common resource fields.
It then replays actual source/cache and disjoint-development acceptance with those fields.
It repeats frozen development acceptance after selecting audit caps.
The final complete external review happens afterward, avoiding a circular dependency.

Actual acceptance checks original labels, semantic cache rows, development ownership, disjointness, and saved worker input hashes.
It also recomputes the original development precision analysis.
The complete observation plan must bind the replay, resource snapshot, resource lock, and target profiles.
Every planned observation ID must appear exactly once and in order.
A failed observation blocks policy export for that plan.
Preserve failed plans; any revised trial plan requires a separate prospective decision.

Initial development trial limits must be declared separately before those trials.
This module does not invent their successful outcomes or authorize exploratory compute.
All actual work remains subject to the user's existing local-compute authorization.

## Selection algorithm

The four common worker limits use the unchanged Phase 6 formula.
Memory equals half the minimum observed available-memory limit.
Wall time and CPU time equal `ceil(4 * maximum observed development stage seconds)`.
Workers use one thread.
No new fitted runtime coefficient or extrapolation multiplier is introduced.

For each target profile, one actual observation must jointly cover its declared workload.
Encoder identity, revision, dimensions, outputs, method, unit, scope, solver, and execution settings must match.
Observed original rows must cover target rows.
Observed requests must execute the complete distinct-unit horizon and checkpoint schedule.
Configured capacity alone cannot substitute for completed deletions.
Separate maxima from unrelated observations cannot manufacture a larger joint workload.

The shared pair cap is the larger of the unchanged default and the maximum declared original-graph demand.
Every qualifying observation must cover that common cap's count extent.
For `N` original rows and curator dimension `c`, the demand is `N(N-1)c/2`.
This is the original-shape cap proxy, not a measured operation count.
The audit also rescans each retained checkpoint, and tiled scoring evaluates extra masked entries.

Each full summary profile supplies its maximum observed fresh-symbol coordinate count.
The common coefficient cap is the minimum of these per-profile observed maxima.
That cap must reach the unchanged default; the module never lowers the default.
Only observations satisfying the other joint coverage checks enter this selection.
For `M` designated symbolic keys, the count is `M[d(d+1)/2 + dq + 1]`.
The count uses fresh symbolic keys, not stored rank-basis size or numeric nonzero support.

Future graph-dependent `M` remains unknown in every qualification receipt.
The receipt separately reports the deterministic bound `M <= 2N`.
It reports whether the selected cap covers that full bound.
Otherwise, future overflow remains an explicit state-audit failure, with no waiver.
No observed key count is relabeled as a future primary value.

## What the observations establish

Observed worker wall time, CPU time, and a reported RSS indicator must fit the candidate limits.
The indicator adds auditor lifetime peak RSS to the maximum reported worker RSS indicator.
Worker indicators use the larger of wait4 RSS and sampled process-tree RSS.
This sum is descriptive and may overcount shared or inherited memory.
Sampling can also miss short-lived process peaks.
It is not a rigorous peak-memory bound.
RSS also differs from the address-space limit enforced by workers.

The report separately retains total audit time, auxiliary worker lifecycle, checkpoint times, and final snapshot bytes.
It retains input bytes, graph edges, maximum blocker degree, and feature density.
Missing setup, rescore, serialization, and simultaneous-memory breakdowns remain unknown.
The resource limits do not independently impose a total auditor wall-time limit.

Equal count extents do not bound future runtime or memory.
Graph density, source overlap, symbolic incidence, dictionary operations, and serialization can differ.
Conditioning and decoder iterations can also differ.
Hardware, runtime binaries, allocator behavior, and host load require explicit evidence and review.
The receipt makes no native feasibility guarantee.
It supports a prospective operational setting with retained failures.

## Source and artifact bindings

Audit reports must bind their actual auxiliary and reference worker reports.
Reference services must belong to the accepted development dossier.
Input hashes must match its accepted semantic arrays.
Audit methods, policies, scopes, requests, and actual checkpoint horizons must agree.
The complete current Python dependency map must match both worker reports and their configuration.
The qualifier checks its source bindings again before emitting a policy.
The original tolerances remain unchanged or stricter.
Neither the numerical gate nor human/provenance acceptance is weakened.

The receipt fingerprint excludes its later qualification and observation attachments.
It also excludes final reviewer testimony and the derived execution policy.
These items have separate bindings.
The immutable preliminary development replay digest remains in the fingerprint.
Replacing preliminary testimony with final review can therefore reproduce the receipt.
Changing actual inputs or that replay digest cannot.

These checks establish consistency of supplied bytes and replays.
They cannot establish historical measurement authenticity or review independence by themselves.
The full external review remains mandatory.

## API and CLI

```python
from phase8.resource_policy import qualify_resource_policy, verify_resource_receipt

receipt = qualify_resource_policy(
    group, bundle, development_audits, target_profiles, base_dir=candidate_root)

verified = verify_resource_receipt(
    receipt, group, bundle, development_audits, target_profiles,
    base_dir=candidate_root)
```

Verification recomputes real acceptance and the complete receipt.
A matching seal alone does not suffice.
Blocked or changed receipts raise an error during verification.
Successful verification returns the identical receipt.

```bash
python3 empirical_execution/phase8/resource_policy.py \
  --request LOCAL_CANDIDATE/resource_request.json \
  --out NEW_RESOURCE_RESULT_DIRECTORY
```

The request contains exactly `group`, `bundle`, `development_audits`, and `target_profiles`.
`bundle` accepts the existing bound JSON/NPY descriptor format.
Paths resolve within the stable candidate root.
Observation descriptors contain only `path` and `sha256`.
Each entry identifies `observation_id`, `audit_report`, `audit_service_report`, `reference_service_report`, and `audit_configuration`.
The observation manifest also contains schema `ccu-native-audit-observations-1` and a bound `plan` descriptor.
The plan contains `observation_ids`, `development_source_replay_sha256`, `target_profiles_sha256`, `resource_snapshot_sha256`, and `development_resource_lock_sha256`.
Its evidence role is `actual_disjoint_native_development`.
A role string alone never passes actual acceptance.

Every target profile supplies `profile_id`, `method`, `unit`, `full_state`, `records`, and `horizon`.
It supplies `checkpoints` and the complete `checkpoint_schedule` ending at the horizon.
It also supplies both dimensions, output count, encoder name, encoder ID, and immutable revision.
Finally, it supplies solver, panel, lambda, threshold, batch size, and compaction fraction.
The exact field names are enforced by `validate_profiles`.

The output contains `qualification.json` and a captured-input receipt in `invocation.json`.
Only a qualified result also contains `execution_policy.json`.
The latter uses the unchanged Phase 7 explicit-policy CLI schema.
That schema compatibility does not bypass its dossier or acceptance gates.
Existing output directories are refused.
Read `status`; successful report generation can still describe a blocked qualification.

## Focused verification

```bash
python3 empirical_execution/phase8/check_resource_policy.py \
  --out empirical_execution/phase8/results/resource_checks_new.json
```

Positive software controls use hypothetical measurement metadata and explicitly mocked upstream acceptance.
Empty arrays communicate shape metadata only.
No synthetic empirical dataset, native encoder output, or real measurement is created.
The real observation reader, selector, receipt verifier, and assembler attach/finalize chain execute.
Unmocked source acceptance rejects the same fixture.
The tests preserve unknown future keys and check missing coverage, source changes, altered policies, and receipt recursion.

The authoritative report is `results/resource_checks_final.json`.
The actual missing-input refusal is `results/resource_missing_native/qualification.json`.
No genuine native policy has been exported.
Provisional attempts remain preserved separately.
The exact second-attempt source bytes are recorded in `results/resource_attempt2_sources.json`.
Exact first-attempt intermediate source bytes were not separately retained.
