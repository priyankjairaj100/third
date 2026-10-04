# Bound statistical evidence

`statistics_evidence.py` validates inputs before the extension runs statistical analyses.
The validator reads saved evidence.
It does not run another worker benchmark.

The primary route derives its study jobs from the current recipe compiler.
The engineering route requires explicit `study_registry` and `study_jobs` inputs.
Their job digest must match.
Neither route permits a caller to rename saved results into another analysis population.

## Required dossier fields

| Field | Contents |
|---|---|
| `groups` | Group IDs mapped to acceptance and request-manifest file descriptors |
| `job_outcomes` | Every compiled relevance job, including matched-volume diagnostics |
| `registry` | The sealed analysis registry |
| `records` | One explicit status row for every derived analysis cell |
| `record_bindings` | Identity, job ID, outcome descriptor, method, and prediction descriptor |
| `crossed` | Complete trajectory/checkpoint arrays and their actual evaluation source partition |
| `system_bindings` | Each planned timing job, its outcome, and its P-I/B-E lifecycle receipts |
| `systems` | Paired differences averaged within each sorted trajectory |

Each descriptor contains a relative path and SHA256 value.
The path must remain within the supplied base directory.
The file must exist and match its hash.
These checks bind bytes.
They do not authenticate external testimony.

## Analysis identities and failures

The validator derives corpus, panel, arm, trajectory, and configuration from the compiled job.
The configuration ID is the digest of the complete configuration.
The validator derives checkpoints from the accepted frozen request manifest.
The sealed analysis registry must contain exactly those feasible cells.
Every cell must have an explicit status and evidence binding.

Ordinary R, S, U, and A paths enter the primary paired registry.
Matched-volume R paths remain separate diagnostics.
Every matched path still requires a saved outcome.
Empty request paths remain in the returned job ledger.
They do not create fictitious checkpoint releases.

Successful rows require accepted outputs and completed methods.
Failed rows require accepted inputs and a matching saved failure status.
They do not require a successful output declaration.
Their metrics remain empty.
A complete analysis therefore cannot hide failed trajectories by omitting them.

## Predictions and source partitions

The validator recomputes each metric from its saved prediction file.
The prediction path must belong to the recorded method output.
The saved label hash must match the accepted label hash.
The evaluation record order must match its accepted population.
Decision thresholds must match the accepted calibration lock.

The crossed analysis must use the same complete trajectory and checkpoint set.
Its prediction arrays must match the saved arrays.
Its source IDs and source kinds must match the accepted source replay.
Its decision thresholds must match the accepted calibration lock.
A failed or missing planned trajectory makes the unconditional crossed estimate unavailable.
The extension records that status instead of running a complete-case bootstrap.

## Systems contrasts

Each pair uses P-I and B-E from the same saved timing job.
Each receipt must bind its named successful service report.
The request bytes, input hashes, policy, unit, solver, panel, and persistence must match.
The primary contrast uses cold-process execution and final-only persistence.
The saved head audit must contain every accepted checkpoint.
Every head audit must pass.
A revoked primary outcome cannot supply an accepted pair.

Every planned timing repetition must remain in the input.
The validator averages paired differences within each trajectory.
The bootstrap therefore resamples trajectories, not repeated timing measurements.
An incomplete pair makes that trajectory contrast unavailable.
The unconditional systems estimate then remains unavailable.

## Checks

Run:

```sh
OPENBLAS_NUM_THREADS=1 python empirical_execution/phase6/check_statistics_evidence.py \
  --out empirical_execution/phase6/results/new_statistics_evidence_checks.json
```

The check uses unchanged saved predictions from existing Civil text.
Its metadata wrappers are explicit software fixtures.
They do not create corpus provenance or primary evidence.
The saved result is `results/statistics_evidence_checks.json`.
It checks renamed identities, missing cells, source overrides, threshold overrides, label hashes, and retained failures.
No worker benchmark runs during these checks.

## Final qualification

The current result is `results/statistics_evidence_checks_v2.json`.
It passes 22 focused checks.
The earlier 15-check result remains historical.

The checks also exercise valid paired receipts and averaging across timing repetitions.
They reject wrong methods, reused receipts, missing repetitions, changed requests, and incorrect averages.
A failed pair preserves an unavailable trajectory contrast.
The checks reuse saved service bytes under explicit temporary metadata fixtures.
They do not pool historical measurements into an empirical systems result.

Primary analyses require the fixed metric set.
Successful rows must include all applicable computed metrics.
This includes signed effects, absolute effects, prediction differences, exceedances, admissions, and admission rates.
Zero denominators retain explicit undefined values.
A caller cannot remove an unfavorable metric from primary analysis.

The checker reads saved worker and extension files.
If those directories are archived, restore them before the check:

```sh
python3 tools/phase6_result_archive.py restore worker_engineering_final
python3 tools/phase6_result_archive.py restore worker_engineering_release_v2
python3 tools/phase6_result_archive.py restore extensions_candidate1
```

`build_fixture` exposes the bound engineering dossier for the extension integration check.
That dossier uses unknown source singletons.
It cannot support the genuine-source bootstrap.
Its metadata wrappers remain software fixtures, even when their saved predictions come from natural text.
