# Executable development measurement producer

`resource_measurement.py` supplies the two observation routes consumed by
`resource_profiles.py`. It does not select a policy, fit a convex model, qualify
an optimizer, authenticate data, or authorize the primary study.

```bash
python empirical_execution/phase9/resource_measurement.py \
  --request /absolute/private/dossier/measurement_request.json \
  --out /absolute/private/dossier/new_measurement
```

Inputs must already exist. The request schema is
`ccu-resource-measurement-request-1`, with exactly these common fields:
`schema`, `kind`, `profile_signature`, `input_package`, `policy`, and
`evidence_role`. Paths resolve beneath the request's directory. The output
directory must be new. Use the complete signature produced by the canonical
profile compiler, not a manually retagged observation.

| Route | Additional request fields | Actual operation |
| --- | --- | --- |
| `graph` | `trajectory`, `source_kinds`, `fresh_graph` | Reconstruct/check the package graph, then call frozen `phase6.dispatch.structural_job` on the supplied complete request path. |
| `convex` | `candidate: {path, sha256}`, `selection: {unit, deleted_unit_ids, selected_record_ids}` | Reconstruct the graph and selected original rows, then verify an existing `x/y/w` candidate with the frozen exact Fraction or dyadic backend. |

The graph signature explicitly binds `request_arm`, `fresh_graph`, deletion
unit, full horizon, and checkpoint schedule. The pointwise convex signature
binds feature/output dimensions, backend, regularization and exact certificate
settings; it makes no claim that a deletion trajectory or optimization ran.
Candidate arrays must be aligned FP32 features, FP64 targets and FP64 weights.
The existing parameter tolerance is `1e-8`; work caps come from the supplied
validated policy. A completed verifier execution may fail the tolerance gate.
The receiver must check both separately.

The producer writes exact request bytes, prospective configuration, invocation,
process measurements, stdout/stderr, route outputs, and `measurement.json`.
It binds frozen dependency hashes, producer hashes, input bytes, Python
executable, environment, configuration and outputs. A failed child or a
post-execution integrity error is retained as a failed measurement. Existing
outputs are never overwritten.

Child processes enforce the supplied address-space and CPU limits, with parent
wall timeout enforcement. The process time includes imports, source/input
checks, graph reconstruction, the frozen route and worker output writes.
Parent intake, candidate copying and parent report writes are separately
described and excluded from that process measurement. Wait4 RSS and the maximum
non-atomic process-tree RSS sweep are separate indicators; neither the sweep
nor their maximum is a guarantee of total deployment peak memory. Mapped
dependency binaries are not completely pinned by this producer.

`actual_disjoint_development` outputs receive a local `.gitignore` that keeps
arrays, request payloads and measurements private by default. Publication or
repackaging still requires the project's data release policy. `software_only`
reports cannot qualify actual development, regardless of successful execution.
The consumer independently replays accepted development caches and the
certificate, checks producer bindings, and requires external authenticity
review. A well-formed receipt does not establish who collected the inputs or
where historical measurements ran.

The focused final software check is
`results/resource_measurement_checks_v2/checks.json`: **26 checks passed**.
It executes two tiny four-row algebraic routes and one required cap refusal,
plus metadata checks and one conspicuously marked non-observation mock for
post-execution failure preservation. These are software fixtures, not synthetic
empirical datasets, benchmarks, resource observations or semantic evidence.
