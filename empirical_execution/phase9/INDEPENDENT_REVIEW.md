# Phase 9 independent software review

The final software sources pass this scoped source and evidence review. The
primary semantic study remains unstarted. This review does not approve a real
corpus, semantic cache, human response, native resource observation, or primary
execution. The separate theory and novelty reviews remain their own evidence.

The reviewer inspected the new scheduler, resource receiver, measurement
producer, and graph adapter against the frozen interfaces, sent concrete
findings to their owners, and checked the final source/report bindings. The
reviewer did not rerun benchmarks or the component checkers. The check counts
below are the owners' retained focused checks, not independent experimental
replications or proof of complete native activation.

## Authoritative evidence

All paths below are relative to `empirical_execution/phase9`.

| Component | Final report | Passed checks | Actual scope |
| --- | --- | ---: | --- |
| Study assembly | `results/study_assembly_final.json` | 69 | Canonical ownership and routing controls; positive acceptance and numerical branches explicitly mocked; unmocked missing-input refusal; zero workers |
| Resource profiles | `results/resource_checks_final/checks.json` | 86 | Small algebra and metadata contracts with mocked upstream acceptance; producer-receiver consistency, mixed/projected package replay, exact certificate arithmetic, and refusal controls; zero actual native observations |
| Measurement producer | `results/resource_measurement_checks_v2/checks.json` | 26 | Two positive four-row algebraic subprocess routes, a required cap refusal, and an explicit non-observation integrity mock; no empirical benchmark |
| Human graph adapter | `results/graph_dossier_final/checks.json` | 68 | Existing natural Civil40 lexical graph, serialization and identity parity, and actual frozen blank admission/context routes; no source-gate mocks or human judgments |

The graph routes remain `awaiting_real_human_responses`, with 93 blank admission
assignments and 24 blank context assignments. Neither route accepts a primary
output. The existing engineering ownership remains unknown-singleton ownership;
the adapter does not make it genuine source metadata.

The actual empty-input resource invocation is retained at
`results/resource_missing_native/qualification.json`. It is blocked at missing
development inputs, accepts zero observations, and exports no policy. It tests
this refusal boundary, not every future asset combination.

`results/independent_review.json` binds the exact production files, checkers,
component documents, reports, referenced source maps, retained measurement
artifacts, and final source snapshots. It also records the read-only comparison
showing no changes in frozen Phases 3–8 against the current published baseline.
The baseline commit and exact file counts are in that JSON. No test count is
used as a substitute for the missing scientific evidence.

## Findings resolved before freeze

1. **One canonical ledger with stable input roots.** The coordinator compiles
   exclusive ownership from the unchanged 43-group, 21,332-job registry. Both
   ownership compilers agree for all 20 task families. Caller scope selects
   whole registered groups and extension recipes, not individual favorable
   jobs. Human/statistics ownership remains separate. Task dependencies retain
   their original roots. A reviewed `phase9_stable_root` prevents copied files
   from silently reinterpreting relative paths under a new root.
2. **Review ordering and input bindings.** The resource candidate includes its
   scope and stable root before qualification. The resource input projection
   omits only the standard derived attachments and final testimony while
   retaining the immutable development replay binding. The complete resource
   receipt and evidence are attached before final external review. Execution
   requires the referenced evidence to equal those reviewed attachments and
   cover every selected job. Captured plan bytes are also the bytes hashed.
3. **Complete status and failure accounting.** Each canonical job has one
   result owner and at most one execution attempt. Blocked, unavailable,
   failed, unowned, and interrupted-coordinator outcomes remain represented.
   Final source/input drift revokes central acceptance. Source bindings include
   the measurement producer as a transitive executable dependency. Statistics
   require the successful unchanged final summary and exact complete ledger;
   provisional per-job artifacts cannot stand in for that ledger.
4. **Configuration-specific resource evidence.** Profiles derive through actual
   frozen resolution/request paths and bind both encoders, revisions,
   projections, targets, calibration settings, precision, solver, process,
   persistence, and complete actual schedules. Original accepted cache paths
   keep their existing semantics; observation artifacts stay within the owner
   root. Required state audits bind their exact auxiliary worker configuration,
   inputs, policy, sources, releases, and complete coordinate comparisons.
5. **Separate convex and utility contracts.** Pointwise dense verifier evidence
   covers actual selected-count maxima across the selected canonical jobs.
   It does not qualify an optimizer or trajectory. The utility branch's own
   frozen cap/forecast lock is recomputed from accepted observations; a larger
   global cap cannot silently authorize it. Its rate formula remains a forecast.
6. **Executable measurement intake.** Graph and convex evidence binds the
   actual producer request, configuration, invocation, Python executable,
   process record, package, source map, and output artifacts. Graph arm/fresh
   mode and actual source ownership must match. The receiver replays retained
   graph membership or the exact certificate and refuses software-only roles.
   Changed post-run inputs preserve a failure receipt. Actual-development
   outputs carry a private-output guard. Byte consistency still requires
   accountable external review of historical execution and source authenticity.
7. **Human graph serialization.** The JSON-compatible immutable adapter retains
   exactly the signed record/source/priority/CSR/threshold values while
   providing the frozen graph interface. Hydration preserves the frozen
   fingerprint. Human execution still rebuilds and compares accepted graph,
   records, features, and request frames. Scalar metadata access avoids a graph
   scan per frame item; graph-array/method access retains its disclosed
   allocation and validation costs. This is trusted Python input handling, not
   a hostile-object sandbox.

No concrete unresolved implementation defect was found in this reviewed scope
after the final corrections. That statement is bounded by source inspection
and the retained focused controls. It is not a claim of universal correctness,
complete adversarial security, or successful real-data activation.

## Limits and unfinished work

The new resource adapter applies to its stated worker, graph, and pointwise
verifier configurations. Full-refit, ANN, negative controls, and other routes
outside the added audit/verifier caps retain their frozen branch gates and
common resource limits. The adapter makes no additional feasibility claim for
them. Human/statistical owners retain their separate real-evidence requirements.

Observed count extents do not bound another graph's runtime or peak memory.
Future symbolic key counts, incidence fan-in, graph density, conditioning,
decoder work, serialization, and allocator costs remain relevant. Reported RSS
indicators are not simultaneous-peak or address-space certificates. Parent
intake/copying and repeated qualification work have their own disclosed costs.
No native end-to-end preparation-memory qualification has been performed.

The scheduler composes frozen executors; its metadata routing checks are not
positive native checks of all registered branches. A completed dispatch means
complete accounting, and can contain only blocked jobs. Cross-invocation
pooling and automatic crash recovery are not provided. Original archives,
pinned model/runtime assets, semantic caches, genuine calibration/relevance
judgments, source-disjoint development observations, full primary execution,
and the publication figures/tables remain unfinished.

Historical attempts remain at their original result paths, with their exact
source snapshots where recorded. In particular the producer v1 and provisional
assembly/resource reports have not been relabeled as final-source evidence.

## Reproduction

From the repository root, this read-only command checks every exact artifact
binding in the final review without running experiments:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, json
p = Path('empirical_execution/phase9/results/independent_review.json')
r = json.loads(p.read_text())
bad = [name for name, expected in r['artifact_sha256'].items()
       if not Path(name).is_file()
       or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
assert not bad, bad
assert r['status'] == 'passed_scoped_source_and_evidence_review'
print('Verified', len(r['artifact_sha256']), 'bound artifacts')
PY
```

For an intentional software reproduction in an isolated checkout, use fresh
output paths. These commands refuse collisions and do not replace the retained
reports. The producer command starts only its labeled tiny software processes;
the graph command reconstructs blank natural-fixture forms.

```bash
python3 empirical_execution/phase9/check_study_assembly.py \
  empirical_execution/phase9/results/repro_study_assembly.json
python3 empirical_execution/phase9/check_resource_profiles.py \
  --out empirical_execution/phase9/results/repro_resource_profiles
python3 empirical_execution/phase9/check_resource_measurement.py \
  --out empirical_execution/phase9/results/repro_resource_measurement
python3 empirical_execution/phase9/check_graph_dossier.py \
  --out empirical_execution/phase9/results/repro_graph_dossier
```

The retained source snapshots and scripts specify the controls. Re-running
them does not authenticate a real source, complete a human assignment, or
qualify native hardware.
