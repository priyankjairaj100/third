# Versioned study execution

`dispatch.py` joins the study branches. It preserves each planned job and method outcome.
It does not create corpus records, semantic vectors, source identities, or human judgments.

The dispatcher has two explicit modes.

| Mode | Inputs | Result scope |
|---|---|---|
| `natural_text_engineering` | Existing natural input, exact file hash, labeled engineering parameters | Software execution and numerical checks |
| `accepted_primary_inputs` | Exact recipe registry, replayed assets, reconstructed statistical gates, bound external review | Accepted inputs and checked method outputs |

The second mode is an executable acceptance route.
It does not contain an unconditional refusal.
No available bundle has passed that route.
Authentic semantic assets and genuine human judgments remain absent.

## Input contract

The Python API is:

```python
run_dispatch(registry, jobs, bundles, output_directory,
             mode="natural_text_engineering",
             design=None, base_dir=None, policy=None)
```

`bundles` maps each registered request family to its input bundle.
The bundle uses the frozen `phase5.task_program` array contract.
It contains these items:

- Ordered training and evaluation IDs.
- Original FP64 targets for labeled tasks.
- Complete source ownership and explicit native or unknown source kinds.
- FP32 curator and learner caches with their ordered IDs.
- Calibration records, features, response files, threshold locks, and quality reports.
- Calibration-only regularization locks and tag decisions.
- The full request design and its frozen manifest.
- Fixed projection matrices when the registered group requires them.

Unlabeled structure bundles omit learner targets and evaluation predictions.
They retain the curator and request contracts.
WCEP uses the inherited News calibration dossier.
The source acceptance module checks its matching encoder lineage.

Engineering bundles also provide `evidence_role`, `natural_input_path`, and `natural_input_sha256`.
The role must equal `natural_text_engineering`.
Engineering parameters cannot pass primary acceptance.

Primary bundles also provide `source_acceptance`, `development`, `execution_policy`, and `external_evidence_review`.
See `acceptance.py` for the original archive and full cache replay contract.
The dispatcher reconstructs both calibration sample manifests before it reads their response decisions.
It then reconstructs the quality report and each task lock.
It reconstructs complete request orders, checkpoints, matched controls, and observations.

## External trust boundary

The external review uses schema `ccu-external-evidence-review-1`.
It contains a named reviewer, review time, exact bundle digest, and exact replay digest.
Its six evidence entries require local file hashes, accepted decisions, and written reasons.

Those entries cover archive identity, source meaning, permissions, encoder origin, human collection, and development choices.
The review is a trusted input.
Software does not prove the reviewer's testimony.
An evidence file alone cannot establish authenticity.
No automatic reviewer record is created.

The development dossier contains actual service reports and a local resource snapshot.
It also contains an archived worker input package.
Its record IDs must match accepted original records.
Its semantic arrays must match exact rows from the replayed cache.
Its targets must match the original labels.
Each service report must bind these same input files.
Unlabeled corpora supply a separate labeled development bundle and its registered group.
That bundle receives its own source and cache replay.
WCEP development also excludes the inherited News study records and sources.
The dispatcher reconstructs the resource formula from their recorded stage times.
It also reconstructs the prospective precision analysis.
The common worker policy must match the derived limits.
The external review binds the exact development replay digest.
Historical collection and precision-effect derivation remain part of that review.

## Executed branches

Structure jobs use an independently scored exhaustive graph.
The independent scorer uses ordered FP64 sums and bounded score blocks.
Most checkpoints reuse this graph.
Precommitted fresh-graph audits independently rescore retained records.
They compare the resulting selection with the production graph.
Every admission count retains zero values.
Structural jobs also record eligibility, incidence rank, key counts, and byte forecasts.
Each forecast uses that path's actual initial horizon.
Weighted and adversarial paths do not inherit a uniform workload expectation.

Ridge jobs call the Phase 6 isolated service.
Each method receives the same graph, task arrays, request path, and resource policy.
A fixed seed changes method order before any method outcome.
The dispatcher saves that order.
All eight registered methods have explicit FP32 variants.
Those variants use FP32 state and retain the fixed FP64 curator target.
They form an accuracy frontier.
They do not inherit FP64 agreement claims.

The evaluator checks every saved ridge head with an independent direct solve.
It uses the smaller primal or dual system.
The B-F check uses retained original selection.
Other methods use retained complete curation.
Task predictions and original targets remain available for later analysis.
The evaluator does not count its own checking work as service time.

The first sixteen R and S method paths also require a complete state audit.
The audit runs a separate service with persistence at every release.
Direct replay snapshots must match that audit service's recorded hashes.
The audit compares every abstract state coordinate with a fresh retained construction.
It also compares all heads and final state with the measured service.
This verifies the separately charged audit service.
It does not expose private intermediate states from the measured service.
Floating coordinate comparisons retain their numerical qualifications.
They do not establish bitwise history independence.
The canonical P-R comparison removes components with no stored coordinates and an exact zero coefficient.
It retains their raw metadata differences in the audit record.
Audit failure prevents primary acceptance for the flagged job.
Every other FP64 B-E path also requires the linked membership and moment audit at each release.
Every FP64 P-I, P-S, and P-R path requires the same complete Gram, cross-moment, and count comparison.
This additional check does not claim FP32 state equality.

Convex jobs run all three methods on the same path.
They include matched-volume record controls.
The payload methods own their eligible rows.
Every release requires the exact optimization certificate.
The prospective policy selects the Fraction or exact dyadic verifier.
The integration check explicitly selects the exact dyadic verifier.
Each checkpoint also receives an independent graph check.
A failed method stays blocked on later checkpoints.
Separate logistic decisions apply to the original twenty tag outputs.
Ridge decision thresholds never apply to logistic probabilities.
These convex runs use a shared process.
They do not establish isolated systems speed.

Refit jobs use their own original fitted curator and request graph.
The dispatcher retains five same-seed fits and three alternative-seed fits.
It checks branch heads independently.
Primary refit requires the pinned official backend.
The NumPy backend remains a separate engineering reference.
Legacy engine metadata remains unchanged.
The dispatcher writes its own acceptance and verification records.

ANN jobs report edge recovery and checkpoint selection differences.
They retain the exhaustive graph as their target.
Extension recipes use `extensions.run_extension`.
The extension registry supplies the exact request laws and analysis inputs.

## Output and failure policy

The output directory must not exist.
The dispatcher writes the input lock before execution.
It writes one ledger outcome for every planned job.
Missing inputs, empty arms, method failures, and unsupported variants remain explicit.
An empty input dictionary preserves the complete primary schedule as blocked rows.

Service snapshots use temporary directories.
The final result retains exact reports, heads, prediction arrays, snapshot hashes, and byte counts.
Temporary snapshot removal is archive cleanup.
It is not a physical erasure claim.

Primary results lose acceptance if bound code or inputs change during execution.
The final summary records both initial and final source hashes.
Runtime measurements do not imply exact physical memory traffic.

## Local verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 empirical_execution/phase6/check_dispatch.py \
  empirical_execution/phase6/results/dispatch_new_run
```

The check uses actual Civil comments with lexical features.
It does not create a synthetic empirical corpus.
It exercises A–G branches, all methods, numerical variants, unavailable sources, and missing semantic inputs.
It also checks the complete blocked primary registry.
These observations are software evidence from reused text.
They are not a primary semantic study.
