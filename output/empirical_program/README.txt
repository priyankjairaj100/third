EMPIRICAL PROGRAM — ACL 2027 COUNTERFACTUAL CURATION
Version 1.0 | 3 October 2026

This is a reviewed proposed design, not measured results, a registered study,
or an implemented training/repair runner. No synthetic dataset is included.
The PDF is the narrative authority. The JSON is the operational study manifest.
The original theory specification remains separate and unchanged.

FILES
  study_design.json: decisions, request allocations, contracts and lock fields.
  claim_ledger.csv: what each claim needs and what unfavorable findings permit.
  run_registry_schema.csv: header-only schema; no fabricated example runs.
  annotation_schema.csv: header-only annotation schema; no fabricated judgments.
  memory_preflight.py: analytic byte estimates from measured structural counts.
  validate_design.py: draft checks, release allocation counts, and lock validation.
  counterfactual_curation_empirical_protocol.tex: complete editable PDF source.
  SHA256SUMS.txt: integrity hashes for package contents (excluding this hash file).

START
  python3 validate_design.py study_design.json
  python3 validate_design.py study_design.json --lock
The first checks internal design constraints and prints allocations. The second
MUST fail until acquisition/development outputs populate all lock fields. Fill
them with substantive resolved values, then perform a human lock review. A passed
validator checks manifest completeness/invariants, not scientific truth or that
an experiment was actually preregistered.

ANALYTIC PREFLIGHT (replace values with measured natural-graph counts)
  python3 memory_preflight.py --dimension 768 --outputs 20 \
      --keys ACTUAL_KEYS --eligible ACTUAL_ELIGIBLE --rank ACTUAL_RANK \
      --metadata-bytes ACTUAL_METADATA --workspace-bytes ACTUAL_WORKSPACE
Integer placeholders above are deliberately not executable data. Supply actual
counts from a natural corpus. The calculator includes packed dense FP64 statistics
and an FP32 eligible-row comparison; metadata/workspace inputs must be measured.
It is not an algorithm implementation or runtime predictor. Missing metadata
cannot establish feasibility or a physical-memory claim.

CORE EXECUTION ORDER
1. Acquire originals; pin checksums/schema/record identity; define the source
   partition and exact split/guard manifests. Keep all unfavorable outcomes.
2. Calibrate only on source-disjoint development data; validate semantic quality.
   Lock canonical features, scorer, threshold, lambda, labels and request seeds.
3. Build exhaustive natural graphs and forecast native-dimensional state bytes.
   Keep forecast-infeasible and OOM cells, separately labeled, in results.
4. Implement independent oracle and optimized eligible-payload B-E first; verify
   natural-data correctness, numerical bounds and worker access isolation.
5. Run locked confirmation and log every failure. No filtering after outcomes.
6. Run independent annotation, full-refit and convex panels; apply claim ledger.

KEY BOUNDARIES
  * Primary global ordered suppression is not complete standard SemDeDup.
  * Core release is head-only. An isolated oracle computes selected IDs; the
    proposed statistic summary is not assumed to recover IDs or document text.
  * Source service supports all partition units; random S samples genuine native
    sources only. Identical eligibility universe/horizon applies to every method.
  * Civil article groups are not authors. Stack owner IDs are account proxies.
  * Unknown sources are separate record singletons, never one pooled source.
  * Finite-horizon state is rebuilt only explicitly and with full cost/access.
  * Native d=768 results cannot disappear behind projected d=128 results.
  * Residual diagnostics are not rigorous interval certificates without verified
    accumulation and residual-evaluation error allowances.
  * 256 independent requests with zero events still allow a one-sided95% event
    probability upper bound about1.16%;64 allow about4.57%.
  * Hyperparameters/encoder pretraining remain outside deletion-sensitive D.
  * Raw text redistribution follows original source conditions. Release IDs,
    hashes and acquisition code where redistribution rights are not established.

All engineering changes after protocol lock require versioned amendments and
reruns of affected comparisons. This design cannot eliminate legitimate debugging
or guarantee reviewer acceptance; it prevents target drift and selective evidence.
