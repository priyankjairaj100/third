# Native resource preflight

This command inspects the frozen audit and verifier caps before execution.
It uses integer arithmetic and reads source files.
It does not read corpora, allocate feature arrays, execute workers, or select budgets.
Phase 6 code and defaults remain unchanged.

Reproduce the planned native report with a new output filename:

```bash
python3 empirical_execution/phase7/resource_preflight.py \
  --records 10000 --dimension 768 --curator-dimension 768 --outputs 1 20 \
  --out empirical_execution/phase7/results/resource_native_new.json
```

The saved report is `results/resource_native_10000_d768.json`.
Omitting `--out` prints the report without writing anything.
Existing output files are refused.
Successful report generation exits zero even when a cap would refuse work.
Read each gate's status; a generated report never approves execution.

## Exact count rules

Let `N` denote original raw records and `R` denote retained records at one checkpoint.
Let `c` denote curator dimension, `d` learner dimension, and `q` output count.
Let `S` denote the actual selected count at that checkpoint.
Let `M` denote `len(fresh_symbols)` at the remaining deletion horizon.

| Frozen check | Exact demand | Default cap |
| --- | --- | ---: |
| Original graph audit | `N(N-1)c/2` | 100,000,000 |
| Each retained graph audit | `R(R-1)c/2` | 100,000,000 |
| Full coefficient construction | `M[d(d+1)/2 + dq + 1]` | 4,000,000 |
| Joint convex verification | `Sdq` | 2,000,000 |

The original graph gate runs before the separate audit service.
Every audited checkpoint rebuilds its retained graph and applies the same cap separately.
Light FP64 audits cover B-E, P-I, P-S, and P-R releases.
Flagged full-state calls cover all eight methods.
Only full-state P-I, P-S, and P-R calls construct the complete coefficient arrays.
Convex jobs use their separate verification gate.
A convex-only job need not execute a state audit.
The aggregate status inventories these gates; it does not admit or refuse individual jobs.

All predicates reject only when demand exceeds the cap.
Equality therefore passes the count check.
Both exact convex backends have the same coordinate cap.
Changing the backend does not remove that cap.

The coefficient count uses designated symbolic keys, including symbolic forms whose numerical coordinates might cancel.
It does not use the deletion horizon, stored rank basis size, or actual/fresh comparison union.
Each retained record contributes at most two monomials.
Thus `2R` is a conservative symbolic-key upper bound, not an observed count.
Stored structural zero nodes can enlarge the comparison union without enlarging this particular cap demand.

## Planned native shapes

These values are arithmetic results, not measurements.
Both curator and learner dimensions equal 768 in the saved report.

| Quantity | q = 1 | q = 20 |
| --- | ---: | ---: |
| Original graph demand at N = 10,000 | 38,396,160,000 | 38,396,160,000 |
| Packed coordinates per symbolic key | 296,065 | 310,657 |
| Symbolic keys allowed by the default coefficient cap | 13 | 12 |
| Selected rows allowed by the default verifier cap | 2,604 | 130 |
| All-10,000-selected verifier upper bound | 7,680,000 | 153,600,000 |

The default graph cap admits at most 510 original records at curator dimension 768.
The native 10,000-record audit therefore has a deterministic count refusal under that default.
This conclusion also applies to light state audits.

Actual selected counts and symbolic-key counts remain unknown without the graph and checkpoint horizon.
The saved report preserves those demands as null.
It does not claim that all 10,000 records are selected.
It does not claim that the coefficient or verifier gate necessarily fails for the unknown counts.

## Supplied counts and policy

Supply actual checkpoint counts with `--retained-records`, `--selected-records`, and `--symbolic-keys`.
Counts must describe the same checkpoint.
`--retained-records` defaults to the original count without implying an observed deletion path.
Omitted selection and key counts remain unknown, except for the provably empty panel.
`--curator-dimension` defaults explicitly to the learner dimension.
Specify it when the encoders have different dimensions.
`--audit-scope light` excludes the coefficient cap but retains the graph gate.

Add `--policy LOCAL_INPUT/execution_policy.json` to compare an existing prospective policy.
The command preserves a separate comparison against all frozen defaults.
Unspecified optional caps retain their frozen values.
The policy uses the same complete field shape as the Phase 7 dispatcher CLI.
It requires memory, CPU, wall-time, and thread limits.
Malformed, duplicate-key, nonfinite, boolean-count, negative-cap, and tolerance-weakening policies are refused.
The parsed policy and its hash come from the same captured bytes.
This validation does not authenticate, approve, or issue a development lock.

Larger verifier caps require genuine, suitable development measurements under the existing selector.
Those observations must match learner dimension and output count and satisfy its density requirements.
Their source hashes, input sizes, timings, and memory observations remain bound.
Genuine twenty-output qualification remains absent from the current project evidence.
Audit caps also require prospective review and a common resource policy before confirmation.

## Work and memory limits

These units describe count predicates, not exact total operations or physical traffic.
Tiled graph scoring also evaluates diagonal and masked entries.
Repeated services and checkpoints repeat graph work; no trajectory total is inferred here.
Coefficient construction can sum many records into each coordinate.
It materializes long-double packed arrays and temporary products.
The coefficient cap does not bound every stored key, comparison coordinate, or allocation.

Runtime and peak RSS remain null.
The report provides no timeout, OOM, native feasibility, or constant-memory claim.
Passing count caps cannot qualify a memory or wall-time envelope.
Larger caps never weaken numerical release tolerances, state checks, sigmoid limits, or authentic-input requirements.

## Source binding and focused checks

The command checks supported guard expressions through Python's abstract syntax tree.
It extracts actual defaults and binds every inspected file by SHA256.
It refuses changed guard formulas rather than silently applying stale arithmetic.
The bindings include both verifier backends, the dispatcher, the selector, and the CLI policy source.
This checks specific expressions, not the complete semantics of every bound source file.
No frozen module is imported or executed by the preflight.

Run only the bounded software checks:

```bash
python3 empirical_execution/phase7/check_resource_preflight.py \
  --out empirical_execution/phase7/results/resource_checks_new.json
```

The final check report is `results/resource_checks_final.json`.
It covers exact-cap boundaries, combinatorial enumeration, unknown counts, policy parsing, and source drift.
It also checks separate curator dimensions and initial-graph refusal after a small retained checkpoint.
Temporary shape placeholders test arithmetic predicates only.
They are not synthetic empirical datasets.
No corpus, worker, optimizer, or timing benchmark runs.
The earlier `resource_checks_attempt1.json` remains preserved before the gate-applicability clarification.
Its exact intermediate source bytes were not retained.
Use the final report for reproducible source-bound checks.
