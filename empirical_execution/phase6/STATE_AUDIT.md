# Complete state audit

`state_audit.py` implements the required complete state comparison.
It does not replace this comparison with a model-head check.
The dispatcher assigns it to the first 16 R and first 16 S core trajectories.
It also checks B-E membership at other required core releases.
Actual primary execution still needs authentic study inputs and accepted resource policies.

## Three guarantee levels

The audit reports three separate levels.

1. Exact structural checks cover stable keys, integer counts, alive units, horizons, payload membership, owners, and blocker sets.
2. Exact symbolic checks compare integer linear forms in original record statistics.
3. Numerical checks compare every stored moment coordinate against fresh construction and an independent extended-precision accumulation.

The third level uses prospective absolute and relative tolerances.
It reports each key, coordinate count, maximum difference, Frobenius difference, and bitwise equality separately.
These comparisons are numerical diagnostics.
They are not rational certificates for floating-point accumulation.
They do not establish bitwise independence from update history.
The repository's separate exact rational theory retains its own contract.

For the symbolic check, each record contributes two monomials:

\[
t_v\left(x_{B(v)}-x_{B(v)\cup\{o(v)\}}\right).
\]

Here \(t_v\) is a symbol for that record's complete moment vector.
Same-source blocking cancels the entire contribution.
Terms above the remaining horizon are absent.
The audit independently forms these integer linear forms from fresh retained blockers.
It also substitutes deleted units into the preceding symbolic state.
It decreases the horizon by the number of new unique deleted units.
The two symbolic maps must agree exactly.
This equality concerns the fixed audited paths and stored-input contract.
It does not authenticate corpus metadata or semantic embeddings.

## Actual service and snapshot linkage

The audit runs a separate service with persistence after every release.
It uses the same method, inputs, requests, solver, compaction policy, and resource policy.
This service retains the measured worker implementation and kernel restriction.
No worker source changes are needed.

The existing worker records each serialized checkpoint hash.
The auditor loads that service's initial snapshot and repeats its prescribed updates.
It reconstructs each complete snapshot byte sequence.
Each byte sequence must match the actual audit worker's recorded hash.
The auditor then examines every designated state coordinate.
Thus hashes bind the examined state to actual execution.
They do not replace complete state comparison.

The original measured service supplies separate linkage checks.
Its input hashes, request configuration, policy, success status, release count, and emitted head hashes must match.
Its final snapshot must match its own worker hash.
Every auxiliary head must match the corresponding measured head bitwise.
Initial and final snapshots must match the measured service snapshots.

Intermediate private states belong to the separate audit service.
The report does not claim direct visibility into private intermediate states from the original measured run.
Its original intermediate linkage consists of matching requests, inputs, heads, and reported state information.
The auxiliary snapshots supply the actual complete-state evidence.

The wire inventory remains descriptive.
Its representation can include process-specific alias or set order.
Its metadata hash is therefore not a canonical abstract-state invariant.
A differing wire inventory hash does not replace any required structural or numerical comparison.

## Complete target checks

The auditor recomputes retained pair predicates with bounded tiles.
It uses the frozen ordered numerical scorer and independently forms adjacency.
It does not use the repair method's blocker update.
The same scorer defines both predicates; this is not an independent real-arithmetic cosine certificate.

P-I, P-S, and P-R expose their complete coefficient maps for this external audit.
The auditor maps all integer keys to stable unit IDs.
It compares the complete union of actual and fresh keys.
An absent coefficient represents zero.
No near-zero pruning occurs.
Every packed Gram, cross-moment, and count coordinate enters the comparison.
Count coordinates must match exactly as integers.

P-R additionally compares its component, grounding, omitted-node, and stored-basis rules.
One exact representation convention needs care.
An ungrounded singleton component has no stored coordinates.
Its only coefficient equals the negative sum of an empty basis, which is exactly zero.
The maintained representation can retain that zero component after deletion.
Fresh construction can omit it.
The audit reports this raw metadata difference.
It compares basis relations after removing only these forced-zero singleton components.
Every removed coefficient must equal zero in every coordinate, without tolerance.
Any other component or basis difference remains a failure.

B-E checks every eligible record, selected record, owner, remaining blocker, blocker count, feature row, and target row.
B-A checks the corresponding complete retained payload.
Both compare all Gram and cross-moment coordinates.
Oracle methods retain their declared greater-access scope.
B-F preserves the surviving original selection.
The audit never substitutes fresh curation for B-F's declared target.

Fresh method construction is a second comparison.
It uses the remaining horizon and retained corpus.
This shared-constructor comparison checks update and serialization consistency.
The independent symbolic construction and direct moment accumulation provide separate checks.

## Cost and prospective limits

Audit time remains separate from measured method time.
The report charges the auxiliary service lifecycle and the complete external audit.
It retains snapshot bytes, per-checkpoint audit time, and worker resource reports.
It also reports the auditor process's lifetime peak RSS.
That value includes earlier work in the parent process.
It is not an isolated audit-stage peak.

Default limits are:

| Limit | Default |
| --- | ---: |
| Pair-coordinate evaluations per graph rebuild | 100,000,000 |
| Coefficient coordinates in a complete map | 4,000,000 |
| Absolute moment tolerance | 0.00000000001 |
| Relative moment tolerance | 0.0000000001 |

These development limits intentionally refuse many primary configurations.
At 10,000 records and curator dimension 768, one fresh undirected graph needs 38,396,160,000 pair-coordinate evaluations.
Dense state comparison can also exceed the default coefficient limit.
Primary execution must bind any larger limits prospectively through its reviewed execution policy.
The auditor preserves budget refusals as failed audit gates.
It never skips the gate or raises its limits after observing results.
No native-scale audit is claimed by the present development checks.

## Evidence and preserved attempts

`check_state_audit.py` checks all eight core methods on the existing Civil100 lexical inputs.
Its final release is `results/state_audit_release_v3/`.
The check covers 32 regular checkpoint states and 1,801,800 coefficient coordinates.
It also checks total withdrawal of one genuine record.
That case exposes the exact P-R zero-component convention.
It creates no synthetic empirical corpus.

`check_state_audit_semantics.py` covers the required finite-horizon cases.
These include singleton, batch, reversed order, duplicate, unknown-ID, save/resume, empty, and exhaustion requests.
Natural Civil subsets supply record cases.
Tiny labeled algebra fixtures supply source-contract cases only.
Source and expanded-record requests compare current targets under their separate horizon contracts.
They do not claim that different budget contracts have identical future states.

The initial development attempt remains preserved.
It exposed an audit-report NumPy boolean serialization error.
It also exposed a noncanonical wire-inventory hash comparison.
One saved O-T snapshot differed from its recorded worker hash after execution.
The cause of that artifact mismatch is not established.
The final audit rejects such a reference before comparing states.

The original successful state-audit release remains historical.
Its exact source is archived at `history/state_audit_pre_rank_zero/state_audit.py`.
The dispatcher then exposed a redundant P-R zero component on a partially deleted natural subset.
Its failed attempt remains at `results/dispatch_state_gate_attempt1/`.
The second state-audit release overlapped the final exact-zero guard edit.
Treat it as a superseded development attempt, not the final source-bound result.
The third release runs only after the corrected source freeze.

These checks do not complete the registered native first-16 R and first-16 S primary audit.
That gate remains mandatory when authentic inputs become available.
