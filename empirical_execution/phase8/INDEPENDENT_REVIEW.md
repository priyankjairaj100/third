# Phase 8 independent integration review

The reviewed preparation software passes this scoped source and evidence review.
It does not establish readiness of the complete study or any new scientific result.
The primary semantic study remains unstarted.

This review covers the frozen original-archive exporter, staged dossier assembler, and native-core resource-policy selector. It examines their interfaces together and checks the saved final reports against current source bytes. No empirical experiment, encoder run, human collection, or benchmark was launched by this reviewer. No production source was modified by this reviewer.

## Evidence checked

| Evidence | Result | Scope |
| --- | --- | --- |
| `results/archive_checks_final.json` | 82 checks passed | Retained owner checker; small archive-format software fixtures. |
| `results/archive_independent_review.json` | 31 checks passed | Independent source inspection and interactive format checks; no original archive accepted. |
| `results/dossier_checks_final.json` | 36 checks passed | Existing Civil preview text and original labels; unavailable acceptance and task-choice boundaries explicitly mocked. |
| `results/resource_checks_final.json` | 47 checks passed | Metadata-only controls; real observation reader, selector, receipt verifier, and attach/finalize routing with explicit upstream mocks. |
| `results/resource_missing_native/qualification.json` | Blocked, no policy | Actual incomplete-input CLI refusal at profile validation; not a successful native-acceptance trial. |
| `results/frozen_phase_preservation.json` | 1,958 prior files unchanged | Saved preservation check against the previous manifest; not new scientific validation. |

The counts describe separate sequences and must not be pooled into a claim of scientific coverage. Final binding verification matched 382 recorded source references covering 133 distinct artifacts. The machine-readable companion records exact source and evidence hashes. No owner check sequence was rerun for this final integration review.

The independent archive sequence was supplied interactively, and its exact script was not retained. Its saved report discloses that limitation. The final owner archive checker and final exporter snapshot are retained. Early archive/resource reports also preserve some intermediate hashes without their exact intermediate source bytes; they are historical records, not independently replayable versions. The failed first dossier source was restored against its recorded hashes with an explicit restoration receipt.

## Conclusions from source review

**Original lineage is a separate required gate.** Civil and News use fixed original archive identities. Full replay reconstructs the records, lineage, and member streams through hash/count sinks and compares complete stream digests and counts; every reconstructed byte contributes. It avoids writing a second complete corpus copy; parsing, full archive reads, the first export, and current-row/member working storage remain. Production identity pins cannot be replaced with fixture pins. Civil official splits remain lineage metadata; they do not replace the frozen source-disjoint study partitions. Unknown sources remain unavailable for genuine source withdrawal.

The assembler requires Civil/News adapter inputs to be the replayed export and repeats original-archive replay at relevant stages. Frozen acceptance still checks derived records, guards, source partitions, labels, calibration rows, and actual semantic caches. Neither an archive checksum nor consistent local files authenticate permissions, historical collection, or reviewer independence.

**The final-review ordering is coherent.** Calibration precedes thresholds; unsigned candidates precede supplied development evidence; policy and observation attachments precede final external review. The resource fingerprint excludes its own derived attachments and final testimony while retaining the immutable development replay digest. The complete external review still binds the final bundle, replay, and actual evidence. Resource verification recomputes its receipt instead of trusting a seal. A changed development digest or actual input fails this check.

Candidate stages use stable paths, exact authoritative group identities, successful candidate receipts, and hash-bound descriptors. Failed stages cannot advance via leftover candidates. Final bundle maps are published after frozen acceptance and stage-integrity checks. Dossier roots have an ignore-all publication guard, which is an accidental-publication precaution rather than a security boundary. Full caches remain mapped; selected/calibration/projection gathers, metadata, graphs, encoder work, and operating-system caches still cost memory. Stage-directory byte counts are scoped storage accounting, not peak-memory measurements or repair costs.

**Resource qualification is deliberately narrow.** Version 1 accepts only an exact canonical native `C_full_methods` primary-10,000 group. It requires all eight full-state methods and four mandatory light-audit methods for each available deletion unit. Same-panel alternative groups cannot borrow its qualification. Encoder identity and revision, dimensions, outputs, calibrated threshold, regularization, numerical settings, and actual distinct-deletion checkpoint schedules must match. The complete worker/configuration source maps, accepted development inputs, prospective observation plan, audit/reference reports, and saved artifact hashes are bound. Failed planned observations remain failures.

One jointly matching observation must support each declared profile; unrelated maxima cannot manufacture coverage. The pair-coordinate value is an original-shape count proxy, not measured operations. The full-summary count uses fresh designated symbolic keys, not rank-basis size. Future key counts remain unknown, the deterministic upper bound is separate, and cap overflow remains explicit. Wall and CPU measurements plus descriptive RSS indicators are retained without asserting runtime or true peak-memory bounds for future graphs. Numerical tolerances and the convex cap are unchanged.

## Closed integration findings

- Added original archive-to-export replay beyond the frozen export-to-cache boundary.
- Removed the resource-receipt/final-review fingerprint cycle without dropping the development replay binding.
- Bound group identities to the authoritative registry and refused unsupported group borrowing.
- Required full method/unit coverage, encoder identity, complete source maps, and actual executed distinct-unit schedules.
- Retained failed staging outputs; fixed metadata-receipt loading so descriptor expansion cannot precede its checksum check.
- Clarified review digest fields, partial family scope, preparation storage, unmeasured memory, and observation-versus-feasibility limits.

There are no open concrete findings in these reviewed component scopes. This is not a blanket correctness certificate or a claim that the eventual paper is reviewer-proof.

## Work still required

Authentic original archives, local models/tokenizers and runtime, successful semantic encoding/replay, genuine independent human judgments, source-disjoint development, and actual native measurements remain absent. No genuine resource policy or fully accepted primary dossier was produced. The official SemDeDup backend and complete semantic study remain unexecuted.

Single-family candidate support does not complete shared-root multi-family automation. Mixed encoders, projected/larger panels, other solvers/process settings, and dense-convex resource qualification remain unfinished. Native runtime and memory feasibility remain unknown. All 21,332 primary jobs and later human/statistical/paper obligations remain pending; missing cells cannot be silently omitted.

The Phase 8 README, remaining-task ledger, and checkpoint were inspected for this distinction. Later publication/status edits are not part of the frozen production source map in this review.

## Reproduction

Run from the repository root. The following verifies the exact artifacts bound by this review; it does not repeat experiments:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, json
review = json.loads(Path('empirical_execution/phase8/results/independent_review.json').read_text())
for name, expected in review['bound_artifact_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('All bound artifacts match:', len(review['bound_artifact_sha256']))
PY
```

The retained owner software checks can be repeated into new result paths:

```bash
python3 empirical_execution/phase8/check_archive_export.py --out empirical_execution/phase8/results/archive_checks_reproduction.json
python3 empirical_execution/phase8/check_dossiers.py empirical_execution/phase8/results/dossier_checks_reproduction.json
python3 empirical_execution/phase8/check_resource_policy.py --out empirical_execution/phase8/results/resource_checks_reproduction.json
```

These are explicit software controls, not primary-data runs. Existing output paths must remain preserved. The written source-review judgments and the unretained interactive archive sequence are not replaced by rerunning these commands.
