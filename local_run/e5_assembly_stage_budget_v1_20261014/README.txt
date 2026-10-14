Assembly-only disk budget launcher v1, 2026-10-14

This separate prospective package changes resource bookkeeping only. It preserves
assembly_gate_v1 (manifest94147011564dffa8354680ca3e0a912d3a32667b8a9c791a083ad68858ebc616),
all previously sealed numerical/core/dispatch code and all failed evidence.
The first1.25GiB allowance preflight stopped before intent, receipt, stage or job.
No outputs are deleted and no threshold/scientific acceptance gate is changed.

Exact stage payload:506555097bytes =430848128-byte FP32 NPY +75706969-byte JSONL.
Add16MiB total stage console allowance, unchanged32MiB JSON allowance,
unchanged16MiB filesystem allowance and unchanged256MiB contingency:
842099417bytes. Round upwards to832MiB=872415232bytes.
The eight-GiB reserve remains8589934592bytes. Fresh minimum available quota is
9462349824bytes. The observed snapshot is only a comparison; the launcher reads
quota again immediately before its one submission and stops if insufficient.

Why console allowance is stage-specific: this stage creates only four console
files (one child stdout/stderr pair, one Slurm stdout/stderr pair), launches no
model/tokenizer/worker forward process and has no per-record console printing.
The large record log is fully counted as payload. Both driver and dispatcher
print one final JSON summary.16MiB is a conservative console planning margin,
not an enforced runtime cap or a mathematical bound on arbitrary system errors.
The old128MiB console margin covered the entire64-worker process run plus assembly.
No other contingency has been cut to fit the current quota.

Every old launch guard is retained: independently reviewed terminal metadata,
all32 completed exact task IDs on csis.cn1, same packages/input/repair/qualification
pins, no active matching array or assembly, no preexisting assembly intent,
receipt or stage, fresh mount-verified quota, and fsynced exclusive intent.
The original2.25GiB total output envelope is still enforced using all existing
run files' max(logical,allocated) bytes plus16MiB directory/stat slack plus
this remaining allowance. This is separate from the unchanged8GiB reserve.
The launcher never hashes vector/log bodies on the login node.

This launcher verifies its OWN new manifest separately from the original gate
manifest. The original gate remains imported from the explicitly supplied path.
Its all32/64 metadata checks are unchanged. New intent fields:
- launcher_package_manifest: new package raw manifest bytes/SHA256
- stage_budget_report: this package's stage_budget_report.json bytes/SHA256
- budget_policy: third-assembly-only-capacity-policy-v1
The old gate_package_manifest field continues to bind the original941470 seal.
The unchanged assembly command uses original core/dispatch, cn1,2CPUs,48GiB,
30minutes and the original root/plan. The old intent/receipt schema is retained
so later readers can verify the execution; the versioned reader must additionally
validate the new launcher and policy bindings.

Prospective invocation only, after independent review and sealing:
  python NEW_PACKAGE/assembly_launch_remote.py \
    --metadata-result-sha256 8ceccfbda805bcf7a4e73b304ea88bdd2c218c3af5f1cb584240b9818b09c047 \
    --gate-package ORIGINAL_GATE_DIR \
    --gate-package-sha256 94147011564dffa8354680ca3e0a912d3a32667b8a9c791a083ad68858ebc616 \
    --launcher-package-sha256 REVIEWED_NEW_MANIFEST_SHA

No SSH, credentials, downloads, cancellation or automatic retries are implemented.
No benchmark or other concurrent output job is reserved by this budget. No remote
archive/full-copy, second assembly, or rerun is included. A failure or ambiguous
submission leaves the intent and evidence for reconciliation, without retry.
Actual vector/log/source checks remain inside unchanged Slurm assembly before
cache publication. Resource planning, terminal metadata and successful assembly
are not original-route/full-cache scientific acceptance or confirmatory readiness.

Local checks (no remote actions):
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s NEW_PACKAGE -p 'test_*.py'
The11 tests cover inherited refusal paths, exact fresh capacity boundary,
unchanged reserve/contingency/total envelope, exclusive intent and preservation
of every original gate member. They are software checks only.
