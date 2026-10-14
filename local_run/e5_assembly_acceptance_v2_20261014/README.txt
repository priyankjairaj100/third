Prospective post-assembly metadata reader v2 — 2026-10-14

This package does not execute or schedule assembly. It was prepared locally;
no current cluster state, completed assembly, or cache acceptance is asserted.
This version additionally requires the separately reviewed832MiB assembly-stage
launcher and budget report; the old acceptance_v1 and gate_v1 are unchanged.
The folder name is bookkeeping: a passing metadata report is not scientific
acceptance. The sealed assembly gate94147011...ebc616 remains unchanged.

Dependency and target
The explicit --gate-package directory must contain the unchanged nine-member
assembly_gate_v1 package with manifest SHA256
94147011564dffa8354680ca3e0a912d3a32667b8a9c791a083ad68858ebc616.
Every member is verified before its stdlib helpers are imported.
The assembly receipt must also bind the exact separately sealed
assembly_stage_budget_v1 launcher package, manifest787bytes with SHA
66511a7a14deaca6c73a7d9c04fc925b9942cbef72aa9f39def7c33e6036f092,
and its6117-byte stage_budget_report,
SHA3068158ad2379cbfe251ca38ad313e17c6860488cf292f7b0cff93c71796275c.
The new policy retains16MiB logs,32MiB JSON,16MiB filesystem overhead and256MiB
contingency for an exact506555097-byte payload, rounds to832MiB, and preserves
the8GiB reserve and original2.25GiB total output envelope. Fresh quota remains
mandatory. It changes no source, numerical, metadata or scientific criterion.
The independently reviewed prior metadata result and bundle are additionally
pinned to raw SHAs8ceccfbda805bcf7a4e73b304ea88bdd2c218c3af5f1cb584240b9818b09c047
and393a90c6f4518f9e9292b8ebce07eebc09d4588697bd67c322c40ebabd24feb6.
The fixed full-run root is
/nfs_home/users/poonam/ccu-acl2027/local_workspace/qualification/e5-full-sharded-v2-20261013-a01
and source array is13498. The later assembly job is derived from its successful
assembly_launch_receipt.json, whose raw SHA must be supplied independently.

Interfaces, executed only by root after connection reconciliation
Verify this package's expected manifest SHA and all members before execution.
read_remote.py --gate-package PACKAGE --assembly-launch-sha256 REVIEWED_SHA
  --mode status
Produces bounded available metadata, exact-parent accounting, queue rows, and
up to4KB per terminal-failure diagnostic log. It makes no verification claim.

read_remote.py --gate-package PACKAGE --assembly-launch-sha256 REVIEWED_SHA
  --mode export
Requires absent queue and one exact parent COMPLETED/0:0 on csis.cn1; derives
and checks the exact sbatch command, actual Slurm job, successful dispatch and
child process, package/source/plan/qualification/repair metadata, and all32
manifest shard-pair references. The unchanged core writes cache/manifest.json
last. Success inventory is exactly manifest.json, vectors.npy, chunk_log.jsonl.
There is no cache/result.json on the assemble route; requiring it would reject
valid assembly output. Incomplete/failed assembly cannot produce a success
export. Capture stderr/status separately to retain failure evidence.
The reader rechecks all bounded metadata and file-stat observations, including
those of the prior gate. Each scheduler command times out after10seconds; the
reader has a180second deadline. Export is stdout-only and capped at25MiB.
There are no SSH calls, writes to remote inputs, jobs, retries or model loads.

verify_local.py --gate-package PACKAGE --assembly-launch-sha256 REVIEWED_SHA
  --transport SAVED_EXPORT.json --out NEW_PRIVATE_LOCAL_DIRECTORY
Replays the metadata checks locally, compares the reported summary, then saves
terminal_export.json and local_verification.json exclusively. Remote scheduler
and file-stat observations remain attested. Original metadata/package checks
are replayed locally; remote payload contents are not available in this export.

Claim limits and actual-content checks
The reader hashes metadata only. It checks each final cache payload's actual
file size and identity against its manifest reference; it does NOT read or
rehash the430,848,128-byte vector file or75,706,969-byte chunk-log file. It also
checks the claimed chunk-log SHA against the precomputed exact serializer/source
count, but this does not independently establish the on-disk log's contents.

A terminal successful sealed assembly is evidence that the unchanged core
already checked source/census bindings, every encoded shard and fresh replay,
actual vectors/logs and hashes, complete row order, and each assembled slice
against its source shard before publishing the sealed manifest. This reader
reports that execution evidence separately from independent payload validation.
Original monolithic-route acceptance, source admission, full-population cache
acceptance and confirmatory readiness remain false.

If an independent actual-content audit is still required, the minimal next
route is a separately reviewed READ-ONLY compute-node verifier, not another
assembly or encoder run. It would bind this exact final manifest/plan and current
packages; reuse common.context/require_qualification/load_plan; verify all32
encoded/replay receipts with the frozen checks; hash the actual final vectors
and log; verify dtype/shape/finite vectors and ordered source-log rows; and call
compare_shard_slice for every exact range. A watcher would recheck all inputs
and outputs before emitting a distinct audit receipt. No model forward, cache
copy, overwrite, new labels or changed thresholds is necessary. That route is
described only: this package neither implements nor submits another Slurm job.
Downstream original-route/cache gates would still require their own explicit
protocol decision; engineering parity is not a substitute.

Local software tests
python3 -m unittest discover -s THIS_PACKAGE -p 'test_*.py'
Eight synthetic assembly tests passed, zero skips. They reuse the locally
copied verified preassembly metadata plus synthetic assembly claims, never
read model payloads. They cover terminal/job/node refusal, process commands and failures,
manifest seal/count/order/claim changes, wrong file size/hash, the explicit
absence of result.json, and no payload reads during rechecking. Fixtures rely
on the local checkout/reports and sibling sealed gate tests; this is not an
assertion of remote test execution or a newly measured empirical result.
