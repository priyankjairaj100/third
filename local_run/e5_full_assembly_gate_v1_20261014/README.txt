Prospective metadata gate and explicit assembly launcher — 2026-10-14

Target: array 13498, fixed full-run root
/nfs_home/users/poonam/ccu-acl2027/local_workspace/qualification/e5-full-sharded-v2-20261013-a01

This package was prepared locally after the cluster status channel refused a
request. Preparation and software tests establish no current array state. No
SSH, scheduler request, model call or assembly submission occurred in this work.
Do not retry a failed/refused SSH channel automatically.

Scope
The read-only evidence checks require exactly 32 unique array tasks, all
COMPLETED/0:0 on csis.cn1, absent from the user's queue; JobIDRaw must match each
shard's recorded physical SLURM_JOB_ID. They bind the reviewed full plan, prior
qualification, additive SciPy repair, original array launch, storage budget,
core and dispatch packages, 32 dispatch receipts, 64 child process receipts,
and 64 completion receipts. Every exact range, work counter, runtime contract,
process command, encode/replay claim and metadata hash must agree.

The audit hashes bounded metadata only. The 128 vector/chunk-log files receive
regular-file, no-symlink, exact directory inventory, stat stability, file-size,
and recorded SHA256-reference checks. Their payload contents are NOT read or
rehashed here. Encode/replay matching hashes and equality flags remain recorded
claims until the unchanged core assembly checks actual complete vectors/logs,
source rows, qualification, and byte equality inside Slurm before publication.
All cache acceptance and confirmatory-readiness flags remain false.

Runtime interfaces (root executes only after connection recovery/reconciliation)
1. Stage this entire sealed package in a private directory. Verify the expected
   package_manifest.json SHA and every member before execution. Use Python 3.13
   and PYTHONDONTWRITEBYTECODE=1. No third-party modules are required.
2. Run gate.py --out FULL_ROOT/terminal-metadata-audit-v1-a01
   The output must be new and directly below the fixed full-run root. The script
   writes bundle.json and metadata_gate_result.json, including on failure.
   Scheduler subprocesses have 10-second limits; the audit has a 180-second
   deadline. No automatic retry, submission, source modification or payload read.
   Existing output is never overwritten. A failed attempt must be reconciled;
   changing attempt suffix is not an automatic retry authorization.
3. Run export_remote.py --report-dir FULL_ROOT/terminal-metadata-audit-v1-a01
   Capture stdout as a local transport JSON. Successful evidence is rechecked;
   failure evidence can also be retained. The transport is at most 25 MiB.
4. Locally run verify_local.py --transport TRANSPORT.json --out NEW_LOCAL_DIR
   It revalidates metadata independently of current remote filesystem access,
   binds the exact bundle/result bytes, and preserves the two files plus
   local_verification.json. Remote scheduler/stat observations remain attested.
   Exit status 0 means the metadata gate passed; failed evidence is retained
   with exit status 2. Record the raw metadata_gate_result.json SHA for review.
5. Only after independent review, the explicit launcher accepts
   assembly_launch_remote.py --metadata-result-sha256 REVIEWED_RESULT_SHA
   --gate-package-sha256 REVIEWED_PACKAGE_MANIFEST_SHA
   It requires the a01 audit, rechecks all stable metadata/report fields,
   observations, exact terminal scheduler status and queue, and fresh quota.
   The remaining allowance is 1.25 GiB plus the unchanged 8 GiB reserve.
   Existing materialized run files plus 16 MiB slack must fit within 1 GiB,
   preserving the original total 2.25 GiB output envelope. This is a snapshot,
   not a reservation or formal guarantee against unrelated quota consumption.
   Exactly one unchanged assembly.sbatch submission is permitted by this
   invocation, with 2 CPUs, 48 GiB, 30 minutes on csis.cn1. An exclusive fsynced
   launch intent precedes submission. Existing intent/receipt/stage refuses;
   timeout or ambiguous sbatch outcome is preserved and requires reconciliation.
   No job is submitted by the audit, exporter, verifier or tests.

Software validation
python3 -m unittest discover -s PACKAGE -p 'test_*.py'
20 local tests passed, zero skips. Gate fixtures use already copied pinned
metadata plus synthetic all-32 completion/process records; they contain no
model execution or new empirical evidence. The fixture builder references the
existing local Three checkout and reports, so these tests are local validation,
not a claim of portable cluster preflight execution. The launcher tests use
small temporary files and mocked accounting only. Static compilation passed.

Original repository packages, frozen numerical source and all failed empirical
attempts are unchanged. The package manifest excludes itself and __pycache__.
