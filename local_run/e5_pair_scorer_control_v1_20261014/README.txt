Benchmark launch and terminal export controls — prospective, no jobs launched

This separate package leaves scoring_benchmark_v1 and frozen repository files
unchanged. Root must review and externally pin this control manifest before
execution. No script opens SSH, copies packages, polls, retries, cancels or
removes data. Local tests mock all scheduler calls.

FIRST STAGE THE EXISTING BENCHMARK PACKAGE
The already sealed six-member package plus its manifest must exist privately
(mode0700 directory, no symlinks/extra files) at:
 /nfs_home/users/poonam/ccu-acl2027/local_workspace/tools/e5-pair-scorer-benchmark-v1-20261014
Manifest SHA:
 ce03a07c9458e23c33d3f47df5f5208dbdf53b8eac36737a22783b1d39255c84
Transfer/staging is a separate root action, not performed by these helpers.

ONE EXPLICIT LAUNCH
Run launch_remote.py only on the authenticated cluster login session, with:
 --control-package-sha256 <externally reviewed control manifest SHA>
 --reviewed-assembly-dispatch-sha256 <reviewed final assembly dispatch_result SHA>
The latter is not yet hardcoded because assembly had not completed when this
package was prepared. All other paths/packages/qualification/repair pins are
fixed. The helper requires successful assembly submission, matching stage job ID,
passed bound dispatch/child/cache manifests, fresh COMPLETED0:0 accounting on
csis.cn1, no active array/assembly/benchmark job, and a clean repository.
Only small metadata and the58KB qualification vectors are hashed on login;
full source/ledger/model/payload verification remains in the benchmark Slurm job.

The verified quota parser is imported only after its source SHA check. Mount
must be /dev/sdb1, /nfs_home, ext4. Fresh quota -v -w must leave an unchanged8GiB
reserve plus32MiB output allowance, with snapshot age<=120seconds. The snapshot
is not a reservation. Benchmark launch waits for assembly completion to avoid
competing output commitments. Any existing benchmark root, intent or stage
requires reconciliation and blocks the call before sbatch.

Before the first filesystem mutation, the helper constructs the initial code,
review basis, source pins, command, accounting and quota record. It then creates
exactly this new private root:
 .../local_workspace/qualification/e5-pair-scorer-benchmark-v1-20261014-a01
It fsyncs an exclusive launch_intent.json, calls sbatch ONCE with11arguments,
cn1 and the pinned benchmark wrapper, then saves launch_receipt.json. Logs are
slurm-%j.out/.err in this root; wrapper child stage is root/run. Timeout or
ambiguous success preserves an unknown outcome and never retries. A root created
before an interruption is deliberately not reusable.

BOUNDED FINAL EXPORT
Run export_remote.py with --job-id <receipt job> and the same control manifest
SHA. It uses sacct plus squeue -u poonam filtered to the exact job, so an aged-out
completed ID does not depend on squeue -j. It checks fixed source/package pins,
intent/receipt/command identity and terminal accounting. Successful status also
requires cn1, matching child CLI, finite positive through-exit walltime,16 tests
with0skips,28 completed cases, result artifact hashes and preserved claim flags.
Final detailed case coverage/mode/bin/score checks remain independent review.

Every regular file under the run root is retained with path,size,SHA256,base64;
no file is silently omitted or truncated, including failure outputs. Caps are
8MiB/file,32MiB decoded total,64files,128directory entries and48MiB encoded JSON.
Oversized evidence fails export and remains intact at source. Raw accounting and
source-pin checks are included. A failed terminal run is exportable and remains
failed; no result is fabricated when the worker terminated early. Submission
outcomes without a known job ID require root reconciliation before this exporter.
The exporter writes nothing remotely and does not hash full cache payloads.

LOCAL VERIFICATION
 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s THIS_FOLDER -p test_control.py -v
Twelve tests cover argument maps, active/terminal gates, exclusive intent-before-
submission, timeout/nonzero no-retry evidence, quota age/reserve/mount, bounded
exports, strict JSON and symlink refusal. Local software tests are not cluster
execution, scheduler guarantees, reservations or scientific acceptance.
