# Bounded E5 CPU forward budget, version 1

This prospective engineering wrapper uses the passed full AskUbuntu source
replay and tokenizer census to measure the unchanged CPU E5 encoder. It does
not create a full cache, semantic labels, source admission, or study readiness.
Only this new package may change; all frozen artifacts and prior evidence stay
unchanged. Run it on a Slurm compute node, never on a login node.

The driver verifies the sealed census helper package, frozen encoder/imports,
exact replay and census receipts, complete source frame, original ASCII view,
and the complete pinned E5 assets. It requires the existing `encoder-cpu`
Python 3.13.7 runtime and the exact libraries recorded in `expectations.json`.
It performs no downloads and imports no custom model code.

Before any forward, select two natural calibration records per exact chunk
count 1–8 using the lowest SHA256 of the fixed salt, NUL and record ID (then ID
to break ties). Source and full ordered census bindings are rechecked; the
private sample manifest and complete selected token/chunk ledger are written
before forwards. Run baseline, repeated and reversed document order: 16
documents, 48 complete document encodings, 48 model forwards, 216 chunks and
56,376 padded token slots. No document is truncated or combined with another.

`LocalTransformer` and `encode_document` are unchanged: CPU FP32, one intra-op
and one inter-op thread, deterministic algorithms, eager attention, MKLDNN
disabled, fixed 261-token padding, at most eight chunks per record-local batch.
The actual model input IDs/masks are checked against the frozen chunks, and
the selected token-ID digests must equal the passed full census. Repeated and
reversed vectors must match baseline FP32 bytes exactly. This is same-process
replay on the selected records, not fresh-process or full-population parity.

Pre/post hooks time the synchronous actual model forward after input checks
and before output checks. Backend and document timers include their respective
padding/pooling and tokenization/normalization; these nested times overlap.
Cold load, source/ledger scans, durable writes, all initial/final hashes and
finalization are separately recorded. The Slurm envelope includes the final
result write and child exit. The private outputs contain selected token IDs,
vectors, source-bound IDs and timing evidence; do not publish them wholesale.

Each batch size has six observed model timings. The budget report weights
these by the complete census distribution (147,825 forwards / 312,159 chunks)
and reports minimum/median/maximum sample scenarios for one and two passes.
It separately adds observed nonforward cost per chunk, cold load and fixed
qualification costs. These are planning estimates, not confidence intervals
or scheduler bounds: only one node/process and two documents per shape were
measured; documents above eight chunks are extrapolated up to 60 chunks, and
full-cache serialization/gathers/extra acceptance work are unmeasured. The
full nonresumable encoding stage needs a later scheduling decision after
inspection of these measurements. The report grants no launch permission.

Run the bounded package after syncing only this new folder. The original
sealed census helper folder must already be present unchanged. Root records
the package-manifest SHA separately in the launch receipt. Example variables
refer to existing source/evidence paths, not new downloads:

```bash
repo=/nfs_home/users/poonam/ccu-acl2027/repo
workspace=/nfs_home/users/poonam/ccu-acl2027/local_workspace
view="$workspace/qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu"
replay="$workspace/qualification/source-replay-v1-20261011-a01/replay/result.json"
census="$workspace/qualification/e5-token-census-v1-20261011-a01/census"
run_root="$workspace/qualification/e5-forward-budget-v1-20261011-a01"
mkdir -p "$workspace/qualification"
mkdir "$run_root"
# Supply the separately verified SHA of this package_manifest.json.
package_sha256=REPLACE_WITH_REVIEWED_MANIFEST_SHA
sbatch --output="$run_root/slurm.out" --error="$run_root/slurm.err" \
  "$repo/local_run/e5_forward_budget_v1_20261011/budget.sbatch" \
  "$repo" "$workspace" "$view" "$replay" "$census" "$run_root" "$package_sha256"
```

The job requests `cpu-short`, 2 CPUs, 48 GiB and 30 minutes, excludes
`csis.mn1`, and uses `--propagate=NONE`. The encoder still uses one thread.
No automatic retries, future full encoding, or other-user job changes occur.
Local validation uses tiny software-only fixtures; it performs no model loads:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s local_run/e5_forward_budget_v1_20261011 -p test_budget.py
bash -n local_run/e5_forward_budget_v1_20261011/budget.sbatch
```
