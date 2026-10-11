# Exact CPU E5 shard qualification

Read `CONTRACT.md` first. This new orchestration preserves the frozen numerical
document encoder, with a new cache schema and explicit full-frame bindings.
No frozen source is edited. Original-route cache acceptance and scientific
readiness remain false, including after this engineering gate passes.

The old monolithic CPU pass is measured at roughly 61.3 modeled hours, exceeding
the 24-hour allocation. This package provides a bounded prospective alternative:
an unmodified golden writer on a named 19-record natural engineering subset,
four fresh encoding processes, four independent replay processes, canonical
assembly and exact golden vector/chunk-log equality. Every call still uses the
original CPU FP32 `encode_document`, one thread, eager attention, padding 261,
and record-local batches of eight chunks. All 149 chunks per pass are retained.
The qualification performs 57 document encodings / 447 chunks; no full run is
automatically started. Its true completion depends on executing the job.

The golden dataset is `askubuntu_engineering_subset_19_v1`. Its derived audit
states that selected objects and already passed fixed guards were inherited
from the complete replayed source; original preparation was not rerun on the
subset. The frozen golden writer's fixed population-scope string refers only
to this explicitly named engineering subset and cannot establish completeness
of the scientific 140,250-row population.

Use the existing `encoder-cpu` interpreter and models. Required pinned assets
and all source/census/budget receipts are in `expectations.json`. The package
also imports `phase6.acceptance` before any forwards to expose missing SciPy or
idna dependencies early; it does not execute or claim that acceptance route.
No downloads, dependency installations, or runtime-manifest edits occur here.

Only `qualify.sbatch` is a ready submission entry point. Its eight arguments:

```bash
repo=/nfs_home/users/poonam/ccu-acl2027/repo
workspace=/nfs_home/users/poonam/ccu-acl2027/local_workspace
view="$workspace/qualification/cpu-qualification-20261011T143735+0530-transport-v1-a02/views/askubuntu"
replay="$workspace/qualification/source-replay-v1-20261011-a01/replay/result.json"
census="$workspace/qualification/e5-token-census-v1-20261011-a01/census"
budget="$workspace/qualification/e5-forward-budget-v1-20261011-a01/budget"
run_root="$workspace/qualification/e5-sharded-cpu-v1-20261011-a01"
mkdir -p "$workspace/qualification"
mkdir "$run_root"
package_sha256=REPLACE_WITH_REVIEWED_MANIFEST_SHA
sbatch --output="$run_root/slurm.out" --error="$run_root/slurm.err" \
  "$repo/local_run/e5_sharded_cpu_v1_20261011/qualify.sbatch" \
  "$repo" "$workspace" "$view" "$replay" "$census" "$budget" \
  "$run_root" "$package_sha256"
```

The allocation is CPU-only, 2 CPUs, 48 GiB, 30 minutes, `cpu-short`, excludes
`csis.mn1`, and uses `--propagate=NONE`. The model uses one thread. It captures
each fresh child process through exit; the outer process receipt also measures
all final result publication. No automatic retries occur. An existing attempt
directory is never overwritten or used as a row checkpoint.

After a passing qualification has been independently reviewed, its exact
`result.json` SHA must be supplied using `--qualification-sha256` for every
full plan/worker/assembly invocation. `--qualification-result` points to that
receipt and all original evidence must remain available. The gate rechecks
the actual golden, shard/replay and assembled artifacts, full source frame,
prospective sample selection and current code/runtime fingerprints. Booleans
alone never unlock full work. Full plans bind both bytes and SHA of this first
accepted result. A changed pin needs new qualification, not zero-work reuse.

`driver.py plan-full` prepares 32 contiguous whole-record ranges using measured
maximum batch-shape timing weights; it submits nothing. `worker.py --mode
encode` and `--mode replay` execute exactly one range in fresh processes.
Only complete immutable shards are reusable; partial attempts are preserved
and must restart in a new output directory. `driver.py assemble` requires all
32 completed encodes and all 32 fresh replays. Every vector byte and source
row/token/chunk ledger entry is checked before canonical assembly; all original
input and output bindings are rechecked after writing. A separate reviewed
launcher/resource decision is required before using these full-frame commands.

The main full matrix payload is 430,848,000 bytes. Encoded shards, replay shards
and assembled output coexist (three payload copies), with models, source data,
ledger files, process logs and temporary/gather memory additional. The new full
manifest is `third-e5-sharded-cpu-cache-v1`; downstream use needs an explicit
versioned adapter, never a fabricated old cache manifest or readiness claim.

Local tests use tiny software-only data and fake vectors, not models:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s local_run/e5_sharded_cpu_v1_20261011 -p test_shards.py
bash -n local_run/e5_sharded_cpu_v1_20261011/qualify.sbatch
```
