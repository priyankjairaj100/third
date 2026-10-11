CPU shard dispatch wrappers, version 1, 2026-10-11

These wrappers execute explicitly submitted CPU stages. They never call sbatch,
SSH, download tools, or automatic retries. The separate frozen numerical/core
package is e5_sharded_cpu_v1_20261011, manifest SHA256:
c7619b96b080500aeee439b3128e034d8138dd89657f8799a2a9a0da9f1c6312

Use only after the 19-record qualification has completed and its complete
evidence has been independently reviewed. Supply the exact result.json file SHA.
Then inspect the generated full plan and supply its independent raw file SHA
before launching any shard. A sealed/hashed plan alone is not launch approval.
The parent task decides launch and resource use. Source acceptance, original
route/cache acceptance, and confirmatory scientific readiness remain false.

All scripts use the existing workspace/envs/encoder-cpu/bin/python. They check
this interpreter path, Linux Slurm context, and exclude the login node. No
dependency or environment installation occurs. CPU model settings remain in
the sealed core. Shell exports set offline operation, one numerical thread,
PYTHONHASHSEED=0, and no visible CUDA devices.

Resources:
- plan.sbatch: cpu-short, 2 CPUs, 48 GiB, 10 minutes.
- shard.sbatch: cpu-short, 2 CPUs, 12 GiB, 8 hours PER TASK, including encode
  followed by a separate fresh replay process. Default array is 0-31%8.
- assembly.sbatch: cpu-short, 2 CPUs, 48 GiB, 30 minutes.
All exclude csis.mn1 and use --propagate=NONE. The task should bind the same
qualified CPU node using sbatch --nodelist after reviewing measured RSS and
the actual CPU/runtime fingerprint. No cross-machine byte identity is assumed.
The maximum-eight array concurrency is a submission/header policy; this
wrapper does not independently query live scheduler concurrency. It does
check exactly 32 tasks with min0/max31 and task IDs that are integers0..31.

COMMON TEN POSITIONAL ARGUMENTS, in this exact order:
1 repo                   absolute repository path
2 workspace              absolute private workspace path
3 view                   full replayed ASCII source view directory
4 replay_result          complete source replay result.json
5 census_dir             complete census evidence directory
6 budget_dir             passed timing evidence directory
7 core_package_sha       fixed core manifest SHA shown above
8 qualification_result   independently reviewed qualification result.json
9 qualification_sha      raw file SHA256 of that result.json
10 dispatch_root         EXISTING new attempt root below workspace/qualification

plan.sbatch adds positional argument11: this dispatch package manifest SHA.
shard.sbatch and assembly.sbatch add:
11 full_plan             absolute generated plan.json path
12 full_plan_sha         independently reviewed raw plan.json file SHA256
13 dispatch_package_sha  this dispatch package manifest SHA256

The manifest SHA always comes LAST. Each shell checks both complete sealed
packages using stdlib only before executing dispatch.py. The dispatcher
repeats these checks, checks the qualification/plan file hashes and identity,
then delegates all source/code/assets/runtime gates to the unchanged core.
Both package hashes, qualification hash, and full plan hash are rechecked
after every child and must remain unchanged before the next child can run.

Example argument arrays; replace every REVIEWED placeholder with actual values:
  common=("$repo" "$workspace" "$view" "$replay_result" "$census_dir"
          "$budget_dir" "$core_sha" "$qualification_result"
          "$qualification_sha" "$dispatch_root")
  sbatch --nodelist="$qualified_node" \
    --output="$dispatch_root/plan-%j.out" --error="$dispatch_root/plan-%j.err" \
    "$repo/local_run/e5_sharded_dispatch_v1_20261011/plan.sbatch" \
    "${common[@]}" "$dispatch_package_sha"

The full plan will be dispatch_root/plan-run/output/plan.json. Review this
plan and record its raw SHA256 before submitting the array. The next commands
are separate explicit actions, not a program that submits subsequent jobs:
  sbatch --nodelist="$qualified_node" --array=0-31%8 \
    --output="$dispatch_root/shard-%A-%a.out" --error="$dispatch_root/shard-%A-%a.err" \
    "$repo/local_run/e5_sharded_dispatch_v1_20261011/shard.sbatch" \
    "${common[@]}" "$full_plan" "$full_plan_sha" "$dispatch_package_sha"

After all32 encode/replay pairs complete and their status is reviewed:
  sbatch --nodelist="$qualified_node" \
    --output="$dispatch_root/assembly-%j.out" --error="$dispatch_root/assembly-%j.err" \
    "$repo/local_run/e5_sharded_dispatch_v1_20261011/assembly.sbatch" \
    "${common[@]}" "$full_plan" "$full_plan_sha" "$dispatch_package_sha"

Fresh stage paths:
- dispatch_root/plan-run/output: core full-plan build.
- dispatch_root/shards/shard-NNNN/encode and /replay: core immutable artifacts.
- dispatch_root/assembly-run/cache: core assembled output.
No existing stage directory is reused or overwritten. Partial/failing attempts
are preserved and never trigger an automatic retry. A separately reviewed
new attempt is required after failure; these wrappers do not automate mixed
attempt assembly or resuming an incomplete pair. Core completed shards remain
available as immutable evidence.

Each stage saves launch.json and dispatch_result.json. Every executed child
saves private stdout/stderr and a process receipt with child/launcher PIDs,
return code, exact command, and wall time through the child's exit (including
its final artifact writes). The process timer excludes this dispatcher's log
flush and receipt publication. Zero exit must also have the expected successful
core artifact. A nonzero exit, missing artifact, or changed binding stops work.

The final cache uses the new core manifest schema. These wrappers never forge
the old cache schema, authorize downstream scientific acceptance, or perform
model-based annotation. No source data or models are packaged in this folder.

Software validation:
  python3 -m unittest discover -s local_run/e5_sharded_dispatch_v1_20261011 \
    -p test_dispatch.py
  bash -n local_run/e5_sharded_dispatch_v1_20261011/plan.sbatch
  bash -n local_run/e5_sharded_dispatch_v1_20261011/shard.sbatch
  bash -n local_run/e5_sharded_dispatch_v1_20261011/assembly.sbatch
Tests use tiny synthetic files and short local Python child processes only.
They provide no executed Slurm, model, full-population, or scientific evidence.
