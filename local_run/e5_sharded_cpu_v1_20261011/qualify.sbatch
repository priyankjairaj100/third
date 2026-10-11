#!/usr/bin/env bash
#SBATCH --job-name=third-e5-shard-qual
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
view=$3
replay_result=$4
census_dir=$5
budget_dir=$6
run_root=$7
package_sha256=$8
qualification_python="$workspace/envs/encoder-cpu/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TMPDIR="$run_root"
"$qualification_python" - "$repo" "$package_sha256" <<'PY'
import hashlib, json, os, platform, resource, socket, sys
from pathlib import Path
if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Linux Slurm allocation required')
if socket.gethostname().split('.')[:2] == ['csis', 'mn1']:
    raise SystemExit('Login node prohibited')
limit = resource.getrlimit(resource.RLIMIT_AS)[0]
if limit != resource.RLIM_INFINITY and limit < 40 * 1024**3:
    raise SystemExit('Inherited address-space limit below 40 GiB')
folder = Path(sys.argv[1]) / 'local_run/e5_sharded_cpu_v1_20261011'
raw = (folder / 'package_manifest.json').read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
    raise SystemExit('Shard qualification package manifest mismatch')
for name, expected in json.loads(raw)['files'].items():
    if Path(name).name != name:
        raise SystemExit('Unsafe package member')
    raw = (folder / name).read_bytes()
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise SystemExit('Shard qualification package member mismatch')
PY
"$qualification_python" -m unittest discover -s "$repo/local_run/e5_sharded_cpu_v1_20261011" -p test_shards.py
"$qualification_python" - "$repo" "$workspace" "$view" "$replay_result" "$census_dir" "$budget_dir" "$run_root" "$package_sha256" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
repo, workspace, view, replay, census, budget, run_root, package = sys.argv[1:]
command = [sys.executable, str(Path(repo) / 'local_run/e5_sharded_cpu_v1_20261011/driver.py'), 'qualify',
    '--repo', repo, '--workspace', workspace, '--view', view, '--replay-result', replay,
    '--census-dir', census, '--budget-dir', budget, '--out', str(Path(run_root) / 'qualification'),
    '--package-sha256', package]
start = time.perf_counter()
process = subprocess.run(command, stdin=subprocess.DEVNULL, check=False)
receipt = {'schema': 'third-e5-shard-qualification-process-v1', 'returncode': process.returncode,
    'complete_child_wall_seconds_including_result_write_and_exit': time.perf_counter() - start,
    'command': command, 'receipt_write_excluded_from_own_timer': True,
    'full_population_launch_authorized': False, 'original_route_acceptance': False,
    'full_population_cache_acceptance': False, 'confirmatory_study_ready': False}
with (Path(run_root) / 'process_completion.json').open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
raise SystemExit(process.returncode)
PY
