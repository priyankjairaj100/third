#!/usr/bin/env bash
#SBATCH --job-name=third-e5-budget
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
run_root=$6
package_sha256=$7
budget_python="$workspace/envs/encoder-cpu/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export TMPDIR="$run_root"
"$budget_python" - "$repo" "$package_sha256" <<'PY'
import hashlib, json, os, platform, resource, socket, sys
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID') or platform.system() != 'Linux':
    raise SystemExit('Linux Slurm allocation required')
if socket.gethostname().split('.')[:2] == ['csis', 'mn1']:
    raise SystemExit('Login node prohibited')
limit = resource.getrlimit(resource.RLIMIT_AS)[0]
if limit != resource.RLIM_INFINITY and limit < 40 * 1024**3:
    raise SystemExit('Inherited address-space limit below 40 GiB')
folder = Path(sys.argv[1]) / 'local_run/e5_forward_budget_v1_20261011'
data = (folder / 'package_manifest.json').read_bytes()
if hashlib.sha256(data).hexdigest() != sys.argv[2]:
    raise SystemExit('Budget package manifest mismatch')
for name, expected in json.loads(data)['files'].items():
    if Path(name).name != name:
        raise SystemExit('Unsafe package member')
    data = (folder / name).read_bytes()
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise SystemExit('Budget package member mismatch')
PY
"$budget_python" -m unittest discover -s "$repo/local_run/e5_forward_budget_v1_20261011" -p test_budget.py
# Time the entire child, including all output finalization and interpreter exit.
# This small envelope writes only its own completion receipt after the child.
"$budget_python" - "$repo" "$workspace" "$view" "$replay_result" "$census_dir" "$run_root" "$package_sha256" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
repo, workspace, view, replay, census_dir, run_root, package = sys.argv[1:]
command = [sys.executable, str(Path(repo) / 'local_run/e5_forward_budget_v1_20261011/budget.py'),
    '--repo', repo, '--workspace', workspace, '--view', view, '--replay-result', replay,
    '--census-dir', census_dir, '--out', str(Path(run_root) / 'budget'), '--package-sha256', package]
start = time.perf_counter()
process = subprocess.run(command, stdin=subprocess.DEVNULL, check=False)
elapsed = time.perf_counter() - start
receipt = {'schema': 'third-e5-budget-process-completion-v1', 'exit_code': process.returncode,
    'complete_child_wall_seconds_including_final_result_write_and_exit': elapsed,
    'command': command, 'timing_excludes_this_small_envelope_receipt_write': True,
    'primary_input_acceptance': False, 'full_population_cache_acceptance': False,
    'confirmatory_study_ready': False}
with (Path(run_root) / 'process_completion.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
raise SystemExit(process.returncode)
PY
