#!/usr/bin/env bash
#SBATCH --job-name=third-source-replay
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
view=$3
lineage_result=$4
lineage_result_sha256=$5
lineage_package_sha256=$6
run_root=$7
replay_python=$8
replay_package_sha256=$9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$run_root"
"$replay_python" - "$repo" "$replay_package_sha256" <<'PY'
import hashlib, json, os, platform, resource, socket, sqlite3, sys
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID') or platform.system() != 'Linux':
    raise SystemExit('Linux Slurm allocation required')
if socket.gethostname().split('.')[:2] == ['csis', 'mn1']:
    raise SystemExit('Login node prohibited')
limit = resource.getrlimit(resource.RLIMIT_AS)[0]
if limit != resource.RLIM_INFINITY and limit < 40 * 1024**3:
    raise SystemExit('Inherited address-space limit below 40 GiB')
folder = Path(sys.argv[1]) / 'local_run/source_replay_v1_20261011'
data = (folder / 'package_manifest.json').read_bytes()
if hashlib.sha256(data).hexdigest() != sys.argv[2]:
    raise SystemExit('Replay package manifest mismatch')
manifest = json.loads(data)
for name, wanted in manifest['files'].items():
    if Path(name).name != name:
        raise SystemExit('Unsafe package member')
    data = (folder / name).read_bytes()
    if len(data) != wanted['bytes'] or hashlib.sha256(data).hexdigest() != wanted['sha256']:
        raise SystemExit('Replay package member mismatch')
import numpy, idna
print('replay_environment', platform.python_version(), 'sqlite', sqlite3.sqlite_version, 'numpy', numpy.__version__, 'idna', idna.__version__, flush=True)
PY
"$replay_python" -m unittest discover -s "$repo/local_run/source_replay_v1_20261011" -p test_replay.py
"$replay_python" "$repo/local_run/source_replay_v1_20261011/replay_source.py" \
  --repo "$repo" --workspace "$workspace" --view "$view" --out "$run_root/replay" \
  --lineage-result "$lineage_result" --lineage-result-sha256 "$lineage_result_sha256" \
  --lineage-package-sha256 "$lineage_package_sha256" \
  --replay-package-sha256 "$replay_package_sha256"
