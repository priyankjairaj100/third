#!/usr/bin/env bash
#SBATCH --job-name=third-scipy-import
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
  echo 'Expected: repo workspace new_output_directory package_sha256' >&2
  exit 2
fi
repo=$1
workspace=$2
out=$3
package_sha=$4
python="$workspace/envs/encoder-cpu/bin/python"
folder="$repo/local_run/e5_scipy_import_repair_v1_20261011"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=''
"$python" - "$folder" "$package_sha" <<'PY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);raw=(folder/'package_manifest.json').read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[2]:
    raise SystemExit('Repair package manifest mismatch')
for name,expected in json.loads(raw)['files'].items():
    if Path(name).name!=name or name in ('.','..'):
        raise SystemExit('Unsafe package member')
    path=folder/name
    if path.is_symlink() or not path.is_file():
        raise SystemExit('Regular package member required')
    raw=path.read_bytes()
    if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
        raise SystemExit('Repair package member mismatch')
PY
exec "$python" "$folder/repair.py" --repo "$repo" --workspace "$workspace" \
  --out "$out" --package-sha256 "$package_sha"
