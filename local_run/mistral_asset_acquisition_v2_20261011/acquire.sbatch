#!/bin/bash
#SBATCH --job-name=third-mistral-assets-v2
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
run_root=$3
package_sha=$4
export PYTHONDONTWRITEBYTECODE=1
folder="$repo/local_run/mistral_asset_acquisition_v2_20261011"
python="$workspace/envs/judge-qwen-v3-20261011/bin/python"
"$python" - "$folder" "$package_sha" <<'PY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);raw=(folder/'package_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
for name,expected in json.loads(raw)['files'].items():
    assert Path(name).name==name and name not in ('','.','..')
    raw=(folder/name).read_bytes()
    assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
PY
"$python" -B -m unittest discover -s "$folder" -p test_quota.py
"$python" "$folder/acquire.py" --expectations "$repo/local_run/model_annotation_runtime_mistral_v1_20261011/asset_expectations.json" \
  --cache "$workspace/models/mistral-nemo-hub-v2" --out "$run_root/acquisition" --package-sha256 "$package_sha"
