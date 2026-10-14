#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
[[ $# -eq 5 ]] || exit 2
control=$1
root=$2
control_sha=$3
frame_sha=$4
tests=$5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/encoder-cpu/bin/python
"$python" - "$control" "$control_sha" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);raw=(folder/'package_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
for name,wanted in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    p=folder/name;assert p.is_file() and not p.is_symlink()
    data=p.read_bytes();assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==wanted
VERIFY
cd "$control"
"$python" runner.py --root "$root" --control-package-sha256 "$control_sha" --frame-package-sha256 "$frame_sha" --expected-tests "$tests"
