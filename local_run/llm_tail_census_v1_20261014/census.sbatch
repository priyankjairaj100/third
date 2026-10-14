#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
package=$1
pin=$2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/encoder-cpu/bin/python
"$python" - "$package" "$pin" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'package_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
for name,b in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    f=p/name;assert f.is_file() and not f.is_symlink()
    data=f.read_bytes();assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==b
VERIFY
cd "$package"
"$python" -m unittest -v test_census
"$python" census.py --package-sha256 "$pin"
