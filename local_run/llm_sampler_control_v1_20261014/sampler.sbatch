#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
control=$1
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/encoder-cpu/bin/python
# Verify the complete externally pinned control package before importing runner.
"$python" - "$control" "$@" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);args=sys.argv[2:]
assert args.count('--control-package-sha256')==1
pin=args[args.index('--control-package-sha256')+1]
raw=(folder/'package_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==pin
for name,wanted in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    path=folder/name;assert path.is_file() and not path.is_symlink()
    data=path.read_bytes();assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==wanted
VERIFY
cd "$control"
"$python" runner.py "$@"
