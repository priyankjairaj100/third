#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --nodelist=csis.cn1
#SBATCH --exclude=csis.mn1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
package=$1
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=''
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/judge-qwen-v3-20261011/bin/python
"$python" -B - "$package" "$@" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
folder=Path(sys.argv[1]);args=sys.argv[2:]
assert args.count('--package-sha256')==1
pin=args[args.index('--package-sha256')+1]
raw=(folder/'package_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
for name,b in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    p=folder/name;assert p.is_file() and not p.is_symlink();x=p.read_bytes()
    assert {'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}==b
os.chdir(folder);os.execv(sys.executable,[sys.executable,'-B',str(folder/'runner.py'),*args])
PY
