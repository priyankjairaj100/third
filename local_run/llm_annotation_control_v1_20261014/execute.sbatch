#!/usr/bin/env bash
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
control=$1
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/judge-qwen-v3-20261011/bin/python
"$python" -B - "$control" "$@" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
folder=Path(sys.argv[1]);args=sys.argv[2:]
assert args.count('--control-package-sha256')==1 and args.count('--mode')==1
pin=args[args.index('--control-package-sha256')+1]
raw=(folder/'package_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin
for name,b in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    p=folder/name;assert p.is_file() and not p.is_symlink();x=p.read_bytes()
    assert {'bytes':len(x),'sha256':hashlib.sha256(x).hexdigest()}==b
if args[args.index('--mode')+1]=='qualification':os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir(folder)
os.execv(sys.executable,[sys.executable,'-B',str(folder/'runner.py'),*args])
PY
