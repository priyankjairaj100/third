#!/usr/bin/env bash
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
package=$1
shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python=/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/judge-qwen-v3-20261011/bin/python
"$python" -B - "$package" "$@" <<'VERIFY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);args=sys.argv[2:]
assert args.count('--package-sha256')==1 and args.count('--mode')==1
pin=args[args.index('--package-sha256')+1];raw=(p/'package_manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==pin
for name,b in json.loads(raw)['files'].items():
    assert Path(name).name==name and name!='package_manifest.json'
    f=p/name;assert f.is_file() and not f.is_symlink()
    data=f.read_bytes();assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==b
if args[args.index('--mode')+1] not in ('qwen','mistral'):os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir(p)
os.execv(sys.executable,[sys.executable,'-B',str(p/'worker.py'),*args])
VERIFY
