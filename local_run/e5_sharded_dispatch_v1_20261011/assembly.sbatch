#!/usr/bin/env bash
#SBATCH --job-name=third-e5-assembly
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
if [[ $# -ne 13 ]]; then
  echo 'Expected 13 positional arguments; see dispatch README.' >&2
  exit 2
fi
repo=$1
workspace=$2
core_sha=${7}
dispatch_sha=${!#}
python="$workspace/envs/encoder-cpu/bin/python"
folder="$repo/local_run/e5_sharded_dispatch_v1_20261011"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
"$python" - "$folder" "$dispatch_sha" "$repo/local_run/e5_sharded_cpu_v1_20261011" "$core_sha" <<'PY_VERIFY'
import hashlib,json,sys
from pathlib import Path
for directory,wanted in ((sys.argv[1],sys.argv[2]),(sys.argv[3],sys.argv[4])):
    folder=Path(directory)
    raw=(folder/'package_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=wanted:
        raise SystemExit('Sealed package manifest mismatch')
    files=json.loads(raw)['files']
    if not files:
        raise SystemExit('Empty sealed package')
    for name,expected in files.items():
        if Path(name).name!=name or name in ('.','..'):
            raise SystemExit('Unsafe package member')
        path=folder/name
        if path.is_symlink() or not path.is_file():
            raise SystemExit('Regular package member required')
        raw=path.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise SystemExit('Sealed package member mismatch')
PY_VERIFY
exec "$python" "$folder/dispatch.py" assembly "$@"
