#!/usr/bin/env bash
#SBATCH --job-name=third-score-bench
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
if [[ $# -ne 11 ]]; then
  echo 'Expected package_folder repo workspace view replay census budget qualification_result repair_result new_stage_root package_sha256' >&2
  exit 2
fi
folder=$1
repo=$2
workspace=$3
stage=${10}
package_sha=${11}
python="$workspace/envs/encoder-cpu/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
export THIRD_BENCHMARK_REPO="$repo" THIRD_BENCHMARK_QUALIFICATION="$(dirname "$8")"
# Before importing any benchmark/test code, verify the externally pinned package.
"$python" - "$folder" "$package_sha" <<'PY_VERIFY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1]);manifest=folder/'package_manifest.json'
if manifest.is_symlink() or not manifest.is_file():raise SystemExit('Regular manifest required')
raw=manifest.read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[2]:raise SystemExit('Package manifest mismatch')
files=json.loads(raw)['files']
if not files:raise SystemExit('Empty package')
for name,wanted in files.items():
    if Path(name).name!=name or name in ('.','..','package_manifest.json'):raise SystemExit('Unsafe member')
    p=folder/name
    if p.is_symlink() or not p.is_file():raise SystemExit('Regular member required')
    raw=p.read_bytes()
    if len(raw)!=wanted['bytes'] or hashlib.sha256(raw).hexdigest()!=wanted['sha256']:raise SystemExit('Member mismatch')
PY_VERIFY
cd "$folder"
"$python" - "$folder" "$stage" "$repo" "$workspace" "$4" "$5" "$6" "$7" "$8" "$9" "$package_sha" <<'PY_RUN'
import json,os,sys,time,unittest,subprocess
from pathlib import Path
import benchmark as b
folder,stage=Path(sys.argv[1]),Path(sys.argv[2])
b.new_output(stage,sys.argv[4])
start=time.perf_counter()
suite=unittest.defaultTestLoader.discover(str(folder),pattern='test_benchmark.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
validation={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'elapsed_seconds':time.perf_counter()-start,'passed':result.wasSuccessful() and result.testsRun==16 and not result.skipped}
b.publish(stage/'software_validation.json',validation)
if not validation['passed']:raise SystemExit('Cluster software checks failed')
command=[sys.executable,str(folder/'benchmark.py')]
for name,value in zip(('repo','workspace','view','replay-result','census-dir','budget-dir','qualification-result','repair-result'),sys.argv[3:11]):command+=['--'+name,value]
command+=['--out',str(stage/'benchmark'),'--benchmark-package-sha256',sys.argv[11]]
with (stage/'stdout.txt').open('x') as out,(stage/'stderr.txt').open('x') as err:
    start=time.perf_counter()
    process=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    elapsed_through_exit=time.perf_counter()-start
    for stream in (out,err):stream.flush();os.fsync(stream.fileno())
b.publish(stage/'process_completion.json',{'command':command,'returncode':process.returncode,'child_wall_seconds_through_exit':elapsed_through_exit,'package_sha256':sys.argv[11]})
raise SystemExit(process.returncode)
PY_RUN
