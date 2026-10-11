#!/usr/bin/env bash
#SBATCH --job-name=third-qwen-v5
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
model=$3
run_root=$4
judge_python=$5
prior_run=$6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export THIRD_TEST_TOKENIZER_PATH="$model"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
"$judge_python" - <<'PY'
import resource, os, socket
limit = resource.getrlimit(resource.RLIMIT_AS)
assert limit[0] == resource.RLIM_INFINITY or limit[0] >= 32 * 1024**3, limit
assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
assert socket.gethostname() not in ('csis.mn1', 'csis.mn1.local'), 'Login node prohibited'
print('memory_preflight', limit, 'node', socket.gethostname(), flush=True)
PY
"$judge_python" - "$repo/local_run/model_annotation_runtime_v5_20261011" > "$run_root/software_tests.out" 2> "$run_root/software_tests.err" <<'PYTEST'
import sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern='test_runner.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 44 and not result.skipped else 1)
PYTEST
"$judge_python" "$repo/local_run/model_annotation_runtime_v5_20261011/prepare_pilot.py" \
  --repo "$repo" --workspace "$workspace" --model-path "$model" \
  --output-dir "$run_root/preparation" --prior-run "$prior_run"
"$judge_python" "$repo/local_run/model_annotation_runtime_v5_20261011/run_pilot.py" \
  --manifest "$run_root/preparation/manifest.json" --output-dir "$run_root/inference"
"$judge_python" "$repo/local_run/model_annotation_runtime_v5_20261011/audit_pilot_artifacts.py" \
  "$run_root" --repo "$repo" --source-workspace "$workspace" --tokenizer-path "$model" \
  > "$run_root/independent_artifact_audit.stdout.json"
