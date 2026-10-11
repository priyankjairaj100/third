#!/usr/bin/env bash
#SBATCH --job-name=third-mistral-v1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
set -o noclobber
umask 077
repo=$1
workspace=$2
model=$3
run_root=$4
judge_python=$5
prior_run=$6
runtime_plan_sha256=$7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export THIRD_TEST_TOKENIZER_PATH="$model"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$run_root"
# Authenticate every package member before importing its tests or preparation.
"$judge_python" - "$repo" "$workspace" "$run_root" "$runtime_plan_sha256" <<'PY'
import hashlib, importlib.metadata, json, os, platform, resource, socket, sys
from pathlib import Path
repo, workspace, run_root = map(lambda p: Path(p).resolve(), sys.argv[1:4])
expected_plan = sys.argv[4]
if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Linux Slurm allocation required')
if socket.gethostname().split('.')[:2] == ['csis', 'mn1']:
    raise SystemExit('Login node prohibited')
limit = resource.getrlimit(resource.RLIMIT_AS)[0]
if limit != resource.RLIM_INFINITY and limit < 56 * 1024**3:
    raise SystemExit('Inherited address-space limit below 56 GiB')
if not run_root.is_relative_to(workspace) or not run_root.is_dir() or run_root.stat().st_mode & 0o077:
    raise SystemExit('Existing private0700 run root under workspace required')
for name in ('preparation', 'inference', 'software_tests.out', 'software_tests.err',
             'independent_artifact_audit.json', 'independent_artifact_audit.stdout.json'):
    if (run_root / name).exists():
        raise SystemExit('Run output already exists: ' + name)
folder = repo / 'local_run/model_annotation_runtime_mistral_v1_20261011'
def checked(path, expected):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Package SHA256 mismatch: ' + str(path))
checked(folder / 'runtime_plan.json', expected_plan)
plan = json.loads((folder / 'runtime_plan.json').read_text())
if plan.get('runtime_plan_schema') != 'third-judge-runtime-amendment-mistral-v1' or plan.get('status') == 'prospective_unsealed':
    raise SystemExit('Sealed Mistral runtime amendment required')
required = {'run_pilot.py', 'prepare_pilot.py', 'bounded_reason.py', 'test_runner.py',
            'audit_pilot_artifacts.py', 'pilot.sbatch', 'requirements.txt', 'asset_expectations.json'}
if not required.issubset(plan['package_files']):
    raise SystemExit('Incomplete launch package bindings')
for name, digest in plan['package_files'].items():
    if Path(name).name != name:
        raise SystemExit('Unsafe package member')
    checked(folder / name, digest)
for version in ('v2', 'v3', 'v4', 'v5'):
    inventory = plan['original_' + version + '_folder_files']
    if not inventory:
        raise SystemExit('Empty preserved package inventory')
    for name, digest in inventory.items():
        if Path(name).name != name:
            raise SystemExit('Unsafe preserved member')
        checked(folder.parent / ('model_annotation_runtime_' + version + '_20261011') / name, digest)
if plan['package_files']['bounded_reason.py'] != plan['original_v5_folder_files']['bounded_reason.py']:
    raise SystemExit('Reused v5 guard changed')
for name, version in plan['required_packages'].items():
    if importlib.metadata.version(name) != version:
        raise SystemExit('Runtime package version mismatch: ' + name)
print('sealed_package_preflight_passed', expected_plan, 'node', socket.gethostname(), flush=True)
PY
"$judge_python" - "$repo/local_run/model_annotation_runtime_mistral_v1_20261011" > "$run_root/software_tests.out" 2> "$run_root/software_tests.err" <<'PYTEST'
import sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern='test_runner.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 46 and not result.skipped else 1)
PYTEST
"$judge_python" "$repo/local_run/model_annotation_runtime_mistral_v1_20261011/prepare_pilot.py" \
  --repo "$repo" --workspace "$workspace" --model-path "$model" \
  --output-dir "$run_root/preparation" --prior-run "$prior_run"
"$judge_python" "$repo/local_run/model_annotation_runtime_mistral_v1_20261011/run_pilot.py" \
  --manifest "$run_root/preparation/manifest.json" --output-dir "$run_root/inference"
"$judge_python" "$repo/local_run/model_annotation_runtime_mistral_v1_20261011/audit_pilot_artifacts.py" \
  "$run_root" --repo "$repo" --source-workspace "$workspace" --tokenizer-path "$model" \
  > "$run_root/independent_artifact_audit.stdout.json"
