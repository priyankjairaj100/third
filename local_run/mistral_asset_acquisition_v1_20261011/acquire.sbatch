#!/bin/bash
#SBATCH --job-name=third-mistral-assets
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
repo=$1
workspace=$2
run_root=$3
script_sha=$4
expectations_sha=$5
export PYTHONDONTWRITEBYTECODE=1
"$workspace/envs/judge-qwen-v3-20261011/bin/python" "$repo/local_run/mistral_asset_acquisition_v1_20261011/acquire.py" \
  --expectations "$repo/local_run/model_annotation_runtime_mistral_v1_20261011/asset_expectations.json" \
  --expectations-sha256 "$expectations_sha" --script-sha256 "$script_sha" \
  --cache "$workspace/models/mistral-nemo-hub-v1" --out "$run_root/acquisition"
