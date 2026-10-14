#!/usr/bin/env bash
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --nodelist=csis.cn4
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/judge-qwen-v3-20261011/bin/python -B "$1/probe.py" "$2" "$3/run"
