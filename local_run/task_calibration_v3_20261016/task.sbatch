#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --nodelist=csis.cn1
#SBATCH --exclude=csis.mn1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --propagate=NONE
set -euo pipefail
umask 077
[[ $# -eq 2 ]] || exit 2
package=$1
pin=$2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$package"
exec /nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/encoder-cpu/bin/python -B runner.py "$pin"
