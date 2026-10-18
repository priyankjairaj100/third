#!/usr/bin/env bash
#SBATCH --partition=cpu-short
#SBATCH --exclude=csis.mn1,csis.cn2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:05:00
#SBATCH --propagate=NONE
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
umask 077
ulimit -f 2048
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
package=$1
pin=$2
runroot=$3
cd "$package"
exec /nfs_home/users/poonam/ccu-acl2027/local_workspace/envs/encoder-cpu/bin/python -B job_runner.py "$pin" "$runroot"
