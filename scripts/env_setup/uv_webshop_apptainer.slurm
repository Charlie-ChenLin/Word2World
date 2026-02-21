#!/usr/bin/env bash
#SBATCH -J uv_webshop_setup
#SBATCH -p Gveval2-T
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o /mnt/petrelfs/chenlin1/slurm-%x-%j.out
#SBATCH -e /mnt/petrelfs/chenlin1/slurm-%x-%j.err

set -euo pipefail

echo "[$(date)] job start: ${SLURM_JOB_ID-unknown}"
echo "[$(date)] host: $(hostname)"
echo "[$(date)] cwd: $(pwd)"

export FLASH_ATTN_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-24}"
export FLASH_ATTN_NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS:-4}"
echo "Using FLASH_ATTN_BUILD_JOBS=${FLASH_ATTN_BUILD_JOBS}"

bash /mnt/petrelfs/chenlin1/projects/Word2World/scripts/env_setup/uv_webshop_apptainer.sh

echo "[$(date)] job end: ${SLURM_JOB_ID-unknown}"
