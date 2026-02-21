#!/usr/bin/env bash
#SBATCH -J uv_sciworld_setup
#SBATCH -p Gveval2-T
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o /mnt/petrelfs/chenlin1/slurm-%x-%j.out
#SBATCH -e /mnt/petrelfs/chenlin1/slurm-%x-%j.err

set -euo pipefail

echo "[$(date)] job start: ${SLURM_JOB_ID-unknown}"
echo "[$(date)] host: $(hostname)"
echo "[$(date)] cwd: $(pwd)"

bash /mnt/petrelfs/chenlin1/projects/Word2World/scripts/env_setup/uv_sciworld_apptainer.sh

echo "[$(date)] job end: ${SLURM_JOB_ID-unknown}"
