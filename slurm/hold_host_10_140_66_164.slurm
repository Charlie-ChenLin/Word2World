#!/usr/bin/env bash
#SBATCH -J hold_host_10_140_66_164
#SBATCH -p Gveval2-T
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=HOST-10-140-66-164
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o /mnt/petrelfs/chenlin1/slurm-%x-%j.out
#SBATCH -e /mnt/petrelfs/chenlin1/slurm-%x-%j.err

set -euo pipefail

echo "[$(date)] job start: ${SLURM_JOB_ID-unknown}"
echo "[$(date)] host: $(hostname)"
echo "[$(date)] cwd: $(pwd)"

# Hold the node.
sleep 7200

echo "[$(date)] job end: ${SLURM_JOB_ID-unknown}"
