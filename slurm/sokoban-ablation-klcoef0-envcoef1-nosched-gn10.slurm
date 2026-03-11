#!/bin/bash
#SBATCH --job-name=soko-abl-kl0-e1
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/inspurfs/chenlin1/log/sokoban/%j.out
#SBATCH --error=/mnt/inspurfs/chenlin1/log/sokoban/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "${REPO_ROOT}"

export PROJECT_NAME="${PROJECT_NAME:-word2world-sokoban-loss-ablation}"
export EXP_NAME="${EXP_NAME:-sokoban-verl-agent-low-kl-small-minibatch-ablation-klcoef0-envcoef1-nosched-gn10-task}"
export ENV_FEEDBACK_LOSS_COEF="${ENV_FEEDBACK_LOSS_COEF:-1}"
export FINAL_ENV_FEEDBACK_LOSS_COEF="${FINAL_ENV_FEEDBACK_LOSS_COEF:-1}"
export SCHEDULE_ENV_FEEDBACK_LOSS_COEF="${SCHEDULE_ENV_FEEDBACK_LOSS_COEF:-false}"
export KL_COEF="${KL_COEF:-0}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
export GRAD_CLIP="${GRAD_CLIP:-10}"

bash "${REPO_ROOT}/slurm/sokoban-verl-agent-low-kl-small-minibatch-a800-apptainer-envcoef1-nosched-gn10-wandb-task.slurm"
