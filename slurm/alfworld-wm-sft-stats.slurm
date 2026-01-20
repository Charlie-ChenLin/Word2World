#!/bin/bash
#SBATCH --job-name=alfworld-wm-sft-stats
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=./log/alfworld-wm-sft-stats/%j/output.out
#SBATCH --error=./log/alfworld-wm-sft-stats/%j/error.err

set -euo pipefail
cd /home/lchen/home/Word2World

mkdir -p ./log/alfworld-wm-sft-stats/${SLURM_JOB_ID}
mkdir -p ./debug

# Configurable via env vars when sbatching:
#   INPUT_FILE        : WM SFT data path (default: data/llama_factory/alfworld_train_with_env_54006.json)
#   AGENT_INSTRUCT    : agent init file used for alignment (default: data/init_contexts/alfworld/agent_instruct_train.json)
#   OUTPUT_FILE       : output stats path (default: debug/alfworld_wm_sft_task_stats.json)
INPUT_FILE="${INPUT_FILE:-data/llama_factory/alfworld_train_with_env_54006.json}"
AGENT_INSTRUCT="${AGENT_INSTRUCT:-data/init_contexts/alfworld/agent_instruct_train.json}"
OUTPUT_FILE="${OUTPUT_FILE:-debug/alfworld_wm_sft_task_stats.json}"

python scripts/analysis/alfworld_wm_sft_stats.py \
  --input "${INPUT_FILE}" \
  --agent_instruct "${AGENT_INSTRUCT}" \
  --output "${OUTPUT_FILE}" \
  | tee "./log/alfworld-wm-sft-stats/${SLURM_JOB_ID}/run.log"
