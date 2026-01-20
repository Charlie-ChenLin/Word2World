#!/bin/bash
#SBATCH --job-name=alfworld-debug
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=./log/alfworld-debug/%j/output.out
#SBATCH --error=./log/alfworld-debug/%j/error.err

set -euo pipefail
cd /home/lchen/home/Word2World

mkdir -p ./log/alfworld-debug/${SLURM_JOB_ID}

# Configurable via env vars when sbatching:
#   START_ID   : lowest game id to scan (default 0)
#   END_ID     : highest game id to scan (default 3001)
#   DATA_PATH  : path to ~/.cache/alfworld data
#   CONFIG_PATH: path to AgentGym/agentenv-alfworld/configs/base_config.yaml
#   SHARDS     : split range into N shards (use with --array); default: array size or 1
START_ID=${START_ID:-0}
END_ID=${END_ID:-3001}
DATA_PATH=${DATA_PATH:-$HOME/.cache/alfworld}
CONFIG_PATH=${CONFIG_PATH:-AgentGym/agentenv-alfworld/configs/base_config.yaml}
# If SHARDS not set, auto-detect from array length; fallback 1
if [ -z "${SHARDS:-}" ]; then
  if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ -n "${SLURM_ARRAY_TASK_MIN:-}" ]; then
    SHARDS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
  else
    SHARDS=1
  fi
fi

# If launched as SLURM array, compute shard-specific start/end
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_LEN=$((END_ID - START_ID + 1))
CHUNK_SIZE=$(( (TOTAL_LEN + SHARDS - 1) / SHARDS ))
SHARD_START=$(( START_ID + SHARD_ID * CHUNK_SIZE ))
SHARD_END=$(( SHARD_START + CHUNK_SIZE - 1 ))
if [ ${SHARD_END} -gt ${END_ID} ]; then SHARD_END=${END_ID}; fi
if [ ${SHARD_START} -gt ${END_ID} ]; then
  echo "Shard ${SHARD_ID} out of range; nothing to do."
  exit 0
fi

source ../uv_alfworld/bin/activate

LOG_DIR=./log/alfworld-debug/${SLURM_JOB_ID}

cmd=(python scripts/debug/scan_alfworld_games.py)

if [ -n "${GAME_IDS:-}" ]; then
  echo "Explicit GAME_IDS=${GAME_IDS}" | tee "${LOG_DIR}/range.txt"
  cmd+=("--ids" "${GAME_IDS}")
else
  echo "Shard ${SHARD_ID}/${SHARDS} scanning game_id [${SHARD_START}, ${SHARD_END}]" \
    | tee "${LOG_DIR}/range.txt"
  cmd+=("--start" "${SHARD_START}" "--end" "${SHARD_END}")
fi

cmd+=("--data_path" "${DATA_PATH}" "--config_path" "${CONFIG_PATH}")
if [ "${RAISE_ON_ERROR:-0}" != "0" ]; then
  cmd+=("--raise_on_error")
fi

"${cmd[@]}" | tee "${LOG_DIR}/scan.log"
