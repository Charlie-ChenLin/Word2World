#!/bin/bash
#SBATCH --job-name alfworld-eval
#SBATCH --partition=L40S,ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128      # 默认给后续 srun
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=./log/alfworld/%j/output.out
#SBATCH --error=./log/alfworld/%j/error.err

set -euo pipefail
cd /home/lchen/home/Word2World
mkdir -p ./log/alfworld/${SLURM_JOB_ID}

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
export PYDEV_PREFIX=~/local/python310-dev
# Include root plus both python include subdirs so pyconfig.h can be found.
export CPATH=$PYDEV_PREFIX/usr/include:$PYDEV_PREFIX/usr/include/python3.10:$PYDEV_PREFIX/usr/include/x86_64-linux-gnu/python3.10:${CPATH:-}
export LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Eval configs (override via env or sbatch --export)
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-AgentGym-RL/saves/test-l40s}"
export CKPT_STEP="${CKPT_STEP:-175}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export DATA_ROOT="${DATA_ROOT:-data}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-200}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.95}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${MODEL_SAVE_PATH}/global_step_${CKPT_STEP}/eval_logs}"

mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${ROLLOUT_LOG_DIR}"

# 环境服务器（少量 CPU，0 GPU）
srun --ntasks=1 --cpus-per-task=20 --gres=gpu:0 --overlap \
     --export=ALL,CUDA_VISIBLE_DEVICES= \
     bash scripts/env_server/start_alfworld.sh \
     > ./log/alfworld/${SLURM_JOB_ID}/env_server.log 2>&1 &
server_pid=$!

sleep 5

# 评测（4 GPU）
srun --ntasks=1 --cpus-per-task=100 --gres=gpu:4 \
     --cpu-bind=cores \
     bash examples/eval/alfworld_eval.sh \
     --n_gpus_per_node "${N_GPUS_PER_NODE}" \
     --model_save_path "${MODEL_SAVE_PATH}" \
     --ckpt_step "${CKPT_STEP}" \
     --data_root "${DATA_ROOT}" \
     --max_response_length "${MAX_RESPONSE_LEN}" \
     --rollout_max_model_len "${ROLLOUT_MAX_MODEL_LEN}" \
     --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
     --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL}" \
     --rollout_tp_size "${ROLLOUT_TP_SIZE}" \
     --batch_size "${BATCH_SIZE}" \
     --rollout_log_dir "${ROLLOUT_LOG_DIR}" \
     > ./log/alfworld/${SLURM_JOB_ID}/eval.log 2>&1

eval_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $eval_status
