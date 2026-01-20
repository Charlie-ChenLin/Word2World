#!/bin/bash
#SBATCH --job-name alfworld
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96       # 机上总计 96 核
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=./log/alfworld/%j/output.out
#SBATCH --error=./log/alfworld/%j/error.err

set -euo pipefail
cd /home/lchen/home/Word2World
mkdir -p ./log/alfworld/${SLURM_JOB_ID}

export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-/data1/public/hf/Qwen/${PURE_AGENT_MODEL_NAME}}"
# export CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.1}"
export CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
# export DS_BUILD_OPS=0
# export DS_SKIP_CUDA_CHECK=1
# export TRITON_BUILD_WITH_CLANG=0
# export VLLM_SKIP_TRTON_CUDA_BUILD=1   # vLLM 0.6.x 支持该变量跳过 triton 构建

export PYDEV_PREFIX=~/local/python310-dev
# Include root plus both python include subdirs so pyconfig.h can be found.
export CPATH=$PYDEV_PREFIX/usr/include:$PYDEV_PREFIX/usr/include/python3.10:$PYDEV_PREFIX/usr/include/x86_64-linux-gnu/python3.10:${CPATH:-}
export LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}



mkdir -p "${TRITON_CACHE_DIR}"

# 环境服务器（少量 CPU，0 GPU）
srun --ntasks=1 --cpus-per-task=12 --gres=gpu:0 --overlap \
     --export=ALL,CUDA_VISIBLE_DEVICES= \
     bash scripts/env_server/start_alfworld.sh \
     > ./log/alfworld/${SLURM_JOB_ID}/env_server.log 2>&1 &
server_pid=$!

sleep 5

# 训练（8 GPU + 88 核）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
srun --ntasks=1 --cpus-per-task=80 --gres=gpu:8 \
     --cpu-bind=cores \
     bash examples/train/AgentGym-RL/alfworld_train.sh \
     --n_gpus_per_node 8 \
     --ppo_mini_batch_size 2 \
     --rollout_gpu_mem_util 0.55 \
     --rollout_tp_size 4 \
     --max_response_length 4096 \
     --max_model_len 6144 \
     --pure_agent_model_name "${PURE_AGENT_MODEL_NAME}" \
     --agent_model_path "${AGENT_MODEL_PATH}" \
     > ./log/alfworld/${SLURM_JOB_ID}/train.log 2>&1
train_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $train_status
