#!/bin/bash
# Eval missing steps for test-l40s-env-0-20260119-201111
# Steps: 75, 100, 125, 150

#SBATCH --job-name alfworld-eval-missing
#SBATCH --partition=L40S,ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=./log/alfworld/alfworld-eval-missing-%j.out
#SBATCH --error=./log/alfworld/alfworld-eval-missing-%j.err

set -euo pipefail
cd /home/lchen/home/Word2World

mkdir -p ./log/alfworld
JOB_DIR="./log/alfworld/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
export PYDEV_PREFIX=~/local/python310-dev
export CPATH=$PYDEV_PREFIX/usr/include:$PYDEV_PREFIX/usr/include/python3.10:$PYDEV_PREFIX/usr/include/x86_64-linux-gnu/python3.10:${CPATH:-}
export LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Eval configs (override via env or sbatch --export)
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-AgentGym-RL/saves/test-l40s-env-0-20260119-201111}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export DATA_ROOT="${DATA_ROOT:-data}"
export SAMPLE_NUM="${SAMPLE_NUM:-1}"
export MAX_ROUNDS="${MAX_ROUNDS:-30}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-200}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.9}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"

export ENV_SERVER_PORT="${ENV_SERVER_PORT:-$((36001 + (SLURM_JOB_ID % 1000)))}"
export ENV_SERVER_URL="http://127.0.0.1:${ENV_SERVER_PORT}"

STEPS=(75 100 125 150)
RESULTS_TSV="${JOB_DIR}/results.tsv"

mkdir -p "${TRITON_CACHE_DIR}"
echo -e "model\tstep\tavg\tpass\tstatus" > "${RESULTS_TSV}"

# Start env server (CPU only) on a per-job port to avoid conflicts
srun --ntasks=1 --cpus-per-task=12 --gres=gpu:0 --overlap \
     --export=ALL,CUDA_VISIBLE_DEVICES= \
     bash -lc "source /home/lchen/home/uv_alfworld/bin/activate && alfworld --host 0.0.0.0 --port ${ENV_SERVER_PORT}" \
     > "${JOB_DIR}/env_server.log" 2>&1 &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true; wait ${server_pid} 2>/dev/null || true' EXIT

sleep 5

for step in "${STEPS[@]}"; do
    run_tag="test-l40s-env-0-20260119-201111_step_${step}"
    run_dir="${JOB_DIR}/runs/${run_tag}"
    mkdir -p "${run_dir}"
    eval_log="${run_dir}/eval.log"
    rollout_log_dir="${MODEL_SAVE_PATH}/global_step_${step}/eval_logs/${RUN_TS}"
    mkdir -p "${rollout_log_dir}"

    if srun --ntasks=1 --cpus-per-task=48 --gres=gpu:${N_GPUS_PER_NODE} \
            --cpu-bind=cores \
            bash examples/eval/alfworld_eval.sh \
            --n_gpus_per_node "${N_GPUS_PER_NODE}" \
            --model_save_path "${MODEL_SAVE_PATH}" \
            --ckpt_step "${step}" \
            --data_root "${DATA_ROOT}" \
            --env_server_url "${ENV_SERVER_URL}" \
            --sample_num "${SAMPLE_NUM}" \
            --max_rounds "${MAX_ROUNDS}" \
            --max_prompt_length "${MAX_PROMPT_LEN}" \
            --max_response_length "${MAX_RESPONSE_LEN}" \
            --rollout_max_model_len "${ROLLOUT_MAX_MODEL_LEN}" \
            --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
            --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL}" \
            --rollout_tp_size "${ROLLOUT_TP_SIZE}" \
            --batch_size "${BATCH_SIZE}" \
            --rollout_log_dir "${rollout_log_dir}" \
            > "${eval_log}" 2>&1; then
        avg=$(awk -F': ' '/Avg@'\"${SAMPLE_NUM}\"':/{print $2; exit}' "${eval_log}" | tr -d ' ')
        pas=$(awk -F': ' '/Pass@'\"${SAMPLE_NUM}\"':/{print $2; exit}' "${eval_log}" | tr -d ' ')
        status="OK"
        if [ -z "${avg}" ] || [ -z "${pas}" ]; then
            status="PARSE_FAIL"
        fi
    else
        status="EVAL_FAIL"
        avg=""
        pas=""
    fi

    echo -e "test-l40s-env-0-20260119-201111\t${step}\t${avg}\t${pas}\t${status}" >> "${RESULTS_TSV}"
done

printf "\nDone. Results: %s\n" "${RESULTS_TSV}"
