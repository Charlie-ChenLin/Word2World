#!/bin/bash
#SBATCH --job-name=webshop-eval-all
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-eval-all-%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-eval-all-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/log/webshop_eval"
JOB_DIR="${REPO_ROOT}/log/webshop_eval/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"

export MODEL_ROOT="${MODEL_ROOT:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves}"
export MODEL_FILTER="${MODEL_FILTER:-webshop}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export OUT_DIR="${OUT_DIR:-${JOB_DIR}}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export SAMPLE_NUM="${SAMPLE_NUM:-1}"
export MAX_ROUNDS="${MAX_ROUNDS:-30}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.8}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-256}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"

# Ensure apptainer can see inspurfs ckpts
export APPTAINER_BIND_EXTRA="${APPTAINER_BIND_EXTRA:-/mnt/inspurfs/chenlin1}"

bash examples/eval/webshop_eval_all_apptainer.sh \
  --model_root "${MODEL_ROOT}" \
  --model_filter "${MODEL_FILTER}" \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --n_gpus_per_node "${N_GPUS_PER_NODE}" \
  --sample_num "${SAMPLE_NUM}" \
  --max_rounds "${MAX_ROUNDS}" \
  --max_prompt_length "${MAX_PROMPT_LEN}" \
  --max_response_length "${MAX_RESPONSE_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL}" \
  --rollout_temperature "${ROLLOUT_TEMPERATURE}" \
  --rollout_max_model_len "${ROLLOUT_MAX_MODEL_LEN}" \
  --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
  --rollout_tp_size "${ROLLOUT_TP_SIZE}"
