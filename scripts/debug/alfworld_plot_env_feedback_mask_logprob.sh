#!/usr/bin/env bash
set -euo pipefail
set -x

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_NO_USAGE_STATS=1

task_name="alfworld"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate env (same as examples/train/AgentGym-RL/alfworld_train.sh)
if [[ -f ../uv_agentgym_rl/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../uv_agentgym_rl/bin/activate
else
  echo "[alfworld_plot_env_feedback_mask_logprob.sh] WARN: ../uv_agentgym_rl/bin/activate not found; using current python env." 1>&2
fi

# Offline run (no internet; only connects to local env server)
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export WANDB_MODE=offline
export WANDB_SILENT=true
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1

# Keep wandb files under repo (easier to sync)
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-./wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-./wandb}"

env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36001}"

# Model path (copied from examples/train/AgentGym-RL/alfworld_train.sh; can override via AGENT_MODEL_PATH)
pure_agent_model_name="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
agent_model_path="${AGENT_MODEL_PATH:-/mnt/shared-storage-user/formalverification-shared/openai-community/Qwen/${pure_agent_model_name}}"

train_file="${DATA_ROOT:-data}/train/${task_name}_train.json"

if [[ ! -f "${agent_model_path}/config.json" ]]; then
  echo "[alfworld_plot_env_feedback_mask_logprob.sh] Missing local model at: ${agent_model_path}" 1>&2
  echo "[alfworld_plot_env_feedback_mask_logprob.sh] Set AGENT_MODEL_PATH to a local HuggingFace model directory." 1>&2
  exit 1
fi
if [[ ! -f "${train_file}" ]]; then
  echo "[alfworld_plot_env_feedback_mask_logprob.sh] Missing train file: ${train_file}" 1>&2
  echo "[alfworld_plot_env_feedback_mask_logprob.sh] Set DATA_ROOT (default: ./data)." 1>&2
  exit 1
fi

# Single-node defaults for debugging; override via env vars if needed.
n_gpus_per_node="${N_GPUS_PER_NODE:-1}"
tp_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
rollout_n="${ROLLOUT_N:-1}"
max_rounds="${MAX_ROUNDS:-30}"
max_new_tokens_per_turn="${MAX_NEW_TOKENS_PER_TURN:-200}"
plot_max_tokens="${PLOT_MAX_TOKENS:-0}"
tokens_per_row="${TOKENS_PER_ROW:-800}"
sample_idx="${SAMPLE_IDX:-0}"
item_id="${ITEM_ID:-}"
verl_ckpt="${VERL_CKPT:-}"

out_dir="${OUT_DIR:-${REPO_ROOT}/debug}"
mkdir -p "${out_dir}"

out_png="${OUT_PNG:-${out_dir}/env_feedback_mask_logprob.png}"
dump_jsonl="${DUMP_JSONL:-${out_dir}/env_feedback_mask_logprob.jsonl}"
rollout_log_dir="${ROLLOUT_LOG_DIR:-${out_dir}/executer_logs}"
mkdir -p "${rollout_log_dir}"

extra_args=()
if [[ -n "${item_id}" ]]; then
  extra_args+=(--item_id "${item_id}")
fi
if [[ -n "${verl_ckpt}" ]]; then
  extra_args+=(--verl_ckpt "${verl_ckpt}")
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/debug/plot_env_feedback_mask_logprob.py \
  --override algorithm.adv_estimator=grpo \
  --override trainer.nnodes=1 \
  --override trainer.n_gpus_per_node="${n_gpus_per_node}" \
  --override data.train_file="${train_file}" \
  --override data.train_batch_size=1 \
  --override data.shuffle=false \
  --override data.max_prompt_length=1024 \
  --override data.max_response_length=8192 \
  --override actor_rollout_ref.agentgym.task_name="${task_name}" \
  --override actor_rollout_ref.agentgym.env_addr="${env_server_url}" \
  --override actor_rollout_ref.model.path="${agent_model_path}" \
  --override actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  --override actor_rollout_ref.rollout.rollout_log_dir="${rollout_log_dir}" \
  --override actor_rollout_ref.rollout.tensor_model_parallel_size="${tp_size}" \
  --override actor_rollout_ref.rollout.n="${rollout_n}" \
  --max_rounds "${max_rounds}" \
  --max_new_tokens_per_turn "${max_new_tokens_per_turn}" \
  --sample_idx "${sample_idx}" \
  --plot_max_tokens "${plot_max_tokens}" \
  --tokens_per_row "${tokens_per_row}" \
  --out_png "${out_png}" \
  --dump_jsonl "${dump_jsonl}" \
  "${extra_args[@]}"
