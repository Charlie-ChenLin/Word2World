#!/bin/bash
#SBATCH --job-name webarena-a800-apptainer
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "$REPO_ROOT"
mkdir -p "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_WEBARENA="${VENV_WEBARENA:-$UVENVS_HOME/uv_webarena}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi
if [ ! -f "$APPTAINER_IMG" ]; then
  echo "Error: Apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi
if [ ! -d "$VENV_AGENTGYM_RL" ]; then
  echo "Error: AgentGym-RL venv not found: $VENV_AGENTGYM_RL" >&2
  exit 1
fi

# Ensure scripts expecting ../uv_* can find the uv_envs venvs.
if [ -L "$REPO_ROOT/../uv_webarena" ] && [ ! -e "$REPO_ROOT/../uv_webarena" ]; then
  rm -f "$REPO_ROOT/../uv_webarena"
fi
if [ -L "$REPO_ROOT/../uv_agentgym_rl" ] && [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  rm -f "$REPO_ROOT/../uv_agentgym_rl"
fi
if [ ! -e "$REPO_ROOT/../uv_webarena" ] && [ -d "$VENV_WEBARENA" ]; then
  ln -s "$VENV_WEBARENA" "$REPO_ROOT/../uv_webarena"
fi
if [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  ln -s "$VENV_AGENTGYM_RL" "$REPO_ROOT/../uv_agentgym_rl"
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-7B-Instruct}"
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-$HOME/models/${PURE_AGENT_MODEL_NAME}}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$TRITON_CACHE_DIR"

# WebArena env server port
if [ -z "${WEBARENA_PORT:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    WEBARENA_PORT="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
  else
    WEBARENA_PORT=$((36005 + (${SLURM_JOB_ID:-0} % 1000)))
  fi
fi
export WEBARENA_PORT

PROJECTS_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
APPTAINER_ARGS=(--pwd "$REPO_ROOT" --bind "$PROJECTS_ROOT" --bind "$UVENVS_HOME")
if [ "${APPTAINER_NV:-1}" = "1" ]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi
if [ -n "${AGENT_MODEL_PATH:-}" ] && [ -d "$AGENT_MODEL_PATH" ]; then
  APPTAINER_ARGS+=(--bind "$AGENT_MODEL_PATH")
fi
if [ -n "${DATA_ROOT:-}" ] && [ -d "$DATA_ROOT" ]; then
  APPTAINER_ARGS+=(--bind "$DATA_ROOT")
fi
if [ -d /mnt/inspurfs/chenlin1 ]; then
  APPTAINER_ARGS+=(--bind /mnt/inspurfs/chenlin1)
fi
if [ -n "${APPTAINER_BIND_EXTRA:-}" ]; then
  IFS=',' read -r -a extra_binds <<< "$APPTAINER_BIND_EXTRA"
  for bind_path in "${extra_binds[@]}"; do
    [ -n "$bind_path" ] && APPTAINER_ARGS+=(--bind "$bind_path")
  done
fi

# Optionally start WebArena sites on this node (CPU only, host network)
if [ "${WEBARENA_START_SITES:-0}" = "1" ]; then
  bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_start.sh \
    > "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}/sites.log" 2>&1
  bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_healthcheck.sh --strict \
    > "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}/sites_check.log" 2>&1
fi

# Env server (CPU only)
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash scripts/env_server/start_webarena.sh \
     > "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}/env_server.log" 2>&1 &
server_pid=$!

sleep 5

# Training (GPU)
srun --ntasks=1 --cpus-per-task=52 --gres=gpu:4 \
     --cpu-bind=cores \
     apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash examples/train/AgentGym-RL/webarena_train.sh \
     --n_gpus_per_node "${N_GPUS_PER_NODE:-4}" \
     --env_server_url "http://127.0.0.1:${WEBARENA_PORT}" \
     --exp_name "${EXP_NAME:-test-webarena}-${RUN_TS}" \
     --model_save_dir "${MODEL_SAVE_DIR:-AgentGym-RL/saves}" \
     --max_prompt_length "${MAX_PROMPT_LENGTH:-750}" \
     --max_response_length "${MAX_RESPONSE_LENGTH:-14098}" \
     --max_model_len "${MAX_MODEL_LEN:-32768}" \
     --max_tokens "${MAX_TOKENS:-512}" \
     --pure_agent_model_name "${PURE_AGENT_MODEL_NAME}" \
     --agent_model_path "${AGENT_MODEL_PATH}" \
     --rollout_sample_num "${ROLLOUT_SAMPLE_NUM:-4}" \
     --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
     --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE:-4}" \
     --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
     --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL:-0.7}" \
     --rollout_tp_size "${ROLLOUT_TP_SIZE:-1}" \
     --env_feedback_loss_coef "${ENV_FEEDBACK_LOSS_COEF:-0}" \
     > "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}/train.log" 2>&1
train_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true

if [ "${WEBARENA_START_SITES:-0}" = "1" ]; then
  bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_stop.sh \
    > "/mnt/petrelfs/chenlin1/log/webarena/${SLURM_JOB_ID}/sites_stop.log" 2>&1 || true
fi

exit $train_status
