#!/bin/bash
#SBATCH --job-name sokoban-a800-apptainer
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
cd "$REPO_ROOT"
LOG_ROOT="${LOG_ROOT:-/mnt/inspurfs/chenlin1/Word2World/log}"
JOB_LOG_DIR="${LOG_ROOT}/sokoban/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_SOKOBAN="${VENV_SOKOBAN:-$UVENVS_HOME/uv_sokoban}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi
if [ ! -f "$APPTAINER_IMG" ]; then
  echo "Error: Apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi
if [ ! -d "$VENV_SOKOBAN" ]; then
  echo "Error: Sokoban venv not found: $VENV_SOKOBAN" >&2
  echo "Hint: run scripts/env_setup/uv_sokoban_apptainer.sh" >&2
  exit 1
fi
if [ ! -d "$VENV_AGENTGYM_RL" ]; then
  echo "Error: AgentGym-RL venv not found: $VENV_AGENTGYM_RL" >&2
  exit 1
fi

# Ensure scripts expecting ../uv_* can find the uv_envs venvs.
if [ -L "$REPO_ROOT/../uv_sokoban" ] && [ ! -e "$REPO_ROOT/../uv_sokoban" ]; then
  rm -f "$REPO_ROOT/../uv_sokoban"
fi
if [ -L "$REPO_ROOT/../uv_agentgym_rl" ] && [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  rm -f "$REPO_ROOT/../uv_agentgym_rl"
fi
if [ ! -e "$REPO_ROOT/../uv_sokoban" ]; then
  ln -s "$VENV_SOKOBAN" "$REPO_ROOT/../uv_sokoban"
fi
if [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  ln -s "$VENV_AGENTGYM_RL" "$REPO_ROOT/../uv_agentgym_rl"
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-$HOME/models/${PURE_AGENT_MODEL_NAME}}"
if [ -n "${SFT_OUTPUT_DIR:-}" ]; then
  latest_ckpt="$(ls -d "${SFT_OUTPUT_DIR}/global_step_"* 2>/dev/null | sort -V | tail -1)"
  if [ -z "$latest_ckpt" ]; then
    echo "Error: No SFT checkpoint found under ${SFT_OUTPUT_DIR}" >&2
    exit 1
  fi
  export AGENT_MODEL_PATH="$latest_ckpt"
fi
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
if [ -z "${SOKOBAN_PORT:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    SOKOBAN_PORT="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
  else
    SOKOBAN_PORT=$((37001 + (${SLURM_JOB_ID:-0} % 1000)))
  fi
fi
export SOKOBAN_PORT
export SOKOBAN_DEFAULT_SPLIT="${SOKOBAN_DEFAULT_SPLIT:-train}"
export SOKOBAN_DIM_X="${SOKOBAN_DIM_X:-6}"
export SOKOBAN_DIM_Y="${SOKOBAN_DIM_Y:-6}"
export SOKOBAN_NUM_BOXES="${SOKOBAN_NUM_BOXES:-1}"
export SOKOBAN_MAX_STEPS="${SOKOBAN_MAX_STEPS:-50}"
export SOKOBAN_SEARCH_DEPTH="${SOKOBAN_SEARCH_DEPTH:-30}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
export ENV_FEEDBACK_LOSS_COEF="${ENV_FEEDBACK_LOSS_COEF:-1}"
export SCHEDULE_ENV_FEEDBACK_LOSS_COEF="${SCHEDULE_ENV_FEEDBACK_LOSS_COEF:-false}"
export FINAL_ENV_FEEDBACK_LOSS_COEF="${FINAL_ENV_FEEDBACK_LOSS_COEF:-1}"
export SCHEDULING_ALGO="${SCHEDULING_ALGO:-linear}"
export GRAD_CLIP="${GRAD_CLIP:-10}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL_SAVE_DIR_DEFAULT="/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$MODEL_SAVE_DIR_DEFAULT}"
PROJECT_NAME="${PROJECT_NAME:-word2world-sokoban}"
EXP_BASE="${EXP_NAME:-sokoban-verl-agent-low-kl-small-minibatch-envcoef1-nosched-gn10-task}"
if [ "${APPEND_RUN_TS:-1}" = "1" ]; then
  EXP_NAME_FULL="${EXP_BASE}-${ENV_FEEDBACK_LOSS_COEF}-${RUN_TS}"
else
  EXP_NAME_FULL="${EXP_BASE}"
fi

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

# Env server (CPU only)
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash scripts/env_server/start_sokoban.sh \
     > "${JOB_LOG_DIR}/env_server.log" 2>&1 &
server_pid=$!

sleep 5

# Training
srun --ntasks=1 --cpus-per-task=52 --gres=gpu:4 \
     --cpu-bind=cores \
     apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash examples/train/AgentGym-RL/sokoban_train.sh \
     --n_gpus_per_node "${N_GPUS_PER_NODE:-4}" \
     --env_server_url "http://127.0.0.1:${SOKOBAN_PORT}" \
     --exp_name "${EXP_NAME_FULL}" \
     --project_name "${PROJECT_NAME}" \
     --model_save_dir "${MODEL_SAVE_DIR}" \
     --train_file "${TRAIN_FILE:-${DATA_ROOT:-data}/train/sokoban_train.json}" \
     --total_epochs "${TOTAL_EPOCHS:-1}" \
     --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}" \
     --max_response_length "${MAX_RESPONSE_LENGTH:-12288}" \
     --max_model_len "${MAX_MODEL_LEN:-16384}" \
     --max_tokens "${MAX_TOKENS:-512}" \
     --rounds "${ROUNDS:-15}" \
     --pure_agent_model_name "${PURE_AGENT_MODEL_NAME}" \
     --agent_model_path "${AGENT_MODEL_PATH}" \
     --rollout_sample_num "${ROLLOUT_SAMPLE_NUM:-8}" \
     --train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
     --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE:-8}" \
     --kl_coef "${KL_COEF:-0.001}" \
     --entropy_coeff "${ENTROPY_COEF}" \
     --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
     --grad_clip "${GRAD_CLIP}" \
     --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
     --rollout_tp_size "${ROLLOUT_TP_SIZE:-1}" \
     --env_feedback_loss_coef "${ENV_FEEDBACK_LOSS_COEF}" \
     --schedule_env_feedback_loss_coef "${SCHEDULE_ENV_FEEDBACK_LOSS_COEF}" \
     --final_env_feedback_loss_coef "${FINAL_ENV_FEEDBACK_LOSS_COEF}" \
     --scheduling_algo "${SCHEDULING_ALGO}" \
     --save_freq "${SAVE_FREQ:-50}" \
     > "${JOB_LOG_DIR}/train.log" 2>&1
train_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $train_status
