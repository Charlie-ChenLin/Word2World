#!/bin/bash
#SBATCH --job-name webshop-a800-pcgrad
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/inspurfs/chenlin1/log/webshop/%j.out
#SBATCH --error=/mnt/inspurfs/chenlin1/log/webshop/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World_run_proj_pgplusother_alphaclamp}"
EXPECTED_GIT_BRANCH="${EXPECTED_GIT_BRANCH:-feature/pcgrad-env-credit-v1}"
cd "$REPO_ROOT"

ACTOR_REPO_DIR="${REPO_ROOT}/AgentGym-RL"
if command -v git >/dev/null 2>&1 && [ -d "$ACTOR_REPO_DIR" ]; then
  current_branch="$(git -C "$ACTOR_REPO_DIR" branch --show-current || true)"
  echo "[slurm] Agent repo: ${ACTOR_REPO_DIR}"
  echo "[slurm] Current branch: ${current_branch}"
  if [ -n "${EXPECTED_GIT_BRANCH}" ] && [ "$current_branch" != "${EXPECTED_GIT_BRANCH}" ]; then
    echo "Error: expected branch '${EXPECTED_GIT_BRANCH}', got '${current_branch}'" >&2
    exit 1
  fi
fi

LOG_ROOT="${LOG_ROOT:-/mnt/inspurfs/chenlin1/Word2World/log}"
JOB_LOG_DIR="${LOG_ROOT}/webshop/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_WEBSHOP="${VENV_WEBSHOP:-$UVENVS_HOME/uv_webshop}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi
if [ ! -f "$APPTAINER_IMG" ]; then
  echo "Error: Apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi
if [ ! -d "$VENV_WEBSHOP" ]; then
  echo "Error: Webshop venv not found: $VENV_WEBSHOP" >&2
  exit 1
fi
if [ ! -d "$VENV_AGENTGYM_RL" ]; then
  echo "Error: AgentGym-RL venv not found: $VENV_AGENTGYM_RL" >&2
  exit 1
fi

# Ensure scripts expecting ../uv_* can find the uv_envs venvs.
if [ -L "$REPO_ROOT/../uv_webshop" ] && [ ! -e "$REPO_ROOT/../uv_webshop" ]; then
  rm -f "$REPO_ROOT/../uv_webshop"
fi
if [ -L "$REPO_ROOT/../uv_agentgym_rl" ] && [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  rm -f "$REPO_ROOT/../uv_agentgym_rl"
fi
if [ ! -e "$REPO_ROOT/../uv_webshop" ]; then
  ln -s "$VENV_WEBSHOP" "$REPO_ROOT/../uv_webshop"
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

TRAIN_FILE="${TRAIN_FILE:-/mnt/petrelfs/chenlin1/projects/Word2World/data/train/webshop_train.json}"
if [ ! -f "$TRAIN_FILE" ]; then
  echo "Error: train file not found: $TRAIN_FILE" >&2
  exit 1
fi

export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
if [ -z "${WEBSHOP_PORT:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    WEBSHOP_PORT="$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
  else
    WEBSHOP_PORT=$((36001 + (${SLURM_JOB_ID:-0} % 1000)))
  fi
fi
export WEBSHOP_PORT
export WEBSHOP_RULE_REWARD="${WEBSHOP_RULE_REWARD:-1}"
export WEBSHOP_RULE_REWARD_SUCCESS="${WEBSHOP_RULE_REWARD_SUCCESS:-10}"
export WEBSHOP_RULE_REWARD_FAIL="${WEBSHOP_RULE_REWARD_FAIL:-0}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
export ENV_FEEDBACK_LOSS_COEF="${ENV_FEEDBACK_LOSS_COEF:-1}"
export SCHEDULE_ENV_FEEDBACK_LOSS_COEF="${SCHEDULE_ENV_FEEDBACK_LOSS_COEF:-false}"
export FINAL_ENV_FEEDBACK_LOSS_COEF="${FINAL_ENV_FEEDBACK_LOSS_COEF:-1}"
export SCHEDULING_ALGO="${SCHEDULING_ALGO:-linear}"
export GRAD_CLIP="${GRAD_CLIP:-10}"

export ENV_FEEDBACK_GRAD_PROJ_TO_PG="${ENV_FEEDBACK_GRAD_PROJ_TO_PG:-true}"
export ENV_FEEDBACK_GRAD_PROJ_EPS="${ENV_FEEDBACK_GRAD_PROJ_EPS:-1e-12}"
export ENV_FEEDBACK_GRAD_PROJ_IMPL="${ENV_FEEDBACK_GRAD_PROJ_IMPL:-reordered_exact}"
export ENV_FEEDBACK_GRAD_PROJ_TARGET="${ENV_FEEDBACK_GRAD_PROJ_TARGET:-pg}"
export ENV_FEEDBACK_GRAD_PROJ_ALGO="${ENV_FEEDBACK_GRAD_PROJ_ALGO:-pcgrad}"
export ENV_FEEDBACK_GRAD_PROJ_LEGACY_COMPAT_METRICS="${ENV_FEEDBACK_GRAD_PROJ_LEGACY_COMPAT_METRICS:-false}"
export ENV_FEEDBACK_PCGRAD_LAMBDA_MAX="${ENV_FEEDBACK_PCGRAD_LAMBDA_MAX:-0.2}"
export ENV_FEEDBACK_PCGRAD_LAMBDA_NORM_RATIO="${ENV_FEEDBACK_PCGRAD_LAMBDA_NORM_RATIO:-0.3}"
export ENV_FEEDBACK_PCGRAD_EPS="${ENV_FEEDBACK_PCGRAD_EPS:-1e-12}"
export ENV_FEEDBACK_PCGRAD_LOG_EXTRA_METRICS="${ENV_FEEDBACK_PCGRAD_LOG_EXTRA_METRICS:-true}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL_SAVE_DIR_DEFAULT="/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$MODEL_SAVE_DIR_DEFAULT}"
EXP_BASE="${EXP_NAME:-verl-agent-low-kl-small-minibatch-envcoef1-nosched-gn10-pcgrad-pg-diag}"
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
APPTAINER_ARGS+=(--env WEBSHOP_RULE_REWARD="$WEBSHOP_RULE_REWARD")
APPTAINER_ARGS+=(--env WEBSHOP_RULE_REWARD_SUCCESS="$WEBSHOP_RULE_REWARD_SUCCESS")
APPTAINER_ARGS+=(--env WEBSHOP_RULE_REWARD_FAIL="$WEBSHOP_RULE_REWARD_FAIL")
JAVA11_HOME_DEFAULT="$HOME/jdk-11"
JAVA_HOME="${JAVA_HOME:-}"
if [ -z "$JAVA_HOME" ] && [ -d "$JAVA11_HOME_DEFAULT" ]; then
  JAVA_HOME="$JAVA11_HOME_DEFAULT"
fi
if [ -n "$JAVA_HOME" ] && [ -d "$JAVA_HOME" ]; then
  APPTAINER_ARGS+=(--bind "$JAVA_HOME:$JAVA_HOME")
  APPTAINER_ARGS+=(--bind /usr/lib/jvm:/usr/lib/jvm)
  APPTAINER_ARGS+=(--env JAVA_HOME="$JAVA_HOME")
  APPTAINER_ARGS+=(--env JRE_HOME="$JAVA_HOME")
  APPTAINER_ARGS+=(--env PATH="$JAVA_HOME/bin:$PATH")
  APPTAINER_ARGS+=(--env JVM_PATH="$JAVA_HOME/lib/server/libjvm.so")
fi
if [ -n "${AGENT_MODEL_PATH:-}" ] && [ -d "$AGENT_MODEL_PATH" ]; then
  APPTAINER_ARGS+=(--bind "$AGENT_MODEL_PATH")
fi
if [ -d /mnt/inspurfs/chenlin1 ]; then
  APPTAINER_ARGS+=(--bind /mnt/inspurfs/chenlin1)
fi

# Env server (CPU only) - run inside batch step to avoid step-creation contention.
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash scripts/env_server/start_webshop.sh \
     > "${JOB_LOG_DIR}/env_server.log" 2>&1 &
server_pid=$!

sleep 5

srun --ntasks=1 --cpus-per-task=52 --gres=gpu:4 \
     --cpu-bind=cores \
     apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash examples/train/AgentGym-RL/webshop_train.sh \
     --n_gpus_per_node "${N_GPUS_PER_NODE:-4}" \
     --env_server_url "http://127.0.0.1:${WEBSHOP_PORT}" \
     --exp_name "${EXP_NAME_FULL}" \
     --model_save_dir "${MODEL_SAVE_DIR}" \
     --train_file "${TRAIN_FILE}" \
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
     --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
     --grad_clip "${GRAD_CLIP}" \
     --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
     --rollout_tp_size "${ROLLOUT_TP_SIZE:-1}" \
     --env_feedback_loss_coef "${ENV_FEEDBACK_LOSS_COEF}" \
     --env_feedback_grad_proj_to_pg "${ENV_FEEDBACK_GRAD_PROJ_TO_PG}" \
     --env_feedback_grad_proj_eps "${ENV_FEEDBACK_GRAD_PROJ_EPS}" \
     --env_feedback_grad_proj_impl "${ENV_FEEDBACK_GRAD_PROJ_IMPL}" \
     --env_feedback_grad_proj_target "${ENV_FEEDBACK_GRAD_PROJ_TARGET}" \
     --env_feedback_grad_proj_algo "${ENV_FEEDBACK_GRAD_PROJ_ALGO}" \
     --env_feedback_grad_proj_legacy_compat_metrics "${ENV_FEEDBACK_GRAD_PROJ_LEGACY_COMPAT_METRICS}" \
     --env_feedback_pcgrad_lambda_max "${ENV_FEEDBACK_PCGRAD_LAMBDA_MAX}" \
     --env_feedback_pcgrad_lambda_norm_ratio "${ENV_FEEDBACK_PCGRAD_LAMBDA_NORM_RATIO}" \
     --env_feedback_pcgrad_eps "${ENV_FEEDBACK_PCGRAD_EPS}" \
     --env_feedback_pcgrad_log_extra_metrics "${ENV_FEEDBACK_PCGRAD_LOG_EXTRA_METRICS}" \
     --schedule_env_feedback_loss_coef "${SCHEDULE_ENV_FEEDBACK_LOSS_COEF}" \
     --final_env_feedback_loss_coef "${FINAL_ENV_FEEDBACK_LOSS_COEF}" \
     --scheduling_algo "${SCHEDULING_ALGO}" \
     --save_freq "${SAVE_FREQ:-50}" \
     --webshop_rule_reward "${WEBSHOP_RULE_REWARD}" \
     --webshop_rule_reward_success "${WEBSHOP_RULE_REWARD_SUCCESS}" \
     --webshop_rule_reward_fail "${WEBSHOP_RULE_REWARD_FAIL}" \
     > "${JOB_LOG_DIR}/train.log" 2>&1
train_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $train_status
