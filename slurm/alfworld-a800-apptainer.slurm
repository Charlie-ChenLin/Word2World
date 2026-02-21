#!/bin/bash
#SBATCH --job-name alfworld-a800-apptainer
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=3-00:00:00
#SBATCH --output=./log/alfworld/%j/output.out
#SBATCH --error=./log/alfworld/%j/error.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "$REPO_ROOT"
mkdir -p "./log/alfworld/${SLURM_JOB_ID}"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_ALFWORLD="${VENV_ALFWORLD:-$UVENVS_HOME/uv_alfworld}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi
if [ ! -f "$APPTAINER_IMG" ]; then
  echo "Error: Apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi
if [ ! -d "$VENV_ALFWORLD" ]; then
  echo "Error: AlfWorld venv not found: $VENV_ALFWORLD" >&2
  exit 1
fi
if [ ! -d "$VENV_AGENTGYM_RL" ]; then
  echo "Error: AgentGym-RL venv not found: $VENV_AGENTGYM_RL" >&2
  exit 1
fi

# Ensure scripts expecting ../uv_* can find the uv_envs venvs.
if [ -L "$REPO_ROOT/../uv_alfworld" ] && [ ! -e "$REPO_ROOT/../uv_alfworld" ]; then
  rm -f "$REPO_ROOT/../uv_alfworld"
fi
if [ -L "$REPO_ROOT/../uv_agentgym_rl" ] && [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  rm -f "$REPO_ROOT/../uv_agentgym_rl"
fi
if [ ! -e "$REPO_ROOT/../uv_alfworld" ]; then
  ln -s "$VENV_ALFWORLD" "$REPO_ROOT/../uv_alfworld"
fi
if [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  ln -s "$VENV_AGENTGYM_RL" "$REPO_ROOT/../uv_agentgym_rl"
fi

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-$HOME/models/${PURE_AGENT_MODEL_NAME}}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$TRITON_CACHE_DIR"

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
if [ -n "${APPTAINER_BIND_EXTRA:-}" ]; then
  IFS=',' read -r -a extra_binds <<< "$APPTAINER_BIND_EXTRA"
  for bind_path in "${extra_binds[@]}"; do
    [ -n "$bind_path" ] && APPTAINER_ARGS+=(--bind "$bind_path")
  done
fi

# Env server (CPU only) - run inside batch step to avoid step-creation contention.
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash scripts/env_server/start_alfworld.sh \
     > "./log/alfworld/${SLURM_JOB_ID}/env_server.log" 2>&1 &
server_pid=$!

sleep 5

# Training (1 GPU + 12 cores)
srun --ntasks=1 --cpus-per-task=52 --gres=gpu:4 \
     --cpu-bind=cores \
     apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash examples/train/AgentGym-RL/alfworld_train.sh \
     --n_gpus_per_node 4 \
     --exp_name "${EXP_NAME:-test-a800-env-0}-${RUN_TS}" \
     --max_response_length 4096 \
     --max_model_len 8192 \
     --pure_agent_model_name "${PURE_AGENT_MODEL_NAME}" \
     --agent_model_path "${AGENT_MODEL_PATH}" \
     --env_feedback_loss_coef "${ENV_FEEDBACK_LOSS_COEF:-0}" \
     > "./log/alfworld/${SLURM_JOB_ID}/train.log" 2>&1
train_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $train_status
