#!/bin/bash
#SBATCH --job-name webshop-env-feedback-sft
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/inspurfs/chenlin1/log/webshop/%j.out
#SBATCH --error=/mnt/inspurfs/chenlin1/log/webshop/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "$REPO_ROOT"
LOG_ROOT="${LOG_ROOT:-/mnt/inspurfs/chenlin1/Word2World/log}"
JOB_LOG_DIR="${LOG_ROOT}/webshop/${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
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
if [ -L "$REPO_ROOT/../uv_agentgym_rl" ] && [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  rm -f "$REPO_ROOT/../uv_agentgym_rl"
fi
if [ ! -e "$REPO_ROOT/../uv_agentgym_rl" ]; then
  ln -s "$VENV_AGENTGYM_RL" "$REPO_ROOT/../uv_agentgym_rl"
fi

export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$HOME/models/${PURE_AGENT_MODEL_NAME}}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
SFT_EXP_NAME="${SFT_EXP_NAME:-webshop-env-feedback-sft-step150-n8-maxprompt4096-resp12288-${RUN_TS}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/sft/${SFT_EXP_NAME}}"
SFT_JSONL="${SFT_JSONL:-/mnt/inspurfs/chenlin1/Word2World/tmp/${SFT_EXP_NAME}.jsonl}"

ROLLOUT_EXP_NAME="${ROLLOUT_EXP_NAME:-webshop-pure-rollout-step150-n8-maxprompt4096-resp12288-maxmodel16384-20260212-171733}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/rollouts/${ROLLOUT_EXP_NAME}}"
EXECUTER_LOGS_DIR="${EXECUTER_LOGS_DIR:-${ROLLOUT_SAVE_DIR}/executer_logs}"

POLICY_LR="${POLICY_LR:-1e-6}"
ENV_FEEDBACK_LOSS_COEF_SFT="${ENV_FEEDBACK_LOSS_COEF_SFT:-1.0}"
SFT_LR="$(python3 - <<PY
import os
plr = float(os.environ.get("POLICY_LR", "1e-6"))
coef = float(os.environ.get("ENV_FEEDBACK_LOSS_COEF_SFT", "1.0"))
print(plr * coef)
PY
)"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12288}"
NPROC="${NPROC:-${SLURM_GPUS_ON_NODE:-4}}"
MASTER_PORT="${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}"

PROJECTS_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
APPTAINER_ARGS=(--pwd "$REPO_ROOT" --bind "$PROJECTS_ROOT" --bind "$UVENVS_HOME")
if [ "${APPTAINER_NV:-1}" = "1" ]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi
if [ -d /mnt/inspurfs/chenlin1 ]; then
  APPTAINER_ARGS+=(--bind /mnt/inspurfs/chenlin1)
fi
if [ -n "${APPTAINER_BIND_EXTRA:-}" ]; then
  IFS=, read -r -a extra_binds <<< "$APPTAINER_BIND_EXTRA"
  for bind_path in "${extra_binds[@]}"; do
    [ -n "$bind_path" ] && APPTAINER_ARGS+=(--bind "$bind_path")
  done
fi

apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
  bash -lc "\
    set -euo pipefail; \
    source ${VENV_AGENTGYM_RL}/bin/activate; \
    export MASTER_ADDR=127.0.0.1; \
    export MASTER_PORT=${MASTER_PORT}; \
    export SFT_JSONL=${SFT_JSONL}; \
    export MAX_MODEL_LEN=${MAX_MODEL_LEN}; \
    python3 scripts/build_env_feedback_sft_from_logs.py \
      --executer_logs_dir ${EXECUTER_LOGS_DIR} \
      --model_path ${BASE_MODEL_PATH} \
      --max_step 150 \
      --max_model_len ${MAX_MODEL_LEN} \
      --max_response_len ${MAX_RESPONSE_LEN} \
      --out ${SFT_JSONL}; \
    MAX_LEN=\$(python3 - <<PY
import json, os, sys
path = os.environ['SFT_JSONL']
max_len = 0
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        max_len = max(max_len, len(rec.get('input_ids', [])))
max_model_len = int(os.environ.get('MAX_MODEL_LEN', '32768'))
if max_len > max_model_len:
    print(f'Max sequence length {max_len} exceeds MAX_MODEL_LEN={max_model_len}', file=sys.stderr)
    sys.exit(1)
print(max_len)
PY
    ); \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node ${NPROC} -m verl.agent_trainer.fsdp_sft_trainer \
      data.pre_tokenized=true \
      data.train_files=${SFT_JSONL} \
      data.max_length=\${MAX_LEN} \
      data.truncation=error \
      data.train_batch_size=64 \
      data.micro_batch_size_per_gpu=1 \
      model.partial_pretrain=${BASE_MODEL_PATH} \
      model.enable_gradient_checkpointing=true \
      optim.lr=${SFT_LR} \
      trainer.default_local_dir=${SFT_OUTPUT_DIR} \
      trainer.project_name=webshop_env_feedback_sft \
      trainer.experiment_name=${SFT_EXP_NAME} \
      trainer.total_epochs=1 \
      trainer.logger=['console'] \
  " \
  > "${JOB_LOG_DIR}/sft.log" 2>&1
