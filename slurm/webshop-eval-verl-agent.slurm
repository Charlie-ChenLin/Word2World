#!/bin/bash
# WebShop eval for a single model, evaluate steps >= MIN_STEP in descending order.
# Usage:
#   sbatch --export=ALL,MODEL_NAME=webshop-a800-envfb-0-20260206-161833 slurm/webshop-eval-envfb-steps-desc.slurm
#   sbatch --export=ALL,MODEL_NAME=...,MIN_STEP=250 slurm/webshop-eval-envfb-steps-desc.slurm
#   sbatch --export=ALL,MODEL_NAME=...,STEPS="500 450 400 250" slurm/webshop-eval-envfb-steps-desc.slurm
#   sbatch --export=ALL,MODEL_NAME=...,ROLLOUT_TEMPERATURE=0.7 slurm/webshop-eval-verl-agent.slurm
#   sbatch --export=ALL,MODEL_NAME=...,MODEL_TEMPERATURE=0.7 slurm/webshop-eval-verl-agent.slurm

#SBATCH --job-name=webshop-eval-envfb
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-eval-envfb-%A.out
#SBATCH --error=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-eval-envfb-%A.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-}"
if [ -z "${MODEL_NAME}" ]; then
  echo "[eval-envfb] MODEL_NAME is required." >&2
  exit 2
fi

SAVE_ROOT="${MODEL_ROOT:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves}"
MIN_STEP="${MIN_STEP:-250}"

# Steps to evaluate (descending). Use STEPS if provided; otherwise auto-discover.
if [ -n "${STEPS:-}" ]; then
  read -r -a steps_raw <<< "${STEPS}"
  steps=( $(printf "%s\n" "${steps_raw[@]}" | awk -v min="${MIN_STEP}" '$1>=min' | sort -nr) )
else
  mapfile -t steps < <(ls -d "${SAVE_ROOT}/${MODEL_NAME}"/global_step_* 2>/dev/null \
    | sed 's#.*/global_step_##' \
    | awk -v min="${MIN_STEP}" '$1>=min' \
    | sort -nr)
fi

if [ ${#steps[@]} -eq 0 ]; then
  echo "[eval-envfb] No steps >= ${MIN_STEP} found for ${MODEL_NAME}." >&2
  exit 1
fi

# Use your apptainer system image + uv envs (bind-mounted).
APPTAINER_IMG="${APPTAINER_IMG:-/mnt/petrelfs/chenlin1/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-/mnt/petrelfs/chenlin1/uv_envs}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

PROJECTS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
APPTAINER_ARGS=(--nv --pwd "${REPO_ROOT}" --bind "${PROJECTS_ROOT}" --bind "${UVENVS_HOME}" --bind /mnt/inspurfs/chenlin1)

# Bind Java 11 into the container so pyserini/jnius uses a compatible JVM.
JAVA11_HOME_DEFAULT="$HOME/jdk-11"
JAVA_HOME="${JAVA_HOME:-$JAVA11_HOME_DEFAULT}"
if [ -d "$JAVA_HOME" ]; then
  APPTAINER_ARGS+=(--bind "$JAVA_HOME:$JAVA_HOME")
  APPTAINER_ARGS+=(--bind /usr/lib/jvm:/usr/lib/jvm)
  if [ -d /usr/share/javazi-1.8 ]; then
    APPTAINER_ARGS+=(--bind /usr/share/javazi-1.8:/usr/share/javazi-1.8)
  fi
  APPTAINER_ARGS+=(--env JAVA_HOME="$JAVA_HOME")
  APPTAINER_ARGS+=(--env JRE_HOME="$JAVA_HOME")
  APPTAINER_ARGS+=(--env PATH="$JAVA_HOME/bin:$PATH")
  APPTAINER_ARGS+=(--env JVM_PATH="$JAVA_HOME/lib/server/libjvm.so")
else
  echo "[eval-envfb] Warning: JAVA_HOME not found at $JAVA_HOME; Webshop env server may fail." >&2
fi

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/log/webshop_eval}"
JOB_BASE="${OUT_ROOT}/${SLURM_JOB_ID}_${MODEL_NAME}"
mkdir -p "${JOB_BASE}"

# Eval params (match existing eval defaults).
SAMPLE_NUM="${SAMPLE_NUM:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12288}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.90}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-${MODEL_TEMPERATURE:-${TEMPERATURE:-0.4}}}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-512}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"

EVAL_PATH="${EVAL_PATH:-${REPO_ROOT}/data/AgentEval/webshop}"

# Pick a free local port for the env server.
WEBSHOP_PORT="${WEBSHOP_PORT:-}"
if [ -z "${WEBSHOP_PORT}" ]; then
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
ENV_SERVER_URL="http://127.0.0.1:${WEBSHOP_PORT}"

# Start env server inside apptainer (CPU only).
apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" \
  bash scripts/env_server/start_webshop.sh \
  > "${JOB_BASE}/env_server.log" 2>&1 &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true; wait ${server_pid} 2>/dev/null || true' EXIT
sleep 5

for STEP in "${steps[@]}"; do
  ckpt_actor="${SAVE_ROOT}/${MODEL_NAME}/global_step_${STEP}/actor"
  model_path_hf="${ckpt_actor}/huggingface"

  OUT_DIR="${JOB_BASE}/step_${STEP}"
  mkdir -p "${OUT_DIR}/runs"
  eval_log="${OUT_DIR}/runs/${MODEL_NAME}_step_${STEP}.eval.log"

  echo "[eval-envfb] Host=$(hostname) model=${MODEL_NAME} step=${STEP}"
  echo "[eval-envfb] ckpt_actor=${ckpt_actor}"
  echo "[eval-envfb] out_dir=${OUT_DIR}"

  if [ ! -d "${ckpt_actor}" ]; then
    echo "[eval-envfb] Missing ckpt_actor=${ckpt_actor}" >&2
    continue
  fi

  # Ensure HF weights exist; if missing, merge from FSDP shards first.
  apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" bash -lc "\
    source '${VENV_AGENTGYM_RL}/bin/activate' && \
    if [ ! -f '${model_path_hf}/model.safetensors.index.json' ] && \
       [ ! -f '${model_path_hf}/model.safetensors' ] && \
       [ ! -f '${model_path_hf}/pytorch_model.bin' ] && \
       ! ls '${model_path_hf}'/model-*.safetensors '${model_path_hf}'/pytorch_model-*.bin >/dev/null 2>&1; then \
      python3 AgentGym-RL/scripts/model_merger.py --local_dir '${ckpt_actor}'; \
    fi"

  # Run eval.
  apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" bash -lc "\
    source '${VENV_AGENTGYM_RL}/bin/activate' && \
    HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
      hydra.job.chdir=False \
      trainer.nnodes=1 \
      trainer.n_gpus_per_node=1 \
      data.path='${EVAL_PATH}' \
      data.max_prompt_length='${MAX_PROMPT_LEN}' \
      data.max_response_length='${MAX_RESPONSE_LEN}' \
      data.n_samples='${SAMPLE_NUM}' \
      data.batch_size='${BATCH_SIZE}' \
      agentgym.task_name='webshop' \
      agentgym.env_addr='${ENV_SERVER_URL}' \
      agentgym.max_rounds='${MAX_ROUNDS}' \
      agentgym.timeout=500 \
      model.path='${model_path_hf}' \
      rollout.gpu_memory_utilization='${ROLLOUT_GPU_MEM_UTIL}' \
      rollout.temperature='${ROLLOUT_TEMPERATURE}' \
      rollout.max_model_len='${ROLLOUT_MAX_MODEL_LEN}' \
      rollout.max_tokens='${ROLLOUT_MAX_TOKENS}' \
      rollout.tensor_model_parallel_size='${ROLLOUT_TP_SIZE}' \
      rollout.rollout_log_dir='${SAVE_ROOT}/${MODEL_NAME}/global_step_${STEP}/eval_logs/$(date +%Y%m%d-%H%M%S)'" \
    > "${eval_log}" 2>&1

  avg=$(awk -F': ' "/Avg@${SAMPLE_NUM}:/ {print \$2; exit}" "${eval_log}" | tr -d ' ')
  pass=$(awk -F': ' "/Pass@${SAMPLE_NUM}:/ {print \$2; exit}" "${eval_log}" | tr -d ' ')

  echo -e "model\tstep\tavg\tpass\tstatus\tlog" > "${OUT_DIR}/results.tsv"
  if [ -n "${avg}" ] && [ -n "${pass}" ]; then
    echo -e "${MODEL_NAME}\t${STEP}\t${avg}\t${pass}\tOK\t${eval_log}" >> "${OUT_DIR}/results.tsv"
  else
    echo -e "${MODEL_NAME}\t${STEP}\t\t\tPARSE_FAIL\t${eval_log}" >> "${OUT_DIR}/results.tsv"
  fi
done
