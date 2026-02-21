#!/bin/bash
#SBATCH --job-name webshop-pure-rollout-151-200
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

export PURE_AGENT_MODEL_NAME="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-$HOME/models/${PURE_AGENT_MODEL_NAME}}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"

if [ -z "${WEBSHOP_PORT:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    WEBSHOP_PORT="$(python3 - <<PY
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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12288}"

ROLLOUT_EXP_NAME="${ROLLOUT_EXP_NAME:-webshop-pure-rollout-step151-200-n8-maxprompt4096-fullresp-${RUN_TS}}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/rollouts/${ROLLOUT_EXP_NAME}}"
EXECUTER_LOGS_DIR="${EXECUTER_LOGS_DIR:-${ROLLOUT_SAVE_DIR}/executer_logs}"
ITEM_IDS_JSON="${ITEM_IDS_JSON:-${ROLLOUT_SAVE_DIR}/step_item_ids.json}"
FULL_ITEM_IDS_JSON="${FULL_ITEM_IDS_JSON:-${ROLLOUT_SAVE_DIR}/step_item_ids_1-200.json}"

PROJECTS_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
APPTAINER_ARGS=(--pwd "$REPO_ROOT" --bind "$PROJECTS_ROOT" --bind "$UVENVS_HOME")
if [ "${APPTAINER_NV:-1}" = "1" ]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi
JAVA11_HOME_DEFAULT="$HOME/jdk-11"
JAVA_HOME="${JAVA_HOME:-}"
if [ -z "$JAVA_HOME" ] && [ -d "$JAVA11_HOME_DEFAULT" ]; then
  JAVA_HOME="$JAVA11_HOME_DEFAULT"
fi
if [ -z "$JAVA_HOME" ] && command -v javac >/dev/null 2>&1; then
  JAVAC_BIN="$(command -v javac)"
  JAVA_REAL="$(readlink -f "$JAVAC_BIN" 2>/dev/null || echo "$JAVAC_BIN")"
  JAVA_HOME="$(dirname "$(dirname "$JAVA_REAL")")"
fi
if [ -z "$JAVA_HOME" ] && command -v java >/dev/null 2>&1; then
  JAVA_BIN="$(command -v java)"
  JAVA_REAL="$(readlink -f "$JAVA_BIN" 2>/dev/null || echo "$JAVA_BIN")"
  JAVA_HOME="$(dirname "$(dirname "$JAVA_REAL")")"
  echo "Warning: javac not found; Webshop env server may fail without a JDK." >&2
fi
if [ -n "$JAVA_HOME" ] && [ -d "$JAVA_HOME" ]; then
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
  echo "Warning: JAVA_HOME not found; Webshop env server may fail without Java 11." >&2
fi
if [ -n "${AGENT_MODEL_PATH:-}" ] && [ -d "$AGENT_MODEL_PATH" ]; then
  APPTAINER_ARGS+=(--bind "$AGENT_MODEL_PATH")
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

# Env server (CPU only)
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash scripts/env_server/start_webshop.sh \
     > "${JOB_LOG_DIR}/env_server.log" 2>&1 &
server_pid=$!

sleep 5

# Pure rollout
srun --ntasks=1 --cpus-per-task=52 --gres=gpu:4 \
     --cpu-bind=cores \
     apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
     bash -lc "\
       set -euo pipefail; \
       export VLLM_USE_MODELSCOPE=0; \
       export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
       export VLLM_ATTENTION_BACKEND=XFORMERS; \
       export VLLM_NO_USAGE_STATS=1; \
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
       export VERL_ROLLOUT_APPEND_LOGS=1; \
       source ${VENV_AGENTGYM_RL}/bin/activate; \
       python3 scripts/webshop_first_steps_item_ids.py \
         --train_file ${REPO_ROOT}/data/train/webshop_train.json \
         --steps 200 \
         --train_batch_size 16 \
         --seed 1 \
         --out ${FULL_ITEM_IDS_JSON}; \
       FULL_ITEM_IDS_JSON=${FULL_ITEM_IDS_JSON} ITEM_IDS_JSON=${ITEM_IDS_JSON} \
       python3 -c 'import json, os; full_path=os.environ.get(\"FULL_ITEM_IDS_JSON\"); out_path=os.environ.get(\"ITEM_IDS_JSON\"); data=json.load(open(full_path, \"r\", encoding=\"utf-8\")); steps=data.get(\"step_item_ids\", [])[150:200]; out=dict(data); out[\"steps\"]=len(steps); out[\"step_item_ids\"]=steps; out[\"note\"]=\"steps 151-200 from training sampler\"; json.dump(out, open(out_path, \"w\", encoding=\"utf-8\"), ensure_ascii=True, indent=2)'; \
       python3 scripts/webshop_pure_rollout.py \
         --config ${REPO_ROOT}/AgentGym-RL/verl/agent_trainer/config/generation.yaml \
         --model_path ${AGENT_MODEL_PATH} \
         --env_server_url http://127.0.0.1:${WEBSHOP_PORT} \
         --item_ids_json ${ITEM_IDS_JSON} \
         --rollout_log_dir ${EXECUTER_LOGS_DIR} \
         --global_step 151-200 \
         --batch_size 16 \
         --rollout_n 8 \
         --max_prompt_length 4096 \
         --max_response_length ${MAX_RESPONSE_LEN} \
         --max_model_len ${MAX_MODEL_LEN} \
         --max_tokens 512 \
         --max_rounds 15 \
         --rollout_tp_size 1 \
         --rollout_gpu_mem_util 0.6; \
     " \
     > "${JOB_LOG_DIR}/rollout.log" 2>&1
rollout_status=$?

kill $server_pid 2>/dev/null || true
wait $server_pid 2>/dev/null || true
exit $rollout_status
