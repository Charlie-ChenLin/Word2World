#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF_USAGE'
Usage: bash webshop_eval_all_apptainer.sh [options]
All options override the defaults in this script. They can also be set via env vars.
  --model_root DIR
  --model_filter STR
  --data_root DIR
  --eval_path DIR
  --n_gpus_per_node N
  --sample_num INT
  --max_rounds INT
  --max_prompt_length INT
  --max_response_length INT
  --batch_size INT
  --rollout_gpu_mem_util FLOAT
  --rollout_temperature FLOAT
  --rollout_max_model_len INT
  --rollout_max_tokens INT
  --rollout_tp_size INT
  --rollout_log_subdir STR
  --out_dir DIR
  --step_mod INT
  --step_shards INT
  --step_shard_id INT
  --env_server_url URL
  -h, --help
EOF_USAGE
}

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/AgentGym-RL/saves}"
MODEL_FILTER="${MODEL_FILTER:-webshop}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
EVAL_PATH="${EVAL_PATH:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
SAMPLE_NUM="${SAMPLE_NUM:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.95}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-200}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_LOG_SUBDIR="${ROLLOUT_LOG_SUBDIR:-eval_logs}"
OUT_DIR_DEFAULT="${REPO_ROOT}/log/webshop_eval"
OUT_DIR="${OUT_DIR:-${OUT_DIR_DEFAULT}}"
STEP_MOD="${STEP_MOD:-}"
STEP_SHARDS="${STEP_SHARDS:-}"
STEP_SHARD_ID="${STEP_SHARD_ID:-}"
ENV_SERVER_URL="${ENV_SERVER_URL:-}"

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_WEBSHOP="${VENV_WEBSHOP:-$UVENVS_HOME/uv_webshop}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

while [ $# -gt 0 ]; do
    case "$1" in
        --model_root) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; MODEL_ROOT="$2"; shift 2 ;;
        --model_filter) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; MODEL_FILTER="$2"; shift 2 ;;
        --data_root) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; DATA_ROOT="$2"; shift 2 ;;
        --eval_path) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; EVAL_PATH="$2"; shift 2 ;;
        --n_gpus_per_node) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; N_GPUS_PER_NODE="$2"; shift 2 ;;
        --sample_num) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; SAMPLE_NUM="$2"; shift 2 ;;
        --max_rounds) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; MAX_ROUNDS="$2"; shift 2 ;;
        --max_prompt_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; MAX_PROMPT_LEN="$2"; shift 2 ;;
        --max_response_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; MAX_RESPONSE_LEN="$2"; shift 2 ;;
        --batch_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; BATCH_SIZE="$2"; shift 2 ;;
        --rollout_gpu_mem_util) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_GPU_MEM_UTIL="$2"; shift 2 ;;
        --rollout_temperature) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_TEMPERATURE="$2"; shift 2 ;;
        --rollout_max_model_len) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_MAX_MODEL_LEN="$2"; shift 2 ;;
        --rollout_max_tokens) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_MAX_TOKENS="$2"; shift 2 ;;
        --rollout_tp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_TP_SIZE="$2"; shift 2 ;;
        --rollout_log_subdir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ROLLOUT_LOG_SUBDIR="$2"; shift 2 ;;
        --out_dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; OUT_DIR="$2"; shift 2 ;;
        --step_mod) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; STEP_MOD="$2"; shift 2 ;;
        --step_shards) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; STEP_SHARDS="$2"; shift 2 ;;
        --step_shard_id) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; STEP_SHARD_ID="$2"; shift 2 ;;
        --env_server_url) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ENV_SERVER_URL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" 1>&2; usage; exit 1 ;;
    esac
done

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

if [ -n "${STEP_SHARDS}" ] || [ -n "${STEP_SHARD_ID}" ]; then
  if [ -z "${STEP_SHARDS}" ] || [ -z "${STEP_SHARD_ID}" ]; then
    echo "Error: step_shards and step_shard_id must be set together." >&2
    exit 1
  fi
  if ! [[ "${STEP_SHARDS}" =~ ^[0-9]+$ ]] || ! [[ "${STEP_SHARD_ID}" =~ ^[0-9]+$ ]]; then
    echo "Error: step_shards and step_shard_id must be integers." >&2
    exit 1
  fi
  if [ "${STEP_SHARDS}" -le 0 ] || [ "${STEP_SHARD_ID}" -lt 0 ] || [ "${STEP_SHARD_ID}" -ge "${STEP_SHARDS}" ]; then
    echo "Error: step_shard_id must be in [0, step_shards-1]." >&2
    exit 1
  fi
fi

if [ -z "$EVAL_PATH" ]; then
  EVAL_PATH="${DATA_ROOT}/AgentEval/webshop"
fi

if [ -z "$ENV_SERVER_URL" ]; then
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
    WEBSHOP_PORT=$((36001 + (RANDOM % 1000)))
  fi
  ENV_SERVER_URL="http://127.0.0.1:${WEBSHOP_PORT}"
else
  WEBSHOP_PORT="${ENV_SERVER_URL##*:}"
fi
export WEBSHOP_PORT

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TRITON_CACHE_DIR="/tmp/webshop_eval_${WEBSHOP_PORT}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
if [ "${OUT_DIR}" = "${OUT_DIR_DEFAULT}" ]; then
  OUT_DIR="${OUT_DIR_DEFAULT}/${RUN_TS}"
fi
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${OUT_DIR}"

PROJECTS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
APPTAINER_ARGS=(--pwd "${REPO_ROOT}" --bind "${PROJECTS_ROOT}" --bind "${UVENVS_HOME}")
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
  # Bind the JDK path itself (works even if not under /usr/lib/jvm).
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
if [ -n "${APPTAINER_BIND_EXTRA:-}" ]; then
  IFS=',' read -r -a extra_binds <<< "$APPTAINER_BIND_EXTRA"
  for bind_path in "${extra_binds[@]}"; do
    [ -n "$bind_path" ] && APPTAINER_ARGS+=(--bind "$bind_path")
  done
fi

# Ensure eval data exists
if [ ! -d "${EVAL_PATH}" ]; then
  if [ -f "${DATA_ROOT}/eval/webshop_test.json" ]; then
    mkdir -p "${EVAL_PATH}"
    cp "${DATA_ROOT}/eval/webshop_test.json" "${EVAL_PATH}/webshop_test.json"
    # also create a copy for category summary
    cp "${DATA_ROOT}/eval/webshop_test.json" "${EVAL_PATH}/webshop_all.json" || true
  else
    echo "[webshop_eval_all] Missing ${DATA_ROOT}/eval/webshop_test.json; attempting to download AgentEval dataset..." >&2
    apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
      bash -lc "source '${VENV_AGENTGYM_RL}/bin/activate' && python3 scripts/download_data/download_data.py --output_dir '${DATA_ROOT}'" \
      > "${OUT_DIR}/download_data.log" 2>&1
    if [ ! -f "${DATA_ROOT}/eval/webshop_test.json" ]; then
      echo "[webshop_eval_all] Failed to prepare eval data." >&2
      exit 1
    fi
    mkdir -p "${EVAL_PATH}"
    cp "${DATA_ROOT}/eval/webshop_test.json" "${EVAL_PATH}/webshop_test.json"
    cp "${DATA_ROOT}/eval/webshop_test.json" "${EVAL_PATH}/webshop_all.json" || true
  fi
fi

RESULTS_TSV="${OUT_DIR}/results.tsv"
FAIL_LOG="${OUT_DIR}/failures.log"
MODELS_TXT="${OUT_DIR}/models.txt"
: > "${RESULTS_TSV}"
: > "${FAIL_LOG}"
: > "${MODELS_TXT}"

# Start env server inside apptainer (CPU only)
apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
  bash scripts/env_server/start_webshop.sh \
  > "${OUT_DIR}/env_server.log" 2>&1 &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true; wait ${server_pid} 2>/dev/null || true' EXIT

sleep 5

echo -e "model\tstep\tavg\tpass\tstatus\tlog" > "${RESULTS_TSV}"

mapfile -t model_dirs < <(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*${MODEL_FILTER}*" -printf '%f\n' | sort)
if [ "${#model_dirs[@]}" -eq 0 ]; then
  echo "[webshop_eval_all] No model dirs matched under ${MODEL_ROOT} (filter: ${MODEL_FILTER})" >&2
  exit 1
fi

# Compute shared steps across all matched models to prioritize eval on common checkpoints.
declare -A step_counts=()
for model in "${model_dirs[@]}"; do
  model_path="${MODEL_ROOT}/${model}"
  while IFS= read -r step; do
    [ -z "${step}" ] && continue
    step_counts["${step}"]=$((step_counts["${step}"] + 1))
  done < <(find "${model_path}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n)
done

shared_steps=()
for step in "${!step_counts[@]}"; do
  if [ "${step_counts[${step}]}" -eq "${#model_dirs[@]}" ]; then
    shared_steps+=("${step}")
  fi
done
if [ "${#shared_steps[@]}" -gt 0 ]; then
  IFS=$'\n' shared_steps_sorted=($(printf "%s\n" "${shared_steps[@]}" | sort -n))
else
  shared_steps_sorted=()
fi
for model in "${model_dirs[@]}"; do
  echo "${model}" >> "${MODELS_TXT}"
  model_path="${MODEL_ROOT}/${model}"
  mapfile -t steps < <(find "${model_path}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n)
  if [ "${#steps[@]}" -eq 0 ]; then
    echo -e "${model}\t\t\t\tNO_STEPS\t" >> "${RESULTS_TSV}"
    continue
  fi

  # Build evaluation order: shared steps first (descending), then remaining steps (ascending).
  eval_order=()
  if [ "${#shared_steps_sorted[@]}" -gt 0 ]; then
    for step in $(printf "%s\n" "${shared_steps_sorted[@]}" | sort -nr); do
      for s in "${steps[@]}"; do
        if [ "${s}" = "${step}" ]; then
          eval_order+=("${step}")
          break
        fi
      done
    done
  fi
  for s in "${steps[@]}"; do
    skip=false
    for es in "${eval_order[@]-}"; do
      if [ "${es}" = "${s}" ]; then
        skip=true
        break
      fi
    done
    if [ "${skip}" = "false" ]; then
      eval_order+=("${s}")
    fi
  done

  order_idx=0
  for step in "${eval_order[@]}"; do
    if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [ -n "${STEP_MOD}" ] && (( step % STEP_MOD != 0 )); then
      continue
    fi
    if [ -n "${STEP_SHARDS}" ]; then
      if (( order_idx % STEP_SHARDS != STEP_SHARD_ID )); then
        order_idx=$((order_idx + 1))
        continue
      fi
    fi
    if [ -n "${STEP_SHARDS}" ]; then
      order_idx=$((order_idx + 1))
    fi

    run_tag="${model//\//_}_step_${step}"
    run_dir="${OUT_DIR}/runs/${run_tag}"
    mkdir -p "${run_dir}"
    eval_log="${run_dir}/eval.log"
    rollout_log_dir="${model_path}/global_step_${step}/${ROLLOUT_LOG_SUBDIR}/${RUN_TS}"
    mkdir -p "${rollout_log_dir}"

    ckpt_path="${model_path}/global_step_${step}/actor"
    model_path_hf="${ckpt_path}/huggingface"

    if apptainer exec "${APPTAINER_ARGS[@]}" "$APPTAINER_IMG" \
      bash -lc "\
        source '${VENV_AGENTGYM_RL}/bin/activate' && \
        if [ ! -d '${ckpt_path}' ]; then echo '[webshop_eval_all] Missing checkpoint: ${ckpt_path}' >&2; exit 1; fi && \
        if [ ! -f '${model_path_hf}/model.safetensors.index.json' ] && \
           [ ! -f '${model_path_hf}/model.safetensors' ] && \
           [ ! -f '${model_path_hf}/pytorch_model.bin' ] && \
           ! ls '${model_path_hf}'/model-*.safetensors '${model_path_hf}'/pytorch_model-*.bin >/dev/null 2>&1; then \
          python3 AgentGym-RL/scripts/model_merger.py --local_dir '${ckpt_path}'; \
        fi && \
        HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
          hydra.job.chdir=False \
          trainer.nnodes=1 \
          trainer.n_gpus_per_node='${N_GPUS_PER_NODE}' \
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
          rollout.rollout_log_dir='${rollout_log_dir}'\
      " > "${eval_log}" 2>&1; then
      avg=$(awk -F': ' "/Avg@${SAMPLE_NUM}:/ {print \$2; exit}" "${eval_log}" | tr -d ' ')
      pass=$(awk -F': ' "/Pass@${SAMPLE_NUM}:/ {print \$2; exit}" "${eval_log}" | tr -d ' ')
      status="OK"
      if [ -z "${avg}" ] || [ -z "${pass}" ]; then
        status="PARSE_FAIL"
      fi
    else
      status="EVAL_FAIL"
      avg=""
      pass=""
      echo "${model} step ${step} eval failed" >> "${FAIL_LOG}"
    fi

    echo -e "${model}\t${step}\t${avg}\t${pass}\t${status}\t${eval_log}" >> "${RESULTS_TSV}"
  done
done

TABLE_PASS_CSV="${OUT_DIR}/webshop_pass_table.csv"
TABLE_AVG_CSV="${OUT_DIR}/webshop_avg_table.csv"
export RESULTS_TSV MODELS_TXT TABLE_PASS_CSV TABLE_AVG_CSV
python3 - <<'PY'
import csv
import os

results_tsv = os.environ['RESULTS_TSV']
models_txt = os.environ['MODELS_TXT']
pass_csv = os.environ['TABLE_PASS_CSV']
avg_csv = os.environ['TABLE_AVG_CSV']

models = []
with open(models_txt, 'r', encoding='utf-8') as f:
    for line in f:
        name = line.strip()
        if name:
            models.append(name)

steps = set()
pass_map = {}
avg_map = {}
with open(results_tsv, 'r', encoding='utf-8') as f:
    _ = f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 5:
            continue
        model, step, avg, pas, status = parts[:5]
        if not step.isdigit():
            continue
        step_i = int(step)
        steps.add(step_i)
        if avg != '':
            avg_map.setdefault(model, {})[step_i] = avg
        if pas != '':
            pass_map.setdefault(model, {})[step_i] = pas

steps_sorted = sorted(steps)

def write_table(path, value_map):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + steps_sorted)
        for model in models:
            row = [model] + [value_map.get(model, {}).get(s, '') for s in steps_sorted]
            writer.writerow(row)

write_table(pass_csv, pass_map)
write_table(avg_csv, avg_map)
PY

printf "\nDone. Results written to:\n- %s\n- %s\n- %s\n" "${RESULTS_TSV}" "${TABLE_PASS_CSV}" "${TABLE_AVG_CSV}"
