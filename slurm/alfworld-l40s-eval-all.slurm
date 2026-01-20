#!/bin/bash
#SBATCH --job-name alfworld-eval-all
#SBATCH --partition=L40S,ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=./log/alfworld/alfworld-eval-all-%j.out
#SBATCH --error=./log/alfworld/alfworld-eval-all-%j.err

set -euo pipefail
cd /home/lchen/home/Word2World

mkdir -p ./log/alfworld
JOB_DIR="./log/alfworld/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME="${CUDA_HOME:-/data1/public/cuda/cuda-12.1}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID:-triton_cache}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
export PYDEV_PREFIX=~/local/python310-dev
export CPATH=$PYDEV_PREFIX/usr/include:$PYDEV_PREFIX/usr/include/python3.10:$PYDEV_PREFIX/usr/include/x86_64-linux-gnu/python3.10:${CPATH:-}
export LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$PYDEV_PREFIX/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Eval configs (override via env or sbatch --export)
export MODEL_ROOT="${MODEL_ROOT:-AgentGym-RL/saves}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export DATA_ROOT="${DATA_ROOT:-data}"
export SAMPLE_NUM="${SAMPLE_NUM:-1}"
export MAX_ROUNDS="${MAX_ROUNDS:-30}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"
export ROLLOUT_MAX_TOKENS="${ROLLOUT_MAX_TOKENS:-200}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.95}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-$((36001 + (SLURM_JOB_ID % 1000)))}"
export ENV_SERVER_URL="http://127.0.0.1:${ENV_SERVER_PORT}"

RESULTS_TSV="${JOB_DIR}/results.tsv"
MODELS_TXT="${JOB_DIR}/models.txt"
FAIL_LOG="${JOB_DIR}/failures.log"
TABLE_PASS_CSV="${JOB_DIR}/alfworld_pass_table.csv"
TABLE_AVG_CSV="${JOB_DIR}/alfworld_avg_table.csv"

mkdir -p "${TRITON_CACHE_DIR}"

# Ensure eval data is present (bootstrap from local eval files when possible).
if [ ! -d "${DATA_ROOT}/AgentEval/alfworld" ]; then
    if [ -f "${DATA_ROOT}/eval/alfworld_test.json" ]; then
        echo "[alfworld-eval-all] Bootstrapping ${DATA_ROOT}/AgentEval/alfworld from ${DATA_ROOT}/eval/alfworld_test.json"
        mkdir -p "${DATA_ROOT}/AgentEval/alfworld"
        cp "${DATA_ROOT}/eval/alfworld_test.json" "${DATA_ROOT}/AgentEval/alfworld/alfworld_test.json"
        python3 - <<'PY'
import json
import os

data_root = os.environ.get("DATA_ROOT", "data")
src = os.path.join(data_root, "eval", "alfworld_test.json")
dst = os.path.join(data_root, "AgentEval", "alfworld", "alfworld_all.json")
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f)
PY
    else
        echo "[alfworld-eval-all] Missing ${DATA_ROOT}/eval/alfworld_test.json; downloading dataset..."
        bash -lc "source uv_agentgym_rl/bin/activate && python3 scripts/download_data/download_data.py --output_dir \"${DATA_ROOT}\"" \
            > "${JOB_DIR}/download_data.log" 2>&1
    fi
fi

# Discover models
mapfile -t model_dirs < <(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
: > "${MODELS_TXT}"
for model in "${model_dirs[@]}"; do
    echo "${model}" >> "${MODELS_TXT}"
    # pre-create model row so empty models still show up
    :
done

echo -e "model\tstep\tavg\tpass\tstatus" > "${RESULTS_TSV}"
: > "${FAIL_LOG}"

# Start env server (CPU only) on a per-job port to avoid conflicts
srun --ntasks=1 --cpus-per-task=20 --gres=gpu:0 --overlap \
     --export=ALL,CUDA_VISIBLE_DEVICES= \
     bash -lc "source /home/lchen/home/uv_alfworld/bin/activate && alfworld --host 0.0.0.0 --port ${ENV_SERVER_PORT}" \
     > "${JOB_DIR}/env_server.log" 2>&1 &
server_pid=$!
trap 'kill ${server_pid} 2>/dev/null || true; wait ${server_pid} 2>/dev/null || true' EXIT

sleep 5

for model in "${model_dirs[@]}"; do
    model_path="${MODEL_ROOT}/${model}"
    mapfile -t steps < <(find "${model_path}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n)
    if [ "${#steps[@]}" -eq 0 ]; then
        echo -e "${model}\t\t\t\tNO_STEPS" >> "${RESULTS_TSV}"
        continue
    fi

    for step in "${steps[@]}"; do
        if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        if (( step % 25 != 0 )); then
            continue
        fi

        run_tag="${model//\//_}_step_${step}"
        run_dir="${JOB_DIR}/runs/${run_tag}"
        mkdir -p "${run_dir}"
        eval_log="${run_dir}/eval.log"
        rollout_log_dir="${model_path}/global_step_${step}/eval_logs/${RUN_TS}"
        mkdir -p "${rollout_log_dir}"

        if srun --ntasks=1 --cpus-per-task=100 --gres=gpu:${N_GPUS_PER_NODE} \
                --cpu-bind=cores \
                bash examples/eval/alfworld_eval.sh \
                --n_gpus_per_node "${N_GPUS_PER_NODE}" \
                --model_save_path "${model_path}" \
                --ckpt_step "${step}" \
                --data_root "${DATA_ROOT}" \
                --env_server_url "${ENV_SERVER_URL}" \
                --sample_num "${SAMPLE_NUM}" \
                --max_rounds "${MAX_ROUNDS}" \
                --max_prompt_length "${MAX_PROMPT_LEN}" \
                --max_response_length "${MAX_RESPONSE_LEN}" \
                --rollout_max_model_len "${ROLLOUT_MAX_MODEL_LEN}" \
                --rollout_max_tokens "${ROLLOUT_MAX_TOKENS}" \
                --rollout_gpu_mem_util "${ROLLOUT_GPU_MEM_UTIL}" \
                --rollout_tp_size "${ROLLOUT_TP_SIZE}" \
                --batch_size "${BATCH_SIZE}" \
                --rollout_log_dir "${rollout_log_dir}" \
                > "${eval_log}" 2>&1; then
            avg=$(awk -F': ' '/Avg@'"${SAMPLE_NUM}"':/{print $2; exit}' "${eval_log}" | tr -d ' ')
            pass=$(awk -F': ' '/Pass@'"${SAMPLE_NUM}"':/{print $2; exit}' "${eval_log}" | tr -d ' ')
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

        echo -e "${model}\t${step}\t${avg}\t${pass}\t${status}" >> "${RESULTS_TSV}"
    done
done

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
    header = f.readline()
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

printf "\nDone. Results:\n- %s\n- %s\n- %s\n" "${RESULTS_TSV}" "${TABLE_PASS_CSV}" "${TABLE_AVG_CSV}"
