set -euo pipefail
set -x

usage() {
    cat <<'EOF'
Usage: bash alfworld_eval.sh [options]
All options override the defaults in this script. They can also be set via env vars.
  --task_name NAME
  --env_server_url URL
  --n_gpus_per_node N
  --data_root DIR
  --eval_path DIR
  --sample_num INT
  --max_rounds INT
  --model_save_path DIR
  --ckpt_step INT
  --max_prompt_length INT
  --max_response_length INT
  --batch_size INT
  --rollout_gpu_mem_util FLOAT
  --rollout_temperature FLOAT
  --rollout_max_model_len INT
  --rollout_max_tokens INT
  --rollout_tp_size INT
  --rollout_log_dir DIR
  -h, --help
EOF
}

task_name="${TASK_NAME:-alfworld}"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36001}"
n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
data_root="${DATA_ROOT:-data}"
eval_path="${EVAL_PATH:-}"
sample_num="${SAMPLE_NUM:-1}"
max_rounds="${MAX_ROUNDS:-30}"
model_save_path="${MODEL_SAVE_PATH:-AgentGym-RL/saves/test-l40s}"
ckpt_step="${CKPT_STEP:-175}"
max_prompt_length="${MAX_PROMPT_LEN:-1024}"
max_response_length="${MAX_RESPONSE_LEN:-8192}"
batch_size="${BATCH_SIZE:-32}"
rollout_gpu_mem_util="${ROLLOUT_GPU_MEM_UTIL:-0.95}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-1}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-32768}"
rollout_max_tokens="${ROLLOUT_MAX_TOKENS:-200}"
rollout_tp_size="${ROLLOUT_TP_SIZE:-1}"
rollout_log_dir="${ROLLOUT_LOG_DIR:-executer_logs}"

while [ $# -gt 0 ]; do
    case "$1" in
        --task_name) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; task_name="$2"; shift 2 ;;
        --env_server_url) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_server_url="$2"; shift 2 ;;
        --n_gpus_per_node) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; n_gpus_per_node="$2"; shift 2 ;;
        --data_root) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; data_root="$2"; shift 2 ;;
        --eval_path) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; eval_path="$2"; shift 2 ;;
        --sample_num) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; sample_num="$2"; shift 2 ;;
        --max_rounds) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_rounds="$2"; shift 2 ;;
        --model_save_path) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; model_save_path="$2"; shift 2 ;;
        --ckpt_step) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ckpt_step="$2"; shift 2 ;;
        --max_prompt_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_prompt_length="$2"; shift 2 ;;
        --max_response_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_response_length="$2"; shift 2 ;;
        --batch_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; batch_size="$2"; shift 2 ;;
        --rollout_gpu_mem_util) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_gpu_mem_util="$2"; shift 2 ;;
        --rollout_temperature) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_temperature="$2"; shift 2 ;;
        --rollout_max_model_len) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_max_model_len="$2"; shift 2 ;;
        --rollout_max_tokens) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_max_tokens="$2"; shift 2 ;;
        --rollout_tp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_tp_size="$2"; shift 2 ;;
        --rollout_log_dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_log_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" 1>&2; usage; exit 1 ;;
    esac
done

if [ -z "${eval_path}" ]; then
    eval_path="${data_root}/AgentEval/${task_name}"
fi

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate training env (uv preferred; fall back to conda template used by other example scripts).
if [ -f "uv_agentgym_rl/bin/activate" ]; then
    source uv_agentgym_rl/bin/activate
else
    source activate
    conda activate agentgym-rl
fi

if [ ! -d "${eval_path}" ]; then
    echo "[alfworld_eval.sh] Missing eval data dir: ${eval_path}" 1>&2
    echo "[alfworld_eval.sh] Run: python scripts/download_data/download_data.py" 1>&2
    exit 1
fi

ckpt_path="${model_save_path}/global_step_${ckpt_step}/actor"
model_path="${ckpt_path}/huggingface"
if [ ! -d "${ckpt_path}" ]; then
    echo "[alfworld_eval.sh] Missing checkpoint dir: ${ckpt_path}" 1>&2
    exit 1
fi

# Merge sharded FSDP weights into a single HF checkpoint.
python3 AgentGym-RL/scripts/model_merger.py --local_dir "${ckpt_path}"

HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation  \
    hydra.job.chdir=False \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    data.path="${eval_path}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.n_samples=${sample_num} \
    data.batch_size=${batch_size} \
    agentgym.task_name="${task_name}" \
    agentgym.env_addr="${env_server_url}" \
    agentgym.max_rounds="${max_rounds}" \
    agentgym.timeout=500 \
    model.path="${model_path}" \
    rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    rollout.temperature=${rollout_temperature} \
    rollout.max_model_len=${rollout_max_model_len} \
    rollout.max_tokens=${rollout_max_tokens} \
    rollout.tensor_model_parallel_size=${rollout_tp_size} \
    rollout.rollout_log_dir="${rollout_log_dir}"

status=$?
exit $status
