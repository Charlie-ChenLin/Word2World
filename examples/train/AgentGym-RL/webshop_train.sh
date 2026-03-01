set -euo pipefail
set -x

usage() {
    cat <<'EOF_USAGE'
Usage: bash webshop_train.sh [options]
All options override the defaults in this script. They can also be set via env vars.
  --task_name NAME
  --env_server_url URL
  --n_gpus_per_node N
  --pure_agent_model_name NAME
  --agent_model_path PATH
  --train_file PATH
  --kl_coef FLOAT
  --policy_learning_rate FLOAT
  --rollout_sample_num INT
  --train_batch_size INT
  --ppo_mini_batch_size INT
  --ppo_micro_batch_size_per_gpu INT
  --ppo_inner_epochs INT
  --total_epochs INT
  --total_training_steps INT
  --model_save_dir DIR
  --exp_name NAME
  --max_prompt_length INT
  --max_response_length INT
  --max_model_len INT
  --max_tokens INT
  --rollout_gpu_mem_util FLOAT
  --rollout_tp_size INT
  --actor_sp_size INT
  --ref_sp_size INT
  --critic_sp_size INT
  --use_remove_padding BOOL
  --critic_use_remove_padding BOOL
  --rounds INT
  --project_name NAME
  --save_freq INT
  --grad_clip FLOAT
  --env_feedback_loss_coef FLOAT
  --env_feedback_grad_proj_to_pg BOOL
  --env_feedback_grad_proj_eps FLOAT
  --env_feedback_grad_proj_impl NAME
  --env_feedback_grad_proj_legacy_compat_metrics BOOL
  --schedule_env_feedback_loss_coef BOOL
  --final_env_feedback_loss_coef FLOAT
  --scheduling_algo NAME
  --webshop_rule_reward BOOL
  --webshop_rule_reward_success FLOAT
  --webshop_rule_reward_fail FLOAT
  -h, --help
EOF_USAGE
}

# Default values (can be overridden by env or CLI)
task_name="${TASK_NAME:-webshop}"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36001}"
n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
pure_agent_model_name="${PURE_AGENT_MODEL_NAME:-Qwen2.5-3B-Instruct}"
agent_model_path="${AGENT_MODEL_PATH:-/mnt/shared-storage-user/formalverification-shared/openai-community/Qwen/${pure_agent_model_name}}"
train_file="${TRAIN_FILE:-${DATA_ROOT:-data}/train/webshop_train.json}"
kl_coef="${KL_COEF:-0.001}"
policy_learning_rate="${POLICY_LR:-1e-6}"
rollout_sample_num="${ROLLOUT_N:-8}"
train_batch_size="${TRAIN_BATCH_SIZE:-8}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ppo_inner_epochs="${PPO_INNER_EPOCHS:-1}"
total_epochs="${TOTAL_EPOCHS:-10}"
total_training_steps="${TOTAL_TRAINING_STEPS:-}"
model_save_dir="${MODEL_SAVE_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves}"
exp_name="${EXP_NAME:-test}"
max_prompt_length="${MAX_PROMPT_LEN:-1024}"
max_response_length="${MAX_RESPONSE_LEN:-28672}"
max_model_len="${MAX_MODEL_LEN:-32768}"
max_tokens="${MAX_TOKENS:-200}"
rollout_gpu_mem_util="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
rollout_tp_size="${ROLLOUT_TP_SIZE:-1}"
actor_sp_size="${ACTOR_SP_SIZE:-${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}}"
ref_sp_size="${REF_SP_SIZE:-${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}}"
critic_sp_size="${CRITIC_SP_SIZE:-${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}}"
use_remove_padding="${USE_REMOVE_PADDING:-false}"
critic_use_remove_padding="${CRITIC_USE_REMOVE_PADDING:-${use_remove_padding}}"
rounds="${ROUNDS:-30}"
project_name="${PROJECT_NAME:-xxx}"
save_freq="${SAVE_FREQ:-50}"
grad_clip="${GRAD_CLIP:-1.0}"
env_feedback_loss_coef="${ENV_FEEDBACK_LOSS_COEF:-1.0}"
env_feedback_grad_proj_to_pg="${ENV_FEEDBACK_GRAD_PROJ_TO_PG:-false}"
env_feedback_grad_proj_eps="${ENV_FEEDBACK_GRAD_PROJ_EPS:-1e-12}"
env_feedback_grad_proj_impl="${ENV_FEEDBACK_GRAD_PROJ_IMPL:-reordered_exact}"
env_feedback_grad_proj_legacy_compat_metrics="${ENV_FEEDBACK_GRAD_PROJ_LEGACY_COMPAT_METRICS:-false}"
schedule_env_feedback_loss_coef="${SCHEDULE_ENV_FEEDBACK_LOSS_COEF:-false}"
final_env_feedback_loss_coef="${FINAL_ENV_FEEDBACK_LOSS_COEF:-}"
scheduling_algo="${SCHEDULING_ALGO:-linear}"
webshop_rule_reward="${WEBSHOP_RULE_REWARD:-0}"
webshop_rule_reward_success="${WEBSHOP_RULE_REWARD_SUCCESS:-10}"
webshop_rule_reward_fail="${WEBSHOP_RULE_REWARD_FAIL:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --task_name) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; task_name="$2"; shift 2 ;;
        --env_server_url) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_server_url="$2"; shift 2 ;;
        --n_gpus_per_node) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; n_gpus_per_node="$2"; shift 2 ;;
        --pure_agent_model_name) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; pure_agent_model_name="$2"; shift 2 ;;
        --agent_model_path) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; agent_model_path="$2"; shift 2 ;;
        --train_file) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; train_file="$2"; shift 2 ;;
        --kl_coef) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; kl_coef="$2"; shift 2 ;;
        --policy_learning_rate|--lr) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; policy_learning_rate="$2"; shift 2 ;;
        --rollout_sample_num|--rollout_n) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_sample_num="$2"; shift 2 ;;
        --train_batch_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; train_batch_size="$2"; shift 2 ;;
        --ppo_mini_batch_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ppo_mini_batch_size="$2"; shift 2 ;;
        --ppo_micro_batch_size_per_gpu) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ppo_micro_batch_size_per_gpu="$2"; shift 2 ;;
        --ppo_inner_epochs) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ppo_inner_epochs="$2"; shift 2 ;;
        --total_epochs) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; total_epochs="$2"; shift 2 ;;
        --total_training_steps) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; total_training_steps="$2"; shift 2 ;;
        --model_save_dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; model_save_dir="$2"; shift 2 ;;
        --exp_name) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; exp_name="$2"; shift 2 ;;
        --max_prompt_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_prompt_length="$2"; shift 2 ;;
        --max_response_length) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_response_length="$2"; shift 2 ;;
        --max_model_len) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_model_len="$2"; shift 2 ;;
        --max_tokens) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; max_tokens="$2"; shift 2 ;;
        --rollout_gpu_mem_util) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_gpu_mem_util="$2"; shift 2 ;;
        --rollout_tp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rollout_tp_size="$2"; shift 2 ;;
        --actor_sp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; actor_sp_size="$2"; shift 2 ;;
        --ref_sp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; ref_sp_size="$2"; shift 2 ;;
        --critic_sp_size) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; critic_sp_size="$2"; shift 2 ;;
        --use_remove_padding) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; use_remove_padding="$2"; shift 2 ;;
        --critic_use_remove_padding) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; critic_use_remove_padding="$2"; shift 2 ;;
        --rounds) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; rounds="$2"; shift 2 ;;
        --project_name) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; project_name="$2"; shift 2 ;;
        --save_freq) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; save_freq="$2"; shift 2 ;;
        --grad_clip) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; grad_clip="$2"; shift 2 ;;
        --env_feedback_loss_coef) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_feedback_loss_coef="$2"; shift 2 ;;
        --env_feedback_grad_proj_to_pg) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_feedback_grad_proj_to_pg="$2"; shift 2 ;;
        --env_feedback_grad_proj_eps) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_feedback_grad_proj_eps="$2"; shift 2 ;;
        --env_feedback_grad_proj_impl) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_feedback_grad_proj_impl="$2"; shift 2 ;;
        --env_feedback_grad_proj_legacy_compat_metrics) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; env_feedback_grad_proj_legacy_compat_metrics="$2"; shift 2 ;;
        --schedule_env_feedback_loss_coef) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; schedule_env_feedback_loss_coef="$2"; shift 2 ;;
        --final_env_feedback_loss_coef) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; final_env_feedback_loss_coef="$2"; shift 2 ;;
        --scheduling_algo) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; scheduling_algo="$2"; shift 2 ;;
        --webshop_rule_reward) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; webshop_rule_reward="$2"; shift 2 ;;
        --webshop_rule_reward_success) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; webshop_rule_reward_success="$2"; shift 2 ;;
        --webshop_rule_reward_fail) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; webshop_rule_reward_fail="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" 1>&2; usage; exit 1 ;;
    esac
done

normalize_bool() {
    case "$1" in
        1|true|TRUE|True|yes|YES|y|Y) echo "true" ;;
        0|false|FALSE|False|no|NO|n|N|"") echo "false" ;;
        *) echo "$1" ;;
    esac
}
env_feedback_grad_proj_to_pg="$(normalize_bool "${env_feedback_grad_proj_to_pg}")"
schedule_env_feedback_loss_coef="$(normalize_bool "${schedule_env_feedback_loss_coef}")"
if [ -z "${final_env_feedback_loss_coef}" ]; then
    final_env_feedback_loss_coef="${env_feedback_loss_coef}"
fi

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Activate training env (uv preferred; fall back to conda template used by other example scripts).
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"
if [ ! -f "${VENV_AGENTGYM_RL}/bin/activate" ]; then
    echo "[webshop_train.sh] Missing venv: ${VENV_AGENTGYM_RL}" 1>&2
    exit 1
fi
source "${VENV_AGENTGYM_RL}/bin/activate"

# Offline training (no internet access; only talks to local env server).
# NOTE: This requires `AGENT_MODEL_PATH` to point to a fully local HF model dir.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export WANDB_MODE=offline
export WANDB_SILENT=true
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export WEBSHOP_RULE_REWARD="${webshop_rule_reward}"
export WEBSHOP_RULE_REWARD_SUCCESS="${webshop_rule_reward_success}"
export WEBSHOP_RULE_REWARD_FAIL="${webshop_rule_reward_fail}"

WANDB_DIR="${WANDB_DIR:-/mnt/inspurfs/chenlin1/Word2World/wandb}"
mkdir -p "${WANDB_DIR}"
export WANDB_DIR

OUTPUTS_ROOT="${OUTPUTS_ROOT:-/mnt/inspurfs/chenlin1/Word2World/outputs}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${OUTPUTS_ROOT}/${exp_name}}"
mkdir -p "${OUTPUTS_ROOT}"

if [ ! -f "${agent_model_path}/config.json" ]; then
    echo "[webshop_train.sh] Missing local model at: ${agent_model_path}" 1>&2
    echo "[webshop_train.sh] Set AGENT_MODEL_PATH to a local HuggingFace model directory." 1>&2
    exit 1
fi
if [ ! -f "${train_file}" ]; then
    echo "[webshop_train.sh] Missing train file: ${train_file}" 1>&2
    echo "[webshop_train.sh] Set DATA_ROOT to where data/train lives (default: ./data)." 1>&2
    exit 1
fi

mkdir -p "${model_save_dir}"
model_save_path="${model_save_dir}/${exp_name}"
mkdir -p "${model_save_path}"

extra_trainer_overrides=()
if [ -n "${total_training_steps}" ]; then
    extra_trainer_overrides+=("trainer.total_training_steps=${total_training_steps}")
fi

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 WANDB_MODE=offline python3 -m verl.agent_trainer.main_ppo  \
    hydra.job.chdir=False \
    hydra.run.dir="${HYDRA_RUN_DIR}" \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=${rounds} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    data.train_file="${train_file}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_tokens=${max_tokens} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    critic.model.use_remove_padding=${critic_use_remove_padding} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.env_feedback_loss_coef=${env_feedback_loss_coef} \
    actor_rollout_ref.actor.project_env_feedback_grad_to_pg=${env_feedback_grad_proj_to_pg} \
    actor_rollout_ref.actor.env_feedback_grad_proj_eps=${env_feedback_grad_proj_eps} \
    actor_rollout_ref.actor.env_feedback_grad_proj_impl=${env_feedback_grad_proj_impl} \
    actor_rollout_ref.actor.env_feedback_grad_proj_legacy_compat_metrics=${env_feedback_grad_proj_legacy_compat_metrics} \
    actor_rollout_ref.actor.env_feedback_loss_coef_schedule.enabled=${schedule_env_feedback_loss_coef} \
    actor_rollout_ref.actor.env_feedback_loss_coef_schedule.final_env_feedback_loss_coef=${final_env_feedback_loss_coef} \
    actor_rollout_ref.actor.env_feedback_loss_coef_schedule.scheduling_algo=${scheduling_algo} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    critic.grad_clip=${grad_clip} \
    "${extra_trainer_overrides[@]}"
status=$?
exit $status
