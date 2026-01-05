set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_NO_USAGE_STATS=1

task_name="alfworld"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Activate training env (uv preferred; fall back to conda template used by other example scripts).

source ../uv_agentgym_rl/bin/activate


export VLLM_ATTENTION_BACKEND=XFORMERS

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

export WANDB_DIR="./wandb"


env_server_url="http://127.0.0.1:36001"

# pure_agent_model_name="Qwen2.5-7B-Instruct"
pure_agent_model_name="Qwen2.5-3B-Instruct"
agent_model_path="/mnt/shared-storage-user/formalverification-shared/openai-community/Qwen/${pure_agent_model_name}"
train_file="${DATA_ROOT:-data}/train/${task_name}_train.json"

if [ ! -f "${agent_model_path}/config.json" ]; then
    echo "[alfworld_train.sh] Missing local model at: ${agent_model_path}" 1>&2
    echo "[alfworld_train.sh] Set AGENT_MODEL_PATH to a local HuggingFace model directory." 1>&2
    exit 1
fi
if [ ! -f "${train_file}" ]; then
    echo "[alfworld_train.sh] Missing train file: ${train_file}" 1>&2
    echo "[alfworld_train.sh] Set DATA_ROOT to where AgentItemId lives (default: ./data)." 1>&2
    exit 1
fi

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=8
train_batch_size=16
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1

total_epoches=10

model_save_dir="AgentGym-RL/saves"
mkdir -p ${model_save_dir}
exp_name="test"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline python3 -m verl.agent_trainer.main_ppo  \
    hydra.job.chdir=False \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=30 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    data.train_file="${train_file}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=xxx \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches}
status=$?
exit $status
