set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="alfworld"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate training env (uv preferred; fall back to conda template used by other example scripts).
if [ -f "uv_agentgym_rl/bin/activate" ]; then
    source uv_agentgym_rl/bin/activate
else
    source activate
    conda activate agentgym-rl
fi

export VLLM_ATTENTION_BACKEND=XFORMERS

# AlfWorld env server (start in another terminal):
#   bash scripts/env_server/start_alfworld.sh
env_server_url="http://127.0.0.1:36001"

# Eval set path (download via `python scripts/download_data/download_data.py`).
DATA_ROOT="${DATA_ROOT:-data}"
agent_eval_path="${DATA_ROOT}/AgentEval/${task_name}"

sample_num=1
max_rounds=30

# Set to your checkpoint actor dir, e.g.:
#   MODEL_SAVE_PATH="AgentGym-RL/saves/test" CKPT_STEP="150" bash examples/eval/alfworld_eval.sh
MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-AgentGym-RL/saves/test}"
CKPT_STEP="${CKPT_STEP:-150}"
ckpt_path="${MODEL_SAVE_PATH}/global_step_${CKPT_STEP}/actor"
model_path="${ckpt_path}/huggingface"

# Merge sharded FSDP weights into a single HF checkpoint.
python3 AgentGym-RL/scripts/model_merger.py --local_dir "${ckpt_path}"

HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation  \
    hydra.job.chdir=False \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    data.path="${agent_eval_path}" \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.n_samples="${sample_num}" \
    data.batch_size=32 \
    agentgym.task_name="${task_name}" \
    agentgym.env_addr="${env_server_url}" \
    agentgym.max_rounds="${max_rounds}" \
    agentgym.timeout=500 \
    model.path="${model_path}" \
    rollout.gpu_memory_utilization=0.95 \
    rollout.temperature=1 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=200 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir=executer_logs

status=$?
exit $status
