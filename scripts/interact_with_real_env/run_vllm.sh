#!/usr/bin/env bash
set -euo pipefail

TASK=$1
RUN=${2:-0}
MODEL=$3
MAX_CONCURRENCY=${4:-150}
MAX_ROUND=${5:-20}
NUM_EXAMPLES=${6:--1}
SPLIT=${7:-test}
OUTPUT_ROOT=${8:-outputs}

OUTPUT_DIR=$OUTPUT_ROOT/interaction/real_env/$SPLIT/vllm/${TASK}/$MODEL/${TASK}_maxround${MAX_ROUND}_run${RUN}

# if global_step_ in MODEL, then NEED_MERGE=1
if [[ "$MODEL" == *"global_step_"* ]]; then
    NEED_MERGE=1
else
    NEED_MERGE=0
fi


INFERENCE_FILE=${INFERENCE_FILE:-}
if [ -z "$INFERENCE_FILE" ]; then
    if [ "$SPLIT" = "test" ]; then
        INFERENCE_FILE="data/eval/${TASK}_test.json"
    elif [ "$SPLIT" = "train" ]; then
        INFERENCE_FILE="data/train/${TASK}_train.json"
    elif [ "$SPLIT" = "valid_seen" ]; then
        INFERENCE_FILE="data/eval/${TASK}_valid_seen.json"
    elif [ "$SPLIT" = "valid_unseen" ]; then
        INFERENCE_FILE="data/eval/${TASK}_valid_unseen.json"
    else
        echo "Unsupported SPLIT: $SPLIT" >&2
        exit 1
    fi
fi

ENV_PORT=${ENV_PORT:-$((30000 + RANDOM % (99999-30000+1)))}
ENV_WAIT_SECONDS=${ENV_WAIT_SECONDS:-10}
ENV_LOG_FILE="/tmp/${TASK}_server_${ENV_PORT}.log"
ENV_SERVER_BASE=${ENV_SERVER_BASE:-}
ENV_SERVER_PID=""
VLLM_SERVER_PID=""
VLLM_LOG_FILE=""

cleanup() {
    if [ -n "${VLLM_SERVER_PID:-}" ]; then
        echo "Stopping vLLM..."
        kill "${VLLM_SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${ENV_SERVER_PID:-}" ]; then
        echo "Stopping env server..."
        kill "${ENV_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [ -n "${ENV_START_CMD:-}" ]; then
    export ENV_PORT TASK
    bash -lc "$ENV_START_CMD" >"$ENV_LOG_FILE" 2>&1 &
    ENV_SERVER_PID=$!
    if [ -z "$ENV_SERVER_BASE" ]; then
        ENV_SERVER_BASE="http://127.0.0.1:$ENV_PORT"
    fi
elif [ "$TASK" = "alfworld" ]; then
    source uv_alfworld/bin/activate
    alfworld --host 0.0.0.0 --port "$ENV_PORT" >"$ENV_LOG_FILE" 2>&1 &
    ENV_SERVER_PID=$!
    ENV_SERVER_BASE=${ENV_SERVER_BASE:-"http://127.0.0.1:$ENV_PORT"}
elif [ "$TASK" = "sciworld" ]; then
    source uv_sciworld/bin/activate
    sciworld --host 0.0.0.0 --port "$ENV_PORT" >"$ENV_LOG_FILE" 2>&1 &
    ENV_SERVER_PID=$!
    ENV_SERVER_BASE=${ENV_SERVER_BASE:-"http://127.0.0.1:$ENV_PORT"}
elif [ "$TASK" = "textworld" ]; then
    source uv_textworld/bin/activate
    textworld --host 0.0.0.0 --port "$ENV_PORT" >"$ENV_LOG_FILE" 2>&1 &
    ENV_SERVER_PID=$!
    ENV_SERVER_BASE=${ENV_SERVER_BASE:-"http://127.0.0.1:$ENV_PORT"}
elif [ "$TASK" = "webshop" ]; then
    source uv_webshop/bin/activate
    webshop --host 0.0.0.0 --port "$ENV_PORT" >"$ENV_LOG_FILE" 2>&1 &
    ENV_SERVER_PID=$!
    ENV_SERVER_BASE=${ENV_SERVER_BASE:-"http://127.0.0.1:$ENV_PORT"}
else
    echo "Unsupported built-in task: $TASK" >&2
    echo "Set ENV_START_CMD + ENV_SERVER_BASE (or ENV_PORT), and --task_class_path if needed." >&2
    exit 1
fi
echo "Launching Environment server... (pid=$ENV_SERVER_PID, port=$ENV_PORT)"
sleep "$ENV_WAIT_SECONDS"
echo "Environment server base: $ENV_SERVER_BASE"
echo "Logs: $ENV_LOG_FILE"



## ====== Start vLLM Server ======
source uv_agentgym_rl/bin/activate
if [ $NEED_MERGE -eq 1 ]; then
    echo "Merging model checkpoints for vLLM serving..."
    python AgentGym-RL/scripts/multiple_model_merger.py --local_dir $MODEL --save_dir $MODEL
    MODEL=$MODEL/huggingface
    echo "Merged model saved to $MODEL"
fi

# ===== 1. Pick a free port =====
VLLM_PORT=$((30000 + RANDOM % (99999-30000+1)))
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $NUM_GPUS"

# ===== 2. Start vLLM in background =====
VLLM_LOG_FILE="/tmp/vllm_${VLLM_PORT}.log"
vllm serve "$MODEL" \
    --port $VLLM_PORT \
    --served-model-name vllm_model \
    --tensor-parallel-size $NUM_GPUS \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    </dev/null >"$VLLM_LOG_FILE" 2>&1 &
VLLM_SERVER_PID=$!
echo "Launching vLLM server... (pid=$VLLM_SERVER_PID, port=$VLLM_PORT)"

# ===== 4. Wait for vLLM to be ready =====
wait_for_server() {
    local port=$1
    local log_file=$2
    local max_wait_time=600  # 10 minutes (600 seconds)
    local wait_time=0
    tail -f $log_file &   # This will print logs in real-time
    tail_pid=$!           # Save the process ID of the tail command
    while ! curl -sf "http://localhost:${port}/health" >/dev/null; do
        if [[ $wait_time -ge $max_wait_time ]]; then
            kill $tail_pid  # Stop tail command
            exit 1
        fi
        sleep 2
        ((wait_time+=2))
    done
    # Stop tail when server is ready
    kill $tail_pid
    echo "Server on port $port is ready!"
}
sleep 5
wait_for_server $VLLM_PORT "$VLLM_LOG_FILE"
echo "Model: $MODEL | port: $VLLM_PORT | PID: $VLLM_SERVER_PID | Log: $VLLM_LOG_FILE"
## ==== End of vLLM Server Setup =====


echo "TASK: $TASK"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo "Max Round: $MAX_ROUND"
echo "Num Examples: $NUM_EXAMPLES"
echo "Max Concurrency: $MAX_CONCURRENCY"
echo "Env Port: $ENV_PORT"
echo "Inference File: $INFERENCE_FILE"
echo "Output Dir: $OUTPUT_DIR"
if [ -n "${TASK_CLASS_PATH:-}" ]; then
    echo "Task Class Path: $TASK_CLASS_PATH"
fi

source uv_agentgym_rl/bin/activate
PY_ARGS=(
    --api_key "EMPTY"
    --base_url "http://localhost:$VLLM_PORT/v1"
    --model vllm_model
    --inference_file "$INFERENCE_FILE"
    --output_dir "$OUTPUT_DIR"
    --task_name "$TASK"
    --max_round "$MAX_ROUND"
    --num_examples "$NUM_EXAMPLES"
    --max_concurrency "$MAX_CONCURRENCY"
    --env_server_base "$ENV_SERVER_BASE"
)
if [ -n "${TASK_CLASS_PATH:-}" ]; then
    PY_ARGS+=(--task_class_path "$TASK_CLASS_PATH")
fi
if [ -n "${TASK_CLIENT_ARGS_JSON:-}" ]; then
    PY_ARGS+=(--task_client_args_json "$TASK_CLIENT_ARGS_JSON")
fi
python scripts/interact_with_real_env/run.py "${PY_ARGS[@]}"
