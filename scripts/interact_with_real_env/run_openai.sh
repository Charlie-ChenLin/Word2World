#!/usr/bin/env bash
set -euo pipefail

TASK=$1
RUN=${2:-0}
API_KEY=$3
API_BASE_URL=$4
MODEL=${5:-gpt-4o}
MAX_CONCURRENCY=${6:-150}
MAX_ROUND=${7:-50}
NUM_EXAMPLES=${8:--1}
SPLIT=${9:-test}
OUTPUT_ROOT=${10:-outputs}

OUTPUT_DIR=$OUTPUT_ROOT/interaction/real_env/$SPLIT/${TASK}/$MODEL/${TASK}_maxround${MAX_ROUND}_run${RUN}

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

cleanup() {
    if [ -n "${ENV_SERVER_PID:-}" ]; then
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


echo "TASK: $TASK"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo "Max Concurrency: $MAX_CONCURRENCY"
echo "Max Round: $MAX_ROUND"
echo "Num Examples: $NUM_EXAMPLES"
echo "Port: $ENV_PORT"
echo "Inference File: $INFERENCE_FILE"
echo "Output Dir: $OUTPUT_DIR"
if [ -n "${TASK_CLASS_PATH:-}" ]; then
    echo "Task Class Path: $TASK_CLASS_PATH"
fi

source uv_agentgym_rl/bin/activate
PY_ARGS=(
    --api_key "$API_KEY"
    --base_url "$API_BASE_URL"
    --model "$MODEL"
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
