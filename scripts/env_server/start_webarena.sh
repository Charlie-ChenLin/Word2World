#!/bin/bash
set -euo pipefail

WEBARENA_PORT="${WEBARENA_PORT:-36005}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_WEBARENA="${VENV_WEBARENA:-$UVENVS_HOME/uv_webarena}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if [ -f "$VENV_WEBARENA/bin/activate" ]; then
  source "$VENV_WEBARENA/bin/activate"
elif [ -f "$VENV_AGENTGYM_RL/bin/activate" ]; then
  source "$VENV_AGENTGYM_RL/bin/activate"
else
  echo "Error: missing venv. Set VENV_WEBARENA or VENV_AGENTGYM_RL." >&2
  exit 1
fi

webarena --host 0.0.0.0 --port "$WEBARENA_PORT"
