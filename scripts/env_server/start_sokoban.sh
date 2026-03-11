#!/bin/bash
set -euo pipefail

UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
VENV_SOKOBAN="${VENV_SOKOBAN:-$UVENVS_HOME/uv_sokoban}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

if [ -f "$VENV_SOKOBAN/bin/activate" ]; then
  source "$VENV_SOKOBAN/bin/activate"
elif [ -f "$VENV_AGENTGYM_RL/bin/activate" ]; then
  source "$VENV_AGENTGYM_RL/bin/activate"
else
  echo "Error: missing venv. Set VENV_SOKOBAN or VENV_AGENTGYM_RL." >&2
  exit 1
fi

SOKOBAN_PORT="${SOKOBAN_PORT:-37001}"
export SOKOBAN_DEFAULT_SPLIT="${SOKOBAN_DEFAULT_SPLIT:-train}"
export SOKOBAN_STRICT_ACTION_MATCH="${SOKOBAN_STRICT_ACTION_MATCH:-1}"
export SOKOBAN_REQUIRE_THINK_TAG="${SOKOBAN_REQUIRE_THINK_TAG:-1}"
export SOKOBAN_INVALID_ACTION_PENALTY="${SOKOBAN_INVALID_ACTION_PENALTY:-0.1}"

sokoban --host 0.0.0.0 --port "${SOKOBAN_PORT}"
