#!/usr/bin/env bash
set -euo pipefail

# Install multiple agentic-RL envs via Apptainer + uv into ~/uv_envs (host).
# Skips any env that already exists unless VENV_CLEAR=1.

UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

declare -a TASKS=(
  webshop
  alfworld
  sciworld
  textcraft
  babyai
  searchqa
  sokoban
)

for task in "${TASKS[@]}"; do
  env_name="uv_${task}"
  venv_dir="${UVENVS_HOME}/${env_name}"
  if [ -f "${venv_dir}/bin/activate" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
    echo "[SKIP] ${env_name} exists at ${venv_dir}"
    continue
  fi

  case "$task" in
    webshop)
      script="$PROJECT_ROOT/scripts/env_setup/uv_webshop_apptainer.sh"
      ;;
    alfworld)
      script="$PROJECT_ROOT/scripts/env_setup/uv_alfworld_apptainer.sh"
      ;;
    sciworld)
      script="$PROJECT_ROOT/scripts/env_setup/uv_sciworld_apptainer.sh"
      ;;
    textcraft)
      script="$PROJECT_ROOT/scripts/env_setup/uv_textcraft_apptainer.sh"
      ;;
    babyai)
      script="$PROJECT_ROOT/scripts/env_setup/uv_babyai_apptainer.sh"
      ;;
    searchqa)
      script="$PROJECT_ROOT/scripts/env_setup/uv_searchqa_apptainer.sh"
      ;;
    sokoban)
      script="$PROJECT_ROOT/scripts/env_setup/uv_sokoban_apptainer.sh"
      ;;
    *)
      echo "[WARN] Unknown task: $task"
      continue
      ;;
  esac

  if [ ! -f "$script" ]; then
    echo "[WARN] Missing script: $script"
    continue
  fi

  echo "[RUN] $task -> $script"
  APPTAINER_IMG="$APPTAINER_IMG" \
  UV_BIN="$UV_BIN" \
  UVENVS_HOME="$UVENVS_HOME" \
  VENV_CLEAR="${VENV_CLEAR:-0}" \
  bash "$script"
done

echo "Done."
