#!/usr/bin/env bash
set -euo pipefail

# Install the AlfWorld uv environment inside Apptainer,
# while storing all uv artifacts and the venv under ~/uv_envs.

IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
ENV_NAME="${ENV_NAME:-uv_alfworld}"
VENV_DIR="${VENV_DIR:-$UVENVS_HOME/$ENV_NAME}"
UVROOT="${UVROOT:-$UVENVS_HOME/_uv}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AGENTENV_ALFWORLD="$PROJECT_ROOT/AgentGym/agentenv-alfworld"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi
if [ ! -x "$UV_BIN" ]; then
  echo "Error: uv not found at $UV_BIN" >&2
  echo "Install uv first, or set UV_BIN to its path." >&2
  exit 1
fi
if [ ! -f "$IMG" ]; then
  echo "Error: Apptainer image not found: $IMG" >&2
  exit 1
fi
if [ ! -d "$AGENTENV_ALFWORLD" ]; then
  echo "Error: Expected directory not found: $AGENTENV_ALFWORLD" >&2
  exit 1
fi

mkdir -p "$UVROOT/cache" "$UVROOT/python" "$UVROOT/tools" "$UVROOT/bin"
mkdir -p "$(dirname "$VENV_DIR")"

APPTAINER_ARGS=(--pwd "$PROJECT_ROOT")
if [ "${APPTAINER_NV:-1}" = "1" ]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi

APPTAINER_ENV_ARGS=(
  --env UV_CACHE_DIR="$UVROOT/cache"
  --env UV_PYTHON_INSTALL_DIR="$UVROOT/python"
  --env UV_TOOL_DIR="$UVROOT/tools"
  --env UV_TOOL_BIN_DIR="$UVROOT/bin"
  --env UV_LINK_MODE=copy
  --env SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  --env SSL_CERT_DIR="/etc/ssl/certs"
)
if [ -n "$UV_DEFAULT_INDEX" ]; then
  APPTAINER_ENV_ARGS+=(--env UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX")
fi

apptainer exec "${APPTAINER_ARGS[@]}" \
  "${APPTAINER_ENV_ARGS[@]}" \
  "$IMG" bash -lc "
set -euo pipefail
UV_BIN=\"$UV_BIN\"
VENV_DIR=\"$VENV_DIR\"
AGENTENV_ALFWORLD=\"$AGENTENV_ALFWORLD\"

\"$UV_BIN\" venv \"$VENV_DIR\" --python 3.9
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" alfworld==0.3.3
\"$UV_BIN\" pip uninstall -p \"$VENV_DIR/bin/python\" opencv-python
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" -e \"$AGENTENV_ALFWORLD\"
\"$UV_BIN\" pip list -p \"$VENV_DIR/bin/python\"
"

echo "Done. Venv created at: $VENV_DIR"
