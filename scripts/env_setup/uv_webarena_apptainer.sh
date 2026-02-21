#!/usr/bin/env bash
set -euo pipefail

# Install the WebArena uv environment inside Apptainer,
# while storing all uv artifacts and the venv under ~/uv_envs.

IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
ENV_NAME="${ENV_NAME:-uv_webarena}"
VENV_DIR="${VENV_DIR:-$UVENVS_HOME/$ENV_NAME}"
UVROOT="${UVROOT:-$UVENVS_HOME/_uv}"
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$UVENVS_HOME/_playwright}"

# Optional mirrors
ALIYUN_PYPI="${ALIYUN_PYPI:-https://mirrors.aliyun.com/pypi/simple/}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$ALIYUN_PYPI}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WEBARENA_ROOT="$PROJECT_ROOT/AgentGym/agentenv-webarena"
WEBARENA_PKG="$WEBARENA_ROOT/webarena"

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
if [ ! -d "$WEBARENA_ROOT" ] || [ ! -d "$WEBARENA_PKG" ]; then
  echo "Error: Expected directories not found under $PROJECT_ROOT" >&2
  echo "Missing: $WEBARENA_ROOT or $WEBARENA_PKG" >&2
  exit 1
fi

mkdir -p "$UVROOT/cache" "$UVROOT/python" "$UVROOT/tools" "$UVROOT/bin"
mkdir -p "$(dirname "$VENV_DIR")"
mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"

APPTAINER_ARGS=(--pwd "$PROJECT_ROOT")
if [ "${APPTAINER_NV:-0}" = "1" ]; then
  APPTAINER_ARGS=(--nv "${APPTAINER_ARGS[@]}")
fi

APPTAINER_ENV_ARGS=(
  --env UV_CACHE_DIR="$UVROOT/cache"
  --env UV_PYTHON_INSTALL_DIR="$UVROOT/python"
  --env UV_TOOL_DIR="$UVROOT/tools"
  --env UV_TOOL_BIN_DIR="$UVROOT/bin"
  --env UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

  --env UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX"
  --env PIP_INDEX_URL="$UV_DEFAULT_INDEX"

  --env UV_LINK_MODE=copy
  --env SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  --env SSL_CERT_DIR="/etc/ssl/certs"
  --env PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH"
)

apptainer exec "${APPTAINER_ARGS[@]}" \
  "${APPTAINER_ENV_ARGS[@]}" \
  "$IMG" bash -lc "
set -euo pipefail
UV_BIN=\"$UV_BIN\"
VENV_DIR=\"$VENV_DIR\"
WEBARENA_ROOT=\"$WEBARENA_ROOT\"
WEBARENA_PKG=\"$WEBARENA_PKG\"
UV_DEFAULT_INDEX=\"$UV_DEFAULT_INDEX\"
PLAYWRIGHT_BROWSERS_PATH=\"$PLAYWRIGHT_BROWSERS_PATH\"

echo \"[INFO] UV_DEFAULT_INDEX=\$UV_DEFAULT_INDEX\"
echo \"[INFO] PLAYWRIGHT_BROWSERS_PATH=\$PLAYWRIGHT_BROWSERS_PATH\"

# If venv exists, keep it; set VENV_CLEAR=1 to rebuild cleanly.
if [ \"\${VENV_CLEAR:-0}\" = \"1\" ]; then
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10 --clear
else
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10
fi

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" --upgrade pip

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -r \"\$WEBARENA_PKG/requirements.txt\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" gunicorn
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -e \"\$WEBARENA_PKG\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -e \"\$WEBARENA_ROOT\"

\"\$VENV_DIR/bin/python\" -m playwright install
if [ \"\${PLAYWRIGHT_INSTALL_DEPS:-0}\" = \"1\" ]; then
  \"\$VENV_DIR/bin/python\" -m playwright install-deps || true
fi

# Optional init scripts (no auto-login by default)
if [ \"\${RUN_WEBARENA_INIT:-1}\" = \"1\" ]; then
  if [ -z \"\${SHOPPING:-}\" ] || [ -z \"\${SHOPPING_ADMIN:-}\" ] || [ -z \"\${REDDIT:-}\" ] || \\
     [ -z \"\${GITLAB:-}\" ] || [ -z \"\${WIKIPEDIA:-}\" ] || [ -z \"\${MAP:-}\" ] || [ -z \"\${HOMEPAGE:-}\" ]; then
    echo \"[SKIP] init scripts: missing site URLs. Set SHOPPING/SHOPPING_ADMIN/REDDIT/GITLAB/WIKIPEDIA/MAP/HOMEPAGE or RUN_WEBARENA_INIT=0.\"
  else
    cd \"\$WEBARENA_PKG\"
    \"\$VENV_DIR/bin/python\" scripts/generate_test_data.py
    \"\$VENV_DIR/bin/python\" agent/prompts/to_json.py
    mkdir -p ./.auth
    if [ \"\${RUN_AUTO_LOGIN:-0}\" = \"1\" ]; then
      \"\$VENV_DIR/bin/python\" browser_env/auto_login.py || true
    else
      echo \"[SKIP] auto_login (RUN_AUTO_LOGIN=0)\"
    fi
  fi
fi
"

echo "Done. Venv created at: $VENV_DIR"
