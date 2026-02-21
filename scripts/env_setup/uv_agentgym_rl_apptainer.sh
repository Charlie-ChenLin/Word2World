#!/usr/bin/env bash
set -euo pipefail

# Install the AgentGym-RL uv environment inside Apptainer,
# while storing all uv artifacts and the venv under ~/uv_envs.
# Pure Aliyun mirrors:
# - PyPI index: mirrors.aliyun.com/pypi/simple
# - PyTorch CUDA wheels: mirrors.aliyun.com/pytorch-wheels/cu124 (find-links page)

IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
ENV_NAME="${ENV_NAME:-uv_agentgym_rl}"
VENV_DIR="${VENV_DIR:-$UVENVS_HOME/$ENV_NAME}"
UVROOT="${UVROOT:-$UVENVS_HOME/_uv}"

# ✅ Aliyun mirrors
ALIYUN_PYPI="${ALIYUN_PYPI:-https://mirrors.aliyun.com/pypi/simple/}"
# NOTE: this is a wheel listing page; use as --find-links (pip -f), not --index-url
ALIYUN_TORCH_WHL_PAGE="${ALIYUN_TORCH_WHL_PAGE:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"

# ✅ Defaults (pure Aliyun)
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$ALIYUN_PYPI}"
TORCH_FIND_LINKS="${TORCH_FIND_LINKS:-$ALIYUN_TORCH_WHL_PAGE}"

# ✅ Torch version (allow override)
TORCH_VERSION="${TORCH_VERSION:-2.4.1+cu124}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AGENTGYM_RL="$PROJECT_ROOT/AgentGym-RL"
AGENTENV="$PROJECT_ROOT/AgentGym/agentenv"

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
if [ ! -d "$AGENTGYM_RL" ] || [ ! -d "$AGENTENV" ]; then
  echo "Error: Expected directories not found under $PROJECT_ROOT" >&2
  echo "Missing: $AGENTGYM_RL or $AGENTENV" >&2
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

  # ✅ Pure Aliyun PyPI
  --env UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX"
  --env PIP_INDEX_URL="$UV_DEFAULT_INDEX"

  # ✅ Pure Aliyun torch wheel page (find-links)
  --env TORCH_FIND_LINKS="$TORCH_FIND_LINKS"
  --env TORCH_VERSION="$TORCH_VERSION"

  --env UV_LINK_MODE=copy
  --env UV_HTTP_TIMEOUT=300
  --env SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  --env SSL_CERT_DIR="/etc/ssl/certs"
)

apptainer exec "${APPTAINER_ARGS[@]}" \
  "${APPTAINER_ENV_ARGS[@]}" \
  "$IMG" bash -lc "
set -euo pipefail
UV_BIN=\"$UV_BIN\"
VENV_DIR=\"$VENV_DIR\"
AGENTGYM_RL=\"$AGENTGYM_RL\"
AGENTENV=\"$AGENTENV\"
UV_DEFAULT_INDEX=\"\$UV_DEFAULT_INDEX\"
TORCH_FIND_LINKS=\"\$TORCH_FIND_LINKS\"
TORCH_VERSION=\"\$TORCH_VERSION\"

echo \"[INFO] UV_DEFAULT_INDEX=\$UV_DEFAULT_INDEX\"
echo \"[INFO] TORCH_FIND_LINKS=\$TORCH_FIND_LINKS\"
echo \"[INFO] TORCH_VERSION=\$TORCH_VERSION\"

# If venv exists, keep it; set VENV_CLEAR=1 to rebuild cleanly.
if [ \"\${VENV_CLEAR:-0}\" = \"1\" ]; then
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10 --clear
else
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10
fi

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" --upgrade pip

# ✅ Install CUDA torch from Aliyun wheel page (find-links), while resolving deps from Aliyun PyPI.
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \"torch==\$TORCH_VERSION\" \\
  --index-url \"\$UV_DEFAULT_INDEX\" \\
  --find-links \"\$TORCH_FIND_LINKS\"

FLASH_ATTENTION_WHL=\"\${FLASH_ATTENTION_WHL:-\$HOME/wheels/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}\"
if [ ! -f \"\$FLASH_ATTENTION_WHL\" ]; then
  echo \"Error: flash_attn wheel not found: \$FLASH_ATTENTION_WHL\" >&2
  exit 1
fi
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \"\$FLASH_ATTENTION_WHL\"

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -e \"\$AGENTGYM_RL\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -e \"\$AGENTENV\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" transformers==4.51.3

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" openai azure.identity
VLLM_WHL=\"\${VLLM_WHL:-\$HOME/wheels/vllm-0.6.3-cp38-abi3-manylinux1_x86_64.whl}\"
if [ ! -f \"\$VLLM_WHL\" ]; then
  echo \"Error: vllm wheel not found: \$VLLM_WHL\" >&2
  exit 1
fi
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \"\$VLLM_WHL\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" outlines==0.1.8
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" matplotlib
\"\$UV_BIN\" pip uninstall -p \"\$VENV_DIR/bin/python\" ray || true
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \"ray[default]\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \"click==8.2.1\"

\"\$UV_BIN\" pip list -p \"\$VENV_DIR/bin/python\"
"

echo "Done. Venv created at: $VENV_DIR"
