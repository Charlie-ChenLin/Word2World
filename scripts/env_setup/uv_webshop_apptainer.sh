#!/usr/bin/env bash
set -euo pipefail

# Install the WebShop uv environment inside Apptainer,
# while storing all uv artifacts and the venv under ~/uv_envs.

IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
ENV_NAME="${ENV_NAME:-uv_webshop}"
VENV_DIR="${VENV_DIR:-$UVENVS_HOME/$ENV_NAME}"
UVROOT="${UVROOT:-$UVENVS_HOME/_uv}"

# ✅ Aliyun mirrors (mirrors PyPI + PyTorch CUDA wheels)
ALIYUN_PYPI="${ALIYUN_PYPI:-https://mirrors.aliyun.com/pypi/simple/}"
# NOTE: this is a wheel listing page; use as --find-links (pip -f), not --index-url
ALIYUN_TORCH_WHL_PAGE="${ALIYUN_TORCH_WHL_PAGE:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"

# ✅ Defaults (pure Aliyun)
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$ALIYUN_PYPI}"
TORCH_FIND_LINKS="${TORCH_FIND_LINKS:-$ALIYUN_TORCH_WHL_PAGE}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"

# Optional override (legacy)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-$HOME/wheels/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
WEBSHOP_DIR="$PROJECT_ROOT/AgentGym/agentenv-webshop/webshop"
SEARCH_ENGINE_DIR="$WEBSHOP_DIR/search_engine"
AGENTENV_WEBSHOP="$PROJECT_ROOT/AgentGym/agentenv-webshop"

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
if [ ! -d "$WEBSHOP_DIR" ] || [ ! -d "$SEARCH_ENGINE_DIR" ] || [ ! -d "$AGENTENV_WEBSHOP" ]; then
  echo "Error: Expected directories not found under $PROJECT_ROOT" >&2
  echo "Missing: $WEBSHOP_DIR or $SEARCH_ENGINE_DIR or $AGENTENV_WEBSHOP" >&2
  exit 1
fi
if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
  echo "Error: flash-attn wheel not found: $FLASH_ATTN_WHEEL" >&2
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
  --env UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

  # ✅ Pure Aliyun PyPI
  --env UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX"
  --env PIP_INDEX_URL="$UV_DEFAULT_INDEX"

  # ✅ Pure Aliyun torch wheel page (find-links)
  --env TORCH_FIND_LINKS="$TORCH_FIND_LINKS"
  --env TORCH_VERSION="$TORCH_VERSION"
  --env TORCH_INDEX_URL="$TORCH_INDEX_URL"

  --env UV_LINK_MODE=copy
  --env SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  --env SSL_CERT_DIR="/etc/ssl/certs"
)

apptainer exec "${APPTAINER_ARGS[@]}" \
  "${APPTAINER_ENV_ARGS[@]}" \
  "$IMG" bash -lc "
set -euo pipefail
UV_BIN=\"$UV_BIN\"
VENV_DIR=\"$VENV_DIR\"
WEBSHOP_DIR=\"$WEBSHOP_DIR\"
SEARCH_ENGINE_DIR=\"$SEARCH_ENGINE_DIR\"
AGENTENV_WEBSHOP=\"$AGENTENV_WEBSHOP\"
FLASH_ATTN_WHEEL=\"$FLASH_ATTN_WHEEL\"
UV_DEFAULT_INDEX=\"$UV_DEFAULT_INDEX\"
TORCH_FIND_LINKS=\"$TORCH_FIND_LINKS\"
TORCH_VERSION=\"$TORCH_VERSION\"
TORCH_INDEX_URL=\"$TORCH_INDEX_URL\"

echo \"[INFO] UV_DEFAULT_INDEX=\$UV_DEFAULT_INDEX\"
echo \"[INFO] TORCH_FIND_LINKS=\$TORCH_FIND_LINKS\"
echo \"[INFO] TORCH_VERSION=\$TORCH_VERSION\"
if [ -n \"\$TORCH_INDEX_URL\" ]; then
  echo \"[INFO] TORCH_INDEX_URL=\$TORCH_INDEX_URL\"
fi

\"$UV_BIN\" venv \"$VENV_DIR\" --python 3.10
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" --upgrade pip

cd \"$WEBSHOP_DIR\"
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" packaging wheel setuptools

BUILD_JOBS=\"\${FLASH_ATTN_BUILD_JOBS:-\$(getconf _NPROCESSORS_ONLN)}\"
export MAX_JOBS=\"\$BUILD_JOBS\"
export NINJA_NUM_JOBS=\"\$BUILD_JOBS\"
export CMAKE_BUILD_PARALLEL_LEVEL=\"\$BUILD_JOBS\"
export NVCC_THREADS=\"\${FLASH_ATTN_NVCC_THREADS:-4}\"

TORCH_INDEX_URL=\"\${TORCH_INDEX_URL:-\$UV_DEFAULT_INDEX}\"
if [ -n \"\$TORCH_FIND_LINKS\" ]; then
  \"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" \"torch==\$TORCH_VERSION\" \\
    --index-url \"\$TORCH_INDEX_URL\" \\
    --find-links \"\$TORCH_FIND_LINKS\"
else
  \"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" \"torch==\$TORCH_VERSION\" \\
    --index-url \"\$TORCH_INDEX_URL\"
fi

\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" \"$FLASH_ATTN_WHEEL\"
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" vllm==0.8.2
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" -r requirements.txt --no-deps

\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" mkl faiss-cpu
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" gdown

if command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y openjdk-11-jdk
  else
    echo \"Skipping apt-get: sudo requires a password or is not permitted.\"
  fi
else
  echo \"Skipping apt-get: sudo not found.\"
fi

\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" nltk itsdangerous pyjnius pytz python-dateutil huggingface_hub threadpoolctl onnxruntime setuptools
cd \"$SEARCH_ENGINE_DIR\"

\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" openai azure.identity
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" omegaconf

\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" -e \"$AGENTENV_WEBSHOP\"
\"$UV_BIN\" pip install -p \"$VENV_DIR/bin/python\" thinc numpy==1.26.4 langcodes spacy==3.7.1
\"$VENV_DIR/bin/python\" -m pip install \"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl\"
\"$VENV_DIR/bin/python\" -m pip install \"https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl\"
"

echo "Done. Venv created at: $VENV_DIR"
