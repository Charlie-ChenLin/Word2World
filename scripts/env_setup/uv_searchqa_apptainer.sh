#!/usr/bin/env bash
set -euo pipefail

# Install the SearchQA uv environment inside Apptainer,
# while storing all uv artifacts and the venv under ~/uv_envs (host).

IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UVENVS_HOME="${UVENVS_HOME:-$HOME/uv_envs}"
ENV_NAME="${ENV_NAME:-uv_searchqa}"
VENV_DIR="${VENV_DIR:-$UVENVS_HOME/$ENV_NAME}"
UVROOT="${UVROOT:-$UVENVS_HOME/_uv}"

# Aliyun PyPI mirror (override as needed)
ALIYUN_PYPI="${ALIYUN_PYPI:-https://mirrors.aliyun.com/pypi/simple/}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-$ALIYUN_PYPI}"

# Increase timeout for large wheels (e.g., cuDNN)
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# PyTorch wheels (override if needed)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.0}"

# Retrieval deps (faiss-gpu via conda is preferred; pip default is cpu)
FAISS_PKG="${FAISS_PKG:-faiss-cpu}"
EXTRA_PIP="${EXTRA_PIP:-}"

# Optional: run the dataset download/processing step
SEARCHQA_RUN_SETUP="${SEARCHQA_RUN_SETUP:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
AGENTENV_SEARCHQA="$PROJECT_ROOT/AgentGym/agentenv-searchqa"

if [ -f "$VENV_DIR/bin/activate" ] && [ "${VENV_CLEAR:-0}" != "1" ]; then
  echo "[SKIP] venv exists: $VENV_DIR (set VENV_CLEAR=1 to rebuild)"
  exit 0
fi
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
if [ ! -d "$AGENTENV_SEARCHQA" ]; then
  echo "Error: Expected directory not found: $AGENTENV_SEARCHQA" >&2
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
  --env UV_HTTP_TIMEOUT="$UV_HTTP_TIMEOUT"
  --env UV_DEFAULT_INDEX="$UV_DEFAULT_INDEX"
  --env PIP_INDEX_URL="$UV_DEFAULT_INDEX"
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
AGENTENV_SEARCHQA=\"$AGENTENV_SEARCHQA\"
UV_DEFAULT_INDEX=\"$UV_DEFAULT_INDEX\"
TORCH_INDEX_URL=\"$TORCH_INDEX_URL\"
TORCH_VERSION=\"$TORCH_VERSION\"
TORCHVISION_VERSION=\"$TORCHVISION_VERSION\"
TORCHAUDIO_VERSION=\"$TORCHAUDIO_VERSION\"
FAISS_PKG=\"$FAISS_PKG\"
EXTRA_PIP=\"$EXTRA_PIP\"
SEARCHQA_RUN_SETUP=\"$SEARCHQA_RUN_SETUP\"
HF_ENDPOINT=\"$HF_ENDPOINT\"

echo \"[INFO] UV_DEFAULT_INDEX=\$UV_DEFAULT_INDEX\"
echo \"[INFO] TORCH_INDEX_URL=\$TORCH_INDEX_URL\"
echo \"[INFO] TORCH_VERSION=\$TORCH_VERSION\"
echo \"[INFO] FAISS_PKG=\$FAISS_PKG\"

if [ \"\${VENV_CLEAR:-0}\" = \"1\" ]; then
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10 --clear
else
  \"\$UV_BIN\" venv \"\$VENV_DIR\" --python 3.10
fi

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" --upgrade pip

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \\
  \"torch==\$TORCH_VERSION\" \\
  \"torchvision==\$TORCHVISION_VERSION\" \\
  \"torchaudio==\$TORCHAUDIO_VERSION\" \\
  --index-url \"\$TORCH_INDEX_URL\"

\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" transformers datasets \"\$FAISS_PKG\"
\"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" -e \"\$AGENTENV_SEARCHQA\"

if [ -n \"\$EXTRA_PIP\" ]; then
  \"\$UV_BIN\" pip install -p \"\$VENV_DIR/bin/python\" \$EXTRA_PIP
fi

if [ \"\$SEARCHQA_RUN_SETUP\" = \"1\" ]; then
  echo \"[INFO] Running SearchQA setup.sh (downloads data). This may take a while.\"
  export PATH=\"\$VENV_DIR/bin:\$PATH\"
  export HF_ENDPOINT=\"\$HF_ENDPOINT\"
  cd \"\$AGENTENV_SEARCHQA\"
  bash ./setup.sh
else
  echo \"[INFO] Skip data setup. Set SEARCHQA_RUN_SETUP=1 to download data.\"
fi

\"\$UV_BIN\" pip list -p \"\$VENV_DIR/bin/python\" | grep -E \"(agentenv_searchqa|fastapi|uvicorn|torch|faiss|datasets|transformers)\" || true
"

echo "Done. Venv created at: $VENV_DIR"
