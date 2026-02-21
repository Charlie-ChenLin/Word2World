#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
DATA_DIR="${WEBARENA_DATA_DIR:-/mnt/petrelfs/chenlin1/datasets/webarena_env}"
OUT_DIR="${WEBARENA_APPTAINER_ROOT:-$REPO_ROOT/AgentGym/agentenv-webarena/apptainer_images}"
FORCE="${FORCE_BUILD:-0}"
SKIP_WIKI="${SKIP_WIKI:-0}"

usage() {
  cat <<'USAGE'
Usage: webarena_apptainer_build.sh [--data-dir DIR] [--out-dir DIR] [--force] [--skip-wiki]

Build Apptainer images from WebArena docker-archive tars and kiwix docker image.
Requires Apptainer with setuid support for unprivileged build.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; DATA_DIR="$2"; shift 2 ;;
    --out-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; OUT_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --skip-wiki) SKIP_WIKI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

build_from_tar() {
  local name="$1"; local tar="$2";
  local out="$OUT_DIR/${name}.sif"
  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
    echo "[SKIP] $out exists"
    return 0
  fi
  if [ ! -f "$tar" ]; then
    echo "[FAIL] missing tar: $tar" >&2
    exit 1
  fi
  echo "[BUILD] $out from $tar"
  apptainer build --force "$out" "docker-archive://$tar"
}

build_from_tar shopping "$DATA_DIR/shopping_final_0712.tar"
build_from_tar shopping_admin "$DATA_DIR/shopping_admin_final_0719.tar"
build_from_tar forum "$DATA_DIR/postmill-populated-exposed-withimg.tar"
build_from_tar gitlab "$DATA_DIR/gitlab-populated-final-port8023.tar"

# Kiwix image for wikipedia (optional, requires network)
if [ "$SKIP_WIKI" = "1" ]; then
  echo "[SKIP] wiki build (SKIP_WIKI=1)"
else
  kiwix_out="$OUT_DIR/kiwix-serve-3.3.0.sif"
  if [ -f "$kiwix_out" ] && [ "$FORCE" != "1" ]; then
    echo "[SKIP] $kiwix_out exists"
  else
    echo "[BUILD] $kiwix_out from docker://ghcr.io/kiwix/kiwix-serve:3.3.0"
    apptainer build --force "$kiwix_out" "docker://ghcr.io/kiwix/kiwix-serve:3.3.0"
  fi
fi

cat <<EOF_SUM
[OK] Build complete.
Images in: $OUT_DIR
  - shopping.sif
  - shopping_admin.sif
  - forum.sif
  - gitlab.sif
  - kiwix-serve-3.3.0.sif (if not skipped)
EOF_SUM
