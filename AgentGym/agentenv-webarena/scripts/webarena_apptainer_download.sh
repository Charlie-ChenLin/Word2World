#!/bin/bash
set -euo pipefail

DATA_DIR="${WEBARENA_DATA_DIR:-/mnt/petrelfs/chenlin1/datasets/webarena_env}"
SKIP_WIKI="${SKIP_WIKI:-0}"

usage() {
  cat <<'USAGE'
Usage: webarena_apptainer_download.sh [--data-dir DIR] [--skip-wiki]

Downloads WebArena website images (docker tar) and wikipedia zim.
Default data dir: /mnt/petrelfs/chenlin1/datasets/webarena_env

Env overrides:
  WEBARENA_DATA_DIR  download directory
  SKIP_WIKI=1        skip wikipedia_en_all_maxi_2022-05.zim
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; DATA_DIR="$2"; shift 2 ;;
    --skip-wiki) SKIP_WIKI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$DATA_DIR"

if command -v aria2c >/dev/null 2>&1; then
  DL_TOOL="aria2c"
elif command -v wget >/dev/null 2>&1; then
  DL_TOOL="wget"
elif command -v curl >/dev/null 2>&1; then
  DL_TOOL="curl"
else
  echo "Error: need aria2c/wget/curl for downloads" >&2
  exit 1
fi

download_with_fallback() {
  local out="$1"; shift
  if [ -f "$out" ]; then
    echo "[SKIP] $out exists"
    return 0
  fi
  for url in "$@"; do
    echo "[DL] $url -> $out"
    if [ "$DL_TOOL" = "aria2c" ]; then
      aria2c -c -x 8 -s 8 -k 1M -o "$(basename "$out")" -d "$(dirname "$out")" "$url" && return 0
    elif [ "$DL_TOOL" = "wget" ]; then
      wget -c -O "$out" "$url" && return 0
    else
      curl -L -C - -o "$out" "$url" && return 0
    fi
    echo "[WARN] failed: $url" >&2
  done
  echo "[FAIL] all mirrors failed for $out" >&2
  return 1
}

# Mirror base (fast in CN)
BASE_HTTP="http://metis.lti.cs.cmu.edu/webarena-images"
BASE_ARCHIVE="https://archive.org/download"

# Shopping
shopping_tar="$DATA_DIR/shopping_final_0712.tar"
download_with_fallback "$shopping_tar" \
  "$BASE_HTTP/shopping_final_0712.tar" \
  "$BASE_ARCHIVE/webarena-env-shopping-image/shopping_final_0712.tar"

# Shopping admin
shopping_admin_tar="$DATA_DIR/shopping_admin_final_0719.tar"
download_with_fallback "$shopping_admin_tar" \
  "$BASE_HTTP/shopping_admin_final_0719.tar" \
  "$BASE_ARCHIVE/webarena-env-shopping-admin-image/shopping_admin_final_0719.tar"

# Forum (Reddit)
forum_tar="$DATA_DIR/postmill-populated-exposed-withimg.tar"
download_with_fallback "$forum_tar" \
  "$BASE_HTTP/postmill-populated-exposed-withimg.tar" \
  "$BASE_ARCHIVE/webarena-env-forum-image/postmill-populated-exposed-withimg.tar"

# GitLab
if [ -f "$DATA_DIR/gitlab-populated-final-port8023.tar" ]; then
  gitlab_tar="$DATA_DIR/gitlab-populated-final-port8023.tar"
else
  gitlab_tar="$DATA_DIR/gitlab-populated-final-port8023.tar"
fi

download_with_fallback "$gitlab_tar" \
  "$BASE_HTTP/gitlab-populated-final-port8023.tar" \
  "$BASE_ARCHIVE/webarena-env-gitlab-image/gitlab-populated-final-port8023.tar"

# Wikipedia ZIM
if [ "$SKIP_WIKI" != "1" ]; then
  wiki_zim="$DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"
  download_with_fallback "$wiki_zim" \
    "$BASE_HTTP/wikipedia_en_all_maxi_2022-05.zim" \
    "$BASE_ARCHIVE/webarena-env-wiki-image/wikipedia_en_all_maxi_2022-05.zim"
else
  echo "[SKIP] wikipedia zim (SKIP_WIKI=1)"
fi

cat <<EOF_SUM
[OK] Download complete.
Data dir: $DATA_DIR
Files:
  - shopping_final_0712.tar
  - shopping_admin_final_0719.tar
  - postmill-populated-exposed-withimg.tar
  - gitlab-populated-final-port8023.tar
  - wikipedia_en_all_maxi_2022-05.zim (if not skipped)
EOF_SUM
