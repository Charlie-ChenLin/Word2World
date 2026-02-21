#!/bin/bash
#SBATCH --job-name webarena-dl-wiki
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.dl.wiki.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.dl.wiki.err

set -euo pipefail

DATA_DIR="${WEBARENA_DATA_DIR:-/mnt/petrelfs/chenlin1/datasets/webarena_env}"

mkdir -p /mnt/petrelfs/chenlin1/log/webarena
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

BASE_HTTP="http://metis.lti.cs.cmu.edu/webarena-images"
BASE_ARCHIVE="https://archive.org/download"

file_path="$DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"
download_with_fallback "$file_path" \
  "$BASE_HTTP/wikipedia_en_all_maxi_2022-05.zim" \
  "$BASE_ARCHIVE/webarena-env-wiki-image/wikipedia_en_all_maxi_2022-05.zim"

echo "[OK] Download complete: $file_path"
