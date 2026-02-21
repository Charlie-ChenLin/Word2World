#!/bin/bash
#SBATCH --job-name=ckpt-migrate
#SBATCH --partition=Gveval2-T
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/log/ckpt_migrate/%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/ckpt_migrate/%j.err

set -euo pipefail

SRC_ROOT="${HOME}"
DEST_ROOT="/mnt/inspurfs/chenlin1/ckpts"

mkdir -p /mnt/petrelfs/chenlin1/log/ckpt_migrate
mkdir -p "${DEST_ROOT}"

LOG_DIR="/mnt/petrelfs/chenlin1/log/ckpt_migrate/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

RCLONE_BIN="$(command -v rclone || true)"
if [ -z "${RCLONE_BIN}" ] && [ -x "${HOME}/bin/rclone" ]; then
  RCLONE_BIN="${HOME}/bin/rclone"
fi
if [ -z "${RCLONE_BIN}" ]; then
  echo "Error: rclone not found in PATH or ${HOME}/bin" | tee -a "${LOG_DIR}/move.log"
  exit 1
fi

SEARCH_ROOTS=("${SRC_ROOT}/projects" "${SRC_ROOT}/models" "${SRC_ROOT}/log" "${SRC_ROOT}/tmp" "${SRC_ROOT}/Documents")

FOUND_LIST="${LOG_DIR}/src_dirs.txt"
: > "${FOUND_LIST}"

for root in "${SEARCH_ROOTS[@]}"; do
  [ -d "${root}" ] || continue
  find "${root}" -maxdepth 6 -type d \
    \( -iname 'saves' -o -iname 'ckpt' -o -iname 'ckpts' -o -iname 'checkpoint' -o -iname 'checkpoints' \) \
    -not -path '*/.git/*' -not -path '*/.venv/*' -not -path '*/uv_envs/*' \
    -not -path '*/.cache/*' -not -path '*/.conda/*' -not -path '*/.local/*' \
    -not -path '*/site-packages/*' -not -path '*/lib/*' -not -path '*/lib64/*' \
    -print 2>>"${LOG_DIR}/find.err" >> "${FOUND_LIST}"
done

# De-duplicate
sort -u "${FOUND_LIST}" -o "${FOUND_LIST}"

if [ ! -s "${FOUND_LIST}" ]; then
  echo "No checkpoint-like directories found under ${SRC_ROOT}." | tee -a "${LOG_DIR}/move.log"
  exit 0
fi

echo "Source dirs:" | tee -a "${LOG_DIR}/move.log"
cat "${FOUND_LIST}" | tee -a "${LOG_DIR}/move.log"

while IFS= read -r src; do
  [ -z "${src}" ] && continue
  if [ ! -d "${src}" ]; then
    echo "Skip missing: ${src}" | tee -a "${LOG_DIR}/move.log"
    continue
  fi
  if [[ "${src}" != "${SRC_ROOT}"/* ]]; then
    echo "Skip non-home path: ${src}" | tee -a "${LOG_DIR}/move.log"
    continue
  fi
  if find "${src}" -maxdepth 2 -type d -name 'global_step_*' -print -quit | grep -q . \
    || [ -f "${src}/latest_checkpointed_iteration.txt" ]; then
    rel="${src#${SRC_ROOT}/}"
    dest="${DEST_ROOT}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    echo "Moving ${src} -> ${dest}" | tee -a "${LOG_DIR}/move.log"
    "${RCLONE_BIN}" move "${src}" "${dest}" \
      --transfers 8 \
      --checkers 8 \
      --stats 60s \
      --log-file "${LOG_DIR}/rclone.log" \
      --log-level INFO
    echo "Done ${src}" | tee -a "${LOG_DIR}/move.log"
  else
    echo "Skip (no global_step_* or latest_checkpointed_iteration.txt): ${src}" | tee -a "${LOG_DIR}/move.log"
  fi
done < "${FOUND_LIST}"
