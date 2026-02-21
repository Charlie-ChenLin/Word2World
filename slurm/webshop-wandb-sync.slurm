#!/bin/bash
#SBATCH --job-name=webshop-wandb-sync
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/inspurfs/chenlin1/Word2World/log/webshop_wandb_sync/%j.out
#SBATCH --error=/mnt/inspurfs/chenlin1/Word2World/log/webshop_wandb_sync/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
LOG_ROOT="${LOG_ROOT:-/mnt/inspurfs/chenlin1/Word2World/log}"
# Where to look for offline runs.
# Training scripts may set WANDB_DIR=./wandb, in which case W&B often writes to ./wandb/wandb/.
# Default to ${REPO_ROOT}/wandb and auto-detect the nested wandb/ subdir when present.
WANDB_ROOT="${WANDB_ROOT:-/mnt/inspurfs/chenlin1/Word2World/wandb}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-/mnt/petrelfs/chenlin1/uv_envs/uv_agentgym_rl}"
WANDB_API_KEY="${WANDB_API_KEY:-2e3f0cdfb6089a1e52ee117856ca6dfcf331858b}"
WANDB_PROJECT="${WANDB_PROJECT:-webshop}"

mkdir -p "${LOG_ROOT}/webshop_wandb_sync"
JOB_DIR="${LOG_ROOT}/webshop_wandb_sync/${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}"

if [ -d "${WANDB_ROOT}/wandb" ] && ls -d "${WANDB_ROOT}/wandb/offline-run-"* >/dev/null 2>&1; then
  WANDB_ROOT="${WANDB_ROOT}/wandb"
fi

if [ ! -d "${WANDB_ROOT}" ]; then
  echo "W&B dir not found: ${WANDB_ROOT}" | tee -a "${JOB_DIR}/sync.log"
  exit 1
fi

if [ ! -f "${VENV_AGENTGYM_RL}/bin/activate" ]; then
  echo "Missing venv: ${VENV_AGENTGYM_RL}" | tee -a "${JOB_DIR}/sync.log"
  exit 1
fi

source "${VENV_AGENTGYM_RL}/bin/activate"
if [ -n "${WANDB_API_KEY}" ]; then
  export WANDB_API_KEY
fi
export WANDB_PROJECT

RUNS_FILE="${JOB_DIR}/runs.txt"
: > "${RUNS_FILE}"

if command -v rg >/dev/null 2>&1; then
  search_cmd=(rg -q)
else
  # Many compute nodes don't have ripgrep; grep is more portable.
  search_cmd=(grep -q)
fi

for run_dir in "${WANDB_ROOT}"/offline-run-*; do
  [ -d "${run_dir}" ] || continue
  log_file="${run_dir}/logs/debug.log"
  if [ -f "${log_file}" ]; then
    if "${search_cmd[@]}" "webshop" "${log_file}"; then
      echo "${run_dir}" >> "${RUNS_FILE}"
    fi
  fi
  done

if [ ! -s "${RUNS_FILE}" ]; then
  echo "No webshop offline runs found under ${WANDB_ROOT}" | tee -a "${JOB_DIR}/sync.log"
  exit 0
fi

sort -u "${RUNS_FILE}" -o "${RUNS_FILE}"

while IFS= read -r run_dir; do
  [ -z "${run_dir}" ] && continue
  echo "[wandb-sync] syncing ${run_dir}" | tee -a "${JOB_DIR}/sync.log"
  wandb sync "${run_dir}" --project "${WANDB_PROJECT}" >> "${JOB_DIR}/sync.log" 2>&1 || echo "[wandb-sync] failed: ${run_dir}" | tee -a "${JOB_DIR}/sync.log"
  echo "[wandb-sync] done ${run_dir}" | tee -a "${JOB_DIR}/sync.log"
done < "${RUNS_FILE}"
