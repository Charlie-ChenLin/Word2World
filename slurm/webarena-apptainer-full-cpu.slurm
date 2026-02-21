#!/bin/bash
#SBATCH --job-name webarena-apptainer-full-cpu
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.full.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.full.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
DATA_DIR="${WEBARENA_DATA_DIR:-/mnt/petrelfs/chenlin1/datasets/webarena_env}"
IMG_DIR="${WEBARENA_APPTAINER_ROOT:-$REPO_ROOT/AgentGym/agentenv-webarena/apptainer_images}"
RUN_DIR="${WEBARENA_RUN_DIR:-/tmp/webarena_sites_${SLURM_JOB_ID:-$$}}"

mkdir -p /mnt/petrelfs/chenlin1/log/webarena
cd "$REPO_ROOT"

export WEBARENA_DATA_DIR="$DATA_DIR"
export WEBARENA_APPTAINER_ROOT="$IMG_DIR"
export WEBARENA_RUN_DIR="$RUN_DIR"
export WEBARENA_HOSTNAME="${WEBARENA_HOSTNAME:-127.0.0.1}"

cleanup() {
  bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_stop.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf "[INFO] repo=%s\n" "$REPO_ROOT"
printf "[INFO] data_dir=%s\n" "$DATA_DIR"
printf "[INFO] img_dir=%s\n" "$IMG_DIR"
printf "[INFO] run_dir=%s\n" "$RUN_DIR"

# Step 1: download (includes wiki)
echo "[STEP] download webarena assets (incl. wiki)"
bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_download.sh

# Step 2: build apptainer images (includes wiki if network allows)
echo "[STEP] build apptainer images"
bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_build.sh

# Step 3: start website services (host network)
echo "[STEP] start webarena sites"
bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_start.sh

# Step 4: healthcheck (skip map backend by default)
echo "[STEP] healthcheck (skip map backend)"
WEBARENA_SKIP_MAP=1 bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_healthcheck.sh --strict

echo "[DONE] webarena local apptainer setup completed"
