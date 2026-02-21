#!/bin/bash
# Merge a single missing HF checkpoint for WebShop eval (CPU-only).

#SBATCH --job-name=webshop-merge725
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-merge725-%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-merge725-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/log/webshop_eval"

APPTAINER_IMG="${APPTAINER_IMG:-/mnt/petrelfs/chenlin1/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-/mnt/petrelfs/chenlin1/uv_envs}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-${UVENVS_HOME}/uv_agentgym_rl}"
PROJECTS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

ACTOR_DIR="${ACTOR_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves/webshop-a800-1-oomfix1-20260204-122238/global_step_725/actor}"

# Respect cpus-per-task when model_merger spins up a ThreadPool.
export MODEL_MERGER_MAX_WORKERS="${MODEL_MERGER_MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

echo "[merge725] Host=$(hostname) actor_dir=${ACTOR_DIR} max_workers=${MODEL_MERGER_MAX_WORKERS}"

APPTAINER_ARGS=(--pwd "${REPO_ROOT}" --bind "${PROJECTS_ROOT}" --bind "${UVENVS_HOME}" --bind /mnt/inspurfs/chenlin1)

apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" \
  bash -lc "source '${VENV_AGENTGYM_RL}/bin/activate' && python3 AgentGym-RL/scripts/model_merger.py --local_dir '${ACTOR_DIR}'"

