#!/bin/bash
# Quick HF checkpoint sanity check (inside apptainer + uv env).
#
# Usage:
#   sbatch slurm/hf-ckpt-load-check.slurm
#   MODEL_DIR=... STEP=50 sbatch slurm/hf-ckpt-load-check.slurm
#
# Notes:
# - Runs CPU-only; pins to an already-used node by default to avoid increasing node quota.

#SBATCH --job-name=hf-ckpt-check
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/hf-ckpt-check-%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/hf-ckpt-check-%j.err
#SBATCH --nodelist=HOST-10-140-66-180

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
APPTAINER_IMG="${APPTAINER_IMG:-/mnt/petrelfs/chenlin1/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-/mnt/petrelfs/chenlin1/uv_envs}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

# Default target: a known problematic model/step.
MODEL_DIR="${MODEL_DIR:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves/webshop-a800-0-oomfix1-20260204-122230}"
STEP="${STEP:-25}"

CKPT_DIR="${MODEL_DIR}/global_step_${STEP}/actor"
HF_DIR="${CKPT_DIR}/huggingface"

mkdir -p "${REPO_ROOT}/log/webshop_eval"

echo "[hf-ckpt-check] Node=$(hostname)"
echo "[hf-ckpt-check] CKPT_DIR=${CKPT_DIR}"
echo "[hf-ckpt-check] HF_DIR=${HF_DIR}"

PROJECTS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
APPTAINER_ARGS=(--pwd "${REPO_ROOT}" --bind "${PROJECTS_ROOT}" --bind "${UVENVS_HOME}" --bind /mnt/inspurfs/chenlin1)

if [ ! -d "${CKPT_DIR}" ]; then
  echo "[hf-ckpt-check] ERROR: missing CKPT_DIR=${CKPT_DIR}" >&2
  exit 1
fi
if [ ! -d "${HF_DIR}" ]; then
  echo "[hf-ckpt-check] ERROR: missing HF_DIR=${HF_DIR}" >&2
  exit 1
fi

echo "[hf-ckpt-check] HF dir before merge:"
ls -lh "${HF_DIR}" | sed -n '1,200p'

needs_merge=1
if [ -f "${HF_DIR}/model.safetensors.index.json" ] || ls "${HF_DIR}"/pytorch_model*.bin "${HF_DIR}"/model.safetensors >/dev/null 2>&1; then
  needs_merge=0
fi

if [ "${needs_merge}" = "1" ]; then
  echo "[hf-ckpt-check] Running model_merger.py inside apptainer..."
  apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" \
    bash -lc "source '${VENV_AGENTGYM_RL}/bin/activate' && python3 AgentGym-RL/scripts/model_merger.py --local_dir '${CKPT_DIR}'"
else
  echo "[hf-ckpt-check] Detected weight files in HF dir; skip model_merger."
fi

echo "[hf-ckpt-check] HF dir after merge:"
ls -lh "${HF_DIR}" | sed -n '1,200p'

echo "[hf-ckpt-check] Verifying transformers can load from HF dir..."
apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" \
  bash -lc "source '${VENV_AGENTGYM_RL}/bin/activate' && python3 -c \"from transformers import AutoModelForCausalLM; import torch; hf_dir='${HF_DIR}'; print('Loading:', hf_dir); model=AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype='auto', low_cpu_mem_usage=True); first=next(model.parameters()); print('OK: dtype=', first.dtype, 'device=', first.device, 'n_params=', sum(p.numel() for p in model.parameters()))\""

echo "[hf-ckpt-check] Done."
