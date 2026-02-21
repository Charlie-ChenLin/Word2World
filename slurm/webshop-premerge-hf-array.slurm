#!/bin/bash
# Pre-merge FSDP actor shards into HuggingFace weights for webshop eval.
#
# Why:
#   Eval uses transformers/vLLM and needs weight files under:
#     .../global_step_*/actor/huggingface/{model*.safetensors|pytorch_model*.bin}
#
# Usage:
#   # Recommended: shard the work, but limit concurrency to avoid heavy I/O contention.
#   sbatch --array=0-7%2 slurm/webshop-premerge-hf-array.slurm
#
#   # Override filter/root:
#   MODEL_FILTER=webshop-a800-0-oomfix1 sbatch --array=0-3%1 slurm/webshop-premerge-hf-array.slurm
#
# Notes:
# - Runs CPU-only inside apptainer image `cuda_12.4.1-devel-ubuntu22.04.sif`, using uv envs bind-mounted.
# - We intentionally do NOT pin to a specific node; allow Slurm to place tasks wherever resources exist.

#SBATCH --job-name=webshop-premerge-hf
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-premerge-hf-%A_%a.out
#SBATCH --error=/mnt/petrelfs/chenlin1/projects/Word2World/log/webshop_eval/webshop-premerge-hf-%A_%a.err
# NOTE: Do not pin to a specific node; allow Slurm to place tasks anywhere with available resources.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
cd "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/log/webshop_eval"

# Auto-detect SHARDS from array length unless explicitly set.
if [ -z "${SHARDS:-}" ]; then
  if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ -n "${SLURM_ARRAY_TASK_MIN:-}" ]; then
    SHARDS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
  else
    SHARDS=1
  fi
fi
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

MODEL_ROOT="${MODEL_ROOT:-/mnt/inspurfs/chenlin1/ckpts/projects/Word2World/AgentGym-RL/saves}"
MODEL_FILTER="${MODEL_FILTER:-webshop}"

APPTAINER_IMG="${APPTAINER_IMG:-/mnt/petrelfs/chenlin1/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
UVENVS_HOME="${UVENVS_HOME:-/mnt/petrelfs/chenlin1/uv_envs}"
VENV_AGENTGYM_RL="${VENV_AGENTGYM_RL:-$UVENVS_HOME/uv_agentgym_rl}"

PROJECTS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
APPTAINER_ARGS=(--pwd "${REPO_ROOT}" --bind "${PROJECTS_ROOT}" --bind "${UVENVS_HOME}" --bind /mnt/inspurfs/chenlin1)

# Respect cpus-per-task when model_merger spins up a ThreadPool.
export MODEL_MERGER_MAX_WORKERS="${MODEL_MERGER_MAX_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

echo "[premerge-hf] Host=$(hostname) shards=${SHARDS} shard_id=${SHARD_ID} max_workers=${MODEL_MERGER_MAX_WORKERS}"
echo "[premerge-hf] MODEL_ROOT=${MODEL_ROOT}"
echo "[premerge-hf] MODEL_FILTER=${MODEL_FILTER}"

mapfile -t model_dirs < <(find "${MODEL_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*${MODEL_FILTER}*" -printf '%f\n' | sort)
if [ "${#model_dirs[@]}" -eq 0 ]; then
  echo "[premerge-hf] No model dirs matched under ${MODEL_ROOT} (filter: ${MODEL_FILTER})" >&2
  exit 1
fi

total=0
selected=0
merged=0
skipped=0
failed=0

idx=0
# Build a stable manifest once so sharding is consistent even if tasks start at different times.
# Otherwise, if new global_step_* dirs appear mid-run, some steps can be missed.
manifest_dir="${REPO_ROOT}/log/webshop_eval"
manifest_file="${MANIFEST_FILE:-${manifest_dir}/premerge_manifest_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.txt}"
manifest_lock="${manifest_file}.lock"
(
  flock -w 600 9
  if [ ! -f "${manifest_file}" ]; then
    tmp="${manifest_file}.tmp"
    : > "${tmp}"
    for model in "${model_dirs[@]}"; do
      model_path="${MODEL_ROOT}/${model}"
      mapfile -t steps < <(find "${model_path}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n)
      if [ "${#steps[@]}" -eq 0 ]; then
        continue
      fi
      for step in "${steps[@]}"; do
        if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
          continue
        fi
        echo "${model_path}/global_step_${step}/actor" >> "${tmp}"
      done
    done
    mv "${tmp}" "${manifest_file}"
  fi
) 9>"${manifest_lock}"

mapfile -t actor_dirs < "${manifest_file}"
total="${#actor_dirs[@]}"

for idx in "${!actor_dirs[@]}"; do
  if (( idx % SHARDS != SHARD_ID )); then
    continue
  fi

  actor_dir="${actor_dirs[$idx]}"
  hf_dir="${actor_dir}/huggingface"
  selected=$((selected + 1))

  step_dir="$(basename "$(dirname "${actor_dir}")")"
  step="${step_dir#global_step_}"
  model="$(basename "$(dirname "$(dirname "${actor_dir}")")")"

  if [ ! -d "${actor_dir}" ]; then
    echo "[premerge-hf] MISSING actor_dir=${actor_dir}" >&2
    failed=$((failed + 1))
    continue
  fi
  if [ ! -d "${hf_dir}" ]; then
    echo "[premerge-hf] MISSING hf_dir=${hf_dir}" >&2
    failed=$((failed + 1))
    continue
  fi

  if [ -f "${hf_dir}/model.safetensors.index.json" ] || \
     [ -f "${hf_dir}/model.safetensors" ] || \
     [ -f "${hf_dir}/pytorch_model.bin" ] || \
     ls "${hf_dir}"/model-*.safetensors "${hf_dir}"/pytorch_model-*.bin >/dev/null 2>&1; then
    echo "[premerge-hf] SKIP (weights exist): ${model} step=${step}"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[premerge-hf] MERGE: ${model} step=${step}"
  if apptainer exec "${APPTAINER_ARGS[@]}" "${APPTAINER_IMG}" \
    bash -lc "source '${VENV_AGENTGYM_RL}/bin/activate' && python3 AgentGym-RL/scripts/model_merger.py --local_dir '${actor_dir}'"; then
    merged=$((merged + 1))
  else
    echo "[premerge-hf] ERROR: model_merger failed for ${model} step=${step}" >&2
    failed=$((failed + 1))
  fi
done

echo "[premerge-hf] Done. total=${total} selected=${selected} merged=${merged} skipped=${skipped} failed=${failed}"
if [ "${failed}" -ne 0 ]; then
  exit 2
fi
