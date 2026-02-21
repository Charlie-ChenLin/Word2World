#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/log/webshop/monitor_jobs.tsv}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/log/webshop/monitor_oom.log}"
CHANGE_LOG="${CHANGE_LOG:-$REPO_ROOT/log/webshop/monitor_changes.log}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_ROOT/slurm/webshop-a800-apptainer.slurm}"
SLEEP_SEC="${SLEEP_SEC:-600}"
MAX_RETRIES="${MAX_RETRIES:-1}"

OOM_REGEX='(CUDA )?out of memory|CUBLAS_STATUS_ALLOC_FAILED|std::bad_alloc|oom-kill|Killed process|OOM'

if [ ! -f "$STATE_FILE" ]; then
  echo "[monitor] Missing state file: $STATE_FILE" | tee -a "$LOG_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

log_msg() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

log_change() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$CHANGE_LOG" >> "$LOG_FILE"
}

submit_retry() {
  local coef="$1"
  local label="$2"
  local attempt="$3"
  local next_attempt=$((attempt + 1))
  local oom_label="${label}-oomfix${next_attempt}"
  local out
  out=$(sbatch \
    --export=ALL,ENV_FEEDBACK_LOSS_COEF="$coef",EXP_NAME="webshop-a800-${oom_label}",\
MAX_PROMPT_LENGTH=2048,MAX_RESPONSE_LENGTH=8192,MAX_MODEL_LEN=16384,MAX_TOKENS=256,\
ROLLOUT_SAMPLE_NUM=4,TRAIN_BATCH_SIZE=4,PPO_MINI_BATCH_SIZE=4,PPO_MICRO_BATCH_SIZE_PER_GPU=1,\
ROLLOUT_GPU_MEM_UTIL=0.6,ROLLOUT_TP_SIZE=1 \
    "$SLURM_SCRIPT")
  local jid
  jid=$(echo "$out" | awk '{print $NF}')
  log_change "OOM detected for coef=${coef} label=${label} -> resubmitted as ${oom_label} (job ${jid}); reduced max lengths and batch sizes."
  echo -e "${jid}\t${coef}\t${oom_label}\t${next_attempt}\tPD"
}

while true; do
  tmp_state="${STATE_FILE}.tmp"
  : > "$tmp_state"
  while IFS=$'\t' read -r job_id coef label attempt status; do
    if [ -z "${job_id:-}" ]; then
      continue
    fi
    if [[ "$job_id" == \#* ]]; then
      echo -e "$job_id" >> "$tmp_state"
      continue
    fi
    attempt="${attempt:-0}"
    status="${status:-unknown}"

    queue_state=$(squeue -j "$job_id" -h -o '%T' 2>/dev/null || true)
    if [ -n "$queue_state" ]; then
      echo -e "${job_id}\t${coef}\t${label}\t${attempt}\t${queue_state}" >> "$tmp_state"
      continue
    fi

    sacct_state=$(sacct -j "$job_id" --format=JobID,State -P -n 2>/dev/null | awk -F'|' -v id="$job_id" '$1==id {print $2; exit}')

    log_dir="$REPO_ROOT/log/webshop/${job_id}"
    train_log="$log_dir/train.log"
    err_log="$log_dir/error.err"

    oom_found=0
    if [ -f "$train_log" ] && grep -E -i -m1 "$OOM_REGEX" "$train_log" >/dev/null 2>&1; then
      oom_found=1
    elif [ -f "$err_log" ] && grep -E -i -m1 "$OOM_REGEX" "$err_log" >/dev/null 2>&1; then
      oom_found=1
    fi

    if [ "$oom_found" -eq 1 ]; then
      if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        retry_line=$(submit_retry "$coef" "$label" "$attempt")
        echo -e "$retry_line" >> "$tmp_state"
        echo -e "${job_id}\t${coef}\t${label}\t${attempt}\tOOM" >> "$tmp_state"
      else
        log_msg "OOM detected for coef=${coef} label=${label} (job ${job_id}) but retries exhausted."
        echo -e "${job_id}\t${coef}\t${label}\t${attempt}\tOOM_GIVEUP" >> "$tmp_state"
      fi
      continue
    fi

    if [ -n "$sacct_state" ]; then
      echo -e "${job_id}\t${coef}\t${label}\t${attempt}\t${sacct_state}" >> "$tmp_state"
    else
      # Not in queue, no sacct state, and no OOM pattern found.
      echo -e "${job_id}\t${coef}\t${label}\t${attempt}\tUNKNOWN" >> "$tmp_state"
    fi
  done < "$STATE_FILE"

  mv "$tmp_state" "$STATE_FILE"
  sleep "$SLEEP_SEC"
done
