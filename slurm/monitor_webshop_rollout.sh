#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/mnt/petrelfs/chenlin1/projects/Word2World/slurm/webshop-verl-agent-low-kl-small-minibatch-a800-apptainer-qwen2.5-7b.slurm"
LOG_ROOT="/mnt/inspurfs/chenlin1/Word2World/log/webshop"
TS="$(date +%Y%m%d-%H%M%S)"
MONITOR_LOG="${LOG_ROOT}/monitor_webshop_rollout_${TS}.log"

# Initial job IDs (submitted just before starting this monitor)
JOB_IDS=(5857766 5857767 5857768)
COEFS=(0 0.1 1)

# Rollout adjustment levels (progressively lower memory)
ROLL_SAMPLES=(8 4 2 1)
ROLL_MEM=(0.6 0.5 0.4 0.3)
ROLL_TP=(1 2 4 8)
LEVEL=1

DURATION_SEC=3600
SLEEP_SEC=60
END_TS=$(( $(date +%s) + DURATION_SEC ))

log() {
  echo "[$(date +"%F %T")] $*" | tee -a "$MONITOR_LOG"
}

job_state() {
  local jid="$1"
  local state
  state=$(squeue -h -j "$jid" -o %T 2>/dev/null || true)
  if [ -z "$state" ]; then
    state=$(sacct -n -j "$jid" -o State 2>/dev/null | head -n1 | awk "{print \$1}" || true)
  fi
  echo "${state:-UNKNOWN}"
}

is_oom() {
  local jid="$1"
  local log_file="${LOG_ROOT}/${jid}/train.log"
  if [ -f "$log_file" ]; then
    grep -qiE "out of memory|cuda out of memory|OutOfMemoryError" "$log_file"
  else
    return 1
  fi
}

cancel_jobs() {
  local ids=("$@")
  if [ ${#ids[@]} -gt 0 ]; then
    scancel "${ids[@]}" >/dev/null 2>&1 || true
  fi
}

submit_jobs() {
  local lvl="$1"
  local sample="${ROLL_SAMPLES[$lvl]}"
  local mem="${ROLL_MEM[$lvl]}"
  local tp="${ROLL_TP[$lvl]}"
  local new_ids=()

  log "Submitting jobs with rollout_sample_num=${sample}, rollout_gpu_mem_util=${mem}, rollout_tp_size=${tp}"
  for coef in "${COEFS[@]}"; do
    local out
    out=$(sbatch --export=ALL,ENV_FEEDBACK_LOSS_COEF=${coef},ROLLOUT_SAMPLE_NUM=${sample},ROLLOUT_GPU_MEM_UTIL=${mem},ROLLOUT_TP_SIZE=${tp} "$SCRIPT")
    local jid
    jid=$(echo "$out" | awk "{print \$4}")
    new_ids+=("$jid")
  done
  JOB_IDS=("${new_ids[@]}")
  log "New job IDs: ${JOB_IDS[*]}"
}

log "Monitor start; duration=${DURATION_SEC}s; initial jobs: ${JOB_IDS[*]}"

while [ $(date +%s) -lt $END_TS ]; do
  any_running=false
  any_pending=false
  any_failed=false
  oom_detected=false

  for jid in "${JOB_IDS[@]}"; do
    state=$(job_state "$jid")
    log "Job ${jid} state=${state}"
    case "$state" in
      RUNNING)
        any_running=true
        ;;
      PENDING)
        any_pending=true
        ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)
        any_failed=true
        if is_oom "$jid"; then
          oom_detected=true
        fi
        ;;
    esac
  done

  if $any_failed && $oom_detected; then
    if [ $LEVEL -lt $(( ${#ROLL_SAMPLES[@]} - 1 )) ]; then
      log "OOM detected; lowering rollout settings (level ${LEVEL} -> $((LEVEL+1)))"
      LEVEL=$((LEVEL+1))
      cancel_jobs "${JOB_IDS[@]}"
      submit_jobs "$LEVEL"
      sleep "$SLEEP_SEC"
      continue
    else
      log "OOM detected but already at lowest rollout level; no further changes"
      break
    fi
  fi

  if ! $any_running && ! $any_pending; then
    log "All jobs finished; stopping monitor early"
    break
  fi

  sleep "$SLEEP_SEC"
done

log "Monitor finished"
