#!/usr/bin/env bash
set -u

JOBS=(${JOBS:-5849050 5849051 5849052 5849053})
INTERVAL=${INTERVAL:-600}
LOG=${LOG:-/mnt/inspurfs/chenlin1/Word2World/log/webshop/job_monitor_$(date +%Y%m%d-%H%M%S).log}

job_list_csv() {
  local IFS=,
  echo "${JOBS[*]}"
}

while true; do
  {
    echo "==== $(date) ===="
    squeue -j "$(job_list_csv)" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.9l %.6D %R" 2>&1 || true
    for j in "${JOBS[@]}"; do
      if ! squeue -j "$j" -h >/dev/null 2>&1; then
        sacct -j "$j" --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -n 2>&1 || true
      fi
    done
    echo
  } >> "$LOG"
  sleep "$INTERVAL"
done
