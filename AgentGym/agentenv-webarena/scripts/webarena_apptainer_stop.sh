#!/bin/bash
set -euo pipefail

RUN_DIR="${WEBARENA_RUN_DIR:-/tmp/webarena_sites_${SLURM_JOB_ID:-}}"

stop_instance() {
  local name="$1"
  if apptainer instance list | awk '{print $1}' | grep -qx "$name"; then
    echo "[STOP] instance $name"
    apptainer instance stop "$name" || true
  fi
}

stop_instance webarena-shopping
stop_instance webarena-shopping-admin
stop_instance webarena-forum
stop_instance webarena-gitlab
stop_instance webarena-wiki

# Stop homepage server
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/homepage.pid" ]; then
  pid=$(cat "$RUN_DIR/homepage.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "[STOP] homepage pid $pid"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
fi
