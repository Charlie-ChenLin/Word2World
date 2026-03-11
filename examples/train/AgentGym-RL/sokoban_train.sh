#!/bin/bash
set -euo pipefail

# Thin wrapper on top of webshop_train.sh with Sokoban defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SOKOBAN_DEFAULT_SPLIT="${SOKOBAN_DEFAULT_SPLIT:-train}"
train_file="${TRAIN_FILE:-${DATA_ROOT:-data}/train/sokoban_train.json}"

exec bash "${SCRIPT_DIR}/webshop_train.sh" \
  --task_name sokoban \
  --train_file "${train_file}" \
  "$@"
