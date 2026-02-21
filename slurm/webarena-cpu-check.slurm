#!/bin/bash
#SBATCH --job-name webarena-cpu-check
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
mkdir -p /mnt/petrelfs/chenlin1/log/webarena
cd "$REPO_ROOT"

# Optional: source env file that exports SHOPPING/REDDIT/... etc.
if [ -n "${WEBARENA_ENV_FILE:-}" ] && [ -f "$WEBARENA_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$WEBARENA_ENV_FILE"
fi

# Optional: set all URLs from a single host.
if [ -n "${WEBARENA_HOST:-}" ]; then
  export SHOPPING="http://${WEBARENA_HOST}:7770"
  export SHOPPING_ADMIN="http://${WEBARENA_HOST}:7780/admin"
  export REDDIT="http://${WEBARENA_HOST}:9999"
  export GITLAB="http://${WEBARENA_HOST}:8023"
  export MAP="http://${WEBARENA_HOST}:3000"
  export WIKIPEDIA="http://${WEBARENA_HOST}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
  export HOMEPAGE="http://${WEBARENA_HOST}:4399"
fi

MODE="${WEBARENA_CHECK_MODE:-basic}"
ARGS=()
if [ "$MODE" = "curl" ]; then
  ARGS+=(--curl)
else
  ARGS+=(--basic)
fi
if [ "${WEBARENA_STRICT:-0}" = "1" ]; then
  ARGS+=(--strict)
fi

bash AgentGym/agentenv-webarena/scripts/webarena_env_check.sh "${ARGS[@]}"
