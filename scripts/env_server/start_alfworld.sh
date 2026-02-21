source "$HOME/uv_envs/uv_alfworld/bin/activate"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$REPO_ROOT/.cache/alfworld}"
alfworld --host 0.0.0.0 --port 36001
