#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
IMG_DIR="${WEBARENA_APPTAINER_ROOT:-$REPO_ROOT/AgentGym/agentenv-webarena/apptainer_images}"

SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
FORUM_PORT="${FORUM_PORT:-9999}"

usage() {
  cat <<'USAGE'
Usage: webarena_apptainer_patch_ports.sh [--img-dir DIR]

Patches Apache/Nginx configs inside Apptainer images to listen on high ports.
Requires images to be writable at runtime; this script generates patch commands
that are applied in start script (no in-place modification of SIF).

Env overrides:
  WEBARENA_APPTAINER_ROOT
  SHOPPING_PORT, SHOPPING_ADMIN_PORT, FORUM_PORT
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --img-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; IMG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi

patch_cmd() {
  local port="$1"
  cat <<'PATCH'
patch_apache() {
  local port="$1"
  if [ -f /etc/apache2/ports.conf ]; then
    sed -i "s/Listen 80/Listen ${port}/g" /etc/apache2/ports.conf || true
  fi
  if [ -d /etc/apache2/sites-enabled ]; then
    find /etc/apache2/sites-enabled -type f -maxdepth 1 -print0 | xargs -0 -I{} sed -i "s/<VirtualHost \*:80>/<VirtualHost *:${port}>/g" {} || true
  fi
  if [ -f /etc/apache2/apache2.conf ]; then
    sed -i "s/^User .*/User ${USER}/" /etc/apache2/apache2.conf || true
    sed -i "s/^Group .*/Group ${USER}/" /etc/apache2/apache2.conf || true
  fi
}
patch_nginx() {
  local port="$1"
  if [ -f /etc/nginx/nginx.conf ]; then
    sed -i "s/^user .*/user ${USER};/" /etc/nginx/nginx.conf || true
  fi
  if [ -d /etc/nginx ]; then
    find /etc/nginx -type f -print0 | xargs -0 -I{} sed -i "s/listen 80;/listen ${port};/g" {} || true
    find /etc/nginx -type f -print0 | xargs -0 -I{} sed -i "s/listen \[::\]:80;/listen [::]:${port};/g" {} || true
  fi
}
PATCH
  echo "patch_apache ${port}; patch_nginx ${port};"
}

cat <<EOF_SUM
[INFO] Patch commands (used by start script with --writable-tmpfs):
SHOPPING_PORT=${SHOPPING_PORT}
SHOPPING_ADMIN_PORT=${SHOPPING_ADMIN_PORT}
FORUM_PORT=${FORUM_PORT}

These patches are applied at container start time; SIF files remain unchanged.
EOF_SUM
