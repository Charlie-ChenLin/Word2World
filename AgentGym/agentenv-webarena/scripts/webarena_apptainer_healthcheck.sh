#!/bin/bash
set -euo pipefail

HOSTNAME_VAL="${WEBARENA_HOSTNAME:-127.0.0.1}"
SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
REDDIT_PORT="${REDDIT_PORT:-9999}"
GITLAB_PORT="${GITLAB_PORT:-8023}"
WIKI_PORT="${WIKI_PORT:-8888}"
MAP_PORT="${MAP_PORT:-3000}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-4399}"
STRICT="${WEBARENA_STRICT:-0}"
SKIP_MAP="${WEBARENA_SKIP_MAP:-0}"
SKIP_WIKI="${WEBARENA_SKIP_WIKI:-0}"

usage() {
  cat <<'USAGE'
Usage: webarena_apptainer_healthcheck.sh [--strict] [--skip-map] [--skip-wiki]

Env overrides:
  WEBARENA_HOSTNAME, SHOPPING_PORT, SHOPPING_ADMIN_PORT, REDDIT_PORT,
  GITLAB_PORT, WIKI_PORT, MAP_PORT, HOMEPAGE_PORT, WEBARENA_STRICT
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --skip-map) SKIP_MAP=1; shift ;;
    --skip-wiki) SKIP_WIKI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found in PATH" >&2
  exit 2
fi

fail=0
check() {
  local name="$1"; local url="$2"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || true)
  if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
    echo "[OK]   $name -> $url ($code)"
  else
    echo "[WARN] $name -> $url ($code)"
    fail=$((fail+1))
  fi
}

check "Shopping" "http://${HOSTNAME_VAL}:${SHOPPING_PORT}"
check "Shopping Admin" "http://${HOSTNAME_VAL}:${SHOPPING_ADMIN_PORT}"
check "Reddit" "http://${HOSTNAME_VAL}:${REDDIT_PORT}"
check "GitLab" "http://${HOSTNAME_VAL}:${GITLAB_PORT}"
if [ "$SKIP_WIKI" != "1" ]; then
  check "Wikipedia" "http://${HOSTNAME_VAL}:${WIKI_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
else
  echo "[SKIP] Wikipedia check (WEBARENA_SKIP_WIKI=1)"
fi
if [ "$SKIP_MAP" != "1" ]; then
  check "Map" "http://${HOSTNAME_VAL}:${MAP_PORT}"
  check "Map tile" "http://${HOSTNAME_VAL}:${MAP_PORT}/tile/0/0/0.png"
else
  echo "[SKIP] Map check (WEBARENA_SKIP_MAP=1)"
fi
check "Homepage" "http://${HOSTNAME_VAL}:${HOMEPAGE_PORT}"

if [ "$STRICT" = "1" ] && [ "$fail" -gt 0 ]; then
  exit 1
fi
