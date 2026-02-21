#!/bin/bash
set -euo pipefail

MODE="${WEBARENA_CHECK_MODE:-basic}"
STRICT="${WEBARENA_STRICT:-0}"

usage() {
  cat <<'USAGE'
Usage: webarena_env_check.sh [--curl|--basic] [--strict]
  --basic   Only print resolved URLs (default)
  --curl    Perform HTTP checks via curl
  --strict  Return non-zero if any URL is unreachable

Env overrides:
  SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, HOMEPAGE
  WEBARENA_CHECK_MODE=basic|curl
  WEBARENA_STRICT=0|1
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --basic) MODE="basic"; shift ;;
    --curl) MODE="curl"; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

labels=(SHOPPING SHOPPING_ADMIN REDDIT GITLAB MAP WIKIPEDIA HOMEPAGE)
defaults=(
  "http://127.0.0.1:7770"
  "http://127.0.0.1:7780/admin"
  "http://127.0.0.1:9999"
  "http://127.0.0.1:8023"
  "http://127.0.0.1:3000"
  "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
  "http://127.0.0.1:4399"
)

resolved_urls=()
for i in "${!labels[@]}"; do
  key="${labels[$i]}"
  val="${!key:-${defaults[$i]}}"
  resolved_urls+=("$val")
  printf "%s=%s\n" "$key" "$val"
done

if [ "$MODE" = "basic" ]; then
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found in PATH" >&2
  exit 2
fi

fail=0
for i in "${!labels[@]}"; do
  key="${labels[$i]}"
  url="${resolved_urls[$i]}"
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || true)
  if [ "$code" = "000" ]; then
    echo "[FAIL] $key -> $url (no response)"
    fail=$((fail+1))
    continue
  fi
  if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
    echo "[OK]   $key -> $url ($code)"
  else
    echo "[WARN] $key -> $url ($code)"
    fail=$((fail+1))
  fi
  if [ "$key" = "MAP" ]; then
    tile_url="${url%/}/tile/0/0/0.png"
    tile_code=$(curl -s -o /dev/null -w "%{http_code}" "$tile_url" || true)
    if [ "$tile_code" -ge 200 ] && [ "$tile_code" -lt 400 ]; then
      echo "[OK]   MAP_TILE -> $tile_url ($tile_code)"
    else
      echo "[WARN] MAP_TILE -> $tile_url ($tile_code)"
      fail=$((fail+1))
    fi
  fi
  if [ "$key" = "SHOPPING_ADMIN" ] && [[ "$url" != */admin ]]; then
    echo "[WARN] SHOPPING_ADMIN url usually ends with /admin" >&2
  fi
done

if [ "$STRICT" = "1" ] && [ "$fail" -gt 0 ]; then
  exit 1
fi
