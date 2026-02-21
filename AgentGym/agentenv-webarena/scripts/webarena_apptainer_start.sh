#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/petrelfs/chenlin1/projects/Word2World}"
IMG_DIR="${WEBARENA_APPTAINER_ROOT:-$REPO_ROOT/AgentGym/agentenv-webarena/apptainer_images}"
DATA_DIR="${WEBARENA_DATA_DIR:-/mnt/petrelfs/chenlin1/datasets/webarena_env}"
RUN_DIR="${WEBARENA_RUN_DIR:-/tmp/webarena_sites_${SLURM_JOB_ID:-$$}}"

SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
REDDIT_PORT="${REDDIT_PORT:-9999}"
GITLAB_PORT="${GITLAB_PORT:-8023}"
WIKI_PORT="${WIKI_PORT:-8888}"
MAP_PORT="${MAP_PORT:-3000}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-4399}"

HOSTNAME_VAL="${WEBARENA_HOSTNAME:-127.0.0.1}"

usage() {
  cat <<'USAGE'
Usage: webarena_apptainer_start.sh [--img-dir DIR] [--data-dir DIR] [--run-dir DIR]

Starts WebArena website services via Apptainer (host network, no --net).
Requires: apptainer, images built in IMG_DIR.

Env overrides (ports):
  SHOPPING_PORT=7770 SHOPPING_ADMIN_PORT=7780 REDDIT_PORT=9999
  GITLAB_PORT=8023 WIKI_PORT=8888 MAP_PORT=3000 HOMEPAGE_PORT=4399
  WEBARENA_HOSTNAME=127.0.0.1
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --img-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; IMG_DIR="$2"; shift 2 ;;
    --data-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; DATA_DIR="$2"; shift 2 ;;
    --run-dir) [ $# -ge 2 ] || { echo "Missing arg for $1" >&2; usage; exit 1; }; RUN_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer not found in PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

SHOPPING_IMG="$IMG_DIR/shopping.sif"
SHOPPING_ADMIN_IMG="$IMG_DIR/shopping_admin.sif"
FORUM_IMG="$IMG_DIR/forum.sif"
GITLAB_IMG="$IMG_DIR/gitlab.sif"
KIWIX_IMG="$IMG_DIR/kiwix-serve-3.3.0.sif"
WIKI_ZIM="$DATA_DIR/wikipedia_en_all_maxi_2022-05.zim"

for img in "$SHOPPING_IMG" "$SHOPPING_ADMIN_IMG" "$FORUM_IMG" "$GITLAB_IMG" "$KIWIX_IMG"; do
  if [ ! -f "$img" ]; then
    echo "[FAIL] missing image: $img" >&2
    exit 1
  fi
 done

# write patch script to /tmp (auto bind into container)
PATCH_SCRIPT="$RUN_DIR/webarena_patch_ports.sh"
cat > "$PATCH_SCRIPT" <<'PATCH'
#!/bin/sh
set -e
port="$1"
patch_apache() {
  if [ -f /etc/apache2/ports.conf ]; then
    sed -i "s/Listen 80/Listen ${port}/g" /etc/apache2/ports.conf || true
  fi
  if [ -d /etc/apache2/sites-enabled ]; then
    find /etc/apache2/sites-enabled -maxdepth 1 -type f -print0 | xargs -0 -I{} sed -i "s/<VirtualHost \*:80>/<VirtualHost *:${port}>/g" {} || true
  fi
  if [ -f /etc/apache2/apache2.conf ]; then
    sed -i "s/^User .*/User ${USER}/" /etc/apache2/apache2.conf || true
    sed -i "s/^Group .*/Group ${USER}/" /etc/apache2/apache2.conf || true
  fi
}
patch_nginx() {
  if [ -f /etc/nginx/nginx.conf ]; then
    sed -i "s/^user .*/user ${USER};/" /etc/nginx/nginx.conf || true
  fi
  if [ -d /etc/nginx ]; then
    find /etc/nginx -type f -print0 | xargs -0 -I{} sed -i "s/listen 80;/listen ${port};/g" {} || true
    find /etc/nginx -type f -print0 | xargs -0 -I{} sed -i "s/listen \[::\]:80;/listen [::]:${port};/g" {} || true
  fi
}
patch_apache
patch_nginx
exec /.singularity.d/runscript
PATCH
chmod +x "$PATCH_SCRIPT"

start_instance() {
  local name="$1"; local image="$2"; local port="$3";
  if apptainer instance list | awk '{print $1}' | grep -qx "$name"; then
    echo "[SKIP] instance already running: $name"
    return 0
  fi
  echo "[START] $name on port $port"
  apptainer instance start --writable-tmpfs "$image" "$name" /bin/sh "$PATCH_SCRIPT" "$port"
}

# Start shopping services
start_instance webarena-shopping "$SHOPPING_IMG" "$SHOPPING_PORT"
start_instance webarena-shopping-admin "$SHOPPING_ADMIN_IMG" "$SHOPPING_ADMIN_PORT"
start_instance webarena-forum "$FORUM_IMG" "$REDDIT_PORT"

# GitLab (image already uses port 8023 internally)
if ! apptainer instance list | awk '{print $1}' | grep -qx webarena-gitlab; then
  echo "[START] webarena-gitlab on port $GITLAB_PORT"
  apptainer instance start --writable-tmpfs "$GITLAB_IMG" webarena-gitlab
fi

# Wikipedia (Kiwix)
if ! apptainer instance list | awk '{print $1}' | grep -qx webarena-wiki; then
  if [ ! -f "$WIKI_ZIM" ]; then
    echo "[WARN] missing wikipedia ZIM: $WIKI_ZIM (wiki service skipped)" >&2
  else
    echo "[START] webarena-wiki on port $WIKI_PORT"
    apptainer instance start --writable-tmpfs --bind "$DATA_DIR":/data \
      "$KIWIX_IMG" webarena-wiki \
      /bin/sh -c "kiwix-serve --port ${WIKI_PORT} /data/$(basename "$WIKI_ZIM")"
  fi
fi

# Homepage (simple local HTTP server)
HOMEPAGE_LOG="$RUN_DIR/homepage.log"
HOMEPAGE_PID_FILE="$RUN_DIR/homepage.pid"
if [ ! -f "$HOMEPAGE_PID_FILE" ] || ! kill -0 "$(cat "$HOMEPAGE_PID_FILE")" 2>/dev/null; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PY_BIN=python
  else
    PY_BIN=""
  fi
  if [ -z "$PY_BIN" ]; then
    echo "[WARN] python not found; homepage server skipped" >&2
  else
    echo "[START] homepage on port $HOMEPAGE_PORT"
    SHOPPING="http://${HOSTNAME_VAL}:${SHOPPING_PORT}" \
    SHOPPING_ADMIN="http://${HOSTNAME_VAL}:${SHOPPING_ADMIN_PORT}/admin" \
    REDDIT="http://${HOSTNAME_VAL}:${REDDIT_PORT}" \
    GITLAB="http://${HOSTNAME_VAL}:${GITLAB_PORT}" \
    MAP="http://${HOSTNAME_VAL}:${MAP_PORT}" \
    WIKIPEDIA="http://${HOSTNAME_VAL}:${WIKI_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
    HOMEPAGE_PORT="$HOMEPAGE_PORT" \
    "$PY_BIN" "$REPO_ROOT/AgentGym/agentenv-webarena/scripts/webarena_homepage_server.py" \
      > "$HOMEPAGE_LOG" 2>&1 &
    echo $! > "$HOMEPAGE_PID_FILE"
  fi
fi

# Post-start configuration (best effort)
set +e
apptainer exec instance://webarena-shopping /bin/sh -c "/var/www/magento2/bin/magento setup:store-config:set --base-url=http://${HOSTNAME_VAL}:${SHOPPING_PORT} || true"
apptainer exec instance://webarena-shopping /bin/sh -c "mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value=\"http://${HOSTNAME_VAL}:${SHOPPING_PORT}/\" WHERE path = \"web/secure/base_url\";' || true"
apptainer exec instance://webarena-shopping /bin/sh -c "/var/www/magento2/bin/magento cache:flush || true"

apptainer exec instance://webarena-shopping-admin /bin/sh -c "/var/www/magento2/bin/magento setup:store-config:set --base-url=http://${HOSTNAME_VAL}:${SHOPPING_ADMIN_PORT} || true"
apptainer exec instance://webarena-shopping-admin /bin/sh -c "mysql -u magentouser -pMyPassword magentodb -e 'UPDATE core_config_data SET value=\"http://${HOSTNAME_VAL}:${SHOPPING_ADMIN_PORT}/\" WHERE path = \"web/secure/base_url\";' || true"
apptainer exec instance://webarena-shopping-admin /bin/sh -c "/var/www/magento2/bin/magento cache:flush || true"

apptainer exec instance://webarena-shopping-admin /bin/sh -c "/var/www/magento2/bin/magento config:set admin/security/password_is_forced 0 || true"
apptainer exec instance://webarena-shopping-admin /bin/sh -c "/var/www/magento2/bin/magento config:set admin/security/password_lifetime 0 || true"

apptainer exec instance://webarena-gitlab /bin/sh -c "sed -i \"s|^external_url.*|external_url 'http://${HOSTNAME_VAL}:${GITLAB_PORT}'|\" /etc/gitlab/gitlab.rb || true"
apptainer exec instance://webarena-gitlab /bin/sh -c "gitlab-ctl reconfigure || true"
set -e

cat <<EOF_SUM
[OK] WebArena sites started (best effort)
Run dir: $RUN_DIR
Ports:
  Shopping:        $SHOPPING_PORT
  Shopping admin:  $SHOPPING_ADMIN_PORT
  Reddit/Forum:    $REDDIT_PORT
  GitLab:          $GITLAB_PORT
  Wikipedia:       $WIKI_PORT
  Map:             $MAP_PORT (not started here)
  Homepage:        $HOMEPAGE_PORT

Use: bash AgentGym/agentenv-webarena/scripts/webarena_apptainer_healthcheck.sh --strict
EOF_SUM
