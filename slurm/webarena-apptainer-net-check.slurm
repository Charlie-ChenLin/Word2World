#!/bin/bash
#SBATCH --job-name webarena-apptainer-net-check
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.netcheck.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.netcheck.err

set -euo pipefail

mkdir -p /mnt/petrelfs/chenlin1/log/webarena

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
PORT_HOST="${PORT_HOST:-18080}"
PORT_CONT="${PORT_CONT:-8080}"
export PORT_CONT

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[FAIL] apptainer not found in PATH" >&2
  exit 1
fi

echo "[INFO] apptainer: $(apptainer --version)"

echo "[INFO] slirp4netns: $(command -v slirp4netns || echo 'NOT_FOUND')"

echo "[INFO] image: ${APPTAINER_IMG}"
if [ ! -f "$APPTAINER_IMG" ]; then
  echo "[FAIL] apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi

# Decide a lightweight HTTP server command inside container.
SERVER_CMD=""
if apptainer exec "$APPTAINER_IMG" python3 -V >/dev/null 2>&1; then
  SERVER_CMD="python3 -m http.server ${PORT_CONT}"
elif apptainer exec "$APPTAINER_IMG" python -V >/dev/null 2>&1; then
  SERVER_CMD="python -m http.server ${PORT_CONT}"
elif apptainer exec "$APPTAINER_IMG" perl -e 'use IO::Socket::INET;' >/dev/null 2>&1; then
  PERL_SERVER="/tmp/webarena_netcheck_server.pl"
  cat > "$PERL_SERVER" <<'PL'
use IO::Socket::INET;
my $port = $ENV{PORT_CONT} // 8080;
my $s = IO::Socket::INET->new(LocalPort => $port, Reuse => 1, Listen => 5) or die $!;
while (my $c = $s->accept) {
  print $c "HTTP/1.1 200 OK\r\nContent-Length:2\r\n\r\nOK";
  close $c;
}
PL
  SERVER_CMD="perl $PERL_SERVER"
else
  echo "[FAIL] No python or perl available in container for HTTP server" >&2
  exit 1
fi

echo "[INFO] Starting http server inside container with --net and portmap"
set +e
apptainer exec --net --network-args "portmap=${PORT_HOST}:${PORT_CONT}/tcp" \
  "$APPTAINER_IMG" /bin/sh -c "$SERVER_CMD" \
  >/tmp/webarena_netcheck.server.log 2>&1 &
server_pid=$!
set -e

# wait for server to start
sleep 2

# curl localhost on host side
if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT_HOST}" || true)
  echo "[INFO] curl http://127.0.0.1:${PORT_HOST} -> ${code}"
else
  echo "[WARN] curl not found; skipping HTTP check" >&2
  code=000
fi

# cleanup
kill "$server_pid" >/dev/null 2>&1 || true
wait "$server_pid" >/dev/null 2>&1 || true

if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
  echo "[OK] slirp4netns + portmap appears to work"
  exit 0
fi

# If HTTP failed, show last lines of server log for diagnosis
if [ -f /tmp/webarena_netcheck.server.log ]; then
  echo "[INFO] server log tail:" >&2
  tail -n 50 /tmp/webarena_netcheck.server.log >&2
fi

echo "[FAIL] HTTP check failed (code=${code}); slirp4netns/portmap may be unavailable" >&2
exit 1
