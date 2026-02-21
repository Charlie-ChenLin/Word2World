#!/bin/bash
#SBATCH --job-name webarena-apptainer-hostnet-check
#SBATCH --partition=Gveval2-T
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/petrelfs/chenlin1/log/webarena/%j.hostnet.out
#SBATCH --error=/mnt/petrelfs/chenlin1/log/webarena/%j.hostnet.err

set -euo pipefail

mkdir -p /mnt/petrelfs/chenlin1/log/webarena

APPTAINER_IMG="${APPTAINER_IMG:-$HOME/containers/cuda_12.4.1-devel-ubuntu22.04.sif}"
PORT_CONT="${PORT_CONT:-18081}"
export PORT_CONT

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[FAIL] apptainer not found in PATH" >&2
  exit 1
fi

if [ ! -f "$APPTAINER_IMG" ]; then
  echo "[FAIL] apptainer image not found: $APPTAINER_IMG" >&2
  exit 1
fi

echo "[INFO] apptainer: $(apptainer --version)"

echo "[INFO] Starting HTTP server inside container (host network, no --net) on port ${PORT_CONT}"
PERL_SERVER="/tmp/webarena_hostnet_server.pl"
cat > "$PERL_SERVER" <<'PL'
use IO::Socket::INET;
my $port = $ENV{PORT_CONT} // 18081;
my $s = IO::Socket::INET->new(LocalPort => $port, Reuse => 1, Listen => 5) or die $!;
while (my $c = $s->accept) {
  print $c "HTTP/1.1 200 OK\r\nContent-Length:2\r\n\r\nOK";
  close $c;
}
PL

set +e
apptainer exec "$APPTAINER_IMG" /bin/sh -c "perl $PERL_SERVER" \
  >/tmp/webarena_hostnet.server.log 2>&1 &
server_pid=$!
set -e

sleep 2

if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT_CONT}" || true)
  echo "[INFO] curl http://127.0.0.1:${PORT_CONT} -> ${code}"
else
  echo "[WARN] curl not found; skipping HTTP check" >&2
  code=000
fi

kill "$server_pid" >/dev/null 2>&1 || true
wait "$server_pid" >/dev/null 2>&1 || true

if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
  echo "[OK] host network bind works without --net"
  exit 0
fi

if [ -f /tmp/webarena_hostnet.server.log ]; then
  echo "[INFO] server log tail:" >&2
  tail -n 50 /tmp/webarena_hostnet.server.log >&2
fi

echo "[FAIL] host network bind failed (code=${code})" >&2
exit 1
