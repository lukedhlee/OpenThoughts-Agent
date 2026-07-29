#!/usr/bin/env bash
# Expose a Jupiter-side vLLM endpoint to JURECA dc-cpu bridge workers.
#
# WHY THIS SHAPE (measured 2026-07-29, don't re-derive it):
#   * JURECA -> Jupiter is DEAD in both senses: `No route to host` on Jupiter's
#     public IP and timeouts on every internal IP (10.128.1.2, 10.99.0.2,
#     10.201.15.132), port 22 included. Workers cannot dial Jupiter.
#   * Jupiter -> JURECA :22 is OPEN. So the tunnel must be initiated FROM
#     Jupiter, which is why this script runs on a Jupiter login node.
#   * OpenCode runs INSIDE the SIF on a JURECA compute node and dials the model
#     itself, so the endpoint must resolve from within JURECA. A relay on a
#     JURECA login node is therefore mandatory, not a convenience.
#   * ssh -4 IS REQUIRED. Jupiter's external egress is IPv6 (2001:638:404::84),
#     which is not in the JuDoor from= clause, so an IPv6 attempt fails with
#     "Permission denied (publickey)". Over IPv4 the source is 134.94.0.132,
#     which matches, and auth proceeds to the second factor.
#   * JSC sshd enforces publickey + keyboard-interactive, so ONE interactive
#     TOTP is unavoidable. We pay it once into a ControlMaster socket and every
#     later automated ssh reuses it with no prompt.
#
# USAGE
#   Step 1 (interactive, ONCE per ControlPersist window, needs your TOTP):
#       ssh jupiter                       # from your Mac
#       eval/jureca/jupiter_to_jureca_tunnel.sh master
#   Step 2 (automated, no TOTP):
#       eval/jureca/jupiter_to_jureca_tunnel.sh up <vllm_host> <vllm_port> <remote_port>
#   Check / tear down:
#       eval/jureca/jupiter_to_jureca_tunnel.sh status
#       eval/jureca/jupiter_to_jureca_tunnel.sh down
set -euo pipefail

KEY="${TUNNEL_KEY:-$HOME/.ssh/id_ed25519_jupiter2jureca}"
JURECA_HOST="${JURECA_HOST:-jureca.fz-juelich.de}"
JURECA_USER="${JURECA_USER:-lee27}"
CM_DIR="${CM_DIR:-$HOME/.ssh/cm_jureca}"
CM_PATH="$CM_DIR/%r@%h:%p"
CONTROL_PERSIST="${CONTROL_PERSIST:-12h}"

mkdir -p "$CM_DIR"
chmod 700 "$CM_DIR"

# -4 is load-bearing; see header.
ssh_common=(
  -4
  -i "$KEY"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=auto
  -o ControlPath="$CM_PATH"
  -o ControlPersist="$CONTROL_PERSIST"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes
)

case "${1:-}" in
  master)
    echo "Opening ControlMaster to $JURECA_USER@$JURECA_HOST."
    echo "You WILL be prompted for your JSC TOTP code -- that is expected, once."
    ssh "${ssh_common[@]}" "$JURECA_USER@$JURECA_HOST" \
      'echo "ControlMaster established on $(hostname -f)"; \
       echo -n "GatewayPorts: "; \
       (grep -iE "^[[:space:]]*GatewayPorts" /etc/ssh/sshd_config 2>/dev/null \
         || echo "unset => defaults to no (loopback-only -R binds)")'
    ;;

  up)
    VLLM_HOST="${2:?usage: up <vllm_host> <vllm_port> <remote_port>}"
    VLLM_PORT="${3:?usage: up <vllm_host> <vllm_port> <remote_port>}"
    REMOTE_PORT="${4:?usage: up <vllm_host> <vllm_port> <remote_port>}"

    if [[ ! -S "$(echo "$CM_PATH" | sed "s/%r/$JURECA_USER/;s/%h/$JURECA_HOST/;s/%p/22/")" ]]; then
      echo "ERROR: no ControlMaster socket. Run '$0 master' interactively first." >&2
      exit 1
    fi

    # Try a public bind first. This only works if JURECA's sshd sets
    # GatewayPorts yes|clientspecified; the default (no) silently restricts the
    # bind to loopback, which compute nodes cannot reach. ExitOnForwardFailure
    # turns that into a hard error rather than a tunnel that looks up but isn't.
    echo "Attempting public -R bind 0.0.0.0:$REMOTE_PORT ..."
    if ssh "${ssh_common[@]}" -f -N \
         -R "0.0.0.0:$REMOTE_PORT:$VLLM_HOST:$VLLM_PORT" \
         "$JURECA_USER@$JURECA_HOST" 2>/dev/null; then
      echo "OK: public bind succeeded (GatewayPorts permits it)."
      echo "Workers should use: http://<jureca-login-IB-host>:$REMOTE_PORT/v1"
      exit 0
    fi

    echo "Public bind refused => falling back to loopback -R + a relay on 0.0.0.0."
    LOOPBACK_PORT=$(( REMOTE_PORT + 1 ))
    ssh "${ssh_common[@]}" -f -N \
      -R "127.0.0.1:$LOOPBACK_PORT:$VLLM_HOST:$VLLM_PORT" \
      "$JURECA_USER@$JURECA_HOST"
    echo "Loopback tunnel up on JURECA 127.0.0.1:$LOOPBACK_PORT."

    # Pure-python relay: socat is not guaranteed present on JSC login nodes.
    ssh "${ssh_common[@]}" "$JURECA_USER@$JURECA_HOST" \
      "REMOTE_PORT=$REMOTE_PORT LOOPBACK_PORT=$LOOPBACK_PORT nohup setsid python3 -u -c '
import os, socket, threading, socketserver

LISTEN = int(os.environ[\"REMOTE_PORT\"])
TARGET = int(os.environ[\"LOOPBACK_PORT\"])

def pipe(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            up = socket.create_connection((\"127.0.0.1\", TARGET), timeout=30)
        except OSError:
            return
        t = threading.Thread(target=pipe, args=(self.request, up), daemon=True)
        t.start()
        pipe(up, self.request)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

Server((\"0.0.0.0\", LISTEN), Handler).serve_forever()
' > \$HOME/jureca_vllm_relay_$REMOTE_PORT.log 2>&1 &
       sleep 2
       (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E \":$REMOTE_PORT\b\" \
         && echo 'relay listening on 0.0.0.0:$REMOTE_PORT' \
         || echo 'WARNING: relay did not bind -- check the log'"

    echo
    echo "Workers should use: http://<jureca-login-IB-host>:$REMOTE_PORT/v1"
    echo "(the bridge already reaches a login node this way -- it served :9920)"
    ;;

  status)
    ssh "${ssh_common[@]}" -O check "$JURECA_USER@$JURECA_HOST" 2>&1 || true
    ssh "${ssh_common[@]}" "$JURECA_USER@$JURECA_HOST" \
      'echo "on $(hostname -f)"; pgrep -af "socketserver|jureca_vllm_relay" | head -5 || echo "no relay"; \
       (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E ":99[0-9][0-9]" || echo "no 99xx listeners"'
    ;;

  down)
    ssh "${ssh_common[@]}" "$JURECA_USER@$JURECA_HOST" \
      'pkill -f "socketserver" 2>/dev/null; echo "relay stopped (if any)"' || true
    ssh "${ssh_common[@]}" -O exit "$JURECA_USER@$JURECA_HOST" 2>&1 || true
    echo "tunnel + ControlMaster torn down"
    ;;

  *)
    sed -n '1,32p' "$0"
    exit 2
    ;;
esac
