#!/bin/bash
# node_bridge.sh <gw_dir> <tunnel ports, comma-separated> <bridge port> — run once per node (srun --overlap) by an RL arm's
# sbatch. Waits for the login-side tunnel.sh to open its `ssh -R PORT` SOCKS5 listeners on this node, then keeps
# socks_connect_bridge.py (the relay driver's HTTP CONNECT bridge) running on 127.0.0.1:<bridge port>, restarting it if it
# dies, and touches <gw_dir>/ready-<host> once the bridge reaches the Daytona API. Runs until the job ends.
GW=${1:?gw_dir}; PORTS=${2:?ports}; BP=${3:-18946}
H=$(hostname -s); LOG=$GW/$H.log
OTA=$(cd "$(dirname "$0")/../../../.." && pwd)
PY=${RL_PYTHON:-$HOME/snowball/envs/snowball/bin/python}
log() { echo "[$(date -Is)] $*" >> "$LOG"; }
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy LD_PRELOAD
up=""
for i in $(seq 1 90); do
  up=""; for p in ${PORTS//,/ }; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && up="$up $p"; done
  [ "$(echo $up | wc -w)" = "$(echo ${PORTS//,/ } | wc -w)" ] && break
  sleep 10
done
[ -z "$up" ] && { log "no tunnel on any of $PORTS after 15 min"; exit 1; }
UP=$(echo $up | tr ' ' ','); log "tunnels up on $UP"
( while :; do OMP_NUM_THREADS=1 $PY -u $OTA/data/relay/horizon/socks_connect_bridge.py --port $BP --socks-port $UP --stats-every 300 >> "$LOG" 2>&1
    log "bridge exited $?, restarting"; sleep 2; done ) &
LOOP=$!
trap 'kill $LOOP 2>/dev/null; pkill -P $LOOP 2>/dev/null' EXIT TERM
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -x http://127.0.0.1:$BP https://app.daytona.io/api/health) && [ "$c" = 200 ] && break
  sleep 5
done
log "Daytona health through the bridge: ${c:-none}"
[ "$c" = 200 ] && touch "$GW/ready-$H"
wait $LOOP
