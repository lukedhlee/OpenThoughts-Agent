#!/bin/bash
# tb2_driver.sbatch — one harbor eval run (TB2 / SWE-bench / TB-lite on Daytona) in its own Horizon compute-node job,
# against the model server of another job. By diff from data/relay/horizon/driver.sbatch: the same login-side tunnels
# (tunnel.sh, started by run_tb2.sh) and the same CONNECT bridge for Daytona; instead of run_pilot.sh it runs
# `harbor run -c $CFG` with the pinned harbor clone on PYTHONPATH (as Jupiter's run_tb2.sh ran it on a login node), then
# removes any sandbox of this job left behind and, with SCANCEL_SERVE=1, releases the serve job.
#
#   submitted by run_tb2.sh with CFG, HARBOR_SRC, SERVE_JOB, RUN_NAME, JOBS in the environment
#SBATCH -J tb2_drv
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 08:00:00
set -uo pipefail
CFG=${CFG:?CFG}; HARBOR_SRC=${HARBOR_SRC:?HARBOR_SRC}; SERVE_JOB=${SERVE_JOB:?SERVE_JOB}; RUN_NAME=${RUN_NAME:?RUN_NAME}; JOBS=${JOBS:?JOBS}
SB=${SNOWBALL_ROOT:-$HOME/snowball}; OTA=${OTA:-$SB/ota}; HZ=$OTA/data/relay/horizon
PY=${PY:-$SB/envs/snowball/bin/python}; HARBOR=${HARBOR:-$SB/envs/snowball/bin/harbor}
KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}
TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}; BRIDGE_PORT=${BRIDGE_PORT:-18946}
LOGD=${LOGD:-$(dirname $CFG)/logs/drv_$SLURM_JOB_ID}; mkdir -p $LOGD
export OMP_NUM_THREADS=1 LITELLM_LOCAL_MODEL_COST_MAP=True RES_OPTIONS=${RES_OPTIONS:-timeout:1 attempts:2}
log() { echo "[$(date -Is)] $*"; }
log "tb2 driver job=$SLURM_JOB_ID node=$(hostname -s) serve_job=$SERVE_JOB run=$RUN_NAME harbor=$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD) cfg=$CFG"

# ---- 1. tunnels (login side) and the CONNECT bridge, as in driver.sbatch
up=""
for i in $(seq 1 60); do
  up=""; for p in ${TUNNEL_PORTS//,/ }; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && up="$up $p"; done
  [ "$(echo $up | wc -w)" = "$(echo ${TUNNEL_PORTS//,/ } | wc -w)" ] && break
  sleep 10
done
[ -z "$up" ] && { log "no tunnel on any of $TUNNEL_PORTS after 10 min"; exit 1; }
PORTS=$(echo $up | tr ' ' ','); log "tunnels up on $PORTS"
( while :; do $PY $HZ/socks_connect_bridge.py --port $BRIDGE_PORT --socks-port $PORTS --stats-every 60 >> $LOGD/bridge.log 2>&1
    echo "[$(date -Is)] bridge exited $?, restarting" >> $LOGD/bridge.log; sleep 2; done ) &
BRIDGE_LOOP=$!
trap 'kill $BRIDGE_LOOP 2>/dev/null; pkill -P $BRIDGE_LOOP 2>/dev/null' EXIT
unset HTTP_PROXY http_proxy ALL_PROXY all_proxy
URL=$(sed -n 's#.*api_base: *##p' $CFG | head -1); MHOST=$(echo $URL | sed -E 's#^[a-z]+://([^:/]+).*#\1#')
export HTTPS_PROXY=http://127.0.0.1:$BRIDGE_PORT https_proxy=http://127.0.0.1:$BRIDGE_PORT
export NO_PROXY=localhost,127.0.0.1,.horizon.tacc.utexas.edu,$MHOST no_proxy=localhost,127.0.0.1,.horizon.tacc.utexas.edu,$MHOST
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 https://app.daytona.io/api/health) && [ "${c%% *}" = 200 ] && break
  sleep 5
done
log "Daytona via the bridge: ${c:-no answer}; model server $URL"

# ---- 2. the server answers from here (as run_tb2.sh checks it)
curl -sf --max-time 20 $URL/models | grep -q '"snowball"' || { log "no model 'snowball' at $URL"; exit 1; }
curl -sf --max-time 300 $URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"snowball","messages":[{"role":"user","content":"Print hello in bash."}],"max_tokens":300,"skip_special_tokens":false}' \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); print("server ok:", r["usage"], repr((r["choices"][0]["message"].get("content") or "")[:160]))'

# ---- 3. harbor
set -a; source $KEYF; set +a
log "harbor run -c $CFG"
# MSA_DIR (mini-swe-agent tool-mode policies): upstream mini-swe-agent on the path beside the harbor clone
PYTHONPATH=$HARBOR_SRC${MSA_DIR:+:$MSA_DIR} $HARBOR run -c $CFG -y > $LOGD/harbor.log 2>&1
rc=$?
log "harbor exited $rc"

# ---- 4. sandboxes of this job left behind (harbor deletes its own; this only catches leftovers), then the serve job
$PY $HZ/cleanup_sandboxes_async.py --key-file $KEYF --delete $JOBS/$RUN_NAME 2>&1 | tail -2
[ "${SCANCEL_SERVE:-1}" = 1 ] && { scancel $SERVE_JOB; log "serve job $SERVE_JOB cancelled"; }
log "TB2_DONE rc=$rc bridge: $(grep -c connection_error $LOGD/bridge.log) connection errors, last stats $(grep '"stats"' $LOGD/bridge.log | tail -1)"
