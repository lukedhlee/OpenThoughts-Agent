#!/bin/bash
# driver.sbatch — the relay driver (data/relay/pilot/run_pilot.sh: routers, harbor on Daytona, gates, readout) in its own
# Horizon compute-node job. Horizon's login node has one core, so the driver cannot run there as it did on Jupiter, and it
# cannot run inside the serve job, which it scancels at the end. Compute nodes have no internet: the login-side
# data/r2egym/horizon/tunnel.sh (one `ssh -R PORT` per port in TUNNEL_PORTS; launch_driver.sh starts them) puts a SOCKS5
# proxy on this node's 127.0.0.1:PORT, and socks_connect_bridge.py turns them into one HTTP CONNECT proxy on
# 127.0.0.1:BRIDGE_PORT for the Daytona SDK and curl (HTTPS_PROXY), spreading connections over the tunnels and caching DNS
# (Horizon's first nameserver never answers for outside names: 5 s per uncached lookup). Model traffic is plain http to
# the serve nodes (short names resolve from /etc/hosts) and never touches the proxy.
#
#   launch_driver.sh <serve job> <run name>          (login node: submits this job with the run's settings + the tunnels)
#
# Every run_pilot.sh knob passes through the environment (sbatch --export=ALL); this file only sets Horizon's paths and
# DRIVER_JOB / DRIVER_NODES, so the node-hour cap and run.meta count this node too.
#SBATCH -J relay_drv
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -o /scratch/11584/lukedhlee/experiments/relay/pilot/logs/%x_%j.out
set -uo pipefail
SERVE_JOB=${SERVE_JOB:?SERVE_JOB}; RUN_NAME=${RUN_NAME:?RUN_NAME}; RUN_MODE=${RUN_MODE:-full}
SB=${SNOWBALL_ROOT:-$HOME/snowball}; S=${SCRATCH_DIR:-/scratch/11584/$USER}
OTA=${OTA:-$SB/ota}; HZ=$OTA/data/relay/horizon
TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}; BRIDGE_PORT=${BRIDGE_PORT:-18946}
export PY=${PY:-$SB/envs/snowball/bin/python} HARBOR=${HARBOR:-$SB/envs/snowball/bin/harbor}
export HARBOR_SRC=${HARBOR_SRC:-$SB/harbor-relay/src}
export TREE=${TREE:-$S/relay/calibforge_tree}
export KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}   # the eval org that owns the CalibForge snapshots
export E=${E:-${RELAY_EXP_DIR:-$S/experiments/relay/pilot}}
export JOBS_ROOT=${JOBS_ROOT:-$S/experiments/relay/jobs}
export STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-meta}   # meta: run_pilot.sh reads the served student from the endpoints .meta
[ -f $TREE/TASKS.txt ] && export NTASKS=${NTASKS:-$(grep -c . $TREE/TASKS.txt)}
export DRIVER_JOB=$SLURM_JOB_ID DRIVER_NODES=$SLURM_NNODES
export LITELLM_LOCAL_MODEL_COST_MAP=True   # no GitHub fetch at import
export RES_OPTIONS=${RES_OPTIONS:-timeout:1 attempts:2}   # an uncached outside lookup: 1 s instead of the dead first nameserver's 5 s
LOGD=$E/logs/driver_$SLURM_JOB_ID; mkdir -p $LOGD
log() { echo "[$(date -Is)] $*"; }
log "relay driver job=$SLURM_JOB_ID node=$(hostname -s) serve_job=$SERVE_JOB run=$RUN_NAME mode=$RUN_MODE ota=$(git -C $OTA rev-parse --short=8 HEAD) tunnels=$TUNNEL_PORTS"

# ---- 1. tunnels: the login side opens them once this job runs (pam_slurm_adopt can refuse the first try, tunnel.sh retries)
up=""
for i in $(seq 1 60); do
  up=""; for p in ${TUNNEL_PORTS//,/ }; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && up="$up $p"; done
  [ "$(echo $up | wc -w)" = "$(echo ${TUNNEL_PORTS//,/ } | wc -w)" ] && break
  sleep 10
done
[ -z "$up" ] && { log "no tunnel on any of $TUNNEL_PORTS after 10 min (start launch_driver.sh's tunnel.sh on the login node)"; scancel $SERVE_JOB; exit 1; }
PORTS=$(echo $up | tr ' ' ','); log "tunnels up on $PORTS"

# ---- 2. the CONNECT bridge, restarted if it dies
( while :; do $PY $HZ/socks_connect_bridge.py --port $BRIDGE_PORT --socks-port $PORTS --stats-every 60 >> $LOGD/bridge.log 2>&1
    echo "[$(date -Is)] bridge exited $?, restarting" >> $LOGD/bridge.log; sleep 2; done ) &
BRIDGE_LOOP=$!
trap 'kill $BRIDGE_LOOP 2>/dev/null; pkill -P $BRIDGE_LOOP 2>/dev/null' EXIT
unset HTTP_PROXY http_proxy ALL_PROXY all_proxy
export HTTPS_PROXY=http://127.0.0.1:$BRIDGE_PORT https_proxy=http://127.0.0.1:$BRIDGE_PORT
export NO_PROXY=localhost,127.0.0.1,.horizon.tacc.utexas.edu no_proxy=localhost,127.0.0.1,.horizon.tacc.utexas.edu   # run_pilot adds the model hosts
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 https://app.daytona.io/api/health) && [ "${c%% *}" = 200 ] && break
  sleep 5
done
log "Daytona via the bridge: ${c:-no answer}"

# ---- 3. the driver
bash $OTA/data/relay/pilot/run_pilot.sh $SERVE_JOB $RUN_NAME $RUN_MODE
rc=$?
log "run_pilot.sh exited $rc; bridge: $(grep -c connection_error $LOGD/bridge.log) connection errors, last stats $(grep '"stats"' $LOGD/bridge.log | tail -1)"
exit $rc
