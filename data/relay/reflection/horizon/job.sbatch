#!/bin/bash
# job.sbatch — one reflection experiment (data.relay.reflection.cli run) in one Horizon allocation of 1 + TRAIN nodes.
# The first node runs the controller, harbor's MSA2 collection (Daytona through the login-side tunnels and the CONNECT
# bridge, as tb2_driver.sbatch does) and this experiment's own vLLM server; the other nodes run each native update
# (train_update.sh). Submitted by launch.sh, which also starts the tunnels. Environment from launch.sh:
#   CONFIG        the experiment JSON (in the job checkout)
#   FIXED_SERVE   "" (ROFT: the refresh adapter starts and refreshes the student) | student | teacher: a fixed author
#                 server started once from the config's weights_path (offline arms)
#   SOURCE_YAML   instead of the controller, run this harbor source-collection config (source_job.py)
#   POST_CHECK=1  after the run, refresh the student server to the final export and check /models and a completion
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
set -uo pipefail
CONFIG=${CONFIG:?CONFIG}
OTA=${REFLECTION_OTA:?REFLECTION_OTA}; SB=$HOME/snowball
PY=$SB/envs/snowball/bin/python; HZ=$OTA/data/relay/horizon
KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}
TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}; BRIDGE_PORT=${BRIDGE_PORT:-18946}
NAME=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$CONFIG")
OUTD=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['runtime']['output_dir'])" "$CONFIG")/$NAME
export REFLECTION_SERVE_DIR=$OUTD/serve/job_$SLURM_JOB_ID REFLECTION_OTA=$OTA
export REFLECTION_TRAIN_NODES=$((SLURM_NNODES - 1))
mkdir -p "$REFLECTION_SERVE_DIR"
log() { echo "[$(date -Is)] $*"; }
export CC=gcc CXX=g++ OMP_NUM_THREADS=4 LITELLM_LOCAL_MODEL_COST_MAP=True RES_OPTIONS=${RES_OPTIONS:-timeout:1 attempts:2}
log "reflection job=$SLURM_JOB_ID node0=$(hostname -s) nodes=$SLURM_JOB_NODELIST config=$CONFIG name=$NAME ota=$(git -C "$OTA" rev-parse --short HEAD) marin=$(git -C "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['backend']['marin_root'])" "$CONFIG")" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- 1. tunnels (login side, launch.sh) and the CONNECT bridge for Daytona
up=""
for i in $(seq 1 60); do
  up=""; for p in ${TUNNEL_PORTS//,/ }; do (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && up="$up $p"; done
  [ "$(echo $up | wc -w)" = "$(echo ${TUNNEL_PORTS//,/ } | wc -w)" ] && break
  sleep 10
done
[ -z "$up" ] && { log "no tunnel on any of $TUNNEL_PORTS after 10 min"; exit 1; }
PORTS=$(echo $up | tr ' ' ','); log "tunnels up on $PORTS"
( while :; do $PY $HZ/socks_connect_bridge.py --port $BRIDGE_PORT --socks-port $PORTS --stats-every 60 >> $REFLECTION_SERVE_DIR/bridge.log 2>&1
    echo "[$(date -Is)] bridge exited $?, restarting" >> $REFLECTION_SERVE_DIR/bridge.log; sleep 2; done ) &
BRIDGE_LOOP=$!
cleanup() {
  bash "$OTA/data/relay/reflection/horizon/serve_model.sh" stop student || true
  bash "$OTA/data/relay/reflection/horizon/serve_model.sh" stop teacher || true
  kill $BRIDGE_LOOP 2>/dev/null; pkill -P $BRIDGE_LOOP 2>/dev/null
}
trap cleanup EXIT
unset HTTP_PROXY http_proxy ALL_PROXY all_proxy
export HTTPS_PROXY=http://127.0.0.1:$BRIDGE_PORT https_proxy=http://127.0.0.1:$BRIDGE_PORT
export NO_PROXY=localhost,127.0.0.1,.horizon.tacc.utexas.edu no_proxy=localhost,127.0.0.1,.horizon.tacc.utexas.edu
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 https://app.daytona.io/api/health) && [ "${c%% *}" = 200 ] && break
  sleep 5
done
log "Daytona via the bridge: ${c:-no answer}"
set -a; source "$KEYF"; set +a

# ---- 2. a fixed author server for offline arms (ROFT's student is started by the refresh adapter)
if [ -n "${FIXED_SERVE:-}" ]; then
  W=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['weights_path'])" "$CONFIG" "$FIXED_SERVE")
  bash "$OTA/data/relay/reflection/horizon/serve_model.sh" refresh "$FIXED_SERVE" "$W" 0 || { log "fixed $FIXED_SERVE server failed"; exit 1; }
fi

# ---- 3. the controller: its state.json is the durable resume point; a rerun of this job resumes it
export PYTHONPATH=$OTA:$SB/harbor-msa-relay/src:${MSA_DIR:-$SB/envs/msa-2.4.6}
cd "$OTA"
if [ -n "${SOURCE_YAML:-}" ]; then
  # Source collection only (source_job.py's harbor config against the fixed server); import on the login node after.
  log "source collection: harbor jobs start -c $SOURCE_YAML"
  $SB/envs/snowball/bin/harbor jobs start --config "$SOURCE_YAML" -y < /dev/null > "$REFLECTION_SERVE_DIR/harbor_sources.log" 2>&1
  rc=$?
  log "SOURCES_EXIT rc=$rc"
  HJ=$($PY -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c['jobs_dir']+'/'+c['job_name'])" "$SOURCE_YAML")
  $PY $HZ/cleanup_sandboxes_async.py --key-file "$KEYF" --delete "$HJ" 2>&1 | tail -2
  log "REFLECTION_JOB_DONE rc=$rc"; exit $rc
fi
log "controller start: cli run --config $CONFIG"
$PY -m data.relay.reflection.cli run --config "$CONFIG" < /dev/null
rc=$?
log "CONTROLLER_EXIT rc=$rc state=$(cat "$OUTD/state.json" 2>/dev/null | tr -d '\n' | head -c 600)"

# ---- 4. smoke: the last export must load in vLLM, be reported as the served root and complete a request
if [ "$rc" = 0 ] && [ "${POST_CHECK:-0}" = 1 ]; then
  M=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$OUTD/state.json")
  V=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['iteration'])" "$OUTD/state.json")
  mkdir -p "$OUTD/post_check"
  $PY - "$CONFIG" "$M" "$V" "$OUTD/post_check" <<'PY'
import json, sys
from pathlib import Path
config, model, version, out = json.load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
(out / "refresh.request.json").write_text(json.dumps({"model": model, "weights_version": version, "config": config, "output_dir": str(out)}))
PY
  $PY -m data.relay.reflection.refresh --request "$OUTD/post_check/refresh.request.json" --response "$OUTD/post_check/refresh.response.json" &&
  curl -sf --noproxy '*' --max-time 600 localhost:29010/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"cf-student","messages":[{"role":"user","content":"Write a bash command that counts the lines of /etc/hosts."}],"max_tokens":2048,"skip_special_tokens":false}' \
    > "$OUTD/post_check/completion.json"
  prc=$?
  log "POST_CHECK rc=$prc root=$(grep -o '"root": *"[^"]*"' "$OUTD/post_check/refresh.response.json" | head -1) completion=$(head -c 300 "$OUTD/post_check/completion.json" 2>/dev/null)"
  [ "$prc" = 0 ] || rc=$prc
fi
HJ=$(ls -d "$OUTD"/update-*/harbor/*/ 2>/dev/null)
[ -n "$HJ" ] && $PY $HZ/cleanup_sandboxes_async.py --key-file "$KEYF" --delete $HJ 2>&1 | tail -2
log "REFLECTION_JOB_DONE rc=$rc"
exit $rc
