#!/bin/bash
# launch.sh CONFIG TRAIN_NODES TIME [FIXED_SERVE] — submit job.sbatch for one reflection experiment (1 + TRAIN_NODES
# nodes) from the job checkout ~/snowball/ota-reflection and start its two login-side tunnels. Login node only; it
# returns once the job is submitted. Rerunning it on the same config resumes from state.json.
#   POST_CHECK=1 bash launch.sh data/relay/reflection/configs/smoke/roft.json 4 03:00:00          # the ROFT smoke
#   bash launch.sh data/relay/reflection/configs/teacher_teacher.json 4 24:00:00 teacher           # an offline arm
set -euo pipefail
CONFIG=$(readlink -f "${1:?config}"); TRAIN=${2:?training nodes}; TIME=${3:?time limit}; FIXED=${4:-}
OTA=$(cd "$(dirname "$0")/../../../.." && pwd)
case "$CONFIG" in "$OTA"/*) ;; *) echo "config must live in this checkout ($OTA)"; exit 1;; esac
PY=$HOME/snowball/envs/snowball/bin/python
NAME=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$CONFIG")
OUT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['runtime']['output_dir'])" "$CONFIG")/$NAME
PYTHONPATH=$OTA $PY -c "import sys; from data.relay.reflection.core import load_config; load_config(sys.argv[1])" "$CONFIG"
mkdir -p "$OUT/logs"
J=$(CC=gcc CXX=g++ CONFIG=$CONFIG REFLECTION_OTA=$OTA FIXED_SERVE=$FIXED POST_CHECK=${POST_CHECK:-0} \
  sbatch --parsable -N $((TRAIN + 1)) -t "$TIME" -J "cfr_$NAME" -o "$OUT/logs/job_%j.out" --export=ALL \
  "$OTA/data/relay/reflection/horizon/job.sbatch")
for p in 18080 18081; do
  setsid nohup bash "$OTA/data/r2egym/horizon/tunnel.sh" "$J" "$p" > "$OUT/logs/tunnel_${J}_$p.log" 2>&1 < /dev/null &
done
echo "$J $((TRAIN + 1)) nodes x $TIME -> $OUT/logs/job_$J.out"
