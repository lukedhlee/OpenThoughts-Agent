#!/bin/bash
# push_rollouts.sh <run name> [more run names] — Horizon login node. Packs each finished relay run (its run dir and its
# harbor job dirs, CHUNK trial dirs per .tar.zst) with a sha256 manifest and episode counts, and uploads it to the
# private HF dataset repo REPO (created private on first use). Idempotent and resumable: re-running skips every unit
# the repo already has; a run is complete once <run>/DONE is in the repo. One process at the lowest CPU and I/O
# priority (the login node has one core). Jupiter side: pull_rollouts.sh. Token: ~/.cache/huggingface/token (laion write).
#   bash push_rollouts.sh relay_hz_a1_20260930            # in tmux or with setsid nohup; ~1-2 min per 1,000 episodes
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: push_rollouts.sh <run name> ..."; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
S=${SCRATCH_DIR:-/scratch/11584/$USER}
PY=${PY:-$HOME/snowball/envs/snowball/bin/python}
E=${E:-${RELAY_EXP_DIR:-$S/experiments/relay/pilot}}; JOBS_ROOT=${JOBS_ROOT:-$S/experiments/relay/jobs}
REPO=${REPO:-laion/relay-rollouts-horizon}; CHUNK=${CHUNK:-500}; STAGE=${STAGE:-$S/relay/push_stage}
export OMP_NUM_THREADS=1 HF_XET_HIGH_PERFORMANCE=0 HF_HUB_DISABLE_PROGRESS_BARS=1
export RES_OPTIONS=${RES_OPTIONS:-timeout:1 attempts:2}   # Horizon's first nameserver never answers outside names (5 s per lookup)
for NAME in "$@"; do
  echo "[$(date -Is)] push $NAME -> $REPO"
  nice -n 19 ionice -c 3 $PY $HERE/rollout_transfer.py push --repo $REPO --name $NAME --run-dir $E/runs/$NAME \
    --jobs-root $JOBS_ROOT --stage $STAGE --chunk $CHUNK ${PUSH_ARGS:-} || { echo "[$(date -Is)] push $NAME FAILED"; exit 1; }
done
