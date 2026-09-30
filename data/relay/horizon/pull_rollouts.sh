#!/bin/bash
# pull_rollouts.sh [run name ...] — Jupiter login node. Downloads the relay runs push_rollouts.sh finished on Horizon
# (every run with a DONE in the private repo REPO, or the named ones) into DEST/<run>/, checks each tar's sha256 against
# the run's manifest, unpacks it (DEST/<run>/run = the run dir, DEST/<run>/jobs = its harbor job dirs, run/jobs -> ../jobs),
# and links $E/runs/<run> -> DEST/<run>/run so readout.py, merge_runs.py and select_kept.py take it like a Jupiter run.
# Idempotent: a unit is unpacked once. Everything lands on /e/data1/mmlaion (40M-inode fileset); only the one symlink goes
# on reformo fscratch. MAX_NEW_FILES caps the files one call may create (default 400k; a 64k relay trial is ~11 files).
# Login-node safe: one process, OMP_NUM_THREADS=1, xet off (its downloader starts a thread per core).
#   bash pull_rollouts.sh                   # every finished run not yet pulled
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python}
REPO=${REPO:-lukeleeai/relay-rollouts-horizon}; DEST=${DEST:-/e/data1/mmlaion/lee27/relay/horizon}
E=${E:-/e/fscratch/reformo/lee27/experiments/relay/pilot}; MAX_NEW_FILES=${MAX_NEW_FILES:-400000}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $DEST
if [ $# -eq 0 ]; then set -- ""; fi
for NAME in "$@"; do
  echo "[$(date -Is)] pull ${NAME:-all finished runs} <- $REPO"
  nice -n 10 $PY $HERE/rollout_transfer.py pull --repo $REPO --dest $DEST ${NAME:+--name $NAME} --link-runs $E/runs \
    --max-new-files $MAX_NEW_FILES ${PULL_ARGS:-} || { echo "[$(date -Is)] pull ${NAME:-all} FAILED"; exit 1; }
done
echo "[$(date -Is)] inodes on /e/data1: $(df -i /e/data1 | awk 'NR==2{print $3 " used of " $2}')"
