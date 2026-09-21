#!/bin/bash
# gepa_serve.sbatch — the STANDING serve job for the GEPA loop: N Jupiter nodes, one vLLM server each, held up
# while the Opus session reflects, so the GPUs never idle waiting on a human-in-the-loop step.
#
# This replaces "one probe sbatch per candidate". The measured reason (2026-09-20): a probe job spends ~0.5 h x nodes
# loading the 67B MoE across four engines before its first trial, which is ~4 node-hours per job on 8 nodes. A wave
# that reflects between candidates paid that over and over. One standing server pays it once.
#
# Each node runs gepa_serve_node.sh (serve_snowball.sbatch's body, adapted by diff) and publishes
# $GEPA_ENDPOINTS/<jobid>.<rank>. gepa_runner.py picks endpoints up from there.
#
# IDLE WATCHDOG: every 60 s the head checks the queue. With nothing queued and nothing running for IDLE_MIN
# consecutive minutes, it scancels this job. That is the whole safety story for a standing allocation -- a loop
# that is abandoned mid-session releases its nodes by itself instead of burning the wall.
#
#   NODES=8 IDLE_MIN=20 sbatch -N 8 gepa_serve.sbatch      (gepa_serve.sh does this for you, behind SUBMIT=1)
#
#SBATCH --job-name=gepa_serve
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/gepa/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
G=${GEPA_CODE:-/e/project1/transfernetx/lee27/code/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
export GEPA_ENDPOINTS=${GEPA_ENDPOINTS:-$E/endpoints}
export GEPA_LOGS=${GEPA_LOGS:-$E/logs}
export MODEL DRAFT SERVED MAX_NUM_SEQS MAX_MODEL_LEN
IDLE_MIN=${IDLE_MIN:-20}
mkdir -p "$GEPA_ENDPOINTS" "$GEPA_LOGS" "$E/queue" "$E/running" "$E/done"
echo "$(date -Is) gepa_serve job $SLURM_JOB_ID on $SLURM_NNODES nodes; idle cancel after ${IDLE_MIN} min"

# --- idle watchdog (head node, background) -------------------------------------------------------------------
(
  idle=0
  # grace: do not count idle minutes until the servers are actually up, or a slow start would cancel the job
  while [ "$(ls -1 "$GEPA_ENDPOINTS"/$SLURM_JOB_ID.* 2>/dev/null | wc -l)" -eq 0 ]; do
    sleep 60
    squeue -h -j "$SLURM_JOB_ID" -o %T | grep -q RUNNING || exit 0
  done
  echo "$(date -Is) watchdog: servers up, idle timer armed"
  while true; do
    sleep 60
    n=$(( $(ls -1 "$E/queue" 2>/dev/null | grep -c . ) + $(ls -1 "$E/running" 2>/dev/null | grep -c . ) ))
    if [ "$n" -gt 0 ]; then
      idle=0
    else
      idle=$((idle + 1))
      [ $((idle % 5)) -eq 0 ] && echo "$(date -Is) watchdog: idle ${idle}/${IDLE_MIN} min"
      if [ "$idle" -ge "$IDLE_MIN" ]; then
        echo "$(date -Is) watchdog: ${IDLE_MIN} min with an empty queue -- cancelling $SLURM_JOB_ID"
        scancel "$SLURM_JOB_ID"
        exit 0
      fi
    fi
  done
) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null; rm -f "$GEPA_ENDPOINTS"/$SLURM_JOB_ID.*' TERM EXIT

# --- one server per node -------------------------------------------------------------------------------------
srun --kill-on-bad-exit=0 --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash "$G/gepa_serve_node.sh"
