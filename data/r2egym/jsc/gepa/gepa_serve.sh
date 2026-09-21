#!/bin/bash
# gepa_serve.sh — bring up (or report on) the standing GEPA serve job. Prints the cost line; submits only with SUBMIT=1.
#
#   bash gepa_serve.sh                        # status: the job, its endpoints, the queue, node-hours burned so far
#   bash gepa_serve.sh up                     # dry run: validate model + draft + code, print the COST line, submit nothing
#   SUBMIT=1 bash gepa_serve.sh up            # submit
#   SUBMIT=1 NODES=1 HOURS=6 ... up           # the PILOT -- one node, the first live action (SKILL.md 4.0a)
#   bash gepa_serve.sh down                   # scancel the serve job (the queue is left alone)
#
# NODES is honoured end to end: `sbatch -N` overrides the template's #SBATCH --nodes, the sbatch sruns exactly
# SLURM_NNODES server tasks, and gepa_runner.py sizes shards from the LIVE ENDPOINT COUNT rather than a constant.
# So the 1-node pilot and the 8-node loop are the same code with a different NODES.
#
# The serve job's cost line is stated ONCE, at session start: nodes x expected hours. After that every enqueue is
# free of a new go -- the allocation is already paid for and the queue only decides whether it idles. What the
# session owes instead is a node-hours-burned figure in every status line, which `gepa_serve.sh` prints.
set -uo pipefail
CMD=${1:-status}
NODES=${NODES:-8}; HOURS=${HOURS:-12}; IDLE_MIN=${IDLE_MIN:-20}; ACCOUNT=${ACCOUNT:-laionize}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
export OMP_NUM_THREADS=1
mkdir -p "$E/queue" "$E/running" "$E/done" "$E/endpoints" "$E/logs"

jobid() { squeue -h -u "$USER" -n gepa_serve -o '%i %T' 2>/dev/null | awk '$2=="RUNNING"||$2=="PENDING"{print $1; exit}'; }

if [ "$CMD" = status ]; then
  J=$(jobid)
  if [ -z "$J" ]; then
    echo "serve job: none running or pending"
  else
    squeue -h -j "$J" -o 'serve job %i  %T  elapsed %M  nodes %D  %R'
    EL=$(squeue -h -j "$J" -o %M)
    echo "endpoints up: $(ls -1 "$E"/endpoints/$J.* 2>/dev/null | wc -l) of $(squeue -h -j "$J" -o %D)"
    ND=$(squeue -h -j "$J" -o %D)
    python3 - "$EL" "$ND" <<'PY'
import sys
el, nd = sys.argv[1], int(sys.argv[2])
p = el.split("-"); d = int(p[0]) if len(p) > 1 else 0; t = p[-1].split(":")
s = int(t[-1]) + 60 * int(t[-2]) + 3600 * (int(t[-3]) if len(t) > 2 else 0) + 86400 * d
print("node-hours burned by this serve job so far: %.1f (%d nodes x %.2f h)" % (nd * s / 3600.0, nd, s / 3600.0))
PY
  fi
  echo "queue: $(ls -1 "$E/queue" 2>/dev/null | grep -c .) waiting, $(ls -1 "$E/running" 2>/dev/null | grep -c .) running, $(ls -1 "$E/done" 2>/dev/null | grep -c .) done"
  ls -1 "$E/queue" 2>/dev/null | sed 's/^/  queued: /'
  ls -1 "$E/running" 2>/dev/null | sed 's/^/  running: /'
  exit 0
fi

if [ "$CMD" = down ]; then
  J=$(jobid); [ -z "$J" ] && { echo "no serve job to cancel"; exit 0; }
  scancel "$J"; echo "cancelled serve job $J (queue left in place)"
  exit 0
fi

if [ "$CMD" != up ]; then echo "usage: gepa_serve.sh [status|up|down]"; exit 2; fi

J=$(jobid); [ -n "$J" ] && { echo "serve job $J is already up; nothing to do"; exit 0; }
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$DRAFT/model.safetensors" ] || { echo "no EAGLE-3 draft at $DRAFT"; exit 1; }
[ -f "$G/gepa_serve_node.sh" ] || { echo "missing $G/gepa_serve_node.sh"; exit 1; }
[ -f "$G/gepa_serve.sbatch" ] || { echo "missing $G/gepa_serve.sbatch"; exit 1; }
bash -n "$G/gepa_serve_node.sh" && bash -n "$G/gepa_serve.sbatch" || { echo "serve scripts do not parse"; exit 1; }

echo
echo "COST: standing GEPA serve job = $NODES nodes x up to $HOURS h wall = up to $((NODES * HOURS)) GPU node-hours,"
echo "      self-cancelling after $IDLE_MIN idle minutes; we want one warm server pool so candidates run back to back"
echo "      while the session reflects, instead of paying ~4 node-hours of engine load per candidate."
echo "      Actual spend is whatever the loop uses before the queue empties -- 'gepa_serve.sh status' prints it."
if [ "$NODES" = 1 ]; then
  echo "      PILOT layout (SKILL.md 4.0a): one node proves serve -> harbor run -> score -> ledger and the reap path"
  echo "      before 8 nodes are held. On one node, node-hours equal wall-hours."
fi
echo "model:  $MODEL"
echo "draft:  $DRAFT (EAGLE-3, 3 speculative tokens)"
echo "sbatch: $G/gepa_serve.sbatch"
if [ "${SUBMIT:-0}" != 1 ]; then
  echo
  echo "DRY RUN -- nothing submitted. Show the cost line to Luke, get the go, then:"
  echo "  SUBMIT=1 NODES=$NODES HOURS=$HOURS bash $G/gepa_serve.sh up"
  exit 0
fi
JID=$(MODEL=$MODEL DRAFT=$DRAFT IDLE_MIN=$IDLE_MIN GEPA_CODE=$G GEPA_DIR=$E \
      sbatch --parsable -A "$ACCOUNT" -N "$NODES" --time="${HOURS}:00:00" "$G/gepa_serve.sbatch")
echo "$JID" > "$E/serve_job"
echo "SUBMITTED serve job $JID ($NODES nodes)"
echo "next: bash $G/gepa_queue.sh start     # the runner tmux, then enqueue candidates"
