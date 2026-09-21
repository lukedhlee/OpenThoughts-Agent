#!/bin/bash
# gepa_serve.sh — bring up (or report on) the standing GEPA serve job. Prints the cost line; submits only with SUBMIT=1.
#
#   bash gepa_serve.sh                        # status: the job, its endpoints, the queue, node-hours burned + remaining
#   bash gepa_serve.sh up                     # dry run: validate model + draft + code, print the COST line, submit nothing
#   SUBMIT=1 bash gepa_serve.sh up            # submit
#   SUBMIT=1 PILOT=1 NODES=1 HOURS=6 ... up   # the PILOT -- one node, the first live action (SKILL.md 4.0a).
#                                             # PILOT=1 keeps it OUT of the 100 node-hour loop budget.
#   SUBMIT=1 NODES=8 HOURS=13 ... up          # the loop, fast: ~12.5 h of wall buys the whole budget
#   SUBMIT=1 NODES=4 HOURS=26 ... up          # the loop, roomy: same attempts over ~25 h, more slack to reflect in
#   bash gepa_serve.sh down                   # scancel the serve job (the queue is left alone)
#
# BUDGET. The loop gets 100 GPU node-hours after the pilot (Luke, 2026-09-21). Every serve job is recorded in
# $E/budget.jsonl and `status` prints burned and remaining, because a standing allocation is invisible otherwise: it
# costs NODES node-hours for every wall-hour it is up, busy or idle. Once the budget is spent `up` refuses a further
# serve job unless BUDGET_OVERRIDE=1, which prints a fresh cost line for Luke to approve.
#
# NODES is honoured end to end: `sbatch -N` overrides the template's #SBATCH --nodes, the sbatch sruns exactly
# SLURM_NNODES server tasks, and gepa_runner.py sizes shards from the LIVE ENDPOINT COUNT rather than a constant.
# So 1, 4 and 8 nodes are the same code with a different NODES.
#
# The serve job's cost line is stated ONCE, at session start. After that every enqueue is free of a new go -- the
# allocation is already paid for and the queue only decides whether it idles. What the session owes instead is the
# burned/remaining figure in every status line.
set -uo pipefail
CMD=${1:-status}
NODES=${NODES:-8}; HOURS=${HOURS:-12}; IDLE_MIN=${IDLE_MIN:-20}; ACCOUNT=${ACCOUNT:-laionize}
BUDGET=${BUDGET:-100}; PILOT=${PILOT:-0}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
export OMP_NUM_THREADS=1
mkdir -p "$E/queue" "$E/running" "$E/done" "$E/endpoints" "$E/logs"
LEDGER=$E/budget.jsonl

jobid() { squeue -h -u "$USER" -n gepa_serve -o '%i %T' 2>/dev/null | awk '$2=="RUNNING"||$2=="PENDING"{print $1; exit}'; }

# Burned node-hours over every non-pilot serve job this loop has had, from sacct (which knows finished jobs too).
burned() {
  [ -f "$LEDGER" ] || { echo 0; return; }
  python3 - "$LEDGER" <<'PY'
import json, subprocess, sys
tot = 0.0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("pilot"):
        continue          # the pilot is proving the chain works, not spending the loop's budget
    jid = str(r.get("jid", ""))
    el = subprocess.run(["sacct", "-j", jid, "-X", "-n", "--format=Elapsed"],
                        capture_output=True, text=True).stdout.strip().splitlines()
    if not el:
        continue
    p = el[0].strip().split("-")
    d = int(p[0]) if len(p) > 1 else 0
    t = p[-1].split(":")
    try:
        s = int(t[-1]) + 60 * int(t[-2]) + 3600 * (int(t[-3]) if len(t) > 2 else 0) + 86400 * d
    except ValueError:
        continue
    tot += r.get("nodes", 0) * s / 3600.0
print("%.1f" % tot)
PY
}

if [ "$CMD" = status ]; then
  J=$(jobid)
  if [ -z "$J" ]; then
    echo "serve job: none running or pending"
  else
    squeue -h -j "$J" -o 'serve job %i  %T  elapsed %M  nodes %D  %R'
    echo "endpoints up: $(ls -1 "$E"/endpoints/$J.* 2>/dev/null | wc -l) of $(squeue -h -j "$J" -o %D)"
  fi
  B=$(burned)
  python3 -c "
b=$B; T=$BUDGET
print('BUDGET: %.1f of %d node-hours burned, %.1f remaining (the pilot does not count)' % (b, T, T-b))
if b >= T: print('  budget SPENT -- gepa_serve.sh up will refuse a new serve job without BUDGET_OVERRIDE=1')
"
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

B=$(burned)
REMAIN=$(python3 -c "print('%.1f' % ($BUDGET - $B))")
SPENT=$(python3 -c "print(1 if $B >= $BUDGET else 0)")
if [ "$PILOT" != 1 ] && [ "$SPENT" = 1 ] && [ "${BUDGET_OVERRIDE:-0}" != 1 ]; then
  echo "BUDGET SPENT: $B of $BUDGET node-hours already burned on serve jobs."
  echo "A further serve job needs a new decision, not a re-run. If Luke approves more:"
  echo "  BUDGET_OVERRIDE=1 SUBMIT=1 NODES=$NODES HOURS=$HOURS bash $G/gepa_serve.sh up"
  exit 1
fi

echo
if [ "$PILOT" = 1 ]; then
  echo "COST: PILOT serve job = $NODES nodes x up to $HOURS h wall = up to $((NODES * HOURS)) GPU node-hours."
  echo "      It does NOT come out of the loop's $BUDGET node-hour budget; it proves serve -> harbor run -> score ->"
  echo "      ledger and the reap path before any of the budget is committed. On one node, node-hours = wall-hours."
else
  echo "COST: standing GEPA serve job = $NODES nodes x up to $HOURS h wall = up to $((NODES * HOURS)) GPU node-hours,"
  echo "      against a loop budget of $BUDGET (burned so far $B, remaining $REMAIN)."
  python3 -c "
n=$NODES; T=$BUDGET
print('      At %d nodes the whole budget is %.1f h of wall time: the seed wave, about three reflect/gate/accept' % (n, T/n))
print('      waves, and the k=2 confirmation. 8 nodes finishes in ~%.1f h; 4 nodes takes ~%.1f h for the same' % (T/8.0, T/4.0))
print('      attempts, with more slack to reflect in while candidates run.')
"
  [ "${BUDGET_OVERRIDE:-0}" = 1 ] && echo "      BUDGET_OVERRIDE=1: this job is ON TOP of the $BUDGET already agreed. It needs its own go."
fi
echo "      The job self-cancels after $IDLE_MIN idle minutes, so an abandoned loop stops charging."
echo "model:  $MODEL"
echo "draft:  $DRAFT (EAGLE-3, 3 speculative tokens)"
echo "sbatch: $G/gepa_serve.sbatch"
if [ "${SUBMIT:-0}" != 1 ]; then
  echo
  echo "DRY RUN -- nothing submitted. Show the cost line to Luke, get the go, then:"
  echo "  SUBMIT=1 ${PILOT:+PILOT=$PILOT }NODES=$NODES HOURS=$HOURS bash $G/gepa_serve.sh up"
  exit 0
fi
JID=$(MODEL=$MODEL DRAFT=$DRAFT IDLE_MIN=$IDLE_MIN GEPA_CODE=$G GEPA_DIR=$E \
      sbatch --parsable -A "$ACCOUNT" -N "$NODES" --time="${HOURS}:00:00" "$G/gepa_serve.sbatch")
echo "$JID" > "$E/serve_job"
python3 -c "
import json, time
json.dump({'jid': '$JID', 'nodes': $NODES, 'hours': $HOURS, 'pilot': bool($PILOT),
           'submitted': time.strftime('%Y-%m-%dT%H:%M:%S')}, open('$LEDGER', 'a'))
open('$LEDGER', 'a').write('\n')
"
echo "SUBMITTED serve job $JID ($NODES nodes, $HOURS h wall$([ "$PILOT" = 1 ] && echo ', PILOT -- off budget'))"
echo "next: bash $G/gepa_queue.sh start     # the runner tmux, then enqueue candidates"
