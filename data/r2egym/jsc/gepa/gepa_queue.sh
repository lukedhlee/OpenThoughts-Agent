#!/bin/bash
# gepa_queue.sh — the Opus session's handle on the GEPA candidate queue.
#
#   bash gepa_queue.sh start                 # start the runner tmux (gepa_runner)
#   bash gepa_queue.sh add <wave> <cand> [k] # enqueue one candidate of a built wave tree
#   bash gepa_queue.sh add-all <wave> [k]    # enqueue every arm of that wave, control included
#   bash gepa_queue.sh list                  # queued / running / done
#   bash gepa_queue.sh drop <wave> <cand>    # remove a QUEUED candidate (never touches a running one)
#   bash gepa_queue.sh drain                 # remove everything still queued
#   bash gepa_queue.sh stop                  # stop the runner (running shards keep going; they are their own tmuxes)
#
# Enqueueing is free of a new cost line: the serve job is already paid for and the queue only decides whether it
# idles. What the session owes is the node-hours figure from `gepa_serve.sh status` in every status report.
# Keep at least two candidates queued at all times, so the servers never wait on a reflection step.
set -uo pipefail
CMD=${1:-list}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
T=${GEPA_TASKS:-/e/fscratch/reformo/lee27/tasks}
export OMP_NUM_THREADS=1
mkdir -p "$E/queue" "$E/running" "$E/done" "$E/runs" "$E/logs"

case "$CMD" in
start)
  tmux has-session -t gepa_runner 2>/dev/null && { echo "runner tmux already up"; exit 0; }
  tmux new -d -s gepa_runner "export OMP_NUM_THREADS=1 GEPA_DIR=$E GEPA_CODE=$G; python3 $G/gepa_runner.py >> $E/logs/runner.log 2>&1"
  sleep 2; tail -5 "$E/logs/runner.log" 2>/dev/null
  echo "runner tmux gepa_runner up; log $E/logs/runner.log"
  ;;
stop)
  tmux kill-session -t gepa_runner 2>/dev/null && echo "runner stopped" || echo "no runner tmux"
  echo "note: shards already launched keep running in their own tmuxes (tmux ls | grep gepa_)"
  ;;
add|add-all)
  WAVE=${2:?wave name}
  if [ "$CMD" = add ]; then CAND=${3:?candidate id}; K=${4:-1}; else CAND=""; K=${3:-1}; fi
  MAN=$E/$WAVE/cands.json
  [ -f "$MAN" ] || { echo "no wave manifest at $MAN -- run gepa_tree.py --wave $WAVE first"; exit 1; }
  python3 - "$MAN" "$E" "$T" "$WAVE" "$CAND" "$K" <<'PY'
import json, os, sys, time
man_p, E, T, wave, cand, k = sys.argv[1:7]
man = json.load(open(man_p))
tree = man.get("dst") or os.path.join(T, "gepa-" + wave)
arms = man["arms"] if not cand else [cand]
bad = [c for c in arms if c not in man["arms"]]
if bad:
    sys.exit("%s is not an arm of wave %s (arms: %s)" % (bad[0], wave, ",".join(man["arms"])))
for c in arms:
    dirs = ["%s-p%s" % (t, c) for t in man["tasks"]]
    missing = [d for d in dirs if not os.path.isdir(os.path.join(tree, d))]
    if missing:
        sys.exit("%d task dirs missing under %s, e.g. %s" % (len(missing), tree, missing[:2]))
    for state in ("running", "done"):
        if os.path.exists("%s/%s/%s.%s.json" % (E, state, wave, c)):
            print("skip %s.%s: already %s" % (wave, c, state)); break
    else:
        p = "%s/queue/%s.%s.json" % (E, wave, c)
        if os.path.exists(p):
            print("skip %s.%s: already queued" % (wave, c)); continue
        json.dump({"wave": wave, "cand": c, "tree": tree, "tasks": dirs, "k": int(k),
                   "split": man.get("split"), "attempt": 1,
                   "enqueued": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(p, "w"), indent=1)
        print("queued %s.%s: %d tasks, k=%s" % (wave, c, len(dirs), k))
PY
  ;;
drop)
  WAVE=${2:?wave}; CAND=${3:?cand}
  P=$E/queue/$WAVE.$CAND.json
  [ -f "$P" ] || { echo "$WAVE.$CAND is not queued (running items are not dropped here)"; exit 1; }
  rm -f "$P"; echo "dropped $WAVE.$CAND"
  ;;
drain)
  n=$(ls -1 "$E/queue"/*.json 2>/dev/null | wc -l); rm -f "$E/queue"/*.json 2>/dev/null
  echo "drained $n queued candidates (running ones untouched)"
  ;;
list)
  for s in queue running done; do
    echo "== $s =="
    for f in "$E/$s"/*.json; do
      [ -e "$f" ] || { echo "  (none)"; break; }
      python3 - "$f" <<'PY'
import json, sys
it = json.load(open(sys.argv[1]))
extra = ""
if it.get("shards"):
    extra = "  shards %d  %s" % (len(it["shards"]), ",".join(s["name"] for s in it["shards"]))
if it.get("status"):
    extra += "  status %s  trials %s" % (it["status"], it.get("trials"))
print("  %-10s %-8s tasks %-5d k=%s  attempt %s%s" % (
    it["wave"], it["cand"], len(it["tasks"]), it.get("k"), it.get("attempt"), extra))
PY
    done
  done
  ;;
*) echo "usage: gepa_queue.sh [start|stop|add <wave> <cand> [k]|add-all <wave> [k]|list|drop|drain]"; exit 2;;
esac
