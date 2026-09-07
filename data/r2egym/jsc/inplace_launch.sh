#!/bin/bash
# inplace_launch.sh — the whole in-place-clause A/B, from a cold login01 to two submitted probes.
#
# The A/B: two 150-task trees off `tasks/r2egym-tt-v2-idval` that differ by ONE sentence in instruction.md
# (edit_target/tt_prompt `variant=inplace`: grading reads /testbed at episode end, there is no patch file, edit in
# place). Same Stage-3 base, same 64k budget, k=8, same fleet, same bridge, same hour.
#
# PREREQUISITE, and the only thing this script cannot do itself: a JUWELS ControlMaster owned by THIS login node.
# `jupiter` (login02) and `jupiter01` are different machines and an ssh master socket is node-local, so login02's
# ~/.ssh/cm_juwels/bridge gives login01 ECONNREFUSED. Luke, from the Mac, one TOTP:
#   ssh -t jupiter01 'ssh -M -S ~/.ssh/cm_juwels/bridge01 -o ControlPersist=12h -fN juwels && ssh -O check -S ~/.ssh/cm_juwels/bridge01 juwels'
# (a separate socket name on purpose: removing login02's `bridge` socket would kill its master if it is still alive.)
#
# Usage:  bash inplace_launch.sh [fleet|probes|all]     default all
set -uo pipefail
C=/e/project1/transfernetx/lee27/code/snowball
EXPMM=/e/data1/mmlaion/lee27/experiments          # Luke 2026-09-07: mmlaion, not fscratch
O=$EXPMM/inplace_probe
TREES=/e/data1/mmlaion/lee27/tasks
PORT=${PORT:-9928}                                 # 9922/9924/9926/9927 are other work; 9926 is live on login01
SOCK=${SOCK:-~/.ssh/cm_juwels/bridge01}
JWLOGIN=${JWLOGIN:-jwlogin03i}
NODES=${NODES:-64}                                 # 8 workers/node; the two probes ask for 256 seats each
WALL=${WALL:-08:00:00}
STAGE=${1:-all}
mkdir -p $O/logs
LOCALIP=$(getent hosts $(hostname) | awk '{print $1}' | head -1)   # login01 = 10.128.1.1
BRIDGE_LOCAL=http://$LOCALIP:$PORT

say() { echo "$(date '+%F %T') | $*" | tee -a $O/logs/launch.log; }

start_bridge() {
  if curl -s -m 5 http://127.0.0.1:$PORT/status > /dev/null 2>&1; then say "bridge $PORT already up on $(hostname)"; return 0; fi
  tmux new -d -s apptainer_bridge_$PORT "BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 python3 /e/project1/transfernetx/lee27/code/harbor/src/harbor/environments/apptainer/server.py --port $PORT --host 0.0.0.0 >> $O/logs/bridge_$PORT.log 2>&1"
  for i in $(seq 1 20); do sleep 2; curl -s -m 5 http://127.0.0.1:$PORT/status > /dev/null 2>&1 && { say "bridge $PORT up"; return 0; }; done
  say "BRIDGE $PORT DID NOT COME UP -- see $O/logs/bridge_$PORT.log"; return 1
}

fleet() {
  ssh -O check -S $SOCK juwels 2>&1 | grep -q "Master running" || {
    say "NO JUWELS MASTER at $SOCK. Luke must run the one-TOTP command in this script's header."; return 1; }
  SOCK=$SOCK LOGIN=$JWLOGIN FORWARDS="$PORT:$PORT" bash $C/juwels_bridge2_setup.sh 2>&1 | tee -a $O/logs/launch.log
  local W="ssh -o BatchMode=yes -S $SOCK juwels"
  local existing; existing=$($W "squeue -h -u \$USER -n apptainer_workers_juwels_inplace -o %i" | tr '\n' ' ')
  if [ -n "$existing" ]; then say "fleet already submitted: $existing"; else
    $W "cd /p/project1/synthlaion/lee27/fleet && HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src BRIDGE_URL=http://$JWLOGIN:$PORT STAGING_BASE=/tmp/apptainer_staging WORKERS_PER_NODE=8 sbatch --partition=batch --nodes=$NODES --time=$WALL --job-name=apptainer_workers_juwels_inplace juwels_workers.sbatch" 2>&1 | tee -a $O/logs/launch.log
  fi
  say "waiting for workers to register on the bridge (up to 40 min)"
  for i in $(seq 1 80); do
    local st; st=$(curl -s -m 8 http://127.0.0.1:$PORT/status)
    echo "$st" | grep -q '"workers_alive": true' && { say "workers alive: $st"; return 0; }
    sleep 30
  done
  say "WORKERS NEVER REGISTERED -- check the fleet .out on JUWELS"; return 1
}

probes() {
  curl -s -m 8 http://127.0.0.1:$PORT/status | grep -q '"workers_alive": true' || { say "refusing: no workers on bridge $PORT"; return 1; }
  for t in $TREES/r2egym-tt-v2-idval-ctl $TREES/r2egym-tt-v2-idval-inp; do
    [ -d "$t" ] || { say "missing tree $t"; return 1; }
    [ "$(ls $t | wc -l)" = "150" ] || { say "tree $t is not 150 tasks"; return 1; }
  done
  for pair in "snowball_inplace_ctl:r2egym-tt-v2-idval-ctl" "snowball_inplace_inp:r2egym-tt-v2-idval-inp"; do
    local name=${pair%%:*} tree=${pair##*:}
    squeue -h -u $USER -n $name -o %i | grep -q . && { say "$name already in squeue, skipping"; continue; }
    say "launching $name on $tree via $BRIDGE_LOCAL"
    SNOWBALL_EXP=$EXPMM bash $C/probe_ckpt.sh $name $TREES/$tree base $BRIDGE_LOCAL 2>&1 | tee -a $O/logs/launch.log
  done
  # death-watcher: one 60 s squeue poll for both jobs, so a failure shows up in ~a minute
  tmux kill-session -t inplace_deathwatch 2>/dev/null
  tmux new -d -s inplace_deathwatch "bash -c 'while true; do for n in snowball_inplace_ctl snowball_inplace_inp; do s=\$(squeue -h -u \$USER -n \$n -o %T | head -1); echo \"\$(date +%T) \$n \${s:-GONE}\"; done; sleep 60; done' >> $O/logs/deathwatch.log 2>&1"
  say "submitted; watchers: probe_watch_snowball_inplace_{ctl,inp}, inplace_deathwatch"
}

case "$STAGE" in
  fleet)  start_bridge && fleet ;;
  probes) probes ;;
  all)    start_bridge && fleet && probes ;;
  *) echo "usage: $0 [fleet|probes|all]"; exit 2 ;;
esac
