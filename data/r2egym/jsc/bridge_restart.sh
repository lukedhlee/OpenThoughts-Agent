#!/bin/bash
# bridge_restart.sh — restart the apptainer bridge on jpbl-s01-02 with the larger listen backlog and the 2400 s reaper.
# ONLY when no trial is live (bridge env state is in memory; every live env is orphaned by a restart).
set -u
S=/e/project1/transfernetx/lee27/code/harbor/src/harbor/environments/apptainer/server.py
LOG=/e/fscratch/reformo/lee27/apptainer_bridge/server_9922.log
live=$(squeue -h -u $USER -o %j | grep -c "pool60k\|probe_val\|grpo\|smoke")
[ "$live" -gt 0 ] && [ "${1:-}" != "--force" ] && { echo "refusing: $live trial-running jobs in squeue (use --force)"; squeue -h -u $USER -o "%i %j"; exit 1; }
before=$(curl -s -m 5 localhost:9922/status | python3 -c "import sys,json; d=json.load(sys.stdin); print('ready',d['envs']['ready'],'jobs',d['active_jobs'])" 2>/dev/null)
echo "$(date) before: $before"
P=$(tmux list-panes -t apptainer_bridge_9922 -F "#{pane_pid}" 2>/dev/null); B=$(pgrep -P "$P" -f "server.py --port 9922" 2>/dev/null | head -1)
tmux kill-session -t apptainer_bridge_9922 2>/dev/null; [ -n "$B" ] && kill "$B" 2>/dev/null; sleep 3
pgrep -f "python3 $S --port 9922" >/dev/null && pkill -f "python3 $S --port 9922"; sleep 2
grep -c "request_queue_size" $S >/dev/null || { echo "server.py lacks the backlog patch"; exit 1; }
tmux new -d -s apptainer_bridge_9922 "BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 python3 $S --port 9922 --host 0.0.0.0 >> $LOG 2>&1"
for i in $(seq 1 24); do sleep 5; st=$(curl -s -m 5 localhost:9922/status 2>/dev/null); [ -n "$st" ] && { echo "$(date) up after $((i*5))s: $(echo $st | head -c 200)"; break; }; done
for i in $(seq 1 36); do sleep 5; alive=$(curl -s -m 5 localhost:9922/status | python3 -c "import sys,json; print(json.load(sys.stdin)['workers_alive'])" 2>/dev/null); [ "$alive" = "True" ] && { echo "$(date) workers_alive after $((i*5))s"; exit 0; }; done
echo "$(date) WARNING: workers not alive after 180 s"; exit 2
