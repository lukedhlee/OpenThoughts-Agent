#!/bin/bash
# bridge01_start.sh — run ON login01 (ssh jupiter01 after Luke opens that ControlMaster with one TOTP: `ssh -fN jupiter01`).
# Starts the wave's dedicated apptainer bridge on port 9924 on THIS login node (not login02, where the running arm's
# bridge 9922 lives and shares the 4,096-pid cgroup), with the close-after-response flag, and prints the 10.128.x IP the
# compute nodes and the JUWELS forward must use. Light: one python3 process, OMP_NUM_THREADS=1.
set -u
export OMP_NUM_THREADS=1
S=/e/project1/transfernetx/lee27/code/harbor/src/harbor/environments/apptainer/server.py
LOG=/e/fscratch/reformo/lee27/apptainer_bridge/server_9924_$(hostname -s).log
mkdir -p "$(dirname "$LOG")"
if curl -s -m 3 localhost:9924/status >/dev/null 2>&1; then
  echo "bridge 9924 already up on $(hostname -s)"
else
  tmux new -d -s apptainer_bridge_9924 "BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 python3 $S --port 9924 --host 0.0.0.0 >> $LOG 2>&1"
  for i in $(seq 1 24); do sleep 5; curl -s -m 3 localhost:9924/status >/dev/null 2>&1 && { echo "bridge 9924 up on $(hostname -s) after $((i*5))s"; break; }; done
fi
IP=$(hostname -I | tr ' ' '\n' | grep '^10\.128\.' | head -1)
echo "BRIDGE_IP=$IP"
echo "next, on login02: bash /e/fscratch/reformo/lee27/experiments/p2o/relaunch_wave.sh $IP"
