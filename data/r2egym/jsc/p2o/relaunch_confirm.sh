#!/bin/bash
# relaunch_confirm.sh — run ON login02. Bridge 9924 on login02, JUWELS forward, 6-node fleet (no chain), probe p2oAc_s0,
# pool_watch, id-based death-watcher. Every id goes to $E/p2o/confirm_ids.txt. Cancels only what it submits.
set -u
export OMP_NUM_THREADS=1
C=/e/project1/transfernetx/lee27/code; E=/e/fscratch/reformo/lee27/experiments; P=p2oAc_s0; IP=10.128.1.2
SOCK=$(eval echo ~/.ssh/cm_juwels/bridge); W="ssh -o BatchMode=yes -S $SOCK juwels"
IDS=$E/p2o/confirm_ids.txt; LOG=$E/p2o/confirm_watch.log
# 0. preconditions
squeue -h -u $USER -o "%i %j %T" | grep -vE " p2o(_|Ac)" | grep -q . && { echo "lee27 has non-P2O jobs in squeue; refusing:"; squeue -h -u $USER -o "%i %j %T"; exit 1; }
ssh -O check -S $SOCK juwels 2>&1 | grep -q "Master running" || { echo "JUWELS master dead"; exit 1; }
[ "$(hostname -I | tr ' ' '\n' | grep -c "^$IP$")" = 1 ] || { echo "not on login02 ($IP)"; exit 1; }
# 1. bridge on this login node
S=/e/project1/transfernetx/lee27/code/harbor/src/harbor/environments/apptainer/server.py
BL=/e/fscratch/reformo/lee27/apptainer_bridge/server_9924_$(hostname -s).log
if ! curl -s -m 3 localhost:9924/status >/dev/null 2>&1; then
  tmux new -d -s apptainer_bridge_9924 "BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 python3 $S --port 9924 --host 0.0.0.0 >> $BL 2>&1"
  for i in $(seq 1 24); do sleep 5; curl -s -m 3 localhost:9924/status >/dev/null 2>&1 && break; done
fi
curl -s -m 5 http://$IP:9924/status | grep -q workers_alive || { echo "bridge 9924 not answering on $IP"; exit 1; }
echo "bridge 9924 up on $(hostname -s) ($IP)"
# 2. JUWELS forward jwlogin03i:9925 -> login02:9924
JW=$($W "getent hosts jwlogin03i" | awk '{print $1}'); [ -n "$JW" ] || { echo "jwlogin03i did not resolve"; exit 1; }
for old in "localhost:9924" "10.128.1.1:9924" "$IP:9924"; do ssh -O cancel -R "${JW}:9925:${old}" -S $SOCK juwels 2>/dev/null; done
ssh -O forward -R "${JW}:9925:${IP}:9924" -S $SOCK juwels || { echo "forward failed"; exit 1; }
$W "curl -s -m 8 http://jwlogin03i:9925/status | grep -q workers_alive" || { echo "forward not answering from JUWELS"; exit 1; }
echo "forward jwlogin03i:9925 -> $IP:9924 ok"
# 3. fleet: 6 nodes x 32 workers, 3:30, no chain
F=$($W "cd /p/project1/synthlaion/lee27/fleet && sbatch --parsable --nodes=6 --time=03:30:00 --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=9925,MAX_CHAIN=0 -J apptainer_workers_juwels_p2oAc juwels_workers.sbatch")
[ -n "$F" ] || { echo "fleet sbatch failed"; exit 1; }
echo "fleet $F $(date -Is)" >> $IDS; echo "FLEET $F"
ok=0; for i in $(seq 1 72); do sleep 5; curl -s -m 3 http://$IP:9924/status | grep -q '"workers_alive": true' && { ok=1; echo "workers_alive after $((i*5))s"; break; }; done
[ $ok = 1 ] || { echo "fleet did not register in 6 min; cancelling $F"; $W "scancel $F"; exit 2; }
$W "grep -m3 -i 'Workers/node\|Bridge URL\|Chain:' /p/scratch/synthlaion/lee27/dc_agent_eval/logs/apptainer_workers_juwels_$F.out"
# 4. probe
tmux kill-session -t pool_launch_$P 2>/dev/null; tmux kill-session -t pool_watch_p2oAc 2>/dev/null
tmux new -d -s pool_launch_p2oAc "bash $C/snowball/pool_launch.sh p2oAc 1 150"; sleep 25
J=$(squeue -h -n $P -o %i | head -1); [ -n "$J" ] || { echo "probe not in squeue; cancelling fleet $F"; $W "scancel $F"; exit 3; }
echo "probe $J $(date -Is)" >> $IDS; echo "PROBE $J"
tmux new -d -s pool_watch_p2oAc "bash $C/snowball/pool_watch.sh p2oAc 1"
# 5. death-watcher by id: releases fleet (+ any chain ids in the ids file) the minute the probe leaves squeue, then the bridge 10 min later
cat > $E/p2o/confirm_watch.sh <<EOW
#!/bin/bash
E=$E; SOCK=$SOCK; LOG=$LOG; IDS=$IDS; PJ=$J; seen=0; last=""
while true; do
  S=\$(squeue -h -j \$PJ -o "%T %M" 2>/dev/null)
  if [ -n "\$S" ]; then seen=1; [ "\$S" != "\$last" ] && echo "\$(date '+%m-%d %H:%M') probe \$PJ \$S" >> \$LOG; last=\$S
  else
    FL=\$(grep -E "^(fleet|chain) " \$IDS | awk '{print \$2}' | tr '\n' ' ')
    echo "\$(date '+%m-%d %H:%M') probe \$PJ gone -> scancel \$FL" >> \$LOG
    ssh -o BatchMode=yes -S \$SOCK juwels "scancel \$FL; sleep 2; squeue -h -u \\\$USER -o '%i %j %T'" >> \$LOG 2>&1
    echo "\$(date '+%m-%d %H:%M') fleet released; bridge stops in 10 min" >> \$LOG
    sleep 600; tmux kill-session -t apptainer_bridge_9924 2>/dev/null; pkill -u \$USER -f "server.py --port 9924" 2>/dev/null
    echo "\$(date '+%m-%d %H:%M') bridge 9924 stopped on \$(hostname -s)" >> \$LOG; exit 0
  fi
  sleep 60
done
EOW
chmod +x $E/p2o/confirm_watch.sh; tmux kill-session -t p2oAc_watch 2>/dev/null; tmux new -d -s p2oAc_watch "bash $E/p2o/confirm_watch.sh"
echo "watchers: pool_watch_p2oAc, p2oAc_watch; ids $IDS; log $LOG"
squeue -h -u $USER -o "%i %j %T %M %D"
