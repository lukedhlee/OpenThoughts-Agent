#!/bin/bash
# distill_launch.sh — run ON login02. Bridge 9924 (harbor-marin server: send-outside-lock + close-after-response), JUWELS
# forward jwlogin03i:9925 -> login02:9924, a 17-node fleet (544 seats, no chain, 7 h), then the P2O distillation arm pinned
# to that bridge, then distill_watch.sh. Every id goes to $E/p2o/distill_ids.txt; the watcher cancels only what this
# script submitted. Env: DST (run name), FLEET_NODES, FLEET_WALL, ARM_WALL.
set -u
export OMP_NUM_THREADS=1
C=/e/project1/transfernetx/lee27/code; E=/e/fscratch/reformo/lee27/experiments; OTA=$C/OpenThoughts-Agent
DST=${DST:-snowball_ttband_p2o_distill_a}; IP=10.128.1.2
FLEET_NODES=${FLEET_NODES:-17}; FLEET_WALL=${FLEET_WALL:-07:00:00}; ARM_WALL=${ARM_WALL:-07:00:00}
SOCK=$(eval echo ~/.ssh/cm_juwels/bridge); W="ssh -o BatchMode=yes -S $SOCK juwels"
IDS=$E/p2o/distill_ids.txt; LOG=$E/p2o/distill_watch.log
# 0. preconditions
squeue -h -u $USER -o "%i %j %T" | grep -q . && { echo "lee27 has jobs in squeue; refusing:"; squeue -h -u $USER -o "%i %j %T"; exit 1; }
ssh -O check -S $SOCK juwels 2>&1 | grep -q "Master running" || { echo "JUWELS master dead"; exit 1; }
[ "$(hostname -I | tr ' ' '\n' | grep -c "^$IP$")" = 1 ] || { echo "not on login02 ($IP)"; exit 1; }
[ -e $E/$DST ] && { echo "$E/$DST exists; pick another DST"; exit 1; }
# 1. bridge on this login node
S=$C/harbor-marin/src/harbor/environments/apptainer/server.py
grep -q BRIDGE_CLOSE_AFTER_RESPONSE $S || { echo "bridge server lacks the close-after-response mode: $S"; exit 1; }
BL=/e/data1/mmlaion/lee27/apptainer_bridge/server_9924_$(hostname -s).log; mkdir -p $(dirname $BL)
if curl -s -m 3 localhost:9924/status >/dev/null 2>&1; then echo "reusing the bridge already answering on 9924"; else
tmux new -d -s apptainer_bridge_9924 "BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 BRIDGE_MAX_HANDLER_THREADS=1500 python3 $S --port 9924 --host 0.0.0.0 >> $BL 2>&1"
for i in $(seq 1 24); do sleep 5; curl -s -m 3 localhost:9924/status >/dev/null 2>&1 && break; done
fi
curl -s -m 5 http://$IP:9924/status | grep -q workers_alive || { echo "bridge 9924 not answering on $IP"; exit 1; }
echo "bridge 9924 up on $(hostname -s) ($IP) from $S"
# 2. JUWELS forward jwlogin03i:9925 -> login02:9924
JW=$($W "getent hosts jwlogin03i" | awk '{print $1}'); [ -n "$JW" ] || { echo "jwlogin03i did not resolve"; exit 1; }
for old in "localhost:9924" "10.128.1.1:9924" "$IP:9924"; do ssh -O cancel -R "${JW}:9925:${old}" -S $SOCK juwels 2>/dev/null; done
ssh -O forward -R "${JW}:9925:${IP}:9924" -S $SOCK juwels || { echo "forward failed"; exit 1; }
$W "curl -s -m 8 http://jwlogin03i:9925/status | grep -q workers_alive" || { echo "forward not answering from JUWELS"; exit 1; }
echo "forward jwlogin03i:9925 -> $IP:9924 ok"
# 3. fleet
F=$($W "cd /p/project1/synthlaion/lee27/fleet && sbatch --parsable --nodes=$FLEET_NODES --time=$FLEET_WALL --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=9925,MAX_CHAIN=0 -J apptainer_workers_juwels_p2od juwels_workers.sbatch")
[ -n "$F" ] || { echo "fleet sbatch failed"; exit 1; }
echo "fleet $F $(date -Is)" >> $IDS; echo "FLEET $F ($FLEET_NODES nodes x 32 seats, $FLEET_WALL, no chain)"
ok=0; for i in $(seq 1 120); do sleep 5; curl -s -m 3 http://$IP:9924/status | grep -q '"workers_alive": true' && { ok=1; echo "workers_alive after $((i*5))s"; break; }; done
[ $ok = 1 ] || { echo "fleet did not register in 10 min; cancelling $F"; $W "scancel $F"; exit 2; }
# 4. arm (built, then pinned to this bridge and capped at ARM_WALL, then submitted)
bash $E/p2o/build_distill_arm.sh $DST 0 > $E/p2o/build_${DST}.log 2>&1 || { echo "arm build FAILED, see $E/p2o/build_${DST}.log"; $W "scancel $F"; exit 3; }
grep -E "^extra:|prompted tree ok|trials_dir on tmpfs|model_path" $E/p2o/build_${DST}.log
SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed -i "/^export DCFT_RL_ENV=/a export APPTAINER_BRIDGE_URL=http://$IP:9924" $SB
grep -q "^export APPTAINER_BRIDGE_URL=http://$IP:9924$" $SB || { echo "bridge pin failed"; $W "scancel $F"; exit 3; }
sed -i "s/^#SBATCH --time=12:00:00$/#SBATCH --time=$ARM_WALL/" $SB
grep -q "^#SBATCH --time=$ARM_WALL$" $SB || { echo "walltime sed failed"; $W "scancel $F"; exit 3; }
J=$(cd $OTA && DCFT=$PWD sbatch --parsable $SB); [ -n "$J" ] || { echo "arm sbatch failed"; $W "scancel $F"; exit 3; }
echo "arm $J $(date -Is)" >> $IDS; echo "ARM $J ($DST, $ARM_WALL)"
# 5. watcher
tmux kill-session -t p2od_watch 2>/dev/null; tmux new -d -s p2od_watch "bash $E/p2o/distill_watch.sh $J $F $DST"
echo "watcher tmux p2od_watch; ids $IDS; log $LOG"
squeue -h -u $USER -o "%i %j %T %M %D"
