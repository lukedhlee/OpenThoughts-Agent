#!/bin/bash
# relaunch_wave.sh <bridge_ip> — run ON login02 (where the JUWELS ControlMaster and the OTA checkout live) AFTER
# bridge01_start.sh reported the bridge up on login01. Does, in order, with a gate after each step:
#   1. point the single-shard probe sbatch at the login01 bridge (APPTAINER_BRIDGE_URL=http://<bridge_ip>:9924)
#   2. re-point the JUWELS reverse forward jwlogin03i:9925 -> <bridge_ip>:9924 on the live master ~/.ssh/cm_juwels/bridge
#   3. submit the JUWELS fleet: 12 nodes x 32 workers = 384 seats (256 live), 4 h wall, MAX_CHAIN=1 (a second 4 h job is
#      queued afterany; the watcher cancels BY NAME so the chain is released too); planned <= 2 x 12 x 4 x 48 = 4,608 core-h
#   4. wait for workers_alive on the bridge, then submit the probe via pool_launch.sh + pool_watch.sh (table + scancel
#      after the eval block; .compacted marker keeps the raw trajectories)
#   5. start the death-watcher (60 s squeue poll; releases the fleet by name the minute the probe leaves squeue)
# Constraints honoured: never touches job 1771720, bridge 9922 or its fleets; cancels only jobs named *_p2o / p2o6all_s0.
set -u
export OMP_NUM_THREADS=1
IP=${1:?bridge ip (10.128.x.x of login01)}
C=/e/project1/transfernetx/lee27/code; E=/e/fscratch/reformo/lee27/experiments; P=p2o6all_s0
SOCK=~/.ssh/cm_juwels/bridge; SOCK=$(eval echo $SOCK); W="ssh -o BatchMode=yes -S $SOCK juwels"
SB=$E/$P/sbatch/${P}_rl.sbatch
curl -s -m 5 http://$IP:9924/status | grep -q workers_alive || { echo "no bridge at http://$IP:9924"; exit 1; }
ssh -O check -S $SOCK juwels 2>&1 | grep -q "Master running" || { echo "JUWELS master dead at $SOCK"; exit 1; }
squeue -h -n $P -o %i | grep -q . && { echo "a job named $P is already in squeue"; exit 1; }
# 1. probe sbatch -> login01 bridge
sed -i "s#^export APPTAINER_BRIDGE_URL=http://[0-9.]*:9924\$#export APPTAINER_BRIDGE_URL=http://$IP:9924#" $SB
grep -q "APPTAINER_BRIDGE_URL=http://$IP:9924" $SB || { echo "sbatch bridge url not set"; exit 1; }
echo "probe sbatch -> http://$IP:9924"
# 2. JUWELS forward
JW=$($W "getent hosts jwlogin03i" | awk '{print $1}'); [ -n "$JW" ] || { echo "jwlogin03i did not resolve"; exit 1; }
ssh -O cancel -R "${JW}:9925:localhost:9924" -S $SOCK juwels 2>/dev/null
ssh -O cancel -R "${JW}:9925:${IP}:9924" -S $SOCK juwels 2>/dev/null
ssh -O forward -R "${JW}:9925:${IP}:9924" -S $SOCK juwels || { echo "forward failed"; exit 1; }
$W "curl -s -m 8 http://jwlogin03i:9925/status | grep -q workers_alive" || { echo "forward not answering from JUWELS"; exit 1; }
echo "forward jwlogin03i:9925 -> $IP:9924 ok"
# 3. fleet (4 h, 12 nodes, chain 1)
F=$($W "cd /p/project1/synthlaion/lee27/fleet && sbatch --parsable --nodes=12 --time=04:00:00 --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=9925,MAX_CHAIN=1 -J apptainer_workers_juwels_p2o juwels_workers.sbatch")
echo "FLEET $F  ($(date '+%H:%M CEST'))"; echo "$F $(date -Is)" >> $E/p2o/fleet_ids.txt
ok=0; for i in $(seq 1 60); do sleep 5; curl -s -m 3 http://$IP:9924/status | grep -q '"workers_alive": true' && { ok=1; echo "workers_alive after $((i*5))s"; break; }; done
[ $ok = 1 ] || { echo "fleet did not register in 5 min; cancelling it"; $W "scancel -n apptainer_workers_juwels_p2o"; exit 2; }
$W "head -12 /p/scratch/synthlaion/lee27/dc_agent_eval/logs/apptainer_workers_juwels_$F.out | grep -i 'workers/node\|bridge url\|nodes'"
# 4. probe
touch $E/$P/.compacted
tmux new -d -s pool_launch_p2o6all "bash $C/snowball/pool_launch.sh p2o6all 1 150"; sleep 20
tmux new -d -s pool_watch_p2o6all "bash $C/snowball/pool_watch.sh p2o6all 1"
J=$(squeue -h -n $P -o %i | head -1); echo "PROBE $J"; echo "$J $(date -Is)" >> $E/p2o/probe_ids.txt
# 5. death-watcher, releases the fleet chain by name
cat > $E/p2o/wave_watch.sh <<EOF
#!/bin/bash
E=$E; SOCK=$SOCK; LOG=\$E/p2o/wave_watch.log; seen=0; last=""
while true; do
  J=\$(squeue -h -n $P -o "%i %T %M %R" | head -1)
  if [ -n "\$J" ]; then seen=1; [ "\$J" != "\$last" ] && echo "\$(date '+%m-%d %H:%M') probe \$J" >> \$LOG; last=\$J
  elif [ \$seen = 1 ]; then echo "\$(date '+%m-%d %H:%M') probe gone -> scancel -n apptainer_workers_juwels_p2o" >> \$LOG; ssh -o BatchMode=yes -S \$SOCK juwels "scancel -n apptainer_workers_juwels_p2o; sleep 2; squeue -h -n apptainer_workers_juwels_p2o -o '%i %T'" >> \$LOG 2>&1; echo "\$(date '+%m-%d %H:%M') fleet released" >> \$LOG; exit 0; fi
  sleep 60
done
EOF
chmod +x $E/p2o/wave_watch.sh; tmux new -d -s p2o_watch "bash $E/p2o/wave_watch.sh"
echo "watchers: pool_watch_p2o6all, p2o_watch; log $E/p2o/wave_watch.log"
squeue -h -u $USER -o "%i %j %T %M %D"
