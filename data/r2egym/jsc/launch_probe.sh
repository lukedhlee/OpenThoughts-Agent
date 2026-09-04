#!/bin/bash
# launch_probe.sh <shards> — on Jupiter: shard the allowlisted tt tree, generate the pool60k-recipe probes on bridge 9924
# (verifier 1200 s), then start pool_launch + pool_watch tmuxes. Refuses if bridge 9924 has no live workers.
set -e
N=${1:-6}
ssh -o ConnectTimeout=25 jupiter "set -e; cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent
alive=\$(curl -s -m 5 localhost:9924/status | python3 -c 'import json,sys;print(json.load(sys.stdin)[\"workers_alive\"])')
[ \"\$alive\" = True ] || { echo 'bridge 9924 has no live workers; not launching'; exit 1; }
export OMP_NUM_THREADS=1
python3 /e/project1/transfernetx/lee27/code/snowball/make_tt_wave.py --tasks /e/fscratch/reformo/lee27/tasks/r2egym-tt-raw \
  --allow /e/fscratch/reformo/lee27/tasks/allowlist_r2egym_tt_v1.txt --prefix tt60k --shards $N --bridge-url http://10.128.1.2:9924 \
  --verifier-timeout 1200 --wall 10:00:00
tmux new -d -s pool_launch_tt60k \"bash /e/project1/transfernetx/lee27/code/snowball/pool_launch.sh tt60k $N 150\"
tmux new -d -s pool_watch_tt60k \"bash /e/project1/transfernetx/lee27/code/snowball/pool_watch.sh tt60k $N\"
sleep 5; squeue -u lee27 -n tt60k_s0 -o '%.9i %.14j %.4t %.6M %R'"
