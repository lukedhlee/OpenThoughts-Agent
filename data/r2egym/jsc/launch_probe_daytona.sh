#!/bin/bash
# launch_probe_daytona.sh <allowlist> [shards=2] [wall=20:00:00] [conc=256] — on Jupiter: shard the Daytona-shape tt tree
# (/e/fscratch/reformo/lee27/tasks/r2egym-tt-daytona) by the given allowlist, generate the pool60k-recipe probes on the DAYTONA
# backend (make_tt_wave.py --daytona: preset SOCKS via the jpbl-s01-02 microsocks, eval-org key file, setup-files-hook harbor on
# PYTHONPATH), then start pool_launch + pool_watch tmuxes. Refuses if the microsocks (10.128.1.2:7011) is not listening.
set -e
ALLOW=${1:?allowlist file (local)}; N=${2:-2}; WALL=${3:-20:00:00}; CONC=${4:-256}; PREFIX=${PREFIX:-ttd60k}
scp -q "$ALLOW" jupiter:/e/fscratch/reformo/lee27/tasks/allowlist_r2egym_tt_daytona.txt
ssh -o ConnectTimeout=25 jupiter "set -e; cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent
ss -ltn 2>/dev/null | grep -q ':7011 ' || { echo 'microsocks 10.128.1.2:7011 not listening; not launching'; exit 1; }
export OMP_NUM_THREADS=1
python3 /e/project1/transfernetx/lee27/code/snowball/make_tt_wave.py --tasks /e/fscratch/reformo/lee27/tasks/r2egym-tt-daytona \
  --allow /e/fscratch/reformo/lee27/tasks/allowlist_r2egym_tt_daytona.txt --prefix $PREFIX --shards $N --daytona \
  --conc $CONC --verifier-timeout 1200 --wall $WALL
tmux kill-session -t pool_launch_$PREFIX 2>/dev/null || true; tmux kill-session -t pool_watch_$PREFIX 2>/dev/null || true
tmux new -d -s pool_launch_$PREFIX \"bash /e/project1/transfernetx/lee27/code/snowball/pool_launch.sh $PREFIX $N 150\"
tmux new -d -s pool_watch_$PREFIX \"bash /e/project1/transfernetx/lee27/code/snowball/pool_watch.sh $PREFIX $N\"
sleep 20; squeue -u lee27 -o '%.9i %.14j %.2t %.8M %.5D %R' | grep -E \"$PREFIX|JOBID\""
