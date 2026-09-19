#!/bin/bash
# launch_screen_daytona.sh <REFRESH_NAME> <MODEL export dir> — learnability screen of one checkpoint over the current 1,247-task
# training pool with rollouts in Daytona sandboxes (refresh_screen.py DAYTONA=1): rounds of 2 attempts, a task stops once solved,
# unsolved tasks get up to CAP=4 attempts (Luke 2026-09-19: "2 passes for all, +2 for the surviving unsolved, repeat; 4 for now").
# No apptainer bridge, no JUWELS fleet. Two screens at once share the org's 5 creates/s (SHARES = 2 x coordinators) and stay
# under one user's ~1,000 sandboxes (CONC 448 each). Geometry as the Kimi screens: 8 GPU nodes = 4 policy + 4 engines.
# Usage: bash launch_screen_daytona.sh screen_otastd_20260919 /e/data1/.../export-step630-hf-bf16   (SUBMIT=0 builds round 0 only)
set -uo pipefail
export REFRESH_NAME=$1 MODEL=$2 DAYTONA=1 POOL=train EVAL_SPREAD=1 STAGES=ab
export ROUND_ATTEMPTS=${ROUND_ATTEMPTS:-2} CAP=${CAP:-4}
export GPU_NODES=8 ENGINES=4 CONC=${CONC:-448} NUM_COORDINATORS=16 SHARES=${SHARES:-32} CPU_NODES=0 WORKERS_PER_NODE=0
export TASKS_PER_SHARD=2000 CANARY_TASKS=0 GPU_WALL=${GPU_WALL:-05:00:00} CANARY_GPU_WALL=${GPU_WALL:-05:00:00}
export SUBMIT=${SUBMIT:-1} SCREEN_ACCOUNT=${SCREEN_ACCOUNT:-open-sci-mm} OMP_NUM_THREADS=1
C=/e/project1/transfernetx/lee27/code/snowball; E=/e/fscratch/reformo/lee27/experiments; PY=/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
timeout 5 bash -c "</dev/tcp/10.128.1.2/7011" 2>/dev/null || { echo "microsocks 10.128.1.2:7011 (the compute nodes' path) is not reachable"; exit 1; }
[ -d $E/$REFRESH_NAME ] || $PY $C/refresh_screen.py prepare 2>&1 | tail -3
if [ "$SUBMIT" = 0 ]; then $PY $C/refresh_screen.py run; exit; fi
tmux new-session -d -s $REFRESH_NAME "cd $C && env REFRESH_NAME=$REFRESH_NAME MODEL=$MODEL DAYTONA=1 POOL=$POOL EVAL_SPREAD=1 STAGES=ab ROUND_ATTEMPTS=$ROUND_ATTEMPTS CAP=$CAP GPU_NODES=$GPU_NODES ENGINES=$ENGINES CONC=$CONC NUM_COORDINATORS=$NUM_COORDINATORS SHARES=$SHARES CPU_NODES=0 WORKERS_PER_NODE=0 TASKS_PER_SHARD=$TASKS_PER_SHARD CANARY_TASKS=0 GPU_WALL=$GPU_WALL CANARY_GPU_WALL=$GPU_WALL SCREEN_ACCOUNT=$SCREEN_ACCOUNT OMP_NUM_THREADS=1 $PY $C/refresh_screen.py run >> $E/$REFRESH_NAME/controller.log 2>&1"
sleep 20; tail -3 $E/$REFRESH_NAME/controller.log; squeue -h -u $USER -o "%i %j %T %D %M" | grep -i screen
