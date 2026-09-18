#!/bin/bash
# launch_screen_tailsft.sh — learnability screen of TailSFT (full-lr5e-5-ep3-tail25) over the same 1,247-task training pool as the
# vanilla SFT screen, to compare unlocked/learnable bands (Luke 2026-09-18). Adapted by diff from launch_screen_kimiswe.sh: only REFRESH_NAME + MODEL change.
# (8 GPU nodes = 4 policy + 4 EAGLE-3 engines, conc 512, 24 coordinators, eval spread, 16 x 32 sparse workers on JUWELS,
# bridge 9930). Canary a0 = 64 tasks x 4 (short walls), then one 1,183-task shard, then stage B top-ups to 8.
set -uo pipefail
export REFRESH_NAME=screen_tailsft_20260918 POOL=train EVAL_SPREAD=1 STAGES=ab   # STAGES=ab: JSC sets a global $STAGES (module dir); refresh_screen reads it, so pin ours
export MODEL=/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-kimi_swesmith-sft/full-lr5e-5-ep3-tail25/export-step96-hf-bf16
export GPU_NODES=8 ENGINES=4 CONC=512 NUM_COORDINATORS=24 CPU_NODES=16 WORKERS_PER_NODE=32
export TASKS_PER_SHARD=600 CANARY_TASKS=64 CANARY_GPU_WALL=03:00:00 CANARY_CPU_WALL=03:30:00 GPU_WALL=05:00:00 CPU_WALL=05:30:00
export OMP_NUM_THREADS=1
C=/e/project1/transfernetx/lee27/code/snowball; E=/e/fscratch/reformo/lee27/experiments; PY=/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python
# the idle s24-continuation bridge holds port 9930; the screen starts its own bridge there (exact-name match, no prefix kills)
tmux has-session -t =cont_s24_bridge 2>/dev/null && { curl -s -m 5 http://127.0.0.1:9930/status | grep -q '"active_jobs": 0' && tmux kill-session -t =cont_s24_bridge && echo "stopped idle cont_s24_bridge"; }
sleep 3
[ -d $E/$REFRESH_NAME ] || $PY $C/refresh_screen.py prepare 2>&1 | tail -3
tmux new-session -d -s $REFRESH_NAME "cd $C && env REFRESH_NAME=$REFRESH_NAME POOL=$POOL EVAL_SPREAD=1 STAGES=ab PRIOR_SHARDS=${PRIOR_SHARDS:-} SCREEN_ACCOUNT=${SCREEN_ACCOUNT:-laionize} MODEL=$MODEL GPU_NODES=$GPU_NODES ENGINES=$ENGINES CONC=$CONC NUM_COORDINATORS=$NUM_COORDINATORS CPU_NODES=$CPU_NODES WORKERS_PER_NODE=$WORKERS_PER_NODE TASKS_PER_SHARD=$TASKS_PER_SHARD CANARY_TASKS=$CANARY_TASKS CANARY_GPU_WALL=$CANARY_GPU_WALL CANARY_CPU_WALL=$CANARY_CPU_WALL GPU_WALL=$GPU_WALL CPU_WALL=$CPU_WALL OMP_NUM_THREADS=1 $PY $C/refresh_screen.py run 2>&1 | tee -a $E/$REFRESH_NAME/controller.log"
sleep 60; tail -5 $E/$REFRESH_NAME/controller.log; squeue -h -u $USER -o "%i %j %T %D %M" | grep -i screen
