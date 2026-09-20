#!/bin/bash
# rst_chain.sh <tag> <model export> [smoke] — the held-out Recursive-Task-Synthesis probe end to end: serve the export on
# ONE GH200 node (serve_snowball.sbatch, POLICY=trained as every probe), wait for the endpoint, run the held-out tasks on
# Daytona (run_tb2.sh + rst_policy.yaml, harbor = code/harbor-hook so setup_files/setup.sh replays each task's Dockerfile
# at trial start), summarize, release the node. ~1.5 node-hours per model. Log: $E/logs/rst_chain_<tag>.log
#
# The task tree is /e/data1/mmlaion/lee27/tasks/rst_heldout_daytona (data/rst/rst_daytona_tree.py): one shared image per
# base (7 harbor snapshots, auto_snapshot), the real environment built by the setup hook as root with internet.
set -uo pipefail
TAG=${1:?tag}; MODEL=${2:?model export dir}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
TASKS=${TASKS:-/e/data1/mmlaion/lee27/tasks/rst_heldout_daytona_v2}   # v2 = teacher-parity sandboxes (2 CPU / 4 GB, 1,800 s agent), centos vault fix
NTASKS=${NTASKS:-$(ls -d $TASKS/*/ | wc -l)}
NAME=rstho_${TAG}_$(date +%Y%m%d_%H%M); [ "$MODE" = smoke ] && NAME=${NAME}_smoke
LOG=$E/logs/rst_chain_$TAG.log; mkdir -p $E/logs
say() { echo "[$(date -u +%FT%TZ)] [rst $TAG] $*" | tee -a $LOG; }
J=$(MODEL=$MODEL POLICY=trained sbatch --parsable $W/serve_snowball.sbatch) || { say "sbatch failed"; exit 1; }
say "serve job $J model=$MODEL tasks=$TASKS ($NTASKS)"
for i in $(seq 1 90); do [ -f $E/endpoints/$J ] && break; squeue -h -j $J >/dev/null 2>&1 || { say "job $J left the queue before the endpoint came up"; exit 1; }; sleep 20; done
[ -f $E/endpoints/$J ] || { say "no endpoint after 30 min"; scancel $J; exit 1; }
say "endpoint $(cat $E/endpoints/$J)"
RL=$E/logs/run_$NAME.log; rm -f $RL
HARBOR_SRC=$C/harbor-hook/src TASKS=$TASKS NTASKS=$NTASKS POLICY_FILE=rst_policy.yaml bash $W/run_tb2.sh $J $NAME $MODE 2>&1 | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || { say "run_tb2 failed"; scancel $J; exit 1; }
sleep 20; tmux has-session -t tb2_$NAME 2>/dev/null || grep -q RUN_DONE $RL 2>/dev/null || { say "harbor tmux tb2_$NAME did not start"; scancel $J; exit 1; }
for i in $(seq 1 480); do grep -q RUN_DONE $RL 2>/dev/null && break; squeue -h -j $J >/dev/null 2>&1 || { say "serve job $J died mid-run"; break; }; sleep 30; done
say "run finished: $(grep RUN_DONE $RL | tail -1)"
python3 $W/summarize_tb2.py /e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME 2>&1 | tee -a $LOG | tail -5
scancel $J; say "released node ($J). results /e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME"
