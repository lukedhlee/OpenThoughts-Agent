#!/bin/bash
# ifv2_chain.sh <tag> <model export> [smoke] — the held-out if-v2 probe end to end on ONE GH200 node: serve + bridge
# (ifv2_apptainer.sbatch), wait for the endpoint, run the 477 tasks (run_tb2.sh + ifv2_policy.yaml), summarize,
# release the node. ~1.5 node-hours per model. Log: $E/logs/ifv2_chain_<tag>.log
set -uo pipefail
TAG=${1:?tag}; MODEL=${2:?model export dir}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
# the run name carries the clock: a reused name made the chain read the PREVIOUS attempt's RUN_DONE marker and
# release the node while harbor was still connecting (smokes 2-4, 2026-09-19)
NAME=ifv2ho_${TAG}_$(date +%Y%m%d_%H%M); [ "$MODE" = smoke ] && NAME=${NAME}_smoke
LOG=$E/logs/ifv2_chain_$TAG.log; mkdir -p $E/logs
say() { echo "[$(date -u +%FT%TZ)] [ifv2 $TAG] $*" | tee -a $LOG; }
J=$(MODEL=$MODEL sbatch --parsable $W/ifv2_apptainer.sbatch) || { say "sbatch failed"; exit 1; }
say "serve+bridge job $J model=$MODEL"
for i in $(seq 1 90); do [ -f $E/endpoints/$J.bridge ] && [ -f $E/endpoints/$J ] && break; squeue -h -j $J >/dev/null 2>&1 || { say "job $J left the queue before the endpoint came up"; exit 1; }; sleep 20; done
[ -f $E/endpoints/$J.bridge ] || { say "no endpoint after 30 min"; scancel $J; exit 1; }
say "endpoint $(cat $E/endpoints/$J) bridge $(cat $E/endpoints/$J.bridge)"
RL=$E/logs/run_$NAME.log; rm -f $RL
TASKS=/e/data1/mmlaion/lee27/tasks/ifv2_heldout477 NTASKS=477 POLICY_FILE=ifv2_policy.yaml bash $W/run_tb2.sh $J $NAME $MODE 2>&1 | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || { say "run_tb2 failed"; scancel $J; exit 1; }
sleep 20; tmux has-session -t tb2_$NAME 2>/dev/null || grep -q RUN_DONE $RL 2>/dev/null || { say "harbor tmux tb2_$NAME did not start"; scancel $J; exit 1; }
for i in $(seq 1 360); do grep -q RUN_DONE $RL 2>/dev/null && break; squeue -h -j $J >/dev/null 2>&1 || { say "serve job $J died mid-run"; break; }; sleep 30; done
say "run finished: $(grep RUN_DONE $RL | tail -1)"
python3 $W/summarize_tb2.py /e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME 2>&1 | tee -a $LOG | tail -5
scancel $J; say "released node ($J). results /e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME"
