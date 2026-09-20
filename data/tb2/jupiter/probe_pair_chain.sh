#!/bin/bash
# probe_pair_chain.sh <tag> <model export> — the two cheap behavioural probes on ONE GH200 node at the same time:
# the held-out if-v2 tasks (477, apptainer bridge on the serve node) and the held-out Recursive-Task-Synthesis tasks
# (337, Daytona + the setup-files hook). One ifv2_apptainer.sbatch job serves the model and runs the bridge; both harbor
# runs hit its endpoint (32 + 32 agents, the load eval_chain.sh puts on a server). ~2 node-hours per model.
# Readouts: ifv2_rows.json in each run dir (probe_rows.py) -> ifv2_pair.py <arm rows> <base rows>, base = the run of
# THIS chain on the base model (same load; the 1,800 s agent budget is wall-clock, so pairs must share concurrency).
# Log: $E/logs/pair_chain_<tag>.log
set -uo pipefail
TAG=${1:?tag}; MODEL=${2:?model export dir}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
PY=$C/envs/snowball-v2/bin/python; J_ROOT=/e/data1/mmlaion/lee27/experiments/tb2_jobs
RST_TASKS=${RST_TASKS:-/e/data1/mmlaion/lee27/tasks/rst_heldout_daytona_v2}
D=$(date +%Y%m%d_%H%M); NIF=ifv2ho_${TAG}_$D; NRST=rstho_${TAG}_$D
LOG=$E/logs/pair_chain_$TAG.log; mkdir -p $E/logs
say() { echo "[$(date -u +%FT%TZ)] [pair $TAG] $*" | tee -a $LOG; }
[ -f "$MODEL/config.json" ] || { say "no model at $MODEL"; exit 1; }
J=$(MODEL=$MODEL sbatch --parsable $W/ifv2_apptainer.sbatch) || { say "sbatch failed"; exit 1; }
say "serve+bridge job $J model=$MODEL runs $NIF $NRST"
for i in $(seq 1 90); do [ -f $E/endpoints/$J.bridge ] && [ -f $E/endpoints/$J ] && break; squeue -h -j $J >/dev/null 2>&1 || { say "job $J left the queue before the endpoint came up"; exit 1; }; sleep 20; done
[ -f $E/endpoints/$J.bridge ] || { say "no endpoint after 30 min"; scancel $J; exit 1; }
say "endpoint $(cat $E/endpoints/$J) bridge $(cat $E/endpoints/$J.bridge)"
rm -f $E/logs/run_$NIF.log $E/logs/run_$NRST.log
TASKS=/e/data1/mmlaion/lee27/tasks/ifv2_heldout477 NTASKS=477 POLICY_FILE=ifv2_policy.yaml bash $W/run_tb2.sh $J $NIF full 2>&1 | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || { say "run_tb2 failed (if)"; scancel $J; exit 1; }
HARBOR_SRC=$C/harbor-hook/src TASKS=$RST_TASKS NTASKS=$(ls -d $RST_TASKS/*/ | wc -l) POLICY_FILE=rst_policy.yaml bash $W/run_tb2.sh $J $NRST full 2>&1 | tee -a $LOG
[ "${PIPESTATUS[0]}" = 0 ] || { say "run_tb2 failed (rst)"; scancel $J; exit 1; }
sleep 30
for n in $NIF $NRST; do tmux has-session -t tb2_$n 2>/dev/null || grep -q RUN_DONE $E/logs/run_$n.log 2>/dev/null || { say "harbor tmux tb2_$n did not start"; scancel $J; exit 1; }; done
for i in $(seq 1 420); do   # up to 3.5 h
  grep -q RUN_DONE $E/logs/run_$NIF.log 2>/dev/null && grep -q RUN_DONE $E/logs/run_$NRST.log 2>/dev/null && break
  squeue -h -j $J >/dev/null 2>&1 || { say "serve job $J died mid-run"; break; }; sleep 30
done
for n in $NIF $NRST; do
  say "$n: $(grep RUN_DONE $E/logs/run_$n.log 2>/dev/null | tail -1)"
  $PY $W/summarize_tb2.py $J_ROOT/$n 2>&1 | tee -a $LOG | tail -3
  OMP_NUM_THREADS=1 $PY $W/probe_rows.py $J_ROOT/$n 2>&1 | tail -1 | tee -a $LOG
done
scancel $J; say "released node ($J). rows: $J_ROOT/$NIF/ifv2_rows.json $J_ROOT/$NRST/ifv2_rows.json"
echo "PAIR_DONE $NIF $NRST" | tee -a $LOG
