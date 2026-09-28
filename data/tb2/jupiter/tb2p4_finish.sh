#!/bin/bash
# tb2p4_finish.sh <serve-jobid> <run-name>: the tail of tb2p4_chain.sh without its per-server recovery pass. Waits for the
# harbor run to end, writes $E/final_<name>_pre.json and $E/final_<name>.txt, touches $E/tb2p4_rundone_<name>, releases the
# serve node. Infrastructure losses of several runs are rerun together afterwards on one server (tb2p4_recover.sh), which
# costs one node for one recovery wave instead of one per run.
# Run in tmux pl2_<name> (never tb2_<name>). Log: $E/pipeline_<name>.log
set -o pipefail; JOB=${1:?serve job id}; NAME=${2:?run name}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
J=/e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME; PY=$C/envs/snowball-v2/bin/python
export OMP_NUM_THREADS=1
log() { echo "$(date -Is) $*" | tee -a $E/pipeline_$NAME.log; }
log "finish watcher: waiting for RUN_DONE (no per-server recovery; pooled recovery later)"
GONE=0
while ! grep -q RUN_DONE $E/logs/run_$NAME.log 2>/dev/null; do
  if [ $GONE = 0 ] && ! squeue -h -j $JOB -o %T | grep -q RUNNING; then GONE=$(date +%s); log "serve job $JOB ended before the run finished (wall); in-flight trials go unscored and are rerun by the pooled recovery"; fi
  if [ $GONE != 0 ] && [ $(( $(date +%s) - GONE )) -gt 1500 ]; then   # harbor's bounded API retries should have ended the run by now
    log "run still open 25 min after the server ended: stopping its harbor session tb2_$NAME"; tmux kill-session -t tb2_$NAME; break
  fi
  sleep 60
done
log "run finished: $(grep RUN_DONE $E/logs/run_$NAME.log | tail -1)"
$PY $W/summarize_tb2.py $J --json > $E/final_${NAME}_pre.json
$PY $W/summarize_tb2.py $J | tee $E/final_$NAME.txt | tee -a $E/pipeline_$NAME.log
touch $E/tb2p4_rundone_$NAME
log "releasing serve job $JOB"; scancel $JOB
