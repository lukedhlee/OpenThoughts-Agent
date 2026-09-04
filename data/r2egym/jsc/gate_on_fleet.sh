#!/bin/bash
# gate_on_fleet.sh <fleet jobid> [conc=384] — runs on the JURECA login node once OUR fleet is RUNNING (dc-cpu, offline):
# stop the devel sweeps (ours), then gate every non-empty-issue task of the tt list on the fleet nodes with a fresh tag
# `tt_fleet` (envgate_par is idempotent per parts dir, so a re-run resumes).
cd /p/scratch/synthlaion/lee27 || exit 1
FLEET=${1:?fleet jobid}; CONC=${2:-384}
LOG=/p/scratch/synthlaion/lee27/gate_on_fleet.log; exec >>$LOG 2>&1
echo "=== $(date) gate_on_fleet fleet=$FLEET conc=$CONC"
st=$(squeue -h -j $FLEET -o %T); [ "$st" = RUNNING ] || { echo "fleet $FLEET not RUNNING ($st)"; exit 1; }
pkill -u lee27 -f envgate_par_tt.sh; pkill -u lee27 -f continue_gate.sh; pkill -u lee27 -f resume_gate2.sh; sleep 3
for j in $(squeue -h -u lee27 -n tt_gate_devel,tt_gate_devel2,tt_gate_devel3,tt_gate_devel4 -o %i); do scancel $j && echo "cancelled devel sleeper $j"; done
grep -v -x -F -f tt_empty_issue_tasks.txt tt_gate_full.txt > tt_gate_nonempty.txt; echo "tasks: $(wc -l < tt_gate_nonempty.txt)"
FLEET=$FLEET EG_STAGE_ROOT=/p/scratch/synthlaion/lee27/envgate_tt EG_TAG=tt_fleet EG_SCRIPT=envgate_tt.sh EG_TASKLIST=/p/scratch/synthlaion/lee27/tt_gate_nonempty.txt EG_CONC=$CONC EG_SRUN_MIN=45 setsid nohup bash envgate_par_tt.sh >/dev/null 2>&1 </dev/null &
sleep 15; echo "$(date) sweep tt_fleet started: $(tail -1 envgate_par_tt_fleet.log)"
