#!/bin/bash
# cont_s24_steps.sh — one row per training step of an RL arm, appended to a TSV on the login node (kill-check fields only):
# reward_given_done, declared_done_fraction, masked trajectories, agent/verifier timeouts, fully-masked groups, raw reward, pass@8,
# mean tokens. Runs in a tmux next to cont_s24_pipeline.sh so the step readout survives a Mac login expiry.
# Usage: bash cont_s24_steps.sh <arm_log> <out.tsv> <job_id>
set -uo pipefail
L=${1:?arm log}; OUT=${2:?out tsv}; JOB=${3:?job id}; export TZ=America/Los_Angeles OMP_NUM_THREADS=1
[ -f $OUT ] || printf "time_pt\tstep\treward_given_done\tdeclared_done\tmasked_traj\tagent_timeouts\tverifier_timeouts\tfully_masked_groups\tavg_raw_reward\tpass8\tavg_tokens\n" > $OUT
g(){ echo "$2" | grep -o "\"$1\": [-0-9.e]*" | head -1 | awk '{print $2}'; }
while true; do
  last=$(tail -n 1 $OUT | cut -f2); [[ "$last" =~ ^[0-9]+$ ]] || last=0
  grep -a "trainer/global_step" $L 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | while read -r line; do
    s=$(g trainer/global_step "$line"); [[ "$s" =~ ^[0-9]+$ ]] && [ "$s" -gt "$last" ] || continue
    printf "%s\t%s\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.0f\n" "$(date +%H:%M)" "$s" \
      "$(g diag/reward_given_done "$line")" "$(g diag/declared_done_fraction "$line")" "$(g generate/num_masked_trajectories "$line")" \
      "$(g generate/errors/AgentTimeoutError "$line")" "$(g generate/errors/VerifierTimeoutError "$line")" \
      "$(g async/rejected_count/fully_masked "$line")" "$(g reward/avg_raw_reward "$line")" "$(g reward/avg_pass_at_8 "$line")" \
      "$(g generate/avg_num_tokens "$line")" >> $OUT
    last=$s
  done
  out=$(squeue -h -j "$JOB" -o %T 2>&1); rc=$?
  if { [ $rc -eq 0 ] && [ -z "$out" ]; } || echo "$out" | grep -q "Invalid job id"; then
    sacct -j "$JOB" -X -n -o State 2>/dev/null | grep -qE "COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL" \
      && { printf "%s\tjob %s gone: %s\n" "$(date +%H:%M)" "$JOB" "$(sacct -j "$JOB" -X -n -o State,Elapsed | head -1 | tr -s ' ')" >> $OUT; break; }
  fi
  sleep 60
done
