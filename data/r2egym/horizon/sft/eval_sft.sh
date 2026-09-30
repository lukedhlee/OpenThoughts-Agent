#!/bin/bash
# eval_sft.sh — the relay SFT arms' agentic evals for one Horizon-trained checkpoint, REPS independent runs of each set:
# TB2.1 (88 tasks, train-fasttext excluded), SWE-bench Verified random-100 (2 shards of 50) and TB-lite (openthoughts
# tblite 2.0, 2 shards of 50). Every run is Jupiter's (the tb21/swe/tblite_6516_A_* runs): harbor-p0924 @ 761fb516,
# tb2_marin_policy_0924_65k16k.yaml (65,536 in / 16,384 out, 1,800 s agent budget, Daytona), 16 concurrent trials, one
# trial per task, against its own 1-node student serve (serve_relay.sbatch N_STUDENT=1 = serve_snowball POLICY=trained:
# EAGLE-3 draft, TP1 x DP4 x EP). The harness is data/tb2/horizon/run_tb2.sh + tb2_driver.sbatch; this script only
# submits a serve per run, waits for its endpoint, starts the run, and releases nothing itself (the driver cancels its
# serve when harbor ends).
#
#   MODEL=<HF export dir> TAG=<short model tag> [REPS="1 2 3"] [SETS="tb21 swe_s0 swe_s1 tblite_s0 tblite_s1"] \
#     bash eval_sft.sh        (login node, inside tmux)
#
# Runs are named <set>_<TAG>_r<rep>_<YYYYMMDD>; results under $S/experiments/sft_eval/tb2_jobs/<run>. Serve and driver
# logs, endpoints and rendered policies live under $S/experiments/sft_eval/ (not the relay dirs). Two nodes per run
# (serve + driver), all runs side by side: at the defaults 15 runs = 30 nodes and up to 240 sandboxes.
set -uo pipefail
MODEL=${MODEL:?HF export dir}; TAG=${TAG:?model tag}
REPS=${REPS:-1 2 3}; SETS=${SETS:-tb21 swe_s0 swe_s1 tblite_s0 tblite_s1}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd "$HERE/../../../.." && pwd)
S=${SCRATCH_DIR:-/scratch/11584/$USER}; T=$S/tasks
EV=$S/experiments/sft_eval
export RELAY_EXP_DIR=$EV/serve RELAY_PILOT_DIR=$OTA/data/relay/pilot TB2_EXP_DIR=$EV/tb2 JOBS=$EV/tb2_jobs
export STUDENT_MODEL=$MODEL
export HARBOR_SRC=${HARBOR_SRC:-$HOME/snowball/harbor-p0924/src} HARBOR_SHA=${HARBOR_SHA:-761fb516}
export POLICY_FILE=${POLICY_FILE:-tb2_marin_policy_0924_65k16k.yaml}
SERVE_TIME=${SERVE_TIME:-07:00:00}
DAY=${DAY:-$(date +%Y%m%d)}
mkdir -p "$RELAY_EXP_DIR/logs" "$RELAY_EXP_DIR/endpoints" "$TB2_EXP_DIR/logs" "$JOBS"; LOG=$EV/eval_$TAG.log
say() { echo "[$(date -u +%FT%TZ)] [$TAG] $*" | tee -a "$LOG"; }
[ -f "$MODEL/config.json" ] || { say "no export at $MODEL"; exit 1; }
say "EVAL_START model=$MODEL reps='$REPS' sets='$SETS' ota=$(git -C "$OTA" rev-parse --short HEAD) harbor=$HARBOR_SHA policy=$POLICY_FILE"

set_env() {  # the task tree, count, exclusions and shard of one set, as Jupiter ran it
  unset SHARD EXCLUDE_TASKS
  case $1 in
    tb21)      TASKS=$T/terminal_bench_2_1 NTASKS=89; export EXCLUDE_TASKS=train-fasttext;;
    swe_s0)    TASKS=$T/swebench_verified_random100 NTASKS=100; export SHARD=0/2;;
    swe_s1)    TASKS=$T/swebench_verified_random100 NTASKS=100; export SHARD=1/2;;
    tblite_s0) TASKS=$T/openthoughts_tblite_2_0 NTASKS=100; export SHARD=0/2;;
    tblite_s1) TASKS=$T/openthoughts_tblite_2_0 NTASKS=100; export SHARD=1/2;;
    *) return 1;;
  esac
  export TASKS NTASKS
}

one_run() {  # $1 set, $2 rep: serve -> endpoint -> run_tb2.sh; one retry of the serve if it dies before its endpoint
  local set=$1 rep=$2 name=${1}_${TAG}_r${2}_$DAY try j st
  [ -d "$JOBS/$name" ] && { say "$name exists; skipping"; return 0; }
  for try in 1 2; do
    j=$(bash "$OTA/data/relay/horizon/serve_submit.sh" 1 1 "$SERVE_TIME" "esrv_$name" 2>&1 | tail -1)
    [ "$j" -eq "$j" ] 2>/dev/null || { say "$name serve submit failed: $j"; return 1; }
    say "$name serve job $j (try $try)"
    while :; do
      st=$(squeue -h -j "$j" -o %T 2>/dev/null)
      [ -f "$RELAY_EXP_DIR/endpoints/$j.student" ] && [ "$st" = RUNNING ] && break
      if [ -f "$RELAY_EXP_DIR/endpoints/$j.DEAD" ] || [ -z "$st" ]; then say "$name serve $j died before its endpoint"; scancel "$j" 2>/dev/null; j=; break; fi
      sleep 60
    done
    [ -n "$j" ] && break
  done
  [ -n "$j" ] || { say "$name FAILED: no serve after 2 tries"; return 1; }
  set_env "$set" || { say "unknown set $set"; scancel "$j"; return 1; }
  if bash "$OTA/data/tb2/horizon/run_tb2.sh" "$j" "$name" >> "$LOG" 2>&1; then say "$name started (serve $j)"
  else say "$name FAILED: run_tb2.sh refused (see $LOG)"; scancel "$j"; return 1; fi
}

for rep in $REPS; do
  for set in $SETS; do
    one_run "$set" "$rep" &
    sleep 20   # spread the serve starts (EAGLE-3 compile) and the tunnel setups
  done
done
wait
say "EVAL_SUBMITTED; results: $JOBS/<set>_${TAG}_r<rep>_$DAY; readout: data/r2egym/horizon/sft/eval_readout.py"
