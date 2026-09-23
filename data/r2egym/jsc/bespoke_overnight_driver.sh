#!/bin/bash
# overnight_driver.sh (2026-09-23, Luke asleep): bespoke SFT LR sweep -> pick -> noGLM arms -> TB2 -> SWE on the best.
# Runs in tmux on Jupiter login02 so it survives the Mac's ssh ControlMaster expiring. Log: $L/overnight_driver.log
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft; L=$S/logs; EXP=$S/experiments/snowball-bespoke-sft
C=/e/project1/transfernetx/lee27/code/snowball; W=/e/project1/transfernetx/lee27/code/tb2; T=/e/fscratch/reformo/lee27/experiments/tb2
LOG=$L/overnight_driver.log; D=20260923
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
XENV="OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async"
SCORE_ENV="HELDOUT_MAX_LEN=65536 HELDOUT_MAX_MODEL_LEN=66560 HELDOUT_MAX_POS=131072"
BAD=jpbo-056-38,jpbo-081-44

# wait until no job whose name matches $1 is queued; $2 = max seconds
wait_jobs() { local t=0; while squeue -u lee27 -h -o %j | grep -qE "$1"; do sleep 300; t=$((t+300)); [ $t -ge $2 ] && { log "wait_jobs '$1' timed out after $2 s"; return 1; }; done; return 0; }
# wait until every lane in $@ (sw_*.out / nl_*.out files) has LANE_DONE or failed
wait_lanes() { local t=0; while :; do local n=0; for f in "$@"; do grep -qE "LANE_DONE|failed|FATAL" $L/$f && n=$((n+1)); done; [ $n -ge $# ] && return 0; sleep 300; t=$((t+300)); [ $t -ge 14400 ] && { log "wait_lanes timed out"; return 1; }; done; }
# common held-out score for every export a lane submitted (dependency on its export job)
common_scores() { for f in "$@"; do grep EXPORT_SUBMITTED $L/$f | while read -r l; do
  st=$(echo "$l" | grep -oE "step [0-9]+" | cut -d" " -f2); jx=$(echo "$l" | grep -oE "job [0-9]+" | cut -d" " -f2)
  ex=$(echo "$l" | grep -oE "\-> [^ ]+" | cut -c4-); arm=$(echo "$ex" | awk -F/ '{print $(NF-2)"_"$(NF-1)}')
  N=common_${arm}_step$st; [ -f $L/heldout_nll_$N.json ] && continue
  js=$(env $SCORE_ENV SBATCH_TIMELIMIT=00:45:00 sbatch --parsable --exclude=$BAD --dependency=afterok:$jx --account=laionize \
       --export=ALL,$(echo $SCORE_ENV | tr ' ' ','),MODEL="$ex",PARQUET="$S/data/bespoke_v1/common_heldout/heldout-00000-of-00001.parquet",NAME="$N" $C/heldout_nll.sbatch)
  log "common score $N job $js (after export $jx)"; done; done; }
# best checkpoint of a variant by its own held-out NLL: prints "<nll> <export_dir>"
best_of() { python3 - "$1" <<'PY'
import glob, json, sys
v = sys.argv[1]; best = None
for f in glob.glob(f"/e/data1/mmlaion/lee27/snowball-sft/logs/heldout_nll_bespoke_{v}-bespoke_{v}*-step*.json"):
    d = json.load(open(f))
    if best is None or d["heldout_nll"] < best[0]: best = (d["heldout_nll"], d["tokenizer_dir"])
print(f"{best[0]:.4f} {best[1]}" if best else "none none")
PY
}
lr_of() { echo "$1" | grep -oE "/lr[0-9e.-]+-sched3/" | sed -E 's#/lr([0-9e.-]+)-sched3/#\1#'; }
step_of() { echo "$1" | grep -oE "export-step[0-9]+" | grep -oE "[0-9]+"; }
# serve + TB2 pipeline + post_run + reaper (cancels the serve node when the final lands); $1 name $2 model $3 policy file
tb2() { local N=$1_t1_v01_$D; local j; j=$(cd $W && MODEL=$2 POLICY=trained sbatch --parsable serve_snowball.sbatch) || { log "TB2 serve submit failed for $N"; return 1; }
  tmux new-session -d -s pl_$N "cd $W && POLICY_FILE=$3 bash pipeline.sh $j $N"
  tmux new-session -d -s post_$N "cd $W && POLICY_FILE=$3 bash post_run.sh $N"
  reap $j $N & log "TB2 $N serve=$j policy=$3 model=$2"; echo "$N $j $2 $3" >> $L/overnight_tb2.txt; }
reap() { local t=0; while [ ! -f $T/final_$2.DONE ]; do grep -q FAILED $T/pipeline_$2.log 2>/dev/null && break; squeue -h -j $1 | grep -q . || return; sleep 600; t=$((t+600)); [ $t -ge 43200 ] && break; done; scancel $1; log "released serve $1 for $2 ($( [ -f $T/final_$2.DONE ] && echo final || echo 'no final'))"; }
score_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pass@1_mean_of_task_means'])" $T/final_$1.json 2>/dev/null || echo nan; }

log "DRIVER_START"
# 0. the LR 3e-5 checkpoints barely moved (held-out within 0.03 of base): drop their held TB2 serves
scancel 1967253 1967254 1967256 1967257 2>/dev/null; log "cancelled held 3e-5 TB2 serves"
reap 1967259 grugdk0921think_t1_v01_$D &

# 1. sweep: wait for the six lanes, score every export on the common set, wait for all scores
SW="sw_think_all_1e-4.out sw_think_all_3e-4.out sw_think_all_1e-3.out sw_fold_all_1e-4.out sw_fold_all_3e-4.out sw_fold_all_1e-3.out"
wait_lanes $SW; log "sweep lanes done"
common_scores $SW
wait_jobs "snowball-export|heldout_nll" 10800; log "sweep exports + scores done"
read -r TNLL TBEST <<<"$(best_of think_all)"; read -r FNLL FBEST <<<"$(best_of fold_all)"
TLR=$(lr_of $TBEST); FLR=$(lr_of $FBEST)
log "PICK think_all nll=$TNLL lr=$TLR step=$(step_of $TBEST) $TBEST"
log "PICK fold_all  nll=$FNLL lr=$FLR step=$(step_of $FBEST) $FBEST"

# 2. TB2 on the two picks now; the noGLM arms at the picked LRs in parallel
[ "$TBEST" != none ] && tb2 bspTA_lr${TLR}_s$(step_of $TBEST) $TBEST tb2_marin_policy.yaml
[ "$FBEST" != none ] && tb2 bspFA_lr${FLR}_s$(step_of $FBEST) $FBEST tb2_marin_policy_nothink.yaml
for pair in "bespoke_think_noglm $TLR" "bespoke_fold_noglm $FLR"; do set -- $pair; [ -z "$2" ] && continue
  f=nl_${1#bespoke_}_$2.out
  tmux new-session -d -s ${f%.out} "cd $C && $XENV STAGE=$1 LR=$2 LANE=${1}_lr$2 bash bespoke_arm.sh > $L/$f 2>&1"; log "noGLM arm $1 lr=$2 ($f)"; sleep 20; done
NL="nl_think_noglm_$TLR.out nl_fold_noglm_$FLR.out"
sleep 120; wait_lanes $NL; log "noGLM lanes done"; common_scores $NL
wait_jobs "snowball-export|heldout_nll" 10800
read -r TNNLL TNBEST <<<"$(best_of think_noglm)"; read -r FNNLL FNBEST <<<"$(best_of fold_noglm)"
log "PICK think_noglm nll=$TNNLL $TNBEST"; log "PICK fold_noglm nll=$FNNLL $FNBEST"
# only the new-LR arms are candidates here (the 3e-5 ones are in the glob too; take them only if they win)
[ "$TNBEST" != none ] && tb2 bspTN_lr$(lr_of $TNBEST)_s$(step_of $TNBEST) $TNBEST tb2_marin_policy.yaml
[ "$FNBEST" != none ] && tb2 bspFN_lr$(lr_of $FNBEST)_s$(step_of $FNBEST) $FNBEST tb2_marin_policy_nothink.yaml

# 3. SWE-bench random-100 on the best TB2 bespoke checkpoint (once the first two TB2 finals are in)
t=0; while :; do n=0; while read -r N j M P; do [ -f $T/final_$N.DONE ] && n=$((n+1)); done < $L/overnight_tb2.txt; [ $n -ge 2 ] && break; sleep 600; t=$((t+600)); [ $t -ge 43200 ] && break; done
best=""; bs=-1; while read -r N j M P; do s=$(score_of $N); log "TB2 final $N pass@1=$s"; [ "$s" != nan ] && awk -v a=$s -v b=$bs 'BEGIN{exit !(a>b)}' && { bs=$s; best="$N $M $P"; }; done < $L/overnight_tb2.txt
if [ -n "$best" ]; then set -- $best; SN=${1%_t1_v01_$D}_swe_t1_v01_$D
  js=$(cd $W && MODEL=$2 POLICY=trained sbatch --parsable serve_snowball.sbatch) && tmux new-session -d -s pl_$SN "cd $W && POLICY_FILE=$3 bash swe_chain.sh $js $SN" && log "SWE $SN serve=$js on $2 (TB2 pass@1=$bs)"
fi
log "DRIVER_MAIN_DONE (remaining TB2 finals are reaped in the background)"
wait
log "DRIVER_EXIT"
