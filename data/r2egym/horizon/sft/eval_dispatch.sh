#!/bin/bash
# eval_dispatch.sh — evaluate SFT arms as they finish, one (arm, rep) unit at a time: a unit is eval_sft.sh GROUPED=1 with
# one REP, i.e. one 5-node serve job running TB2.1, SWE random-100 s0/s1 and TB-lite s0/s1 once (up to 80 Daytona
# sandboxes). Units are ordered rep-major (rep 1 of every arm, then rep 2, then rep 3), so every arm gets a first read
# before any arm gets its third. An arm is ready when its relay_sft_arm.sh driver logs ARM_DONE with that arm's output
# dir. A unit starts while the eval org's started sandboxes (every user) + 80 stay within DAYTONA_CAP (default 850 of
# the ~1,000 the org allows) and at most MAXUNITS units run (Horizon allows 20 running jobs per user).
#
#   ARMS="tag:stage:epochs ..." [REPS="1 2 3"] [MAXUNITS=10] [DAYTONA_CAP=850] bash eval_dispatch.sh   (login node, tmux)
#   e.g. ARMS="e5B245:relay_qwen:5 h2think:relay_athink:3"
# Each unit runs in tmux evq_<tag>_r<rep>; the dispatcher log is $S/experiments/sft_eval/dispatch.log.
set -uo pipefail
ARMS=${ARMS:?"tag:stage:epochs ..."}; REPS=${REPS:-1 2 3}; MAXUNITS=${MAXUNITS:-10}; DAYTONA_CAP=${DAYTONA_CAP:-850}
HERE=$(cd "$(dirname "$0")" && pwd)
SS=/scratch/11584/$USER/snowball-sft; EXP=$SS/experiments/snowball-relay-sft
EV=/scratch/11584/$USER/experiments/sft_eval; mkdir -p "$EV"; LOG=$EV/dispatch.log
DAY=${DAY:-$(date +%Y%m%d)}
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
say "DISPATCH_START arms='$ARMS' reps='$REPS' maxunits=$MAXUNITS daytona_cap=$DAYTONA_CAP day=$DAY"
UNITS=(); for r in $REPS; do for a in $ARMS; do UNITS+=("$a:$r"); done; done
declare -A started
while :; do
  active=$(tmux ls 2>/dev/null | grep -c '^evq_'); pending=0; ds=
  for u in "${UNITS[@]}"; do
    IFS=: read -r tag stage ep rep <<<"$u"
    [ -n "${started[$tag:$rep]:-}" ] && continue
    pending=$((pending + 1))
    out=$EXP/$stage/lr3e-4-sched$ep
    grep -q "ARM_DONE exports: .*$out/" "$SS/logs/$stage/arm.log" 2>/dev/null || continue
    [ "$active" -lt "$MAXUNITS" ] || break
    [ -n "$ds" ] || ds=$(python3 "$HERE/daytona_started.py")
    [ $((ds + 80)) -le "$DAYTONA_CAP" ] || { say "$tag r$rep: ready, waiting for Daytona room ($ds started)"; break; }
    model=$(ls -d "$out"/export-step*-hf-bf16 | sort -V | tail -1)
    [ -f "$model/config.json" ] || { say "$tag: no final export under $out"; started[$tag:$rep]=failed; continue; }
    tmux new -d -s "evq_${tag}_r$rep" "MODELS='$tag=$model' REPS=$rep GROUPED=1 DAY=$DAY bash $HERE/eval_sft.sh; sleep 300"
    started[$tag:$rep]=1; active=$((active + 1)); pending=$((pending - 1)); ds=$((ds + 80))
    say "$tag r$rep: eval unit started on $model (tmux evq_${tag}_r$rep)"
  done
  [ "$pending" -eq 0 ] && break
  sleep 120
done
say "DISPATCH_DONE (every unit started)"
