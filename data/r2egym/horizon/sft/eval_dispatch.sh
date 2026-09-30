#!/bin/bash
# eval_dispatch.sh — start eval_sft.sh (GROUPED=1: 3 runs each of TB2.1, SWE random-100, TB-lite) for each SFT arm as soon
# as its relay_sft_arm.sh driver logs ARM_DONE with that arm's output dir, at most MAXARMS arms evaluating at once (each
# arm = 3 five-node serve jobs and up to 240 Daytona sandboxes). Arms are started in the order given.
#
#   ARMS="tag:stage:epochs ..." [MAXARMS=3] bash eval_dispatch.sh        (login node, tmux)
#   e.g. ARMS="e5B245:relay_qwen:5 h2think:relay_athink:3"
# Each arm's queue runs in tmux evq_<tag>; the dispatcher log is $S/experiments/sft_eval/dispatch.log.
set -uo pipefail
ARMS=${ARMS:?"tag:stage:epochs ..."}; MAXARMS=${MAXARMS:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
SS=/scratch/11584/$USER/snowball-sft; EXP=$SS/experiments/snowball-relay-sft
EV=/scratch/11584/$USER/experiments/sft_eval; mkdir -p "$EV"; LOG=$EV/dispatch.log
DAY=${DAY:-$(date +%Y%m%d)}
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
say "DISPATCH_START arms='$ARMS' maxarms=$MAXARMS day=$DAY"
declare -A started
while :; do
  active=0; pending=0
  for a in $ARMS; do tmux has-session -t "evq_${a%%:*}" 2>/dev/null && active=$((active + 1)); done
  for a in $ARMS; do
    IFS=: read -r tag stage ep <<<"$a"
    [ -n "${started[$tag]:-}" ] && continue
    pending=$((pending + 1))
    out=$EXP/$stage/lr3e-4-sched$ep
    grep -q "ARM_DONE exports: .*$out/" "$SS/logs/$stage/arm.log" 2>/dev/null || continue
    [ "$active" -lt "$MAXARMS" ] || continue
    model=$(ls -d "$out"/export-step*-hf-bf16 | sort -V | tail -1)
    [ -f "$model/config.json" ] || { say "$tag: no final export under $out"; started[$tag]=failed; continue; }
    tmux new -d -s "evq_$tag" "MODELS='$tag=$model' GROUPED=1 DAY=$DAY bash $HERE/eval_sft.sh; sleep 600"
    started[$tag]=1; active=$((active + 1)); pending=$((pending - 1))
    say "$tag: eval queue started on $model (tmux evq_$tag)"
  done
  [ "$pending" -eq 0 ] && break
  sleep 120
done
say "DISPATCH_DONE (every arm's eval queue started)"
