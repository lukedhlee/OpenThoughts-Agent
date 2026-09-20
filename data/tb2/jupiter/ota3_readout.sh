#!/bin/bash
# ota3_readout.sh <STAGE> [trials] — the readout of one 2026-09-20 SFT arm, launched once the arm's lane has ended:
#   * cheap ladder: probe_pair_chain.sh (if-v2 477 + RST held-out 337 on one node, ~2 node-h) on the FIRST kept
#     checkpoint (the one-third dose, the RL cold-start candidate) and the FINAL one;
#   * targets: eval_chain.sh (SWE-bench Verified random-100 + Terminal-Bench 2 on one serve node, ~4.5 node-h) on the
#     final export, `trials` independent trials (default 2; tags <STAGE>t<i>).
# Waits for LANE_DONE in the lane log, then for each export's held-out NLL file (submitted afterok the export job, so
# the export is complete). Runs in tmux on the login node. Log: $S/logs/<STAGE>/readout.log
set -uo pipefail
STAGE=${1:?stage}; TRIALS=${2:-2}
S=/e/data1/mmlaion/lee27/snowball-sft; C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
ARM=lr1e-4-sched2; OUT=$S/experiments/snowball-ota-sft/$STAGE/$ARM; LANE_LOG=$S/logs/$STAGE/lane_$STAGE.log
LOG=$S/logs/$STAGE/readout.log
say() { echo "[$(date -u +%FT%TZ)] [readout $STAGE] $*" | tee -a $LOG; }
say "waiting for LANE_DONE in $LANE_LOG"
for i in $(seq 1 96); do grep -q LANE_DONE $LANE_LOG 2>/dev/null && break; sleep 300; done   # up to 8 h
grep -q LANE_DONE $LANE_LOG 2>/dev/null || { say "no LANE_DONE after 8 h"; exit 1; }
grep -q ARM_FAILED $LANE_LOG && { say "lane reports ARM_FAILED; stopping"; exit 1; }
steps=$(ls $OUT/checkpoints | grep -oE '^step-[0-9]+$' | cut -d- -f2 | sort -n)
first=$(echo "$steps" | head -1); final=$(echo "$steps" | tail -1)
say "kept checkpoints: $(echo $steps | tr '\n' ' '); ladder on $first and $final; targets x$TRIALS on $final"
for st in $first $final; do
  EX=$OUT/export-step$st-hf-bf16; NLL=$S/logs/heldout_nll_$STAGE-$STAGE-$ARM-step$st.json
  for i in $(seq 1 90); do [ -f $NLL ] && [ -f $EX/config.json ] && break; sleep 120; done   # up to 3 h (export queue)
  [ -f $EX/config.json ] || { say "no export at $EX after 3 h; skipping step $st"; continue; }
  say "step $st: export ready, held-out NLL $(grep -oE '"nll": *[0-9.]+' $NLL | head -1)"
  tmux new-session -d -s pair_${STAGE}_s$st "bash $W/probe_pair_chain.sh ${STAGE}s$st $EX; sleep 600"
  say "launched pair probe tmux pair_${STAGE}_s$st"
  if [ "$st" = "$final" ]; then
    for t in $(seq 1 $TRIALS); do
      J=$(MODEL=$EX POLICY=trained sbatch --parsable $W/serve_snowball.sbatch) || { say "serve sbatch failed (trial $t)"; continue; }
      tmux new-session -d -s eval_${STAGE}_t$t "bash $W/eval_chain.sh $J ${STAGE}t$t; sleep 600"
      say "launched targets trial $t: serve $J, tmux eval_${STAGE}_t$t (runs ${STAGE}t${t}swe_v01_* and ${STAGE}t${t}_v01_*)"
      sleep 90
    done
  fi
done
say "READOUT_LAUNCHED"
