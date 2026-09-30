#!/bin/bash
# s5_arm_eval.sh — PedaGEPA stage 5: train one arm (relay_sft_arm.sh, arm A's recipe) and, when its last export exists,
# run the stage-5 evals on it: TB2.1 + SWE random-100 (2 shards), 3 independent runs each, eval_sft.sh GROUPED=1, in a
# separate eval root (SCRATCH_DIR) so the train-port session's sft_eval state is never touched.
#   ARM=pgc0|pgc1|pgp ROWS=<jsonl> bash s5_arm_eval.sh          (login node, inside tmux)
set -uo pipefail
ARM=${ARM:?}; ROWS=${ROWS:?}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd "$HERE/../.." && pwd)
export CC=gcc CXX=g++
EVROOT=${EVROOT:-/scratch/11584/$USER/experiments/pedagepa/s5/evroot}
if [ -n "${WAIT_ARM_LOG:-}" ]; then   # the arm was started elsewhere: wait for its ARM_DONE
  until grep -q "ARM_DONE\|ARM_FAILED" "$WAIT_ARM_LOG"; do sleep 60; done
  grep -q ARM_FAILED "$WAIT_ARM_LOG" && { echo "ARM $ARM FAILED"; exit 1; }
else
ARM=$ARM ROWS=$ROWS DATASET_REVISION=pg20260930 MARIN_ROOT=$HOME/snowball/marin-peda WANDB_PROJECT_ARM=horizon-pedagepa-sft \
  bash "$OTA/data/r2egym/horizon/sft/relay_sft_arm.sh" || { echo "ARM $ARM FAILED"; exit 1; }
fi
OUT=/scratch/11584/$USER/snowball-sft/experiments/snowball-relay-sft/relay_$ARM/lr3e-4-sched3
EX=$(for d in "$OUT"/export-step*-hf-bf16; do echo "$(basename "$d" | grep -oE '[0-9]+') $d"; done | sort -n | tail -1 | cut -d' ' -f2)
echo "eval $ARM on $EX"
SCRATCH_DIR=$EVROOT MODELS="$ARM=$EX" SETS="tb21 swe_s0 swe_s1" REPS="1 2 3" GROUPED=1 MAXJOBS=${MAXJOBS:-30} \
  bash "$OTA/data/r2egym/horizon/sft/eval_sft.sh"
echo "S5_ARM_EVAL_DONE $ARM"
