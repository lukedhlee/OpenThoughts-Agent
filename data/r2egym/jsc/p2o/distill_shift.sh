#!/bin/bash
# distill_shift.sh <model_dir> <label> — the fixed-token-set mechanism readout: score the confirmation probe's prompted
# trajectories (p2oAc_s0, block A) under the prompted and the bare first user message with the given checkpoint
# (p2o_score.sbatch, 1 node, 1 h wall, forward-KL direction). Output $E/p2o/shift_<label>/summary.tsv
# (mean_logratio_per_tok = the gap; compare labels on the same traces, e.g. base vs s24). Non-blocking; prints the job id.
set -u
E=/e/fscratch/reformo/lee27/experiments; MODEL=$1; LABEL=$2
[ -f $MODEL/config.json ] || { echo "no config.json in $MODEL"; exit 1; }
TR="$E/p2oAc_s0/p2oAc_s0/trace_jobs/eval_sessions/*"; OUT=$E/p2o/shift_$LABEL; mkdir -p $OUT
J=$(cd $E/p2o && sbatch --parsable --time=01:00:00 -J p2o_shift_$LABEL --export=ALL,MODEL=$MODEL,MODE=wave,TRACES="$TR",OUT=$OUT,WALL_SEC=3600 p2o_score.sbatch)
echo "shift $J $LABEL $(date -Is)" >> $E/p2o/distill_ids.txt; echo "SHIFT $J ($LABEL -> $OUT/summary.tsv)"
