#!/bin/bash
# ota_tail_ref.sh — the TailSFT reference pass on one OTA cache: forward-only, initial weights, one loss per
# document, written to REF_OUT (.npy). The loader samples with replacement, so the pass is launched with
# EPOCHS=6 and stops on its own once every document is scored (marin 6ef8da34d). Waits for the job.
# Required env: CACHE, REF_OUT. Optional REF_EPOCHS (6; 4 on the full cache so 4 x ~600 stays under the ceiling). Cost: ~2 s per batch of 64 docs + ~7 min start-up on 16 nodes.
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
MOE=/e/project1/transfernetx/lee27/code/marin-sft/experiments/june_tpu_67b_a2b/moe
: "${CACHE:?}" "${REF_OUT:?}"
export MARIN_ROOT=/e/project1/transfernetx/lee27/code/marin-sft
export MARIN_PYTHON=/e/project1/transfernetx/lee27/code/envs/marin-grug-sft/bin/python
export SNOWBALL_SCRATCH=$S SNOWBALL_STAGE=ota
export SNOWBALL_CACHE=$CACHE
export SNOWBALL_INIT=$S/experiments/snowball-r2egym-sft/init-s3-step1888
export SNOWBALL_TOKENIZER=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
export SNOWBALL_OUTPUT=${REF_OUT%.npy}-run
export SNOWBALL_RUN_ID=snowball-ota-tailscore-$(basename "${REF_OUT%.npy}")
export EPOCHS=${REF_EPOCHS:-6}   # must keep EPOCHS x epoch under the stage ceiling (2,600): 4 on the full set
export SNOWBALL_WALL=${SNOWBALL_WALL:-01:30:00}
export SNOWBALL_TAIL_SCORE_OUT=$REF_OUT
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
mkdir -p "$(dirname "$REF_OUT")"
[ -f "$REF_OUT" ] && { echo "reference exists: $REF_OUT"; exit 0; }
out=$(bash -l "$MOE/launch_jupiter_snowball_r2egym.sh" 2>&1); rc=$?; echo "$out"
[ $rc -eq 0 ] || { echo "launch failed rc=$rc"; exit 1; }
j=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1)
[ -n "$j" ] || { echo "no job id"; exit 1; }
echo "TAILREF_JOB $j"
while squeue -j "$j" -h -o %T 2>/dev/null | grep -q .; do sleep 60; done
st=$(sacct -j "$j" -X -n -o State%24 | head -1 | awk '{print $1}')
echo "TAILREF_END $j $st"
[ -f "$REF_OUT" ] && { echo "TAILREF_OK $REF_OUT"; exit 0; } || { echo "TAILREF_FAILED: no $REF_OUT"; exit 1; }
