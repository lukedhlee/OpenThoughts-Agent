#!/bin/bash
# ota_sweep.sh — the small lr sweep on the OTA v2 SUBSET (~6 % of the train rows), standard SFT and TailSFT.
# Cost, stated before starting: prep 1 node ~10 min (0.2 node-h); each arm 16 nodes x ~20 min (~5.3 node-h);
# standard lane 3 arms (2e-5 5e-5 1e-4) ~16 node-h; TailSFT lane = reference pass (~2.7 node-h) + arm at 5e-5
# (~5.3), and a second tail arm at the standard winner if it is not 5e-5 (run by hand from the readout);
# exports 4 nodes x 6 min each (0.4), scoring 1 node x ~12 min each (0.2). About 30 node-hours in all.
# Stop rule (written before results): an lr counts if held-out NLL < base with think-NLL <= base's; the pick is
# the lowest held-out NLL at epoch 3; two lrs within 0.005 -> the lower lr; the top of the grid winning by
# > 0.01 -> flag before scaling. Then the full pair (ota_full_pair.sh) at the picks.
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
D=$S/data/ota_sft_100k_v2
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-sub-v2
HELDOUT=$D/ota-sub-heldout-00000-of-00001.parquet
LOG=$S/logs/ota/sweep.log; mkdir -p "$S/logs/ota"
say() { echo "[$(date -u +%FT%TZ)] [sweep] $*" | tee -a "$LOG"; }

# 1. cache for the subset (chain prep step; the chain's per-stage prep marker is removed first so the ota
#    stage builds THIS cache rather than reporting the full one as done).
if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/ota/.done.prep"
  SNOWBALL_STAGE=ota SNOWBALL_PARQUET_LIST=$D/parquet.sub.list SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K \
  SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  SBATCH_ACCOUNT=laionize CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(cat "$CACHE/train/.stats.json" 2>/dev/null | head -c 300)"

# 2. base reference score on the sweep held-out (independent of the arms)
jb=$(sbatch --parsable --account=laionize --export=ALL,MODEL=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888,PARQUET="$HELDOUT",NAME=ota-sub-base "$C/heldout_nll.sbatch") && say "base score job $jb"

# 3. standard lane, then (5 min later) the tail lane: reference pass, then the 5e-5 tail arm
tmux new -d -s ota_std "LANE=std CACHE=$CACHE PARQUET_LIST=$D/parquet.sub.list LRS='2e-5 5e-5 1e-4' EPOCHS=3 HELDOUT=$HELDOUT OUT_ROOT=$EXP/sweep SNOWBALL_WALL=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "std lane started (tmux ota_std)"
for i in $(seq 1 60); do squeue -u lee27 -h -o "%j %T" | grep -q "snowball-ota-std.* RUNNING" && break; sleep 30; done
sleep 300
tmux new -d -s ota_tail "CACHE=$CACHE REF_OUT=$EXP/tail/ref_sub_v2.npy bash -l $C/ota_tail_ref.sh && LANE=tail CACHE=$CACHE PARQUET_LIST=$D/parquet.sub.list LRS='5e-5' EPOCHS=3 HELDOUT=$HELDOUT OUT_ROOT=$EXP/sweep TAIL_FRACTION=0.25 TAIL_REF=$EXP/tail/ref_sub_v2.npy SNOWBALL_WALL=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "tail lane started (tmux ota_tail)"
say "SWEEP_LAUNCHED; readout: for f in $S/logs/heldout_nll_ota-sub-base.json $S/logs/heldout_nll_ota-std-*.json $S/logs/heldout_nll_ota-tail-*.json; do echo \$f; done"
