#!/bin/bash
# relay_sft_arm.sh — Horizon port of data/relay/sft/relay_sft_arm.sh: one arm of the 2026-09-28 relay SFT comparison on
# Grug Datakit 09-21, on the same rows Jupiter trained (private HF lukeleeai/relay-calibforge-jupiter-0928, copied to
# /scratch/11584/$USER/relay/jupiter_0928/). ARM=relay is arm A (09-21's episodes with Qwen3.8 taking over, every 09-21
# turn masked), ARM=qwen is arm B (Qwen3.8 alone).
#
# The recipe is Jupiter's, unchanged: marin stage relay_<ARM> (rows in as ids + loss, no template), the 09-21 import
# init (pending_qb_betas = -router_bias, router bias frozen), LR 3e-4, 3 real passes on one cosine (a pass = ceil(packs /
# 16) steps), warmup 5 %, seed 0, 16 x 65,536 on 4 nodes, one kept checkpoint per pass, an HF export per pass.
# What differs, all environment:
#   - the rows -> parquet conversion and the pack count run in a 1-node job (the converter holds the rows in Python
#     lists, several GB: too much for Horizon's shared login node);
#   - the chain's prep / run / export steps are called directly, as pair_kimi0921.sh does (marin branch
#     lukedhlee/horizon-snowball-sft-0921), instead of through snowball_sft_chain.sh + ota_lane.sh;
#   - held-out NLL is not scored here (Jupiter's common held-out set is not on Horizon).
# The pack count must equal Jupiter's (A 82, B 49 steps per pass = its 246 / 147 steps), or the arm stops before
# training: a different count means different rows or a different packer.
#
#   ARM=relay|qwen [EPOCHS=3] bash relay_sft_arm.sh       (login node, inside tmux; it only submits and polls)
#   ARM=<new> ROWS=<jsonl> DATASET_REVISION=<stage pin> bash relay_sft_arm.sh   (stage relay_<new> must exist)
#
# Steps, each skipped when its artifact exists: rows (1 node, ~5 min) -> prep (cache, 1 node, ~5 min) -> run (4 nodes;
# A ~1.1 h, B ~0.7 h) -> 3 exports (4 CPU nodes each, side by side, ~10 min) -> W&B sync + a per-step loss TSV.
# Cost: A ~6 node-h, B ~4 node-h. Native checkpoints (~625 GiB each, 3 per arm) are kept; delete them by hand once the
# exports are checked.
set -uo pipefail
ARM=${ARM:?relay | qwen}
case $ARM in
  relay) ROWS=${ROWS:-/scratch/11584/$USER/relay/jupiter_0928/relay_full_relaym6_20260928/final_v2A_rendered_think16k_clean_afnone.jsonl}
         EXPECT=${EXPECT_EPOCH_STEPS:-82};;
  qwen)  ROWS=${ROWS:-/scratch/11584/$USER/relay/jupiter_0928/relay_full_baseline6m_20260926/final_v2_rendered_think16k_clean.jsonl}
         EXPECT=${EXPECT_EPOCH_STEPS:-49};;
  *) : "${ROWS:?ARM=$ARM needs ROWS=<rendered jsonl> and a marin stage relay_$ARM}"   # a new arm (e.g. the 09-30 experiments)
     EXPECT=${EXPECT_EPOCH_STEPS:-any};;
esac
STAGE=relay_$ARM
OTA=$(cd "$(dirname "$0")/../../../.." && pwd)
S=${SNOWBALL_SCRATCH:-/scratch/11584/$USER/snowball-sft}
MARIN=${MARIN_ROOT:-$HOME/snowball/marin-sft}
MOE=$MARIN/experiments/june_tpu_67b_a2b/moe
PYM=${MARIN_PYTHON:-$HOME/snowball/envs/marin-grug-sft/bin/python}
BASE=${BASE_0921:-/scratch/11584/$USER/models/grug-datakit-sft-20260921}
INIT=${SNOWBALL_INIT:-$S/experiments/snowball-base-inits/init-dk0921-step0}
WANDB=${WANDB_CLI:-$HOME/snowball/envs/snowball/bin/wandb}
KEYS=${KEYS:-$HOME/.config/otagent/secrets.env}
WANDB_ENTITY_ARM=lukedhlee-marin; WANDB_PROJECT_ARM=${WANDB_PROJECT_ARM:-horizon-relay-sft}
LR=${LR:-3e-4}; EPOCHS=${EPOCHS:-3}   # LR: arm A's 3e-4 unless overridden (outputs go to lr<LR>-sched<EPOCHS>); passes; EPOCHS=5 on arm B = arm A's step count (245 vs 246)
D=$S/data/relay_v2/$ARM
EXP=$S/experiments/snowball-relay-sft
CACHE=$EXP/cache-$STAGE-v1
OUT=$EXP/$STAGE/lr$LR-sched$EPOCHS
RUN_ID=snowball-$STAGE-horizon-lr$LR-sched$EPOCHS
LOGD=$S/logs/$STAGE; mkdir -p "$LOGD" "$EXP" "$(dirname "$OUT")"; LOG=$LOGD/arm.log   # the preflight wants the output's parent to exist
say() { echo "[$(date -u +%FT%TZ)] [$STAGE] $*" | tee -a "$LOG"; }
die() { say "ARM_FAILED: $*"; exit 1; }
jobid() { grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1; }
wait_job() {  # $1 job, $2 log, $3 marker regex. COMPLETING counts as ended: sacct already holds the final state, and
  local st     # Horizon's epilog keeps a finished job in squeue for 5-10 min
  while squeue -h -j "$1" -t PD,R,CF,S,RQ,RS 2>/dev/null | grep -q .; do sleep 30; done
  for _ in 1 2 3 4 5; do st=$(sacct -j "$1" -X -n -o State%20 2>/dev/null | head -1 | awk '{print $1}'); [ -n "$st" ] && break; sleep 20; done
  say "job $1 ended ${st:-?}"
  [ "$st" = COMPLETED ] && grep -qE "$3" "$2"
}
export SNOWBALL_SCRATCH=$S MARIN_ROOT=$MARIN MARIN_PYTHON=$PYM SNOWBALL_STAGE=$STAGE SNOWBALL_TOKENIZER=$BASE
export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-$ARM SNOWBALL_DATASET_REVISION=${DATASET_REVISION:-final_v2}   # = the stage's pins
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# layout: 16 x 65,536 on 4 nodes, set before prep as Jupiter's launcher does: the prerendered format's row limit is the
# packing length SNOWBALL_SEQ_LEN (default 32,768), so a prep without it refuses every row over 32,768 tokens
export SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_NODES=4 SNOWBALL_DEVICES=16
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async OMP_NUM_THREADS=1
[ -f "$BASE/config.json" ] || die "no 09-21 export at $BASE"
[ -f "$INIT/snowball_base.json" ] || die "no 09-21 init sidecar at $INIT (pair_kimi0921.sh's import step makes it)"
say "ARM_START marin=$(git -C "$MARIN" rev-parse --short HEAD) ota=$(git -C "$OTA" rev-parse --short HEAD) rows=$ROWS out=$OUT"

# 1. rows -> the one-shard parquet + the pack count (1 node); the converter refuses any row the chain would refuse or cut
if [ ! -f "$D/pack_epoch_steps.json" ]; then
  [ -f "$ROWS" ] || die "no rows at $ROWS"
  mkdir -p "$D"; rm -f "$D/parquet.list"
  jr=$(sbatch -A CCR24067 -p debug -N 1 --ntasks-per-node=1 --cpus-per-task=144 --gres=gpu:4 -t ${ROWS_TIME:-00:40:00} -J "relay-rows-$ARM" -o "$LOGD/rows.%j.log" <<EOF 2>&1 | jobid
#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
sha256sum "$ROWS" > "$D/rows.sha256"
"$PYM" "$OTA/data/relay/sft/relay_rows_to_parquet.py" "$ROWS" "$D"
cd "$MARIN"
PYTHONPATH=$MARIN/lib/levanter/src:$MARIN/lib/haliax/src:$MARIN/lib/rigging/src:$MARIN/lib/fray/src:$MARIN/lib/zephyr/src:$MARIN/lib/iris/src \
  "$PYM" "$OTA/data/relay/sft/pack_epoch_steps.py" "$D/train-00000-of-00001.parquet" --seq-len 65536 --batch 16 | tail -1 > "$D/pack_epoch_steps.json.tmp"
mv "$D/pack_epoch_steps.json.tmp" "$D/pack_epoch_steps.json"
echo ROWS_DONE
EOF
)
  [ -n "$jr" ] || die "rows submit"; say "rows job $jr"
  wait_job "$jr" "$LOGD/rows.$jr.log" "ROWS_DONE" || die "rows job $jr ($LOGD/rows.$jr.log)"
fi
say "rows: $(cut -c1-64 "$D/rows.sha256") parquet: $(tr -d '\n ' < "$D/pack_epoch_steps.json")"
EPOCH_STEPS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pack_epoch_steps"])' "$D/pack_epoch_steps.json") || die "no pack_epoch_steps"
[ "$EXPECT" = any ] || [ "$EPOCH_STEPS" = "$EXPECT" ] || die "pack count $EPOCH_STEPS steps per pass != Jupiter's $EXPECT"
STEPS=$((EPOCHS * EPOCH_STEPS))
WARMUP=$(( (STEPS * 5 + 50) / 100 )); [ "$WARMUP" -ge 2 ] || WARMUP=2
say "plan: $EPOCH_STEPS steps per pass x $EPOCHS = $STEPS steps, warmup $WARMUP, lr $LR"

# 2. cache (prep, 1 node): the prerendered stage copies ids/loss; provenance pins the shard hash and format
if [ ! -f "$CACHE/train/.stats.json" ]; then
  jp=$(sbatch -o "$LOGD/prep.%j.log" --export=ALL,SNOWBALL_ENV="$(dirname "$(dirname "$PYM")")",SNOWBALL_CACHE="$CACHE",SNOWBALL_PARQUET_LIST="$D/parquet.list" \
       "$MOE/horizon_r2egym_prep.sbatch" 2>&1 | jobid); [ -n "$jp" ] || die "prep submit"; say "prep job $jp"
  wait_job "$jp" "$LOGD/prep.$jp.log" "PREP_DONE" || die "prep job $jp ($LOGD/prep.$jp.log)"
fi
say "cache: $(head -c 300 "$CACHE/train/.stats.json")"

# 3. train: 3 passes on one cosine, one kept checkpoint per pass
if [ ! -f "$LOGD/run_$RUN_ID.done" ]; then
  export SNOWBALL_LR=$LR SNOWBALL_WARMUP=$WARMUP EPOCHS=$EPOCHS SNOWBALL_SCHEDULE_EPOCHS=$EPOCHS SNOWBALL_RESUME=0
  export SNOWBALL_EPOCH_STEPS=$EPOCH_STEPS SNOWBALL_KEEP_PER_EPOCH=1
  # temporary (resume) saves: this script never resumes (SNOWBALL_RESUME=0); 625 GB every 30 min from several arms
  # saturated scratch on 10-04, so save rarely unless SAVE_INTERVAL_MIN says otherwise
  export SNOWBALL_SAVE_INTERVAL_MIN=${SAVE_INTERVAL_MIN:-100000}
  export SNOWBALL_INIT=$INIT SNOWBALL_CACHE=$CACHE SNOWBALL_OUTPUT=$OUT SNOWBALL_RUN_ID=$RUN_ID SNOWBALL_WALL=${WALL:-02:30:00}
  out=$(bash "$MOE/launch_horizon_snowball_sft.sh" 2>&1); rc=$?
  echo "$out" >> "$LOG"; echo "$out" | grep -E '^(stage|epochs|steps|layout|lr|init) ' | tee -a "$LOG"
  [ $rc -eq 0 ] || die "launcher rc=$rc"
  jt=$(echo "$out" | jobid); [ -n "$jt" ] || die "no run job id"; say "run job $jt (log $S/logs/snowball-$STAGE.$jt.log)"
  wait_job "$jt" "$S/logs/snowball-$STAGE.$jt.log" "GUARDED_RUN_EXIT rc=0" || die "run job $jt"
  echo "$jt" > "$LOGD/run_$RUN_ID.done"
fi

# 4. one HF export per kept checkpoint (4 CPU nodes each, side by side); 09-21's config values, templates, tokenizer
points=$(ls "$OUT/checkpoints" 2>/dev/null | grep -oE '^step-[0-9]+$' | cut -d- -f2 | sort -n | tr '\n' ' ')
[ -n "$points" ] || die "no kept checkpoints under $OUT/checkpoints"
say "kept checkpoints: $points"
declare -A jx=()
for st in $points; do
  EX=$OUT/export-step$st-hf-bf16
  [ -f "$EX/config.json" ] && { say "export step $st exists"; continue; }
  j=$(sbatch -o "$S/logs/snowball-export.%j.log" \
      --export=ALL,SNOWBALL_EXPORT_CHECKPOINT="$OUT/checkpoints/step-$st",SNOWBALL_EXPORT_OUTPUT="$EX",SNOWBALL_EXPORT_TOKENIZER="$BASE",SNOWBALL_EXPORT_BASE="$BASE" \
      "$MOE/horizon_snowball_export.sbatch" 2>&1 | jobid); [ -n "$j" ] || die "export submit step $st"
  jx[$st]=$j; say "export step $st job $j -> $EX"
done
for st in "${!jx[@]}"; do
  wait_job "${jx[$st]}" "$S/logs/snowball-export.${jx[$st]}.log" "EXPORT_CHECK_OK" || die "export job ${jx[$st]} (step $st)"
done

# 5. W&B: sync the offline run and write the per-step loss
WB=$(ls -d "$MARIN"/wandb/offline-run-*-"$RUN_ID" 2>/dev/null | tail -1); [ -n "$WB" ] || die "no offline W&B run for $RUN_ID under $MARIN/wandb"
( set -a; source "$KEYS"; set +a; cd "$(dirname "$WB")" && WANDB_MODE=online "$WANDB" sync --entity "$WANDB_ENTITY_ARM" --project "$WANDB_PROJECT_ARM" --id "$RUN_ID" "$(basename "$WB")" ) >> "$LOG" 2>&1 \
  && say "W&B synced: $WANDB_ENTITY_ARM/$WANDB_PROJECT_ARM/$RUN_ID" || say "W&B sync FAILED (the TSV below still has the curve)"
"$PYM" - "$WB" "$LOGD/loss_$RUN_ID.tsv" <<'PY' && say "loss TSV: $LOGD/loss_$RUN_ID.tsv"
import glob, json, sys
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
ds = datastore.DataStore(); ds.open_for_scan(glob.glob(sys.argv[1] + "/*.wandb")[0])
rows = {}
while (d := ds.scan_data()) is not None:
    r = pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type") == "history":
        h = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item}
        if "train/loss" in h: rows[h["_step"]] = (h["train/loss"], h.get("optim/learning_rate"))
with open(sys.argv[2], "w") as f:
    f.write("step\ttrain_loss\tlr\n")
    for s in sorted(rows): f.write(f"{s}\t{rows[s][0]:.6f}\t{rows[s][1]}\n")
PY
say "ARM_DONE exports: $(ls -d "$OUT"/export-step*-hf-bf16 | tr '\n' ' ')"
