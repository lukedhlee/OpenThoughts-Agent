#!/bin/bash
# relay_sft_arm.sh — one arm of the relay SFT comparison on Grug Datakit 09-21 (ARM=relay: 09-21's episodes with Qwen3.8
# taking over; ARM=qwen: Qwen3.8 alone; ARM=relayaf: relay with the autofixed 09-21 actions trained), through the Levanter Grug chain (snowball_sft_chain.sh via ota_lane.sh), the
# way bespoke_arm.sh ran the Bespoke SFT. What differs from bespoke_arm.sh:
#   data      the rendered rows themselves (render.py / render_think_limit.py jsonl: ids + a per-token loss), converted
#             by relay_rows_to_parquet.py and cached by the prerendered stage relay_<ARM> (marin
#             grug_datakit_chat.SnowballPrerenderedChatFormat): no template, no re-tokenization, the trainer sees these
#             ids and this mask (loss on Qwen's turns only, 09-21's turns masked)
#   epochs    a pass is ceil(packs / 16) steps (pack_epoch_steps.py -> SNOWBALL_EPOCH_STEPS), not tokens / (16 x 65,536):
#             relay rows fill ~75 % of a 65,536 pack, so the token-derived "3 epochs" was ~2.2-2.3 real passes
#   recipe    LR 3e-4 (the Bespoke pick), 3 passes on one cosine, warmup 5 % (min 2), seed 0, router bias frozen at
#             09-21's (stage default, checked by the init sidecar), 16 x 65,536 on 4 nodes, one checkpoint per pass,
#             export + common held-out NLL per kept checkpoint
# Modes (Jupiter LOGIN node, inside tmux; every heavy step is a Slurm job):
#   ARM=relay ROWS=<final rendered jsonl> PREP_ONLY=1 bash relay_sft_arm.sh   # parquet + cache (1 node) + plan, then stop
#   ARM=relay ROWS=<final rendered jsonl> bash relay_sft_arm.sh               # the full arm
# ROWS is read once (the parquet under $D is reused after that; a new ROWS needs a new DATA_REV).
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
MARIN=/e/project1/transfernetx/lee27/code/marin-sft
PYM=/e/project1/transfernetx/lee27/code/envs/marin-grug-sft/bin/python
: "${ARM:?relay | qwen | relayaf}"
# relay = arm A (every 09-21 turn masked, render --autofix-loss none), qwen = arm B, relayaf = arm C (the relay rows
# with the autofixed 09-21 actions trained, render --autofix-loss content); marin stages relay_relay/relay_qwen/relay_relayaf
case $ARM in relay|qwen|relayaf) ;; *) echo "unknown ARM=$ARM" >&2; exit 1;; esac
STAGE=relay_$ARM
DATA_REV=${DATA_REV:-v2}
RUN=$STAGE$([ "$DATA_REV" = v2 ] || echo "_$DATA_REV")
D=${DATA_DIR:-$S/data/relay_$DATA_REV/$ARM}
EXP=$S/experiments/snowball-relay-sft
CACHE=$EXP/cache-$RUN-v1
HELDOUT=${HELDOUT:-$S/data/bespoke_v1/common_heldout/heldout-00000-of-00001.parquet}   # 20 plain Qwen traces: a sanity NLL
LR=${LR:-3e-4}
EPOCHS=${EPOCHS:-3}

# base: 09-21, the same one-time import the Bespoke arms used (qk_mult 1.75, pending_qb_betas = -router_bias)
export SNOWBALL_BASE=grug0921
export SNOWBALL_HF_BASE=/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921
export SNOWBALL_TOKENIZER=$SNOWBALL_HF_BASE
export SNOWBALL_INIT=$S/experiments/snowball-base-inits/init-dk0921-step0
# layout: 16 x 65,536 on 4 nodes, one packed sequence per GPU; the JAX memory settings that made 64k fit (09-23)
NODES=4
export SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_NODES=$NODES SNOWBALL_DEVICES=$((NODES * 4))
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async OMP_NUM_THREADS=1
export SNOWBALL_STAGE=$STAGE
export SNOWBALL_DATASET_ID=private/relay-calibforge-sft-$ARM SNOWBALL_DATASET_REVISION=final_v2   # = the stage's pins
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
export HELDOUT_MAX_LEN=$SNOWBALL_SEQ_LEN HELDOUT_MAX_MODEL_LEN=$((SNOWBALL_SEQ_LEN + 1024)) HELDOUT_MAX_POS=$((2 * SNOWBALL_SEQ_LEN))

mkdir -p "$S/logs/$RUN" "$EXP"; LOG=$S/logs/$RUN/arm.log
say() { echo "[$(date -u +%FT%TZ)] [$RUN] $*" | tee -a "$LOG"; }
[ -f "$SNOWBALL_HF_BASE/config.json" ] || { say "no 09-21 base at $SNOWBALL_HF_BASE"; exit 1; }
[ -f "$SNOWBALL_INIT/snowball_base.json" ] || { say "no 09-21 init sidecar at $SNOWBALL_INIT (run the chain's import step)"; exit 1; }
say "ARM_START lr=$LR epochs=$EPOCHS layout=${SNOWBALL_BATCH}x${SNOWBALL_SEQ_LEN} nodes=$NODES rows=${ROWS:-reuse} data=$D prep_only=${PREP_ONLY:-0}"

# 1. rows -> the one-shard parquet (login node, one process); refuses any row the chain would refuse or cut
if [ ! -f "$D/parquet.list" ]; then
  : "${ROWS:?ROWS=<final rendered jsonl> is needed to build $D}"
  mkdir -p "$D"
  OMP_NUM_THREADS=1 nice "$PYM" "$C/relay_rows_to_parquet.py" "$ROWS" "$D" | tee -a "$LOG" || { say "parquet conversion refused the rows"; rm -f "$D/parquet.list"; exit 1; }
  sha256sum "$ROWS" > "$D/rows.sha256"
fi
# 2. cache (chain prep, 1 node): the prerendered stage copies ids/loss; provenance pins the shard hash and format
if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/$STAGE/.done.prep"
  SBATCH_TIMELIMIT=01:00:00 SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"
# 3. steps per real pass over the packs
PK=$(cd "$MARIN" && PYTHONPATH=$MARIN/lib/levanter/src:$MARIN/lib/haliax/src:$MARIN/lib/rigging/src:$MARIN/lib/fray/src:$MARIN/lib/zephyr/src:$MARIN/lib/iris/src \
     nice "$PYM" "$C/pack_epoch_steps.py" "$D/train-00000-of-00001.parquet" --seq-len "$SNOWBALL_SEQ_LEN" --batch "$SNOWBALL_BATCH" 2>/dev/null | tail -1)
EPOCH_STEPS=$(echo "$PK" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pack_epoch_steps"])') || { say "pack count failed: $PK"; exit 1; }
export SNOWBALL_EPOCH_STEPS=$EPOCH_STEPS
STEPS=$((EPOCHS * EPOCH_STEPS))
WARMUP=$(( (STEPS * 5 + 50) / 100 )); [ "$WARMUP" -ge 2 ] || WARMUP=2
export SNOWBALL_WARMUP=$WARMUP
say "packs: $PK -> $EPOCH_STEPS steps per pass; $STEPS steps, warmup $WARMUP"
[ "${PREP_ONLY:-0}" = 1 ] && { say "PREP_DONE"; exit 0; }

# 4. the arm: train (chain run step via the lane; one kept checkpoint per pass), then export + held-out NLL per pass
SNOWBALL_RESUME=0 SNOWBALL_SCHEDULE_EPOCHS=$EPOCHS LANE=${LANE:-$RUN} CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS="$LR" \
EPOCHS=$EPOCHS EXPORT_EPOCHS=kept HELDOUT=$HELDOUT OUT_ROOT=$EXP/$RUN SNOWBALL_WALL=${WALL:-02:00:00} \
SCORE_TIME=${SCORE_TIME:-00:45:00} bash -l "$C/ota_lane.sh"
say "ARM_DONE"
