#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --output=/scratch/11584/lukedhlee/experiments/rl/exports/logs/%x_%j.out
#SBATCH --job-name=snowball_export_hf
#SBATCH -p debug
#SBATCH --account CCR24067
#SBATCH --gres=gpu:4
# Horizon port of Jupiter's code/snowball/export_hf.sbatch (by diff): paths, no modules, XALT unset; BASE = the arm's own
# start checkpoint (config/tokenizer/chat template of the 09-21 relay SFT export), not the Stage-3 base.
# export_hf.sbatch — offline FSDP2 -> HF consolidation of a Snowball GRPO checkpoint (CPU only, one node, ~858 GB host RAM).
# Usage: RUN=<run name> STEP=<N> BASE=<start export> sbatch --export=ALL,RUN=...,STEP=...,BASE=... export_hf.sbatch
#   -> /scratch/11584/lukedhlee/experiments/rl/exports/<RUN>_step<N>/model  (config/tokenizer copied from BASE)
# Route A of runbooks/rl-training.md § "FSDP -> HF export". Runs --inspect first; aborts before writing on any layout it cannot reassemble.
set -euo pipefail
unset LD_PRELOAD   # TACC XALT pollutes command output
export CC=gcc CXX=g++ OMP_NUM_THREADS=16
PY=$HOME/snowball/envs/snowball/bin/python
SCRIPT=$HOME/snowball/marinskyrl-marin/skyrl-train/scripts/consolidate_sharded_checkpoint.py
: "${BASE:?set BASE (the arm's start export)}"
E=/scratch/11584/lukedhlee/experiments/rl
: "${RUN:?set RUN}"; : "${STEP:?set STEP}"
SRC=${SRC:-$E/$RUN/$RUN/checkpoints/global_step_$STEP/policy}
DST=${DST:-$E/exports/${RUN}_step${STEP}/model}
echo "$(date) RUN=$RUN STEP=$STEP SRC=$SRC DST=$DST host=$(hostname)"; ls "$SRC" | head -5; ls "$SRC" | wc -l
du -sh "$SRC" 2>/dev/null | tail -1
echo "=== inspect ==="; $PY "$SCRIPT" --src "$SRC" --inspect
echo "=== consolidate ==="; mkdir -p "$(dirname "$DST")"; ( time $PY "$SCRIPT" --src "$SRC" --dst "$DST" --aux-from "$BASE" ) 2>&1 | tail -60
echo "=== verify ==="; ls "$DST" | wc -l; du -sh "$DST" | tail -1
$PY - "$DST" "$BASE" <<'PY'
import json, os, sys
dst, base = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
files = set(idx["weight_map"].values()); missing = [f for f in files if not os.path.exists(os.path.join(dst, f))]
print("index tensors:", len(idx["weight_map"]), "shard files:", len(files), "missing files:", missing)
cfg = json.load(open(os.path.join(dst, "config.json"))); bcfg = json.load(open(os.path.join(base, "config.json")))
print("model_type:", cfg.get("model_type"), "architectures:", cfg.get("architectures"), "max_position_embeddings:", cfg.get("max_position_embeddings"), "tie:", cfg.get("tie_word_embeddings"))
diff = {k: (bcfg.get(k), cfg.get(k)) for k in set(bcfg) | set(cfg) if bcfg.get(k) != cfg.get(k)}
print("config diffs vs base:", diff)
try:
    bidx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    a, b = set(idx["weight_map"]), set(bidx["weight_map"]); print("tensor-name set equal to base:", a == b, "only-export:", sorted(a - b)[:5], "only-base:", sorted(b - a)[:5])
except Exception as e: print("base index compare skipped:", e)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    print(f, os.path.exists(os.path.join(dst, f)))
PY
echo "$(date) done"
