#!/bin/bash
# Summarize a Vista gsm8k MoE GRPO run into the experiment dir + a short local-ready note.
#
# Usage (on Vista):
#   bash hpc/skyrl_standard/vista/summarize_gsm8k_run.sh [JOB_NAME] [SLURM_JOB_ID]
#
# Writes:
#   $EXPERIMENTS_DIR/$JOB_NAME/SUMMARY.md
#   $EXPERIMENTS_DIR/$JOB_NAME/metrics_snip.txt
set -euo pipefail

: "${SCRATCH:=/scratch/11584/lukedhlee}"
: "${EXPERIMENTS_DIR:=$SCRATCH/experiments}"
: "${JOB_NAME:=vista_moe30b_gsm8k_grpo10}"
: "${WANDB_PROJECT:=vista-moe-gsm8k-grpo}"

if [[ $# -ge 1 ]]; then JOB_NAME="$1"; fi
JOB_ID="${2:-}"

JOB_DIR="$EXPERIMENTS_DIR/$JOB_NAME"
LOG_DIR="$JOB_DIR/logs"
SUMMARY="$JOB_DIR/SUMMARY.md"
SNIP="$JOB_DIR/metrics_snip.txt"

if [[ ! -d "$JOB_DIR" ]]; then
  echo "ERROR: missing $JOB_DIR"
  exit 1
fi

if [[ -z "$JOB_ID" ]]; then
  # Newest *.out under logs/
  newest=$(ls -1t "$LOG_DIR"/*_*.out 2>/dev/null | head -1 || true)
  if [[ -n "$newest" ]]; then
    JOB_ID=$(basename "$newest" | sed -E 's/.*_([0-9]+)\.out/\1/')
  fi
fi

OUT=""
ERR=""
if [[ -n "$JOB_ID" ]]; then
  OUT="$LOG_DIR/${JOB_NAME}_${JOB_ID}.out"
  # some templates only write .out; keep optional .err
  ERR="$LOG_DIR/${JOB_NAME}_${JOB_ID}.err"
fi

STATE="unknown"
ELAPSED="unknown"
EXIT_CODE="unknown"
if [[ -n "$JOB_ID" ]] && command -v sacct >/dev/null 2>&1; then
  read -r STATE EXIT_CODE ELAPSED <<<"$(sacct -j "$JOB_ID" -n -X -o State,ExitCode,Elapsed | head -1)"
fi

# Pull reward / accuracy lines from logs (SkyRL + WandB echo patterns).
: > "$SNIP"
for f in "$OUT" "$ERR"; do
  [[ -f "$f" ]] || continue
  {
    echo "===== $(basename "$f") ====="
    grep -Ei \
      'reward|accuracy|eval/|train/|mean_reward|ExactMatch|wandb|error|traceback|OOM|NCCL|step [0-9]+' \
      "$f" | tail -n 200 || true
  } >> "$SNIP"
done

WANDB_HINT="project=${WANDB_PROJECT} run_name=${JOB_NAME}"
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_HINT="https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}/runs (filter name=${JOB_NAME}) entity=${WANDB_ENTITY}"
fi

TS="$(date -Is)"
cat > "$SUMMARY" <<EOF
# ${JOB_NAME} — gsm8k MoE GRPO smoke

- **When summarized:** ${TS}
- **SLURM job:** ${JOB_ID:-unknown}
- **State / exit / elapsed:** ${STATE} / ${EXIT_CODE} / ${ELAPSED}
- **Cluster artifacts:** \`${JOB_DIR}\`
- **WandB:** ${WANDB_HINT}
- **Model:** Qwen/Qwen3-30B-A3B (8× GH200, EP=4×FSDP=2, max_steps=10, verifier-only)

## Verdict checklist
- [ ] Job reached train loop (not env/conda fail)
- [ ] Eval @ 0 / 5 / 10 logged
- [ ] **Verifier** mean reward or accuracy moved (not just shaped reward)
- [ ] Ready for pymethods2test agentic smoke

## Metrics snip
See \`metrics_snip.txt\` in this directory (grep of reward/accuracy/errors).

## Next
If green → tiny Harbor + Daytona on \`DCAgent/exp_rpt_pymethods2test-large\` (or a local few-task subset).
EOF

echo "Wrote $SUMMARY"
echo "Wrote $SNIP"
echo "--- SUMMARY ---"
cat "$SUMMARY"
