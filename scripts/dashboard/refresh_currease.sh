#!/usr/bin/env bash
# Refresh the Curriculum-Easy LR Sweep Runboard end-to-end.
#
# The SkyRL trainer logs wandb OFFLINE on Jupiter (WANDB_MODE=offline, one run per
# Slurm attempt under each experiment dir), so NOTHING reaches the cloud project
# without this script. It rsyncs the offline runs off the cluster, `wandb sync`s the
# ones that actually grew since last time, then re-pulls and rebuilds the dashboard.
#
# Re-syncing an offline run re-uploads it under the same run id, so live attempts
# simply gain their new steps. A size manifest keeps finished attempts from being
# re-uploaded on every refresh (~8s each, 20+ of them).
#
# Usage: scripts/dashboard/refresh_currease.sh [--open]
set -euo pipefail

REPO=/Users/lukedhlee/OpenThoughts-Agent
PY=/Users/lukedhlee/miniforge3/bin/python
WANDB=/Users/lukedhlee/miniforge3/bin/wandb
WORK=${RUNBOARD_WORK:-$HOME/.cache/runboard/currease}
REMOTE=jupiter
F=/e/fscratch/reformo/lee27
PROJECT=jupiter-currease30b-grpo
ENTITY=lukeleeai
SPEC="$REPO/scripts/dashboard/specs/currease30b_grpo_arms.json"
ARMS=(currease30b_grpo_instr2507 currease30b_grpo_instr_lr1e6 currease30b_grpo_instr_lr5e6)

echo "==> rsync offline runs from $REMOTE"
mkdir -p "$WORK"
for a in "${ARMS[@]}"; do
  mkdir -p "$WORK/$a"
  # fchmodat "Operation not permitted" is cosmetic GPFS noise; --no-perms silences it
  rsync -a --no-perms --no-owner --no-group \
    "$REMOTE:$F/experiments/$a/wandb/wandb/" "$WORK/$a/" 2>/dev/null || \
    echo "    WARN: rsync failed for $a (cluster unreachable?) — using cached copy"
done

echo "==> wandb sync (chronological: created_at ordering is what pull_wandb stitches by)"
MANIFEST="$WORK/.synced_sizes"
touch "$MANIFEST"
cd "$WORK"
n_sync=0
# sort by the offline-run-<timestamp> component so new cloud runs are created in
# attempt order — pull_wandb.py resolves per-step collisions latest-attempt-wins.
for d in $(ls -d */offline-run-* 2>/dev/null | sort -t/ -k2); do
  wf=$(ls "$d"/*.wandb 2>/dev/null | head -1 || true)
  [ -n "$wf" ] || continue
  size=$(stat -f%z "$wf")
  prev=$(awk -v k="$d" '$1==k {print $2}' "$MANIFEST" | tail -1)
  [ "$size" = "${prev:-}" ] && continue
  rm -f "$d"/*.synced
  if "$WANDB" sync --project "$PROJECT" --entity "$ENTITY" "$d" >/dev/null 2>&1; then
    awk -v k="$d" '$1!=k' "$MANIFEST" > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
    echo "$d $size" >> "$MANIFEST"
    echo "    synced $d"
    n_sync=$((n_sync + 1))
  else
    echo "    WARN: sync failed for $d"
  fi
done
echo "    $n_sync run(s) uploaded"

echo "==> pull + build"
"$PY" "$REPO/scripts/dashboard/pull_wandb.py" "$SPEC"
"$PY" "$REPO/scripts/dashboard/build.py"

[ "${1:-}" = "--open" ] && open "$REPO/scripts/dashboard/dist/dashboard.html"
exit 0
