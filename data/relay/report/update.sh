#!/usr/bin/env bash
# update.sh — regenerate the teacher-relay status gist and push it.
#   1. pull the run readouts and the verify-note replay from Jupiter (read-only: tar/scp of small files, plus
#      live_status.py and, once, arm_quality.py on the login node, which only read) into $WORK/cache;
#   2. draw the figures (status_figs.py figs);
#   3. push them to the secret figures gist, then write README.md with raw URLs pinned to that commit and push it to
#      the secret text gist.
# It never touches a job, a tmux session or a worktree.   Usage: bash update.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/Users/lukedhlee/miniforge3/bin/python3}
WORK=${WORK:-/private/tmp/claude-503/-Users-lukedhlee-OpenThoughts-Agent/5368fd8c-b9cb-4b09-8f5d-7153da0124c4/scratchpad/relay_gist}
FIG_GIST=${FIG_GIST:-e4f7a2c39bff4723a2738cd8ee1a3bcd}   # figures: Teacher relay status (secret)
TXT_GIST=${TXT_GIST:-d8fc1454ed134eff585448c4982458c9}   # Teacher relay: status (secret)
OWNER=lukedhlee
R=/e/fscratch/reformo/lee27/experiments/relay/pilot/runs
V=/e/fscratch/reformo/lee27/experiments/relay/verify_note
JPY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python   # has `tokenizers` (arm_quality.py); no torch
CACHE=$WORK/cache; OUT=$WORK/out
mkdir -p "$CACHE/runs" "$CACHE/verify_note" "$OUT"

echo "[1/3] pull readouts"
ssh -o BatchMode=yes jupiter "cd $R && tar czf - \$(ls -d */readout*.json */readout*.txt */check_decision*.json \
  */decide*.json */full_decision.json */run.meta */ABORT */SUPERSEDED *.plan.json *.launch.log 2>/dev/null)" \
  | tar xzf - -C "$CACHE/runs"
for f in classified.jsonl replies.jsonl; do   # the replay is finished; fetch once
  [ -s "$CACHE/verify_note/$f" ] || scp -q "jupiter:$V/$f" "$CACHE/verify_note/$f"
done
if [ ! -s "$CACHE/arm_quality.jsonl" ]; then   # the final SFT arms are fixed; per-row facts once (~1 min, 6 procs)
  ssh -o BatchMode=yes jupiter "OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false nice $JPY -" \
    < "$HERE/arm_quality.py" > "$CACHE/arm_quality.jsonl.tmp"
  mv "$CACHE/arm_quality.jsonl.tmp" "$CACHE/arm_quality.jsonl"
fi
ssh -o BatchMode=yes jupiter python3 - < "$HERE/live_status.py" > "$CACHE/live_status.json.tmp"
mv "$CACHE/live_status.json.tmp" "$CACHE/live_status.json"

echo "[2/3] figures"
"$PY" "$HERE/status_figs.py" figs --cache "$CACHE" --out "$OUT"

push() {   # push <gist id> <files...>
  local g=$1; shift; local d=$WORK/gist_$g
  [ -d "$d/.git" ] || git clone -q "https://gist.github.com/$g.git" "$d"
  git -C "$d" pull -q --rebase
  cp "$@" "$d/"
  git -C "$d" add -A
  git -C "$d" diff --cached --quiet || git -C "$d" commit -qm "update $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M') PT"
  git -C "$d" push -q
}

echo "[3/3] push"
push "$FIG_GIST" "$OUT"/fig*.png
SHA=$(git -C "$WORK/gist_$FIG_GIST" rev-parse HEAD)
"$PY" "$HERE/status_figs.py" readme --cache "$CACHE" --out "$OUT" \
  --fig-base "https://gist.githubusercontent.com/$OWNER/$FIG_GIST/raw/$SHA/"
push "$TXT_GIST" "$OUT/README.md"
echo "https://gist.github.com/$OWNER/$TXT_GIST"
