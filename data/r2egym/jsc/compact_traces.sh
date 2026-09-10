#!/bin/bash
# compact_traces.sh <trace_jobs dir>  — shrink a FINISHED probe/pool-shard trace tree ~5x without losing analysis inputs.
# Removes the two byte-identical copies of each attempt result (attempts/*/lifecycle-result.json and the trial-level
# result.json), rewrites attempts/*/result.json without indentation (the token-id arrays are ~3/4 whitespace), and
# truncates terminus_2.pane files above 2 MB to their last 2 MB. Keeps trajectory.json, verifier/, config.json, trial.log.
# Only run on a tree no job is writing to (shard tabled + scancelled). Idempotent.
D=$1; [ -d "$D/eval_sessions" ] || { echo "no eval_sessions under $D"; exit 1; }
before=$(du -s "$D" | cut -f1)
find "$D/eval_sessions" -mindepth 5 -maxdepth 5 -path "*/attempts/*/lifecycle-result.json" -delete
find "$D/eval_sessions" -mindepth 3 -maxdepth 3 -name result.json -delete
find "$D/eval_sessions" -mindepth 5 -maxdepth 5 -path "*/attempts/*/result.json" -size +1M -print0 \
  | xargs -0 -P 8 -n 50 python3 /e/project1/transfernetx/lee27/code/snowball/compact_json.py
find "$D/eval_sessions" -name terminus_2.pane -size +2M -exec sh -c 'tail -c 2097152 "$1" > "$1.tmp" && mv "$1.tmp" "$1"' _ {} \;
after=$(du -s "$D" | cut -f1)
echo "$(date) compacted $D: $((before/1024/1024)) GB -> $((after/1024/1024)) GB"
