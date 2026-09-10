#!/bin/bash
# llm_ts.sh <trial>... — run ON the batch host: model endpoint URLs from one trial's config.json, a sample of trial.log's line
# format, then per trial the first timeout line's timestamp (for the sync-window attribution of LLM ConnectTimeouts).
T=/tmp/otagent-artifact-stores/snowball_ttband_lr5e7_kl01_fulldist_a-d8f52925f261/trace_jobs
python3 - "$T/$1/attempts/000/config.json" <<'PY'
import json, re, sys
s = json.dumps(json.load(open(sys.argv[1])))
print("urls:", sorted(set(re.findall(r'https?://[^"\\ ]+', s)))[:8])
PY
echo "=== trial.log sample ==="; sed -n 1,2p $T/$1/attempts/000/trial.log | cut -c1-160; grep -m1 -E "ConnectTimeout|timed out" $T/$1/attempts/000/trial.log | cut -c1-200
echo "=== per-trial first timeout timestamp ==="
for t in "$@"; do f=$T/$t/attempts/000/trial.log; [ -f $f ] || { echo "$t nolog"; continue; }
  ts=$(grep -m1 -E "ConnectTimeout|LLM request timed out|APITimeoutError|Request timed out" $f | grep -o -E "20[0-9]{2}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
  echo "$t ${ts:-nots}"
done
