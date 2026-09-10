#!/bin/bash
# llm_ts2.sh <trial>... — ON the batch host: per LLM-timeout trial, the last completed agent step's timestamp (UTC, from
# agent/trajectory.json) and result.json finished_at; the failing request was issued right after that step's observation.
T=/tmp/otagent-artifact-stores/snowball_ttband_lr5e7_kl01_fulldist_a-d8f52925f261/trace_jobs
echo "=== peer-port histogram of ESTAB sockets ==="; ss -tan state established | awk 'NR>1{split($4,a,":"); print a[length(a)]}' | sort | uniq -c | sort -rn | head -6
echo "=== proxy env in coordinator procs ==="; for p in $(pgrep -f RolloutCoordinator | head -2); do tr '\0' '\n' < /proc/$p/environ | grep -i -E "^(https?_proxy|no_proxy)=" | cut -c1-120; done | sort -u
echo "=== last-step ts (UTC) / finished_at ==="
python3 - "$T" "$@" <<'PY'
import json, sys, os
T = sys.argv[1]
for t in sys.argv[2:]:
    d = f"{T}/{t}/attempts/000"
    try:
        r = json.load(open(f"{d}/result.json")); fin = r.get("finished_at")
        tr = json.load(open(f"{d}/agent/trajectory.json")); steps = tr.get("steps") or []
        last = steps[-1]["timestamp"] if steps else None
        print(t, len(steps), last, fin)
    except Exception as e:
        print(t, "ERR", str(e)[:80])
PY
