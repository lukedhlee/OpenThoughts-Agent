#!/bin/bash
# Jupiter login node, after the gate's last phase: re-run once every trial the gate left unscored (sandbox-start timeouts that
# exhausted harbor's retries, plus the odd bad gateway), one job at a time, at concurrency <= 300.
# The unscored trial dirs are MOVED to resume_archive/<job>/ first (not deleted), so gate_report.py --archive still counts their
# attempts at the job's original concurrency; `harbor jobs resume` then re-runs exactly the missing trials in the same job dir,
# so the readout sees one result per trial. -f prunes nothing further (nothing unscored is left) and is passed for the record.
set -u
D=/e/fscratch/reformo/lee27/experiments/daytona_gate_v3
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
MAXC=${MAXC:-300}
export DAYTONA_API_KEY=$($V/bin/python -c '
import sys
for l in open("/e/fscratch/reformo/lee27/keys/daytona_eval.env"):
    l = l.strip()
    if l.startswith(("DAYTONA_API_KEY=", "export DAYTONA_API_KEY=")):
        print(l.split("=", 1)[1].split("#", 1)[0].strip().strip("\"\x27")); break')
cd $D
for J in jobs/oracle_shard* jobs/nop_shard*; do
  [ -d "$J" ] || continue
  name=$(basename "$J")
  mapfile -t U < <($V/bin/python - "$J" <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "result.json"))):
    try:
        r = json.load(open(f))
    except Exception:
        print(os.path.dirname(f)); continue
    if r.get("verifier_result") is None:
        print(os.path.dirname(f))
EOF
)
  echo "$(date +%T) $name: ${#U[@]} unscored trials"
  [ "${#U[@]}" -eq 0 ] && continue
  mkdir -p resume_archive/$name
  [ -f resume_archive/$name/config.json.orig ] || cp "$J/config.json" resume_archive/$name/config.json.orig
  $V/bin/python - "$J/config.json" "$MAXC" <<'EOF'
import json, sys
p, m = sys.argv[1], int(sys.argv[2])
c = json.load(open(p))
c["n_concurrent_trials"] = min(int(c["n_concurrent_trials"]), m)
json.dump(c, open(p, "w"), indent=4)
print("n_concurrent_trials ->", c["n_concurrent_trials"])
EOF
  for t in "${U[@]}"; do mv "$t" resume_archive/$name/; done
  $V/bin/harbor jobs resume -p "$J" -f EnvironmentStartTimeoutError
  echo "$(date +%T) $name: resume rc=$? results_now=$(ls $J/*/result.json 2>/dev/null | wc -l)"
done
echo "RESUME DONE $(date +%T)"
