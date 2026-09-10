#!/bin/bash
E=/e/fscratch/reformo/lee27/experiments
line=""
for p in snowball_v2rest_base_fixed snowball_v2rest_fulldist_s12 snowball_v2rest_fulldist_s24 snowball_v2val_fulldist_s36; do
  l=$(ls -t $E/$p/logs/${p}_*.out 2>/dev/null | head -1)
  c=0; [ -n "$l" ] && c=$(grep -c "Batch generation complete" "$l")
  st=$(squeue -h -u $USER -n $p -o %T | head -1)
  line="$line ${p#snowball_v2}=${st:-GONE}:${c}"
done
b=$(curl -s -m 5 localhost:9922/status | python3 -c 'import sys,json;d=json.load(sys.stdin);print("ready",d["envs"]["ready"],"jobs",d["active_jobs"],"err",d["stats"]["jobs_errors"])' 2>/dev/null)
echo "$line | $b"
