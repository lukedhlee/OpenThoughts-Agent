#!/bin/bash
# build_allowlist.sh [max_verifier_sec=1000] — pull the gate parts from JURECA (via Jupiter's /p/scratch view), build the
# allowlist + attrition report on the Mac, ship allowlist to Jupiter tasks/. PARTS_TAGS selects the parts dirs (default: the
# devel sweeps; use PARTS_TAGS=tt_fleet for the offline fleet gate).
set -e
S=$(cd "$(dirname "$0")" && pwd); R=/Users/lukedhlee/OpenThoughts-Agent; MAXV=${1:-1000}
TAGS=${PARTS_TAGS:-"tt_sample_devel tt_full_a tt_full_b"}
rm -rf $S/gate_parts; mkdir -p $S/gate_parts
DIRS=""; PARTS=""; for t in $TAGS; do DIRS="$DIRS envgate_parts_$t"; PARTS="$PARTS $S/gate_parts/envgate_parts_$t"; done
ssh jupiter "cd /p/scratch/synthlaion/lee27 && tar -cf - $DIRS" | tar -xf - -C $S/gate_parts
/Users/lukedhlee/miniforge3/bin/python3 $R/data/r2egym/jsc/tt_allowlist.py --map $R/data/r2egym/overlap/tasktrove_v3_upstream_map.tsv \
  --pairs $S/tt_shared_pairs.txt --parts $PARTS \
  --exclude $S/tt_empty_issue_tasks.txt --max-verifier-sec $MAXV --out $S/allowlist_r2egym_tt_v1.txt --report $S/allowlist_r2egym_tt_v1_report.md
scp -q $S/allowlist_r2egym_tt_v1.txt jupiter:/e/fscratch/reformo/lee27/tasks/allowlist_r2egym_tt_v1.txt
echo "shipped: $(wc -l < $S/allowlist_r2egym_tt_v1.txt) tasks (parts: $TAGS)"
