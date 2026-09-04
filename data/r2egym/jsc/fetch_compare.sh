#!/bin/bash
# fetch_compare.sh [out_dir] — pull the pass@8 tables of the tt60k (JSC apptainer) and ttd60k (Daytona) probe shards from Jupiter
# (written by pool_watch/pass8_table.py once a shard's eval block lands) and run compare_pass8.py on whatever is present.
set -e; OUT=${1:-/tmp/tt_compare}; mkdir -p $OUT/jsc $OUT/daytona; R=$(cd "$(dirname "$0")/.." && pwd)
for p in tt60k ttd60k; do
  d=jsc; [ $p = ttd60k ] && d=daytona
  ssh -o ConnectTimeout=25 jupiter "ls /e/fscratch/reformo/lee27/experiments/${p}_s*/pass8_pass8_table.csv 2>/dev/null" | while read -r f; do scp -q "jupiter:$f" "$OUT/$d/$(basename $(dirname $f))_pass8_table.csv"; done
  echo "$p: $(ls $OUT/$d/*.csv 2>/dev/null | wc -l) shard tables"
done
ls $OUT/jsc/*.csv >/dev/null 2>&1 && ls $OUT/daytona/*.csv >/dev/null 2>&1 || { echo "need at least one table on each side"; exit 1; }
PY=${PY:-/Users/lukedhlee/miniforge3/bin/python3}
$PY $R/compare_pass8.py --map $R/overlap/tasktrove_v3_upstream_map.tsv --jsc $OUT/jsc/*.csv --daytona $OUT/daytona/*.csv --out $OUT/compare_pass8.md
echo "report: $OUT/compare_pass8.md"
