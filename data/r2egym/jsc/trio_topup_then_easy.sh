#!/bin/bash
# trio_topup_then_easy.sh — sequencer for the history-think probes after the 18:30 PT bridge outage (2026-09-06).
# 1. wait for the r2egym trio to table; read it out (strict min-scored 8 and relaxed 6).
# 2. top-up: every val441 task that any arm left with < 8 scored attempts goes into a small tree; three top-up probes
#    (keep/drop/last:2) run on it on the EXISTING 768 seats; merged read-out (base + top-up per arm).
# 3. (removed) the six easy probes: cancelled after the adversarial audits disqualified all three sources.
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; T=/e/fscratch/reformo/lee27/tasks
O=$E/hist_readouts; A=$E/easy3_audit; X=$E; mkdir -p $O; export OMP_NUM_THREADS=1
SPL="idval=$X/tt_v2_idval.txt,oodval=$X/tt_v2_oodval.txt,heldout=$X/tt_v2_heldout.txt"
# settled = probe_watch has tabled ($E/<p>/pass8_summary.json) or the job is gone, AND the trace tree is either archived
# (trace_archive.tar present, trace_jobs removed) or still a tree with the job gone for >= 15 min (tar failed or never ran).
settled() { for p in "$@"; do
  d=$E/$p/$p; job=$(squeue -u $USER -h -n $p -o %T)
  { [ -f $E/$p/pass8_summary.json ] || [ -z "$job" ]; } || return 1
  if [ -f $d/trace_archive.tar ] && [ ! -d $d/trace_jobs ]; then continue; fi
  if [ -z "$job" ] && [ -d $d/trace_jobs ]; then m=$(( $(date +%s) - $(stat -c %Y $d/trace_jobs) )); [ $m -ge 900 ] && [ ! -f $d/trace_archive.tar ] && continue; fi
  return 1
done; return 0; }
log() { echo "$(date) $*" >> $O/log; }
log "sequencer start: waiting for the r2egym trio"
until settled snowball_hist_keep_base snowball_hist_drop_base snowball_hist_last2_base; do sleep 60; done
log "trio tabled: read-outs"
python3 $C/hist_readout.py --probes keep=snowball_hist_keep_base drop=snowball_hist_drop_base last2=snowball_hist_last2_base --splits $SPL --out $O/hist_r2egym >> $O/log 2>&1
python3 $C/hist_readout.py --probes keep=snowball_hist_keep_base drop=snowball_hist_drop_base last2=snowball_hist_last2_base --splits $SPL --min-scored 6 --out $O/hist_r2egym_min6 >> $O/log 2>&1
# top-up list: tasks under-sampled in any arm (from the attempts jsonl), plus tasks absent from every arm
python3 - $O/hist_r2egym_attempts.jsonl $T/r2egym-tt-v2-val441 $O/topup.txt <<'PY' >> $O/log 2>&1
import json, sys, os, collections
att, tree, out = sys.argv[1:4]
scored = collections.defaultdict(lambda: collections.Counter())
for l in open(att):
    r = json.loads(l)
    if r.get("reward") is not None: scored[r["probe"]][r["task"]] += 1
tasks = sorted(os.listdir(tree)); arms = sorted(scored) or ["keep", "drop", "last2"]
need = [t for t in tasks if any(scored[a][t] < 8 for a in arms)]
open(out, "w").write("\n".join(need) + ("\n" if need else ""))
print(f"top-up: {len(need)} of {len(tasks)} tasks under-sampled in some arm; per arm under-sampled:", {a: sum(1 for t in tasks if scored[a][t] < 8) for a in arms})
PY
n=$(grep -c . $O/topup.txt 2>/dev/null || echo 0)
MERGE_K=snowball_hist_keep_base; MERGE_D=snowball_hist_drop_base; MERGE_L=snowball_hist_last2_base
if [ "$n" -gt 0 ]; then
  TT=$T/r2egym-tt-v2-topup; rm -rf $TT; mkdir -p $TT
  while read t; do [ -n "$t" ] && cp -r $T/r2egym-tt-v2-val441/$t $TT/; done < $O/topup.txt
  log "top-up tree: $(ls $TT | wc -l) tasks; launching the three top-up probes on the existing seats"
  for m in keep drop last:2; do nm=${m/last:2/last2}; bash $C/probe_history.sh snowball_hist_${nm}_topup $TT base $m >> $O/log 2>&1; done
  sleep 180
  until settled snowball_hist_keep_topup snowball_hist_drop_topup snowball_hist_last2_topup; do sleep 60; done
  log "top-ups tabled: merged read-out"
  MERGE_K=snowball_hist_keep_base+snowball_hist_keep_topup; MERGE_D=snowball_hist_drop_base+snowball_hist_drop_topup; MERGE_L=snowball_hist_last2_base+snowball_hist_last2_topup
  python3 $C/hist_readout.py --probes keep=$MERGE_K drop=$MERGE_D last2=$MERGE_L --splits $SPL --out $O/hist_r2egym_merged >> $O/log 2>&1
fi
# easy-source probes cancelled by Luke 2026-09-06 21:10 PT: the audits disqualify all three sources, no GPU hours on them.
log ALL_DONE
