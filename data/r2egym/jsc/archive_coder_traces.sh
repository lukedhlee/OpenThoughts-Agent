#!/bin/bash
# Archive the concluded Coder-30B campaign trace trees (already compacted) as one tar each on data1, then remove the tree.
# Tables, logs, configs, checkpoints of those experiments stay in place. Log: experiments/archive_coder.log
A=/e/data1/mmlaion/lee27/archive_0903; E=/e/fscratch/reformo/lee27/experiments; LOG=$E/archive_coder.log; echo "$(date) start" >> $LOG
for X in lr1e7v5 lr3e7v5 lr1e6v5 band_full_s0 band_full_s1 band_full_s2 band_full_s3 band_resid_s0 band_resid_s1 band_resid_s2 band_resid_s3 band_r3_s0 band_r3_s1 band_r3_s2 band_r3_s3; do
  T=$E/$X/$X/trace_jobs; [ -d "$T" ] || { echo "skip $X (no trace_jobs)" >> $LOG; continue; }
  n=$(find $T -type f | wc -l); s=$(du -sh $T | cut -f1)
  if tar -cf $A/${X}_trace_jobs.tar -C $E/$X/$X trace_jobs 2>>$LOG; then rm -rf $T; echo "$(date) archived $X: $s, $n files -> $A/${X}_trace_jobs.tar; tree removed" >> $LOG
  else echo "$(date) FAILED tar for $X; tree kept" >> $LOG; fi
done
echo "$(date) done" >> $LOG
