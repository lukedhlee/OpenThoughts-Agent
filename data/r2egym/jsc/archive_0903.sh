#!/bin/bash
# Archive concluded 2026-09-03 probe trace trees the probe_watch way: tar in place (one inode), verify, delete the tree.
E=/e/fscratch/reformo/lee27/experiments; LOG=$E/archive_0903.log
for name in probe2_c128 probe2_c128x16co probe2_c16 probe2_c64 probe4_ta_c128 snowball_probe_val_b60k_p0 snowball_probe_val_b60k_p1 snowball_probe_val_b60k_p2 snowball_probe_val_b60k_p3 swesmith_sample6; do
  d=$E/$name/$name; TJ=$d/trace_jobs
  [ -d "$TJ" ] || { echo "$(date) $name: no trace_jobs" >> $LOG; continue; }
  [ -f "$d/trace_archive.tar" ] && { echo "$(date) $name: archive exists, skipped" >> $LOG; continue; }
  if nice tar -cf "$d/trace_archive.tar" -C "$d" trace_jobs && tar -tf "$d/trace_archive.tar" > /dev/null; then rm -rf "$TJ"; echo "$(date) $name: archived + deleted trace_jobs ($(du -h "$d/trace_archive.tar" | cut -f1))" >> $LOG
  else echo "$(date) $name: tar FAILED, tree kept" >> $LOG; rm -f "$d/trace_archive.tar"; fi
done
echo "$(date) ARCHIVE_0903_DONE" >> $LOG
