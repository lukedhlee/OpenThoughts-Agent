#!/bin/bash
# migrate_runs.sh <list file> [parallel=2] — move finished run dirs from fscratch to the no-purge mmlaion pool and leave a
# symlink at the old path so every script that hard-codes E=/e/fscratch/reformo/lee27/experiments keeps resolving.
# Luke 2026-09-06 ("yes let's use mmlaion"): fscratch (shared 44 TB reformo pool) hit 97 % with three arms live.
# Per run: refuse if a job of that name is in squeue or the path is already a symlink; rsync -aHS (sparse-aware: the 1 TB
# artifact_store.img is sparse) src -> dst; verify with a dry-run itemize (nothing but directory-time entries may remain) and
# an apparent-size + file-count match; only then rm -rf src and ln -s dst src. Log: $DST/_migrate.log. Derived from archive_trees.sh.
set -u
E=/e/fscratch/reformo/lee27/experiments; DST=/e/data1/mmlaion/lee27/experiments; LIST=${1:?list file}; PAR=${2:-2}
mkdir -p $DST; LOG=$DST/_migrate.log; export E DST LOG
echo "$(date '+%m-%d %H:%M') start list=$LIST par=$PAR" >> $LOG
one() { r=$1; src=$E/$r; dst=$DST/$r
  [ -L "$src" ] && { echo "$(date '+%m-%d %H:%M') $r: already a symlink, skip" >> $LOG; return; }
  [ -d "$src" ] || { echo "$(date '+%m-%d %H:%M') $r: src missing, skip" >> $LOG; return; }
  squeue -h -u $USER -n "$r" -o %i | grep -q . && { echo "$(date '+%m-%d %H:%M') $r: RUNNING in squeue, skip" >> $LOG; return; }
  t0=$(date +%s)
  # cp --sparse=always streams at ~5 GB/s on the login node (70 GB in 14 s, 2026-09-06); rsync -S managed ~125 MB/s per
  # process because its block-wise seek+write pattern is hostile to GPFS. rsync is still used below as the verifier.
  mkdir -p "$dst"
  cp -a --sparse=always "$src/." "$dst/" > "$dst.cp.log" 2>&1 || { echo "$(date '+%m-%d %H:%M') $r: CP FAILED (see $dst.cp.log)" >> $LOG; return; }
  left=$(rsync -aHS --dry-run --itemize-changes "$src/" "$dst/" 2>/dev/null | grep -v -E '^\.d\.\.t' | head -3)
  n_src=$(find "$src" -type f | wc -l); n_dst=$(find "$dst" -type f | wc -l)
  b_src=$(du -sb --apparent-size "$src" | cut -f1); b_dst=$(du -sb --apparent-size "$dst" | cut -f1)
  if [ -n "$left" ] || [ "$n_src" != "$n_dst" ] || [ "$b_src" != "$b_dst" ]; then
    echo "$(date '+%m-%d %H:%M') $r: VERIFY FAILED files $n_src/$n_dst bytes $b_src/$b_dst left='$left' - source kept" >> $LOG; return; fi
  real=$(du -sh "$src" | cut -f1); real_dst=$(du -sh "$dst" | cut -f1)   # real_dst == real means the sparse image stayed sparse
  rm -rf "$src" && ln -s "$dst" "$src" && echo "$(date '+%m-%d %H:%M') $r: moved $real (dst real $real_dst, $n_src files) in $(( $(date +%s) - t0 )) s -> symlink" >> $LOG
}
export -f one
grep -v -E '^\s*(#|$)' "$LIST" | xargs -P "$PAR" -I{} bash -c 'one {}'
echo "$(date '+%m-%d %H:%M') done list=$LIST; fscratch now: $(df -h /e/fscratch | tail -1 | awk '{print $3" used, "$5}')" >> $LOG
