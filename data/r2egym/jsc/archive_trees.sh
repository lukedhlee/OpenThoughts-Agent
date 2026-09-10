#!/bin/bash
# archive_trees.sh — tar → verify → delete many-small-file trees from fscratch into the no-purge archive (one inode per tree).
# List file lines: "<name>\t<src dir>"  (src is the directory to tar; it is tarred as `-C $(dirname src) $(basename src)`).
# Verification: entry count of `tar -tf` must equal `find src | wc -l` before the source is removed. Runs 3 tars in parallel.
DST=/e/data1/mmlaion/lee27/archive_0903; LOG=/e/fscratch/reformo/lee27/experiments/archive_0903b.log; LIST=${1:?list file}
mkdir -p $DST; export DST LOG
echo "$(date '+%m-%d %H:%M') start list=$LIST" >> $LOG
one() { name=$1; src=$2; tarf=$DST/$name.tar
  [ -d "$src" ] || { echo "$(date '+%m-%d %H:%M') $name: src missing, skip" >> $LOG; return; }
  n_src=$(find "$src" -not -type s | wc -l); t0=$(date +%s)   # tar skips unix sockets (ray_logs), so exclude them from the count
  if [ -f "$tarf" ] && [ "$(tar -tf "$tarf" 2>/dev/null | wc -l)" = "$n_src" ]; then :; else
    tar -cf "$tarf.part" -C "$(dirname "$src")" "$(basename "$src")" && mv "$tarf.part" "$tarf" || { echo "$(date '+%m-%d %H:%M') $name: TAR FAILED" >> $LOG; rm -f "$tarf.part"; return; }
  fi
  n_tar=$(tar -tf "$tarf" | wc -l)
  if [ "$n_tar" = "$n_src" ]; then rm -rf "$src"; echo "$(date '+%m-%d %H:%M') $name: OK entries=$n_src bytes=$(stat -c %s "$tarf") $(( $(date +%s) - t0 ))s -> $tarf ; src removed" >> $LOG
  else echo "$(date '+%m-%d %H:%M') $name: VERIFY MISMATCH src=$n_src tar=$n_tar; src kept" >> $LOG; fi; }
export -f one
grep -v '^#' "$LIST" | grep . | tr '\t' ' ' | xargs -P 3 -L 1 bash -c 'one "$0" "$1"'
echo "$(date '+%m-%d %H:%M') done list=$LIST" >> $LOG
