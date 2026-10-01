#!/bin/bash
# build_band_tree.sh <band.txt> <source tree> <dest tree> [copies=16] — the RL train tree for a learnable band, as
# Jupiter's screen_..._learnable_x16 trees: <task>, <task>__r1 ... <task>__r15, each a symlink to the source task dir
# (environment/ stays byte-identical, so the Daytona snapshot hash is unchanged). x16 keeps one dataloader epoch long
# enough that the fully-async trainer never crosses an epoch boundary inside a 30-step arm.
set -euo pipefail
BAND=${1:?band.txt}; SRC=${2:?source tree}; DST=${3:?dest tree}; N=${4:-16}
[ -e "$DST" ] && { echo "$DST exists"; exit 1; }
mkdir -p "$DST.tmp"
while read -r t; do
  [ -z "$t" ] && continue
  tgt=$(readlink -f "$SRC/$t"); [ -f "$tgt/task.toml" ] || { echo "no task $t under $SRC"; exit 1; }
  ln -s "$tgt" "$DST.tmp/$t"
  for k in $(seq 1 $((N - 1))); do ln -s "$tgt" "$DST.tmp/${t}__r$k"; done
done < "$BAND"
mv "$DST.tmp" "$DST"
echo "$DST: $(grep -c . "$BAND") tasks x $N = $(ls "$DST" | wc -l) entries"
