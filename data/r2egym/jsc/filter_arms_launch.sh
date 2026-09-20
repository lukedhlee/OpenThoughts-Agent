#!/bin/bash
# filter_arms_launch.sh — the two 2026-09-20 filter arms on the OTA2+RST+IF mix, end to end, on the Jupiter login node
# (tmux `filter_arms`): build both filtered corpora with data/rst/filter_corpus.py (CPU, streaming), then launch
# STAGE=ota3_if_rst_fmt and STAGE=ota3_if_rst_beh through ota3_arm.sh (cache prep + run, 16 nodes, staggered 7 min)
# and their readout watchers (ota3_readout.sh). Cost: 2 x (1 prep + ~23 run + ~13 readout) ≈ 74 node-h.
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft; C=/e/project1/transfernetx/lee27/code; W=$C/tb2
PY=$C/envs/snowball-v2/bin/python; LOG=$S/logs/rst/filter_arms.log; mkdir -p $S/logs/rst
say() { echo "[$(date -u +%FT%TZ)] [filter_arms] $*" | tee -a $LOG; }
SRC=$S/data/ota3_if_rst_v1
if [ -f $S/data/ota3_if_rst_fmt_v1/parquet.list ] && [ -f $S/data/ota3_if_rst_beh_v1/parquet.list ]; then say "both corpora exist"; else
  say "building ota3_if_rst_fmt_v1 + ota3_if_rst_beh_v1 in one pass (filter_corpus.py asserts 81,889 / 83,851 rows)"
  OMP_NUM_THREADS=1 $PY $C/snowball/rst/filter_corpus.py --in $SRC --out-root $S/data >> $LOG 2>&1 || { say "filter_corpus.py failed (see $LOG)"; exit 1; }
  for rule in fmt beh; do say "$rule: $(head -c 300 $S/data/ota3_if_rst_${rule}_v1/census.json | tr '\n' ' ')"; done
fi
for rule in fmt beh; do
  st=ota3_if_rst_$rule
  tmux new-session -d -s arm_$st "STAGE=$st bash -l $C/snowball/ota3_arm.sh; sleep 3600"
  tmux new-session -d -s readout_$st "sleep 900; bash $W/ota3_readout.sh $st 2; sleep 600"
  say "launched arm_$st + readout_$st"
  sleep 420
done
say "FILTER_ARMS_LAUNCHED"
