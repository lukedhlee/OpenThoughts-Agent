#!/bin/bash
# ota3_report.sh [stage ...] — the readout table of the 2026-09-20 arms from what has landed: per arm, the OTA held-out
# NLL ladder, the paired IF-477 and RST held-out deltas vs the pair_base run (ifv2_pair.py on ifv2_rows.json), and the
# SWE-100 / TB2 trials (final_*.txt from post_run.sh, else the live summarize_tb2.py line). Run on the login node.
S=/e/data1/mmlaion/lee27/snowball-sft; C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
J=/e/data1/mmlaion/lee27/experiments/tb2_jobs; PY=$C/envs/snowball-v2/bin/python; export OMP_NUM_THREADS=1
STAGES=${*:-ota3_if rst_if ota3_if_rst ota3_if_rstsucc ota3_if_rst_fmt ota3_if_rst_beh}
BIF=$(ls -d $J/ifv2ho_base_*/ifv2_rows.json 2>/dev/null | sort | tail -1); BRST=$(ls -d $J/rstho_base_2026*/ifv2_rows.json 2>/dev/null | sort | tail -1)
echo "base refs: IF $BIF ; RST $BRST"
echo "base targets:"; for f in $E/final_otabase*swe_v01_*.txt $E/final_otabase*_v01_*.txt; do [ -f $f ] && echo "  $(basename $f .txt): $(grep -oE 'pass@1 = [0-9.]+ over [0-9]+' $f)"; done
for st in $STAGES; do
  echo "=== $st"
  for f in $(ls $S/logs/heldout_nll_$st-$st-*.json 2>/dev/null | sort -t- -k5 -V); do echo "  nll $(basename $f .json | grep -oE 'step[0-9]+'): $(grep -oE '"nll": *[0-9.]+' $f | head -1 | cut -d: -f2)"; done
  for r in $(ls -d $J/ifv2ho_${st}s[0-9]*_2026* $J/rstho_${st}s[0-9]*_2026* 2>/dev/null); do
    [ -f $r/ifv2_rows.json ] || { echo "  $(basename $r): no rows yet"; continue; }
    case $(basename $r) in ifv2ho_*) B=$BIF;; *) B=$BRST;; esac
    echo "  $(basename $r): $($PY $W/ifv2_pair.py $r/ifv2_rows.json $B 2>/dev/null | grep -E 'pass@1|paired' | tr '\n' ' ' | cut -c1-200)"
  done
  for t in 1 2 3 4; do for leg in swe_v01 _v01; do
    for d in $(ls -d $J/${st}t${t}${leg}_2026* 2>/dev/null); do n=$(basename $d)
      if [ -f $E/final_$n.txt ]; then echo "  $n FINAL: $(grep -oE 'pass@1 = [0-9.]+ over [0-9]+' $E/final_$n.txt)"
      else echo "  $n live: $($PY $W/summarize_tb2.py $d 2>/dev/null | grep -oE 'pass@1 = [0-9.]+ over [0-9]+')"; fi
    done
  done; done
done
