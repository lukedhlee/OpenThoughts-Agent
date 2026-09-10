#!/bin/bash
# after_arm3.sh — wait for arm 3 (1671772) step-30 checkpoint (export_probe chain log), prune arm 3, wait for its JUWELS seats to
# free, submit arm 6 (snowball_ttband_lr5e7_kl05_a, KL 0.05, cap, timeouts, bridge 9926). Session 20ae8169, 2026-09-05 02:15 PT.
export OMP_NUM_THREADS=1; E=/e/fscratch/reformo/lee27/experiments; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
L=$E/export_probe_snowball_val_tt5e7_s30_fixed.log; R6=snowball_ttband_lr5e7_kl05_a
echo "$(date +%F_%T) waiting for ckpt30 ready in $L"
for i in $(seq 1 600); do grep -q "ckpt30 ready\|TIMEOUT" $L 2>/dev/null && break; squeue -h -j 1671772 -o %T | grep -q . || break; sleep 30; done
grep -m1 "ckpt30 ready\|TIMEOUT" $L
if squeue -h -j 1671772 -o %T | grep -q .; then echo "$(date +%F_%T) pruning arm 3 (1671772) after its step-30 checkpoint"; scancel 1671772; fi
for i in $(seq 1 40); do squeue -h -j 1671772 -o %T | grep -q . || break; sleep 15; done; echo "$(date +%F_%T) arm 3 gone"
sleep 180
echo "$(date +%F_%T) bridge 9926: $(curl -s -m 5 localhost:9926/status | head -c 200)"
cd $O && J=$(DCFT=$PWD sbatch --parsable $E/$R6/sbatch/${R6}_rl.sbatch) && echo "ARM6 SUBMITTED $J"
squeue -h -u $USER -o "%i %j %T %M %D"; echo "$(date +%F_%T) DONE"
