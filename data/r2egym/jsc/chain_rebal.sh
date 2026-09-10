#!/bin/bash
# chain_rebal.sh — after the grouped_mm timing bench (job 1686910) ends, submit the generator:trainer REBALANCE timing bench in the
# same 40-node slot: 32 policy nodes (fsdp 128, 4 microbatches/GPU instead of 8) + 8 engine nodes (528 seats = 66 per node, ~16 per
# DP rank; decode at ~3 seqs/GPU was pure weight-bandwidth). grouped_mm stays on if the first bench ran clean (>= 3 finished steps, no
# Traceback), else the rebalance runs eager so the two effects are never confounded by a broken kernel. 4 steps, no eval/checkpoints.
# Session 5b45b53d, 2026-09-06 01:20 PT. Log: experiments/chain_rebal.log.
export OMP_NUM_THREADS=1
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; GJ=1686910
G=$E/snowball_ttband_lr5e7_stale2_gmm_bench/logs/snowball_ttband_lr5e7_stale2_gmm_bench_${GJ}.out
D=snowball_ttband_lr5e7_stale2_rebal_bench; LOG=$E/chain_rebal.log
echo "$(date +%F_%T) waiting for gmm bench $GJ to leave squeue" >> $LOG
while squeue -h -j $GJ -o %T 2>/dev/null | grep -q .; do sleep 60; done
steps=$(grep -c "Finished: .step." $G 2>/dev/null); tb=$(grep -c Traceback $G 2>/dev/null)
echo "$(date +%F_%T) gmm bench gone: finished_steps=$steps tracebacks=$tb" >> $LOG
GMM=(trainer.policy.fsdp_config.use_grouped_mm=true trainer.ref.fsdp_config.use_grouped_mm=true)
[ "${steps:-0}" -ge 3 ] && [ "${tb:-0}" -eq 0 ] || { GMM=(); echo "$(date +%F_%T) gmm bench not clean -> rebalance runs EAGER" >> $LOG; }
bash $C/clone_stale.sh $D 2 1 "${GMM[@]}" \
  trainer.policy.fsdp_config.fsdp_size=128 trainer.ref.fsdp_config.fsdp_size=128 \
  trainer.placement.policy_num_nodes=32 trainer.placement.ref_num_nodes=32 \
  generator.num_inference_engines=8 \
  trainer.max_steps=4 trainer.eval_before_train=false trainer.ckpt_interval=999 trainer.hf_save_interval=999 >> $LOG 2>&1
sleep 3; J=$(squeue -h -u $USER -n $D -o %i | head -1); echo "$(date +%F_%T) rebal bench job=${J:-NONE} gmm_flags=${#GMM[@]}" >> $LOG
[ -n "$J" ] || exit 1
prev=; while true; do s=$(squeue -h -j $J -o %T 2>/dev/null); [ -z "$s" ] && s=GONE
  if [ "$s" != "$prev" ]; then echo "$(date +%F_%T) $J $s" >> $E/watch_rebal.log; prev=$s; fi
  [ "$s" = GONE ] && { sacct -j $J -o JobID,State,ExitCode,Elapsed -n >> $E/watch_rebal.log; break; }; sleep 60; done
