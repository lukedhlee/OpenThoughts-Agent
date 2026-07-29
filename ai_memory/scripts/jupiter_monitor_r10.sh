#!/usr/bin/env bash
# Monitor Jupiter GRPO 50-step job; print status + eval/train milestones.
# Usage: bash ai_memory/scripts/jupiter_monitor_r10.sh [job_id]
set -euo pipefail
JOB_ID="${1:-1012293}"
EXP_GLOB="${EXP_GLOB:-/e/scratch/reformo/lee27/experiments/jupiter_moe30b_gsm8k_grpo_4n_r12_50step_eval5*}"

ssh -o BatchMode=yes -o ConnectTimeout=30 jupiter "bash -s" <<EOF
set -euo pipefail
JOB_ID="$JOB_ID"
echo "=== \$(date -Is) job \$JOB_ID ==="
squeue -j "\$JOB_ID" -o "%.18i %.12P %.40j %.8T %.10M %.6D %R" 2>/dev/null || true
sacct -j "\$JOB_ID" --format=JobID,State,ExitCode,Elapsed,NodeList -P 2>/dev/null | head -5
LOG=\$(ls -t $EXP_GLOB/logs/*.out 2>/dev/null | head -1 || true)
if [[ -z "\${LOG:-}" ]]; then
  echo "No .out log yet under $EXP_GLOB/logs"
  exit 0
fi
echo "LOG=\$LOG"
echo "=== tail ==="
tail -n 40 "\$LOG"
echo "=== milestones ==="
grep -nE "pass_at_1|avg_raw_reward|global_step|Started:|Error|Traceback|OOM|NCCL|FLASH|flash_attn|Training finished|max_steps|torchtitan|ThrMma|Validation" "\$LOG" | tail -n 80
EOF
