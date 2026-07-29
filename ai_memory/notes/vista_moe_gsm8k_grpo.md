# Vista MoE gsm8k GRPO (standard / non-agentic)

**Branch:** `lukedhlee/vista-moe-grpo-30b`  
**Model:** `Qwen/Qwen3-30B-A3B` (not Coder)  
**Session log (2026-07-19):** `ai_memory/logs/2026-07-19_vista_moe_gsm8k_grpo.md`  
**Live handoff:** `ai_memory/handoff.md`

## Geometry (proven)
**Disagg only** — colocate OOMs on 30B-A3B.

| Scale | Layout | Partition |
|-------|--------|-----------|
| 4n | 2 policy EP=2 + 2 vLLM | `gh-dev` |
| 8n | 4 policy EP=4 + 4 vLLM | `gh-dev` — **GREEN** (843281) |
| 24n | 16 policy EP=4×FSDP=4 + 8 vLLM | `gh`/`gg` |

## Files
- YAMLs: `hpc/skyrl_yaml/vista/{4,8,24}node_qwen3_30b_a3b_gsm8k_grpo.yaml`
- Launch: `hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh`
- Prep: `hpc/skyrl_standard/vista/prep_gsm8k_parquet.sh`

## Launch
```bash
# 8n smoke
NUM_NODES=8 RL_CONFIG=./hpc/skyrl_yaml/vista/8node_qwen3_30b_a3b_gsm8k_grpo.yaml \
  JOB_NAME=vista_moe30b_gsm8k_grpo_8n bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh

# 24n (after commit/push + pull)
NUM_NODES=24 PARTITION=gh TIME_LIMIT=12:00:00 \
  RL_CONFIG=./hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo.yaml \
  JOB_NAME=vista_moe30b_gsm8k_grpo_24n bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
```

## Results
- WandB: `vista-moe-gsm8k-grpo` / `lukeleeai`
- Disk: `$SCRATCH/experiments/<job_name>/`
