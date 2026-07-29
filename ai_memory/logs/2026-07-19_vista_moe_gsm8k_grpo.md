# 2026-07-19 — Vista Qwen3-30B-A3B gsm8k GRPO bring-up

Session log for future AIs. Live status: `ai_memory/handoff.md`. Background: `vista_moe_*.md`, `tacc_vista.md`.

## Goal / ladder
GRPO on **`Qwen/Qwen3-30B-A3B`** (general MoE, not Coder) on TACC Vista (1× GH200 96GB/node).
1. ~~small smoke~~ → 2. ~~**8n disagg green**~~ → 3. **24n on `gh`/`gg`** (YAML drafted, not submitted).

Autonomy: fix + relaunch without waiting for "go fix it" (`user_preference.md`).

## What worked (end state)
- **843281** (`gh-dev`×8, disagg 4 policy EP=4 + 4 vLLM): multi-step train **GREEN**
  - step1 `pass@4=0.50` grad_norm≈11.8
  - step2 `pass@4=0.81` grad_norm≈3.7
  - step3 `pass@4=0.63` grad_norm≈1.1
  - ~12 min/step (gen ~90s + train ~535s + sync ~100s)
  - WandB: https://wandb.ai/lukeleeai/vista-moe-gsm8k-grpo/runs/3kiuj1sv
- Prior milestone **843253** (4n): generate+reward OK, then OOM late in policy train → motivated 8n.

## Critical learnings

### Disagg required for 30B-A3B on Vista
- **Colocate OOMs** on 2×96GB (weight-sync gather / vLLM `wake_up`). Jobs 843025/843198 era.
- Always `trainer.placement.colocate_all: false` with exclusive vLLM nodes.
- Default ladder geometry:
  - **4n:** 2 policy (EP=2) + 2 vLLM
  - **8n:** 4 policy (EP=4, FSDP=1) + 4 vLLM
  - **24n (Jupiter parity):** 16 policy (EP=4 × FSDP=4) + 8 vLLM — **not on `gh-dev`** (MaxNodes=8)

### Launcher env vars lie; Hydra is truth
- Shell prints `POLICY_NUM_NODES=8` / `NUM_INFERENCE_ENGINES=8` even when Hydra has 4+4.
- Always verify from experiment config JSON / Hydra args:
  `colocate_all`, `policy_num_nodes`, `num_inference_engines`, `expert_model_parallel_size`.

### Model cache trap
- Use **complete** snapshot: `$SCRATCH/cache/hf/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777…`
- **Do not** use stub `$SCRATCH/hub/models--Qwen--Qwen3-30B-A3B/…` (config/merges only → tokenizer explode).

### Env / deps
- penfever `otagent` + Luke user-site `$SCRATCH/python_user`
- Needed: `torchtitan@a1fdd7e`, `tyro==1.0.15` (same PYTHONUSERBASE breaks llamafactory — OK for RL)
- FlashInfer workspace on `$SCRATCH`; gcc 14.2; never JIT into `/home1`

### MarinSkyRL fixes (branch `lukedhlee/fix-ray-255-pg-strategy`)
| Issue | Fix (approx commit) |
|-------|---------------------|
| Ray 2.55 `PlacementGroupSchedulingStrategy` | import path fix |
| Colocate sync OOM | `28b2ce8` offload model before weight sync (still not enough → disagg) |
| `responses (8) vs prompt_token_ids (2)` | `f7c78a1` `generate_batched` + `return_dict=True` chat template |

### Config gotchas that burned time
- Nested `trainer.ref` under `policy` → Hydra fail (un-nest; job ~843023)
- Eval path mismatch on smoke → `eval_before_train: false`, `eval_interval: 999999`
- Grad ckpt: `gradient_checkpointing_use_reentrant: true`
- `enable_db_registration: false` always

### MoE divisibility
`(128/EP) % FSDP == 0` and `EP × FSDP = #policy GPUs`. Vista: each GPU = one node.

## Repos / branches / paths
| What | Where |
|------|--------|
| OT-Agent | branch `lukedhlee/vista-moe-grpo-30b` → `$SCRATCH/OpenThoughts-Agent` |
| MarinSkyRL | `lukedhlee/fix-ray-255-pg-strategy` → `$SCRATCH/MarinSkyRL` |
| Launch | `bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh` |
| 4n YAML | `hpc/skyrl_yaml/vista/4node_qwen3_30b_a3b_gsm8k_grpo.yaml` |
| 8n YAML | `hpc/skyrl_yaml/vista/8node_qwen3_30b_a3b_gsm8k_grpo.yaml` |
| 24n YAML | `hpc/skyrl_yaml/vista/24node_…yaml` (**local only as of EOD — commit/push before submit**) |
| Data | `$SCRATCH/data/gsm8k/{train,validation}.parquet` |
| Experiments | `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_{4n,8n}/` |
| WandB | project `vista-moe-gsm8k-grpo`, entity `lukeleeai` |
| Secrets | Mac `~/.config/otagent/secrets.env` → Vista `$SCRATCH/keys.env` |

## Job timeline (key IDs)
| Job | Nodes | Result |
|-----|-------|--------|
| 843253 | 4 disagg | sync→gen→reward OK; OOM in `fsdp2_clip_grad_norm_` mid policy train |
| 843281 | 8 disagg | **running green** through ≥3 full steps (max_steps=10) |

4n WandB (failed but generate milestone): https://wandb.ai/lukeleeai/vista-moe-gsm8k-grpo/runs/9xsvwjne

## What we did today (ops)
1. Diagnosed 4n train OOM after successful generate (`avg_pass_at_2: 0.25`).
2. Switched 8n YAML from colocate → disagg 4+4; reentrant grad ckpt; skip eval.
3. Submitted/watched **843281**; confirmed Hydra placement; watched through steps 1–3+.
4. Drafted **24n** YAML (16 policy EP=4×FSDP=4 + 8 vLLM, `PARTITION=gh`).
5. Kept background SSH watches; ControlMaster/MFA can die — re-`ssh vista` interactively if BatchMode fails.

## Next session checklist
1. Read `handoff.md`; check `squeue` for 843281 (expect finish ~10 steps or wall 2h).
2. If 8n still healthy: **commit + push** OT tip (24n YAML + launch comment) → `git pull` on Vista.
3. Submit 24n:
   ```bash
   NUM_NODES=24 PARTITION=gh TIME_LIMIT=12:00:00 \
     RL_CONFIG=./hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo.yaml \
     JOB_NAME=vista_moe30b_gsm8k_grpo_24n \
     bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
   ```
4. Re-enable eval only after generate path is trusted at scale.
5. Do **not** kill RUNNING jobs without explicit permission; no cluster SoT hand-edits.

## Don’ts (hard)
- Colocate 30B-A3B on Vista for train
- Use `$SCRATCH/hub` stub model
- Trust launcher `POLICY_NUM_NODES` env print over Hydra
- Edit on cluster / rsync patches (local commit → push → pull)
- Raise Daytona snapshot caps / enable DB registration casually

## 24n launch (same day, evening)

- Committed/pushed `f6dbcb54` — `24node_qwen3_30b_a3b_gsm8k_grpo.yaml` + launch comment.
- Submitted **843341** on **`gg`×24** (not `gh-dev`; gg had ~52 idle vs gh ~8). Immediate RUNNING.
- Geometry: 16 policy (EP=4 × FSDP=4) + 8 vLLM, reentrant ckpt, cpu_offload.
- 8n **843281** left running in parallel on gh-dev.
- Tip for queue: prefer whichever of `gh`/`gg` has more idle at submit time.

## 24n failures + relaunch

1. **843341 gg** — `peer_access` wanted `{GPU:2}` because base `critic_num_gpus_per_node=4`. Fix: YAML `critic_num_gpus_per_node: 1` (`e0fca8a0`).
2. **843399 gg** — peer_access OK; vLLM died with `RuntimeError: Device string must not be empty` (`current_platform.device_type` empty despite `CUDA_VISIBLE_DEVICES=0`). Nodes were i618-* on `gg`.
3. **843411 gh** — relaunched on `gh` (c6xx family, same as working 8n gh-dev). Prefer gh until gg root-caused.
4. **843437 gh** — `gpu_memory_utilization: 0.80` + `policy_strict_spread_pg` (`6c491b70`). Steps **1–14 OK**, hung mid step-15 policy_train; **scancel** 2026-07-20. No val (interval was 20). Continued in `2026-07-20_vista_moe_gsm8k_grpo.md`.
