# Jupiter Megatron 6-node GSM8k bring-up update (2026-07-26)

Audience: next operator continuing "make Megatron work as well as FSDP2" on Jupiter.

## Summary

Megatron 30B GSM8k now has a faster 6-node no-save smoke that completed successfully:

- Job: `1045827`
- Name: `jupiter_qwen3_30b_gsm8k_megatron_6n_nosave_4vllm_r2`
- State: `COMPLETED`, exit `0:0`, elapsed `00:30:38`
- Shape: 6 Jupiter booster nodes, 24 GH200 GPUs total, 4 vLLM rollout GPUs + 16 Megatron policy/ref GPUs (`20/24` Ray GPUs used)
- Model: `Qwen/Qwen3-30B-A3B`
- Dataset: GSM8k
- Steps: `MAX_STEPS=1`
- Checkpointing: `CKPT_INTERVAL=-1`, `HF_SAVE_INTERVAL=-1`, optimizer save/load disabled

Important log markers:

- `InferenceEngineClient initialized with 4 engines.`
- `MegatronPolicyWorkerBase` workers initialized across the 16-GPU policy/ref placement.
- Metrics emitted at `trainer/global_step: 1`.
- `Training done!`

Key metrics from the successful step:

- `timing/step`: `292.65s`
- `timing/generate`: `116.87s`
- `timing/sync_weights`: `97.04s`
- `timing/policy_train`: `54.14s`
- `reward/avg_raw_reward`: `0.28125`
- `reward/avg_pass_at_4`: `0.4375`
- `policy/raw_grad_norm`: `0.4667`

This validates Megatron train/generate/fwd/sync on the faster 6-node geometry.

## Checkpoint/resume validation completed

Model-only Megatron checkpoint/resume is now validated on the same 4-vLLM / 16-Megatron-GPU geometry with optimizer state save/load disabled.

### Save smoke

- `1045839` / `jupiter_qwen3_30b_gsm8k_megatron_6n_ckpt_4vllm_r1`: trained the step but failed during `dist_checkpointing.save` metadata planning with `RuntimeError: NCCL Error 2: unhandled system error`; only a partial `global_step_1/policy/common.pt` existed.
- Patch applied in `MarinSkyRL/skyrl-train/skyrl_train/distributed/megatron/megatron_strategy.py`:
  - route Megatron torch-dist save planning object collectives through an auxiliary CPU/Gloo group
  - set `validate_access_integrity=False`
  - keep saving `client_state` / optional `tag`
- `1045872` / `jupiter_qwen3_30b_gsm8k_megatron_6n_ckpt_4vllm_r2`: **COMPLETED**, exit `0:0`, elapsed `00:30:46`.
  - Loaded 4 vLLM engines.
  - `load_checkpoints` no-op for fresh run.
  - step 1 completed: `timing/step=267.84s`, `generate=106.93s`, `fwd_logprobs_values_reward=22.09s`, `policy_train=44.60s`, post-step `sync_weights=93.92s`.
  - reward: `avg_pass_at_4=0.5625`, `avg_raw_reward=0.3125`.
  - checkpoint save succeeded for `global_step_1`, then end-of-run checkpoint `global_step_2`.
  - checkpoint files include `policy/.metadata`, `policy/__0_0.distcp` ... `__15_0.distcp`, `policy/common.pt`, `policy/metadata.json`, `data.pt`, `trainer_state.pt`, and `latest_ckpt_global_step.txt`.

### Resume smoke

- `1045911` / `jupiter_qwen3_30b_gsm8k_megatron_6n_resume_4vllm_r1`: launched with `RESUME_MODE=from_path`, `RESUME_PATH=.../ckpt_4vllm_r2/.../checkpoints/global_step_1`, `MAX_STEPS=2`.
  - Confirmed trainer and dataloader state loaded from `global_step_1`.
  - Timed out at `01:00:26` stuck in policy checkpoint load.
  - Diagnosis: Megatron `FullyParallelLoadStrategyWrapper` was unnecessary here and its load/exchange path stalled; base `torch_dist` load uses per-rank `checkpoint.load(..., no_dist=True)`.
- Patch applied: remove `FullyParallelLoadStrategyWrapper` from `MegatronStrategy.load_checkpoint`; use the default load sharded strategy directly.
- `1045992` / `jupiter_qwen3_30b_gsm8k_megatron_6n_resume_4vllm_r2`: **COMPLETED**, exit `0:0`, elapsed `00:32:22`.
  - `InferenceEngineClient initialized with 4 engines`.
  - Loaded from `.../checkpoints/global_step_1`.
  - `Resuming from global_step: 1`.
  - Loaded trainer state and dataloader state immediately.
  - `Successfully loaded policy checkpoint`; `Finished: 'load_checkpoints', time cost: 50.96s`.
  - Initial resumed `sync_weights`: `95.90s`.
  - Resumed step 2 completed: `timing/step=256.47s`, `generate=99.76s`, `fwd_logprobs_values_reward=21.51s`, `policy_train=42.71s`, post-step `sync_weights=92.19s`.
  - reward: `avg_pass_at_4=0.6875`, `avg_raw_reward=0.625`.
  - saved `global_step_2`, then end-of-run `global_step_3`; `latest_ckpt_global_step.txt` contains `3`.
  - `Training done!`

## Full optimizer-state checkpoint/resume validation completed

Full Megatron distributed optimizer-state checkpointing is now validated for the same 6-node / TP4 / EP4 topology using Megatron's `dp_reshardable` optimizer checkpoint format.

### Failed optimizer-state attempts

- `1046102` / `jupiter_qwen3_30b_gsm8k_megatron_6n_optckpt_save_r1`: failed during optimizer sharded state dict construction before writing durable checkpoint files.
  - Cause: Megatron defaulted to deprecated `fully_sharded_model_space`, which creates `ShardedTensor.flattened_range`; this MCore checkpointing build rejects flattened ranges.
- `1046202` / `jupiter_qwen3_30b_gsm8k_megatron_6n_optckpt_save_r2`: changed optimizer metadata to `fully_reshardable` + memory-efficient gather; trained step 1 but failed during optimizer state construction.
  - Cause: `fully_reshardable` save calls Megatron `get_parameter_state_dp_zero(... use_gloo_comm=True)` and died with Gloo `Connection closed by peer`.
  - Conclusion: `fully_reshardable` is theoretically more portable, but it is not supported on this Jupiter/Ray setup today.
- `1046522` / `jupiter_qwen3_30b_gsm8k_megatron_6n_optckpt_resume_r1`: after switching to `dp_reshardable`, save worked, but first resume attempt OOM'd during Transformer Engine FusedAdam optimizer restore.
  - Cause: model checkpoint state stayed referenced while optimizer state was being materialized; GPU had only ~78 MiB free and failed on a 12 MiB allocation.

### Patch now in MarinSkyRL

File: `skyrl_train/distributed/megatron/megatron_strategy.py`

- Pass optimizer checkpoint metadata on save/load:
  - default `metadata={"distrib_optim_sharding_type": "dp_reshardable"}`
  - env override: `SKYRL_MEGATRON_OPTIMIZER_SHARDING_TYPE`
  - optional fully-reshardable memory-efficient flag only when explicitly requested
- Keep `fully_sharded_model_space` out of the default path, avoiding unsupported `flattened_range` tensors.
- Keep `FullyParallelLoadStrategyWrapper` disabled for Megatron checkpoint load.
- During load, `pop` and delete model state before optimizer restore, then `gc.collect()` and `torch.cuda.empty_cache()`; do the same after optimizer restore before LR scheduler restore.

### Passing full-state save

- `1046376` / `jupiter_qwen3_30b_gsm8k_megatron_6n_optckpt_save_r3`: **COMPLETED**, exit `0:0`, elapsed `00:33:51`.
  - Fresh run, `SAVE_OPTIMIZER_STATES=true`, `LOAD_OPTIMIZER_STATES=false`, `SAVE_LR_SCHEDULER_STATES=true`, `HF_SAVE_INTERVAL=-1`.
  - Step 1 completed: `timing/step=381.38s`, `generate=188.86s`, `fwd_logprobs_values_reward=30.91s`, `policy_train=70.55s`, post-step `sync_weights=90.78s`.
  - Saved `global_step_1`, then end checkpoint `global_step_2`.
  - Each policy checkpoint has `policy/.metadata`, `policy/common.pt`, `policy/metadata.json`, HuggingFace config/tokenizer, and 16 `__*_0.distcp` shards around 26.7 GB each.
  - `latest_ckpt_global_step.txt` contains `2`.

### Passing full-state resume

- `1046602` / `jupiter_qwen3_30b_gsm8k_megatron_6n_optckpt_resume_r2`: **COMPLETED**, exit `0:0`, elapsed `00:33:40`.
  - Resumed from `.../optckpt_save_r3/.../checkpoints/global_step_2`.
  - Settings: `RESUME_MODE=from_path`, `MAX_STEPS=3`, `LOAD_OPTIMIZER_STATES=true`, `LOAD_LR_SCHEDULER_STATES=true`, `SAVE_OPTIMIZER_STATES=false`, `CKPT_INTERVAL=-1`, `HF_SAVE_INTERVAL=-1`.
  - Load markers:
    - `Loading checkpoint from: .../global_step_2`
    - `Resuming from global_step: 2`
    - `Finished: 'load_checkpoints', time cost: 83.75s`
    - worker printed `Loaded model state dict.`, `Loaded optimizer state dict.`, `Loaded LR scheduler state dict.`
  - Resumed step 3 completed: `timing/step=381.82s`, `generate=179.85s`, `fwd_logprobs_values_reward=31.99s`, `policy_train=76.21s`, post-step `sync_weights=93.41s`.
  - Trainer logged `Reached max training steps (3)` and `Training done!`.

Supported now:

- 6-node Qwen3-30B-A3B GSM8k Megatron train/generate/fwd/sync.
- Model-only Megatron checkpoint save/resume.
- Full Megatron distributed optimizer + LR scheduler checkpoint save/resume on the same topology with `dp_reshardable`.

## Megatron vs FSDP2 speed comparison

Job `1047736` / `jupiter_qwen3_30b_gsm8k_megatron_6n_speed_fsdp2cmp_r1` completed successfully:

- State: `COMPLETED`, exit `0:0`, elapsed `00:46:16`
- Geometry: 6 Jupiter booster nodes, 16 Megatron policy/ref GPUs + 8 vLLM engines, batch size 32
- Settings: `MAX_STEPS=3`, `EVAL_BEFORE_TRAIN=false`, `EVAL_INTERVAL=999999`, `CKPT_INTERVAL=-1`, `HF_SAVE_INTERVAL=-1`, optimizer/LR checkpoint save/load disabled
- Startup note: 8-engine cold start was long but healthy; vLLM shard loading finished, FlashInfer autotuning ran, Megatron TP4/DP4 mesh initialized, then training completed.

Megatron metrics:

- Step 1: `step=336.96s`, `generate=101.68s`, `fwd=32.30s`, `policy_train=81.67s`, `sync_weights=121.22s`
- Step 2: `step=307.37s`, `generate=107.03s`, `fwd=18.39s`, `policy_train=59.41s`, `sync_weights=122.46s`
- Step 3: `step=296.37s`, `generate=97.13s`, `fwd=17.72s`, `policy_train=58.81s`, `sync_weights=122.63s`
- Average steps 1-3: `step=313.57s`, `generate=101.95s`, `fwd=22.80s`, `policy_train=66.63s`, `sync_weights=122.10s`
- Warm average steps 2-3: `step=301.87s`, `generate=102.08s`, `fwd=18.05s`, `policy_train=59.11s`, `sync_weights=122.55s`

FSDP2 comparison parsed from job `1042857` / `jupiter_moe30b_gsm8k_grpo_6n_fast_50step_eval5`:

- Same broad 6-node fast geometry: 16 policy GPUs + 8 vLLM engines, batch size 32
- Average all 47 available train rows: `step=425.61s`, `generate=105.08s`, `fwd=25.33s`, `policy_train=110.68s`, `sync_weights=184.13s`
- First 3-step average: `step=424.06s`, `generate=101.78s`, `fwd=27.59s`, `policy_train=111.45s`, `sync_weights=182.83s`

Interpretation:

- Megatron improves full step wall time by about `1.36x` vs the FSDP2 all-row average (`425.61/313.57`) and `1.35x` vs FSDP2 first-three average (`424.06/313.57`).
- Megatron improves policy train by about `1.66x` vs FSDP2 all-row average (`110.68/66.63`) and `1.67x` vs FSDP2 first-three average (`111.45/66.63`).
- Warm Megatron steps 2-3 are stronger: about `1.41x` faster full step and `1.87x` faster policy train vs FSDP2 all-row average.
- Generate is essentially unchanged; Megatron's gains are mostly policy train and weight sync. GSM8K is therefore a conservative comparison, because generation remains a large fraction of wall time.

Still not supported:

- Megatron optimizer checkpoint portability across changed topology / TP / PP / EP sizes.
- `fully_reshardable` optimizer saves on Jupiter/Ray; current failure is Gloo DP gather connection closure.
- Megatron default `fully_sharded_model_space`; current failure is unsupported `ShardedTensor.flattened_range`.

## Operational lesson

Do not cancel the 30B Megatron run immediately after vLLM reports `Model loading took ...`.

In the passing historical no-save job `1043650`, vLLM model loading finished around `17:09`, but `InferenceEngineClient initialized with 8 engines` did not appear until `17:20` -- about 11 minutes later. In `1045827`, the quiet interval was also real work:

- vLLM started loading around `04:11`.
- vLLM loading ended around `04:12`.
- FlashInfer MoE autotuning ran around `04:23:53-04:23:59`.
- Megatron workers started after that.

Two jobs were canceled too early before this was understood:

- `1045812`: 8-vLLM no-save relaunch, canceled by us after appearing stuck at `8/24` GPUs.
- `1045822`: 4-vLLM no-save relaunch, canceled by us after appearing stuck at `4/24` GPUs.

Those cancellations were premature. Use a post-vLLM-load wait window of at least 12 minutes before calling this class of run stuck.

## Files changed for Megatron

Local OpenThoughts-Agent changes:

- `hpc/rl_config_utils.py`
  - Treats `hf_hub_repo_id: null` as intentional.
  - Allows Megatron `model_config_kwargs` / `transformer_config_kwargs` as optional Hydra `++` patterns.
  - Applies resolved `model_path` to both `trainer.policy.model.path` and `trainer.ref.model.path`.
  - Adds optional save/load optimizer and LR scheduler override patterns.
- `hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh`
  - Preserves `WANDB_PROJECT` through dotenv.
  - Supports env-driven `HF_SAVE_INTERVAL`, `RESUME_MODE`, `RESUME_PATH`, optimizer/LR state save/load knobs.
  - Supports fast geometry env knobs:
    - `TRAIN_BATCH_SIZE`
    - `POLICY_MINI_BATCH_SIZE`
    - `EVAL_BATCH_SIZE`
    - `GENERATOR_NUM_INFERENCE_ENGINES`
    - `GENERATOR_INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE`
- `hpc/skyrl_standard/jupiter/run_gsm8k_megatron_grpo.sh`
  - Defaults Megatron smoke to no HF save and no optimizer state save/load.
- Megatron YAMLs:
  - `hpc/skyrl_yaml/jupiter/2node_qwen3_0p6b_gsm8k_grpo_megatron_smoke.yaml`
  - `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_megatron_smoke.yaml`
  - `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_megatron_ckpt_smoke.yaml`

Local MarinSkyRL changes:

- `skyrl_train/config/ppo_base_config.yaml`
  - Adds trainer knobs for optimizer/LR scheduler state save/load.
- `skyrl_train/distributed/strategy.py`
  - Abstract `save_checkpoint` accepts `**kwargs`.
- `skyrl_train/distributed/megatron/megatron_strategy.py`
  - Saves/loads `client_state` and optional checkpoint `tag`.
  - Routes save-planning object collectives through an auxiliary Gloo group on Jupiter.
  - Uses the default per-rank load strategy directly instead of `FullyParallelLoadStrategyWrapper`.
- `skyrl_train/workers/worker.py`
  - Policy/Critic checkpoint save can omit optimizer and LR scheduler state.
- `skyrl_train/trainer.py`
  - Trainer passes save/load optimizer/LR knobs to workers.

Remote caveat: Jupiter `OpenThoughts-Agent` is on `lukedhlee/vista-moe-grpo-30b`; Jupiter `MarinSkyRL` is on `main`. The synced MarinSkyRL files came from local branch `lukedhlee/fix-stridedshard-torch29`, so the remote working tree has file-level dirty changes beyond the tiny Megatron patch. Do not blindly revert.

## Current recommended next step

Model-only and same-topology full optimizer-state checkpoint/resume are done. The known-good full optimizer-state save knobs are:

- `CKPT_INTERVAL=1`
- `HF_SAVE_INTERVAL=-1`
- `SAVE_OPTIMIZER_STATES=true`
- `LOAD_OPTIMIZER_STATES=false`
- `SAVE_LR_SCHEDULER_STATES=true`
- `LOAD_LR_SCHEDULER_STATES=false`
- `GENERATOR_NUM_INFERENCE_ENGINES=4`
- `GENERATOR_INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1`

The known-good full optimizer-state resume knobs are:

- explicit resume from the same-topology checkpoint with `RESUME_MODE=from_path`
- `MAX_STEPS` greater than the checkpoint global step
- `CKPT_INTERVAL=-1` for a pure load/run smoke
- `HF_SAVE_INTERVAL=-1`
- `SAVE_OPTIMIZER_STATES=false`
- `LOAD_OPTIMIZER_STATES=true`
- `SAVE_LR_SCHEDULER_STATES=false`
- `LOAD_LR_SCHEDULER_STATES=true`
- `GENERATOR_NUM_INFERENCE_ENGINES=4`
- `GENERATOR_INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1`

## Cancellation rule

User gave standing permission: cancel experiments spawned by us, but do not cancel any other jobs.
