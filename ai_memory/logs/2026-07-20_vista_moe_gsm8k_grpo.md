# 2026-07-20 — Vista 24n hang + re-enable validation

Session log. Live status: `ai_memory/handoff.md`. Prior: `2026-07-19_vista_moe_gsm8k_grpo.md`.

## What happened
- **843437** (`gh`×24, tip `6c491b70`): steps **1–14 OK**, hung mid **step-15** policy_train at 24/32 (75%).
  - Last log ~`2026-07-19 13:01:45`; outfile froze; wall ~10h before cancel.
  - Likely NCCL/collective stall (normal policy_train ~140s).
- User OK → **`scancel 843437`** (2026-07-20).

## Train metrics (843437, steps 1–14) — noisy, not ground truth
Train `pass@4` swung ~0.50–0.78 (batch=32, temp=0.7, on-policy). **No validation ran**:
- Config had `eval_before_train: false`, `eval_interval: 20` → first val at step 20; hung at 15.
- Smoke configs had earlier disabled eval entirely (`999999`) after path mismatch.

**Lesson:** judge learning from **val reward**, not train `pass@4`.

## Config change for relaunch
`24node_qwen3_30b_a3b_gsm8k_grpo.yaml`:
- `eval_before_train: true`
- `eval_interval: 5`
- Keep working recipe: `gh`, `critic_num_gpus_per_node: 1`, `gpu_memory_utilization: 0.80`, `policy_strict_spread_pg: true`

## Done this session
1. `scancel 843437` (user OK).
2. Pushed `9916eaa5` — `eval_before_train: true`, `eval_interval: 5`.
3. Submitted **844374** then **`scancel 844374`** — user said do not launch / cancel it. Cluster idle; wait for explicit relaunch.

## Preference note
After a cancel, do **not** auto-relaunch unless asked.

## Later local config update
- Next diagnostic launch should use `eval_interval: 1`, `eval_before_train: true`, `seed: 42`.
- Reason: train `pass@4` is a 32-prompt on-policy stochastic batch metric; held-out val is the reliable curve.

## Relaunch requested + live monitor
- User later explicitly asked to launch and monitor while away.
- Pushed `135ca0a5` with `eval_interval: 1`, `eval_before_train: true`, `seed: 42`.
- Submitted **844625** on `gh`x24 as `vista_moe30b_gsm8k_grpo_24n_eval1`.
- Job started and is running on 24 GH200 GPUs (24 nodes, 1 GPU/node; 16 policy/ref + 8 vLLM).
- Log: `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_eval1/logs/vista_moe30b_gsm8k_grpo_24n_eval1_844625.out`
- Baseline eval step 0: `pass_at_1=0.4412433662`, eval wall `1847.62s`.
- Step 1 train: `400.03s` train-only wall (~9.0 steps/hour), `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`.
- Step 1 eval: `pass_at_1=0.4579226687`, eval wall `1863.94s`.
- Step 2 train: `390.61s`, `reward/avg_raw_reward=0.4296875`, `reward/avg_pass_at_4=0.59375`.
- Step 2 eval: `pass_at_1=0.4427596664`, eval wall `1868.44s`.
- Step 3 train: `384.85s`, `reward/avg_raw_reward=0.546875`, `reward/avg_pass_at_4=0.71875`.
- Step 3 eval: `pass_at_1=0.4632297195`, eval wall `1866.20s`.
- Step 4 train: `381.76s`, `reward/avg_raw_reward=0.546875`, `reward/avg_pass_at_4=0.75`; eval `pass_at_1=0.4586808188`, eval wall `1872.24s`.
- Step 5 train: `385.88s`, `reward/avg_raw_reward=0.5390625`, `reward/avg_pass_at_4=0.71875`; eval `pass_at_1=0.4632297195`, eval wall `1873.57s`.
- Step 6 train: `386.05s`, `reward/avg_raw_reward=0.578125`, `reward/avg_pass_at_4=0.78125`; eval `pass_at_1=0.4586808188`, eval wall `1874.68s`.
- Step 7 train: `385.14s`, `reward/avg_raw_reward=0.65625`, `reward/avg_pass_at_4=0.78125`; eval `pass_at_1=0.4670204701`, eval wall `1880.29s`.
- Step 8 train: `388.19s`, `reward/avg_raw_reward=0.421875`, `reward/avg_pass_at_4=0.59375`; eval `pass_at_1=0.4601971190`, eval wall `1885.05s`.
- Step 9 train: `388.82s`, `reward/avg_raw_reward=0.3203125`, `reward/avg_pass_at_4=0.46875`; eval `pass_at_1=0.4442759666`, eval wall `1870.31s`.
- Step 10 train: `381.94s`, `reward/avg_raw_reward=0.6171875`, `reward/avg_pass_at_4=0.71875`; eval `pass_at_1=0.4730856710`, eval wall `1879.10s`.
- Step 11 train: `385.64s`, `reward/avg_raw_reward=0.375`, `reward/avg_pass_at_4=0.5`; eval started at `07:40:36` TACC local and was at 5/42 (12%) as of `07:43:35`.
- With eval every step, effective cadence is ~1.6 steps/hour; eval dominates runtime. Train-only throughput is ~9.2-9.4 steps/hour.

## Relaunch with lower eval cadence
- User approved cancelling/relaunching after confirming no checkpoint existed.
- **844625** cancelled at ~`2026-07-20 09:18` TACC local. Checkpoint dir was empty (`ckpt_interval=20`, run had not reached step 20).
- Changed 24n config and pushed `00e40bd8`:
  - `eval_interval: 10`
  - `ckpt_interval: 5`
  - kept `eval_before_train: true`, `seed: 42`
- Vista checkout fast-forwarded to `00e40bd8`; submitted **846086**:
  - job name `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5`
  - `gh` partition, 24 nodes / 24 GH200 GPUs
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_846086.out`
  - checkpoint path `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/checkpoints`
  - submitted JSON verified `trainer.eval_interval=10`, `trainer.ckpt_interval=5`, `trainer.eval_before_train=true`, `policy_num_nodes=16`, `generator.num_inference_engines=8`
  - state `RUNNING` as of `09:20`; Ray startup began.
- `846086` startup/train progress:
  - `load_checkpoints`: no checkpoint found, started from scratch.
  - Baseline eval step 0 finished at `10:12:35`: `pass_at_1=0.4397270660`, eval wall `1926.32s` (~32.1 min).
  - Step 1 generation finished at `10:14:09`: `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`.
  - Step 1 policy_train started at `10:14:33`.
  - Step 1 completed at `10:19:20`: step wall `404.12s`; no eval after step 1 because `eval_interval=10`.
  - Step 2 generation finished at `10:20:51`: `reward/avg_raw_reward=0.4375`, `reward/avg_pass_at_4=0.5625`.
  - **Hard stall:** trainer log did not advance past step-2 `Started: 'fwd_logprobs_values_reward'` / `MESH_DISPATCH method=forward issued forward.remote() to 16 actors` from `10:20:51` until Slurm timeout.
    - Ray status query from inside allocation: 24 nodes active, 24/24 GPUs reserved.
    - Actor query: 16 `FSDPPolicyWorkerBase` alive, 8 `AsyncVLLMInferenceEngine` alive; dead entries are init-time `InfoActor`s.
    - No checkpoint was saved; first checkpoint would have been step 5, but only step 1 completed and step 2 stalled.
    - GPU sample near timeout: several high-memory policy/ref GPUs idle at 0% util while many vLLM-sized GPUs were at 100% util.
    - No trainer traceback/CUDA/NCCL fatal surfaced before teardown.
  - Final Slurm state: `TIMEOUT`, elapsed `12:00:27`; killed at `2026-07-20T21:20:14` TACC local due to time limit.
  - Conclusion: `eval_interval=10` solved eval overhead but exposed/reproduced a distributed step-2 forward/logprob stall. Do not relaunch the exact same 24n layout blindly; next launch should change the recipe or add targeted diagnostics.

## Same-config retry requested
- User requested rerunning without fixes to see whether the 24n stall reproduces deterministically.
- Submitted **848839** on `gh`x24 as `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry`.
  - commit/config: same `00e40bd8`, no file changes
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry_848839.out`
  - checkpoint path `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/checkpoints`
  - actual SkyRL command verified `trainer.eval_interval=10`, `trainer.ckpt_interval=5`, `trainer.eval_before_train=true`, `policy_num_nodes=16`, `generator.num_inference_engines=8`
- Progress:
  - Step 0 eval finished at `22:43:16` TACC local: `pass_at_1=0.4397270660`, eval wall `1927.86s`.
  - Step 1 finished at `22:50:00`: step wall `403.85s`, `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`.
  - Step 2 generation finished at `22:51:30`: `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.59375`.
  - Step 2 `fwd_logprobs_values_reward` finished at `22:51:48` in `17.80s`; this retry passed the phase where **846086** stalled.
  - Step 2 `policy_train` finished at `22:54:06` in `137.46s`.
  - **Likely hard stall:** last trainer progress is step-2 `Started: 'sync_weights'` at `22:54:06`; log stat remained frozen at `22:54:07` as of `23:07:11` while Slurm job remained `RUNNING`.
  - No checkpoint yet; first checkpoint is step 5. Do not `scancel` without explicit user OK.

## Debug: 848839 stalled in weight sync
- User asked to debug why the retry is stuck. Times below are KST.
- Current live status at `2026-07-21 13:51:12 KST`: Slurm job `848839` still `RUNNING`, elapsed `2:02:48`, log still frozen at `2026-07-21 12:54:07 KST`.
- Stall phase: step-2 `sync_weights`, immediately after `policy_train` finished at `2026-07-21 12:54:06 KST`.
- Ray status from head node: 24 active nodes, all 24 GPUs reserved in placement groups, no pending demands/failures.
- Actor state: 16 `FSDPPolicyWorkerBase` alive, 8 `AsyncVLLMInferenceEngine` alive. Ray task list retained 100 tasks, all `AsyncVLLMInferenceEngine.update_named_weights`, all `FINISHED`.
- Process evidence: all 16 policy workers are named `ray::FSDPPolicyWorkerBase.broadcast_to_inference_engines`; vLLM actors are alive/idling. Main policy threads show `ep_poll`; `pstack`/`gdb` attach was denied by ptrace permissions.
- NCCL trace prefix `/scratch/11584/lukedhlee/nccl_trace_*` had no files despite the stall being past the configured worker timeout window.
- vLLM log shows step-1 reload/cache reset completed at `2026-07-21 12:50:00 KST`, including `RotaryEmbedding: Failed to load weights` warning then `Successfully reset prefix cache`. No step-2 reset/finish line appears after `2026-07-21 12:54:06 KST`.
- Current interpretation: not eval, not checkpoint, not generation/reward. The live run is wedged inside policy-side `broadcast_to_inference_engines`, likely a policy-rank barrier or await around W13 reload/update/finish after vLLM update tasks have returned.
- Staged remote instrumentation for the next launch:
  - File: `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py`
  - Adds `WEIGHT_SYNC_DEBUG` rank/phase logs around cache reset, W13 begin/finish, every chunk broadcast/update await, every policy barrier, and final barrier.
  - Verified with `python -m py_compile` and `git diff --check`.
  - Does not affect job `848839`; requires a new launch to collect these logs.

## Targeted weight-sync debug relaunch
- User approved cancelling the stuck retry and launching a faster diagnostic run.
- **848839** cancelled at `2026-07-21 14:16 KST`; final Slurm state observed as `COMPLETING`.
- Added/copyed diagnostic config `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug.yaml`.
- Submitted **849329** on `gh`x24:
  - job name `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_849329.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_rl_config.json`
  - checkpoint dir auto-set to `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/checkpoints`, but `ckpt_interval=999999`.
- Debug settings:
  - `trainer.max_steps=3`
  - `trainer.train_batch_size=8`
  - `trainer.policy_mini_batch_size=8`
  - `trainer.eval_interval=999999`
  - `trainer.eval_before_train=false`
  - `trainer.ckpt_interval=999999`
  - `trainer.placement.policy_num_nodes=16`
  - `trainer.placement.ref_num_nodes=16`
  - `generator.num_inference_engines=8`
  - `generator.n_samples_per_prompt=4`
  - `SKYRL_WEIGHTSYNC_DEBUG=1`
- As of `2026-07-21 14:19 KST`, **849329** was `RUNNING`, elapsed `1:47`, still in Ray startup after `Cleaning up existing Ray instances...`.
- Note: universal runner's early env summary says `NUM_INFERENCE_ENGINES=24` and `POLICY_NUM_NODES=24`, but the generated SkyRL hydra args are authoritative and were verified as the intended 16-policy/8-vLLM geometry.
- Startup note: the `Cleaning up existing Ray instances...` pause was sequential per-node cleanup; Ray became ready at `2026-07-21 14:22:57 KST`.
- **849329** failed before training at `2026-07-21 14:23:50 KST`:
  - Assertion: `train_batch_size (8) should be larger than or equal to ... policy_dp_size=16 ... lcm_dp_size=16`.
  - No trainer step ran; this was a debug-config error, not a SkyRL/vLLM runtime stall.
- Corrected debug config to `train_batch_size=16` and `policy_mini_batch_size=16`, copied it to Vista.
- Submitted **849358** on `gh`x24 at `2026-07-21 14:26 KST`:
  - job name `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_849358.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_rl_config.json`
  - verified submitted args: `trainer.max_steps=3`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
  - As of `2026-07-21 14:26:10 KST`, state `RUNNING`, elapsed `0:14`, Ray startup beginning.
- **849358** failed at `2026-07-21 14:47:37 KST` from an instrumentation bug:
  - Final Slurm state `FAILED`, elapsed `23:47`.
  - It reached runtime: Ray ready, vLLM loaded, FSDP policy weights loaded, `init_weight_sync_state` finished in `8.72s`, `load_checkpoints` found no checkpoint.
  - Initial trainer `sync_weights` ran from `2026-07-21 14:47:36 KST` to `14:47:37.855 KST`, logged `Finished: 'sync_weights', time cost: 1.78s`.
  - Error: `UnboundLocalError: cannot access local variable 'os'` in instrumented `broadcast_to_inference_engines`, caused by local `import os` later in the function shadowing module-level `os`.
  - Interpretation: this is not the original distributed stall; debug instrumentation needed a small fix.
- Fixed remote instrumentation by removing the inner `import os`; verified with `py_compile` and `git diff --check`.
- Submitted **849442** on `gh`x24 at `2026-07-21 14:52 KST`:
  - job name `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_849442.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_rl_config.json`
  - verified submitted args: `trainer.max_steps=3`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
  - Startup recovered from a retryable vLLM FlashInfer JIT/FileLock `OSError: [Errno 116] Stale file handle`; all 8 inference engines initialized by `2026-07-21 15:05:10 KST`.
  - `load_checkpoints`: no checkpoint found, started from scratch.
  - Initial `sync_weights` completed at `2026-07-21 15:18:17 KST`, cost `149.97s`.
  - Step 1: generate `48.68s`, raw reward `0.328125`, pass@4 `0.5625`; fwd/logprobs `19.32s`; policy_train `84.64s`; post-step `sync_weights` `157.85s`; step wall `310.61s`.
  - Step 2: generate `45.95s`, raw reward `0.640625`, pass@4 `0.8125`; fwd/logprobs `10.06s`; policy_train `75.54s`; post-step `sync_weights` `156.42s`; step wall `288.03s`.
  - Step 3: generate `45.08s`, raw reward `0.5625`, pass@4 `0.75`; fwd/logprobs `10.18s`; policy_train `78.31s`; final `sync_weights` `156.31s`; step wall `289.95s`.
  - Training reached `3/3` at `2026-07-21 15:33:06 KST`; `WEIGHT_SYNC_DEBUG` showed each sync reached final tensor `lm_head.weight` and final policy barriers.
  - Debug conclusion: this exact 24-node targeted debug run did **not** reproduce the previous step-2 `sync_weights` hang. The hang is intermittent/environment-sensitive.
  - End-of-run checkpoint saved even though periodic checkpointing was disabled:
    `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/checkpoints/global_step_4`
    - verified files include `data.pt`, `trainer_state.pt`, `latest_ckpt_global_step.txt`, `policy/fsdp_config.json`, and all 16 policy model/optim/extra-state rank shards.
  - HF export completed at `2026-07-21 15:37:35 KST`:
    `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/exports/global_step_4/policy`
    - verified files include two `.safetensors` shards, `model.safetensors.index.json`, config/generation config, tokenizer files, and chat template.
  - HF Hub upload failed with `401 Unauthorized` after training completed; not a sync/training failure.
  - Ray stopped and logs were preserved to `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/ray_logs/`.
  - Final Slurm accounting at `2026-07-21 15:41:59 KST`: top-level job `COMPLETED`, elapsed `00:47:55`, exit `0:0`. `squeue` still briefly showed `COMPLETING` while cleanup drained, but `sacct` recorded successful completion.

## Longer instrumented no-eval run
- User asked to submit a longer debug run before going offline.
- Submitted **849721** on `gh`x24 at `2026-07-21 15:48 KST`:
  - job name `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_849721.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_rl_config.json`
  - local config `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug20.yaml`; remote copy created from the 3-step debug config with `trainer.max_steps=20`.
  - verified generated Hydra args: `trainer.max_steps=20`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.hf_save_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`, `SKYRL_WEIGHTSYNC_DEBUG=1`.
  - Slurm allocation started immediately on 24 nodes; as of `2026-07-21 15:49:24 KST`, state `RUNNING`, elapsed `1:09`, still in expected Ray startup cleanup (`Cleaning up existing Ray instances...`).
  - Live status at `2026-07-21 16:40:54 KST`: Slurm job still `RUNNING`, elapsed `00:52:39`; `squeue -u lukedhlee` shows this is the only queued/running user job. Earlier Ray status had 24 active Ray nodes, all 24/24 GPUs reserved, no pending Ray resource demands/failures.
  - Startup passed the relevant distributed phases:
    - Ray cluster ready by `2026-07-21 15:55:09 KST`.
    - `InferenceEngineClient initialized with 8 engines` by `2026-07-21 15:59:13 KST`.
    - `init_weight_sync_state` finished in `8.21s`; no checkpoint found, started from scratch.
    - Initial `sync_weights` completed at `2026-07-21 16:11:27 KST`, cost `123.13s`.
  - Step 1 completed:
    - generate `47.62s`; raw reward `0.328125`, pass@4 `0.5625`
    - fwd/logprobs `16.98s`; policy_train `76.13s`; post-step `sync_weights` `127.41s`
    - step wall `268.26s`; `Training Batches Processed: 1/20`
  - Step 2 reached:
    - generate `46.72s`; raw reward `0.59375`, pass@4 `0.75`
    - fwd/logprobs `10.21s`, so this run passed the prior 846086 step-2 fwd/logprob stall point.
    - `policy_train` started at `2026-07-21 16:16:53 KST`.
  - **Current suspected hard stall:** no log advance after file timestamp `2026-07-21 16:18:23 KST`; last progress line is `Policy Train epoch [1/1]: 15/16 (94%)` at `2026-07-21 16:17:53 KST`. There is no `16/16`, no `Finished: 'policy_train'`, and no step-2 `sync_weights` start.
  - Process/thread sample at `2026-07-21 16:31 KST`: sampled policy worker GPUs remain at 100% utilization; policy main threads are in `ep_poll`; `pt_autograd_0` threads are running at ~81% CPU. This supports an in-training policy/FSDP/autograd/collective stall rather than eval/checkpoint/generation/fwd-logprob/sync.
  - Current interpretation: 849721 is not stalled in eval/checkpoint/generation/fwd-logprob/sync. It is wedged inside step-2 policy-side training, likely around the final mini-batch/optimizer/FSDP collective. Do not `scancel` without explicit user approval.
  - Cancelled on user approval at `2026-07-21 16:43 KST`.
  - Final Slurm accounting: `CANCELLED by 910114`, elapsed `00:55:12`, exit `0:0`.

## Policy-train instrumented debug run
- Added remote SkyRL instrumentation in `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/worker.py`:
  - `POLICY_TRAIN_DEBUG` logs for ppo_train enter, each mini-batch before/after `training_step`, status all-reduce before/after, final policy barrier, and exit.
  - `POLICY_STEP_DEBUG` logs for per-rank `training_step` enter, `to_device`, forward, policy loss, backward, router replay teardown, optimizer step, entropy all-reduce, and return.
  - `faulthandler.dump_traceback_later(..., repeat=True)` enabled during policy_train with `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`, so a stall should dump Python stacks into the Slurm log.
  - Existing `WEIGHT_SYNC_DEBUG` instrumentation in `fsdp_worker.py` remains active.
  - Validated with `py_compile` and `git diff --check` on Vista.
- Submitted **850044** on `gh`x24 at `2026-07-21 16:47 KST`:
  - job name `vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/logs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_850044.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/configs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_rl_config.json`
  - local config `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_policydebug10.yaml`
  - generated Hydra args verified: `trainer.max_steps=10`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `generator.num_inference_engines=8`.
  - sbatch env verified: `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`, `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
  - initial Slurm status at `2026-07-21 16:47:26 KST`: `RUNNING`, elapsed `0:14`, 24 nodes allocated; log showed `Starting Ray cluster with 24 nodes, 1 GPUs/node` and `Cleaning up existing Ray instances...`.
  - Final status checked at `2026-07-21 18:52 KST`: job no longer in `squeue`; `sacct` top-level state `FAILED`, elapsed `01:17:22`, exit `1:0`.
  - Training itself completed all 10 requested steps:
    - `Training Batches Processed: 10/10 (100%)` logged at `2026-07-21 17:56:32 KST`.
    - Steps 1-10 all completed generate, fwd/logprobs/reward, policy_train, and sync_weights; no policy-train/sync stall reproduced.
    - Step 10 timings: generate `47.45s`, fwd/logprobs/reward `8.58s`, policy_train `70.75s`, sync_weights `153.67s`, full step wall `280.50s`.
    - Aggregate throughput: `2836s` for 10 steps = about `12.7` training steps/hour.
    - Step-10 reward metrics: `reward/avg_raw_reward=0.578125`, `reward/avg_pass_at_4=0.6875`.
  - Checkpoint/export artifacts exist:
    - checkpoint: `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints/global_step_11`
    - export: `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/exports/global_step_11/policy`
  - Failure was post-training/save-export, not the targeted training stall:
    - `Successfully saved model weights` logged at `2026-07-21 18:00:26 KST`.
    - Ray raised `WorkerCrashedError` at `2026-07-21 18:02:33 KST`.
    - Raylet reported worker `skyrl_entrypoint` PID `1232885` on IP `129.114.17.45` exited with connection error code 2 / EOF.
    - Memory monitors reported high node memory pressure on one policy worker rank on IP `129.114.18.199`, rising to `91.9%` node memory used (`17.2 GiB` available), during save/export.
  - Interpretation: 850044 is a useful negative sample for the original stall. The training loop was healthy for 10 steps under policy/debug instrumentation; the terminal failure looks like post-training save/export memory pressure or Ray worker crash during teardown/export.
  - Current Slurm queue at `2026-07-21 18:52:17 KST`: no jobs running or queued for `lukedhlee`.

## Resumed long eval/debug run
- User asked to "do the next thing" and will be away for hours, so submitted a longer unattended resumed run rather than restarting from scratch.
- Added/copyed config `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_resume11_eval10_policydebug60.yaml`.
- Submitted **850782** on `gh`x24 at `2026-07-21 18:57 KST`:
  - job name `vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60`
  - log `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/logs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_850782.out`
  - generated config `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/configs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_rl_config.json`
  - resumes from `/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints`, whose `latest_ckpt_global_step.txt` says `11`.
  - verified generated Hydra args: `trainer.max_steps=60`, `trainer.resume_mode=latest`, `trainer.eval_interval=10`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=10`, `trainer.hf_save_interval=999999`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`.
  - verified debug env: `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`, `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
  - Initial Slurm state at `2026-07-21 18:57:18 KST`: `PENDING`, reason `(Priority)`, 24 nodes requested, wall `12:00:00`.
  - Queue detail at `2026-07-21 18:59:45 KST`: still `PENDING (Priority)`. `scontrol` estimated start at `2026-07-22 00:28:23 KST` with a candidate 24-node schedule; this may move.
  - Live check at `2026-07-22 00:26-00:27 KST`: job is `RUNNING`, elapsed `~4:10`, but appears hard-stalled. Log mtime is `2026-07-21 20:46:38 KST` (shown as `2026-07-21 06:46:38 -0500` by `stat`), so no log progress for ~3h40.
  - Resume succeeded and step 11/12 progressed:
    - Step 11 post-resume `sync_weights` completed at `2026-07-21 20:37:34 KST`, cost `138.40s`.
    - Step 12 completed at `2026-07-21 20:42:04 KST`: policy_train `77.93s`, sync_weights `130.31s`, step wall `270.57s`, train reward `avg_raw_reward=0.5625`, train pass@4 `0.6875`.
  - Current stall:
    - Step 13 generate/fwd completed.
    - Step 13 `policy_train` started at `2026-07-21 20:43:00 KST`, finished at `20:44:17 KST`, cost `77.19s`.
    - Stalled in the following `sync_weights`; last precise marker is `WEIGHT_SYNC_DEBUG rank=0 phase=chunk_update_task_await_before chunk=18666 name=model.layers.47.mlp.experts.62.down_proj.weight` at `2026-07-21 20:46:27 KST`.
    - No corresponding `await_after`, no final policy barrier, and no `Finished: 'sync_weights'` after that.
  - Interpretation update: stall timing is nondeterministic/environment-sensitive. It has occurred at different global steps and phases across representative runs; this resumed run provides a clean step-13 weight-sync hang after policy_train completed.
