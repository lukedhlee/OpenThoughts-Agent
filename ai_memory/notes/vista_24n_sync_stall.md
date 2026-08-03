# Vista 24-node MoE GRPO — the intermittent stall investigation (full run record)

Every Vista `gh`×24 Qwen3-30B-A3B gsm8k GRPO job we submitted 2026-07-20 → 2026-07-22, verbatim, with
the per-step timings, the stall phases, and the debug instrumentation added at each stage.
Read this before relaunching anything 24-node on Vista, or before re-deriving whether the stall is
deterministic (**it is not** — it moved between `fwd_logprobs`, `policy_train`, and `sync_weights`).

Carved out of `handoff.md` 2026-07-29 (append-only from here; corrections go at the bottom as dated
entries). Session logs with the narrative: [[2026-07-19_vista_moe_gsm8k_grpo]],
[[2026-07-20_vista_moe_gsm8k_grpo]] under `ai_memory/logs/`. Parent: [[handoff]].

⚠ **EVERY `pass_at_1 ≈ 0.44–0.47` IN THIS FILE IS A FORMAT ARTIFACT, NOT MATH ABILITY** (measured
2026-07-29, [[gsm8k_format_artifact]]). The model answers in `\boxed{N}` and never emits the `####`
the strict scorer requires. Paired re-scoring of identical rollouts: `strict@1024=41.6%` (reproduces
these numbers) vs `flexible@4096=90.67%` (≈ Qwen's published 91.81%). Any upward trend in these
curves is the model learning FORMAT COMPLIANCE, not reasoning. Do not cite as reasoning gains.

---

## Environment

- **NCCL (otagent env on Vista):** `nvidia-nccl-cu12` / `libnccl` **2.28.9**
  (`NCCL version 2.28.9+cuda12.9`); torch `2.11.0+cu128`. Prior debug YAMLs used `NCCL_DEBUG: WARN`.
- GPUs: 24 Vista GH200 GPUs = 24 nodes, 1 GPU/node; layout 16 policy/ref + 8 vLLM.

## Working recipe (as of the last Vista run)

- Partition **`gh`**, `critic_num_gpus_per_node: 1`, `gpu_memory_utilization: 0.80`,
  `policy_strict_spread_pg: true`
- 16 policy EP=4×FSDP=4 + 8 vLLM
- **Eval/checkpoint:** `eval_before_train: true`, `eval_interval: 10`, `ckpt_interval: 5`, `seed: 42`
  — judge by **val**, not train `pass@4`

## Relaunch command (only when asked)

```bash
NUM_NODES=24 PARTITION=gh TIME_LIMIT=12:00:00 \
  RL_CONFIG=./hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo.yaml \
  JOB_NAME=vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5 \
  bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
```

---

## Run-by-run record

### Earlier jobs (2026-07-20)
- **843437** cancelled (hung step-15; no val).
- **844374** submitted then **cancelled on user request** (do not relaunch).
- Tip ready: `00e40bd8`; local next-run config now has `eval_before_train: true`, `eval_interval: 10`,
  `ckpt_interval: 5`, `seed: 42`.

### 844625 — eval-every-step, cancelled on user request
Submitted on `gh`×24 from commit `135ca0a5`, then **cancelled on user request** at
~`2026-07-20 09:18` TACC local.
- Job name: `vista_moe30b_gsm8k_grpo_24n_eval1`
- Slurm state: `COMPLETING` immediately after cancel; no checkpoint was saved because
  `ckpt_interval=20` and it had not reached step 20.
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval1/logs/vista_moe30b_gsm8k_grpo_24n_eval1_844625.out`
- Step-0 eval: `pass_at_1=0.4412433662`, eval wall `1847.62s`.
- Step 1: train step wall `400.03s` (~9.0 train-only steps/hour); train `avg_raw_reward=0.4765625`,
  `avg_pass_at_4=0.65625`; eval `pass_at_1=0.4579226687`, eval wall `1863.94s`.
- Step 2: train step wall `390.61s`; train `avg_raw_reward=0.4296875`, `avg_pass_at_4=0.59375`;
  eval `pass_at_1=0.4427596664`, eval wall `1868.44s`.
- Step 3: train step wall `384.85s`; train `avg_raw_reward=0.546875`, `avg_pass_at_4=0.71875`;
  eval `pass_at_1=0.4632297195`, eval wall `1866.20s`.
- Step 4: train step wall `381.76s`; train `avg_raw_reward=0.546875`, `avg_pass_at_4=0.75`;
  eval `pass_at_1=0.4586808188`, eval wall `1872.24s`.
- Step 5: train step wall `385.88s`; train `avg_raw_reward=0.5390625`, `avg_pass_at_4=0.71875`;
  eval `pass_at_1=0.4632297195`, eval wall `1873.57s`.
- Step 6: train step wall `386.05s`; train `avg_raw_reward=0.578125`, `avg_pass_at_4=0.78125`;
  eval `pass_at_1=0.4586808188`, eval wall `1874.68s`.
- Step 7: train step wall `385.14s`; train `avg_raw_reward=0.65625`, `avg_pass_at_4=0.78125`;
  eval `pass_at_1=0.4670204701`, eval wall `1880.29s`.
- Step 8: train step wall `388.19s`; train `avg_raw_reward=0.421875`, `avg_pass_at_4=0.59375`;
  eval `pass_at_1=0.4601971190`, eval wall `1885.05s`.
- Step 9: train step wall `388.82s`; train `avg_raw_reward=0.3203125`, `avg_pass_at_4=0.46875`;
  eval `pass_at_1=0.4442759666`, eval wall `1870.31s`.
- Step 10: train step wall `381.94s`; train `avg_raw_reward=0.6171875`, `avg_pass_at_4=0.71875`;
  eval `pass_at_1=0.4730856710`, eval wall `1879.10s`.
- Step 11: train step wall `385.64s`; train `avg_raw_reward=0.375`, `avg_pass_at_4=0.5`; eval started
  `2026-07-20 07:40:36` and was at 5/42 (12%) as of `07:43:35`.
- With eval every step, effective cadence is ~1.6 steps/hour because each full eval costs ~31 min.
  Train-only throughput is now ~9.2-9.4 steps/hour.
- Eval curve through step 10: `0.44124`, `0.45792`, `0.44276`, `0.46323`, `0.45868`, `0.46323`,
  `0.45868`, `0.46702`, `0.46020`, `0.44428`, `0.47309`.

### 846086 — first hard stall (step-2 `fwd_logprobs_values_reward`), TIMEOUT
Submitted on `gh`×24 from commit `00e40bd8`, then **TIMED OUT**.
- Job name: `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5`
- Slurm final state: `TIMEOUT`, elapsed `12:00:27`; Slurm killed it at `2026-07-20T21:20:14` CDT due to
  time limit.
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_846086.out`
- Checkpoint path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/checkpoints`
- Submitted config verified: `trainer.eval_interval=10`, `trainer.eval_before_train=true`,
  `trainer.ckpt_interval=5`, `policy_num_nodes=16`, `generator.num_inference_engines=8`.
- Checkpoint load: `No checkpoint found, starting training from scratch`.
- Baseline eval step 0: `pass_at_1=0.4397270660`, eval wall `1926.32s` (~32.1 min).
- Step 1 generation done: `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`;
  policy_train started `2026-07-20 10:14:33` TACC local.
- Step 1 completed at `10:19:20`: step wall `404.12s`, train-only throughput `8.91` steps/hour, no eval
  due `eval_interval=10`.
- Step 2 generation completed at `10:20:51`: `reward/avg_raw_reward=0.4375`,
  `reward/avg_pass_at_4=0.5625`.
- **Hard stall:** log did not advance past step-2 `Started: 'fwd_logprobs_values_reward'` /
  `MESH_DISPATCH method=forward issued forward.remote() to 16 actors` from `10:20:51` until Slurm
  timeout at `21:20`.
  - Ray status: 24 nodes active, 24/24 GPUs reserved; 16 `FSDPPolicyWorkerBase` +
    8 `AsyncVLLMInferenceEngine` actors alive.
  - No trainer traceback/CUDA/NCCL fatal surfaced; several init-time `InfoActor` deaths are present but
    policy/vLLM actors are alive.
  - GPU sample: several ~89GB policy/ref GPUs idle at 0% while many ~21GB vLLM-sized GPUs are at 100%.
- No checkpoint was saved. First checkpoint would have been step 5, but only step 1 completed and step 2
  stalled.
- Do not relaunch the exact same 24n recipe blindly; this looks like distributed
  forward/logprob/collective instability, not eval cadence.

### 848839 — retry without fixes; stall moved to step-2 `sync_weights`
Submitted on `gh`×24 from the same commit/config (`00e40bd8`) at user request to retry without fixes.
- Job name: `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry_848839.out`
- Checkpoint path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/checkpoints`
- Actual SkyRL command verified same intended geometry: `trainer.eval_interval=10`,
  `trainer.ckpt_interval=5`, `trainer.eval_before_train=true`, `policy_num_nodes=16`,
  `generator.num_inference_engines=8`.
- Baseline eval step 0 finished at `22:43:16` CDT: `pass_at_1=0.4397270660`, eval wall `1927.86s`.
- Step 1 completed at `22:50:00`: step wall `403.85s`, `reward/avg_raw_reward=0.4765625`,
  `reward/avg_pass_at_4=0.65625`.
- Step 2 generation completed at `22:51:30`: `reward/avg_raw_reward=0.4765625`,
  `reward/avg_pass_at_4=0.59375`.
- Step 2 `fwd_logprobs_values_reward` **did not stall this time**; it finished in `17.80s`.
- Step 2 `policy_train` finished in `137.46s`.
- **Likely hard stall now at step-2 `sync_weights`:** last trainer progress is `Started: 'sync_weights'`
  at `22:54:06`; log/stat still frozen at `22:54:07` as of `23:07:11` while Slurm job remained
  `RUNNING`.
- No checkpoint yet; first checkpoint is step 5.
- Do not `scancel` without explicit user OK.
- Debug update (2026-07-21 KST):
  - Still `RUNNING` as of `2026-07-21 13:51:12 KST`; log/stat still frozen at
    `2026-07-21 12:54:07 KST`.
  - Ray status: 24 active nodes, all 24 GPUs reserved, no pending demands/failures.
  - Ray actors: 16 `FSDPPolicyWorkerBase` alive, 8 `AsyncVLLMInferenceEngine` alive; `InfoActor` deaths
    are init-time leftovers.
  - Ray tasks retained only the last 100 tasks, all `AsyncVLLMInferenceEngine.update_named_weights` and
    all `FINISHED`.
  - Process names on all 16 policy workers are
    `ray::FSDPPolicyWorkerBase.broadcast_to_inference_engines`; vLLM actors are idle/alive.
  - `/proc`/thread sampling: policy actor main threads are in `ep_poll`; no `pstack`/`gdb` attach due
    ptrace denial; no NCCL trace files were dumped under `/scratch/11584/lukedhlee/nccl_trace_*`.
  - vLLM log shows step-1 weight reload/cache reset completed at `2026-07-21 12:50:00 KST`; no
    equivalent step-2 finish/reset line after `2026-07-21 12:54:06 KST`.
  - Interpretation: current stall is inside policy-side `broadcast_to_inference_engines`, likely at a
    policy barrier or Ray/vLLM reload/update await after the vLLM update tasks have mostly/completely
    returned.
  - Remote SkyRL source staged for next launch with `WEIGHT_SYNC_DEBUG` phase logging in
    `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py`; syntax
    verified with `py_compile` and `git diff --check`. This does not affect the already-running job.
- **848839** cancelled on user approval at `2026-07-21 14:16 KST` after remaining stuck in step-2
  `sync_weights`.

### 849329 — config-validation fail-fast (batch size < policy DP LCM)
Submitted on `gh`x24 at `2026-07-21 14:17 KST` for targeted weight-sync debugging, then failed fast
before training.
- Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_849329.out`
- Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_rl_config.json`
- Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug.yaml`
- Purpose: reproduce the step-2 `sync_weights` stall quickly with `WEIGHT_SYNC_DEBUG=1`.
- Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=8`,
  `trainer.policy_mini_batch_size=8`, `trainer.eval_interval=999999`,
  `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`,
  `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`,
  `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
- Note: runner startup prints generic env `NUM_INFERENCE_ENGINES=24` / `POLICY_NUM_NODES=24`; generated
  SkyRL hydra args are the authoritative geometry and are correct.
- Ray cleanup/startup was healthy; the long `Cleaning up existing Ray instances...` pause was a
  sequential 24-node `ray stop --force` cleanup loop.
- Failed at `2026-07-21 14:23:50 KST` in SkyRL config validation: `train_batch_size (8)` must be at
  least the policy DP LCM (`policy_dp_size=16`, `lcm_dp_size=16`). No training step ran.

### 849358 — instrumentation bug (`UnboundLocalError: os`)
Submitted on `gh`x24 at `2026-07-21 14:26 KST` with corrected minimum batch size for the same targeted
weight-sync debug.
- Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_849358.out`
- Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_rl_config.json`
- Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=16`,
  `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`,
  `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`,
  `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`,
  `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
- Final state: `FAILED`, elapsed `23:47`.
- Progress before failure: Ray ready, vLLM loaded, FSDP policy weights loaded, initial
  `init_weight_sync_state` finished in `8.72s`, `load_checkpoints` found no checkpoint, initial
  `sync_weights` started at `2026-07-21 14:47:36 KST` and logged
  `Finished: 'sync_weights', time cost: 1.78s`.
- Failure cause: instrumentation bug in remote `fsdp_worker.py`:
  `UnboundLocalError: cannot access local variable 'os'` because a later local `import os` shadowed
  module-level `os`. This is not evidence of the original weight-sync stall.

### 849442 — 3 steps clean; stall did NOT reproduce
Submitted on `gh`x24 at `2026-07-21 14:52 KST` after fixing the instrumentation bug.
- Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_849442.out`
- Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_rl_config.json`
- Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=16`,
  `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`,
  `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`,
  `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`,
  `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
- Slurm/job startup:
  - At `2026-07-21 15:04 KST`, one vLLM engine hit retryable FlashInfer JIT/FileLock
    `OSError: [Errno 116] Stale file handle`; vLLM retried and recovered.
  - `InferenceEngineClient initialized with 8 engines` by `2026-07-21 15:05:10 KST`.
  - `load_checkpoints`: no checkpoint found, started from scratch.
- Initial `sync_weights` completed at `2026-07-21 15:18:17 KST`, cost `149.97s`, after streaming all
  `18866` tensors through `WEIGHT_SYNC_DEBUG` to `lm_head.weight`.
- Step 1:
  - generate `48.68s`; `reward/avg_pass_at_4=0.5625`, `reward/avg_raw_reward=0.328125`
  - fwd/logprobs/reward `19.32s`; policy_train `84.64s`
  - post-step `sync_weights` completed at `2026-07-21 15:23:28 KST`, cost `157.85s`
  - step wall `310.61s`; train batches `1/3`
- Step 2:
  - generate `45.95s`; `reward/avg_pass_at_4=0.8125`, `reward/avg_raw_reward=0.640625`
  - fwd/logprobs/reward `10.06s`; policy_train `75.54s`
  - post-step `sync_weights` completed at `2026-07-21 15:28:16 KST`, cost `156.42s`
  - step wall `288.03s`; train batches `2/3`
  - This explicitly passed the previous step-2 `sync_weights` hang point.
- Step 3:
  - generate `45.08s`; `reward/avg_pass_at_4=0.75`, `reward/avg_raw_reward=0.5625`
  - fwd/logprobs/reward `10.18s`; policy_train `78.31s`
  - final `sync_weights` completed at `2026-07-21 15:33:06 KST`, cost `156.31s`
  - step wall `289.95s`; train batches `3/3`; trainer logged `Reached max training steps (3)`.
- Debug conclusion: the 24-node targeted run did **not** reproduce the stall. All three post-train
  weight syncs completed with stable ~156-158s sync cost. The earlier step-2 sync hang is
  intermittent/environment-sensitive, not deterministic for this recipe.
- End-of-run checkpoint saved despite `ckpt_interval=999999`:
  - `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/checkpoints/global_step_4`
  - verified files include `data.pt`, `trainer_state.pt`, `latest_ckpt_global_step.txt`,
    `policy/fsdp_config.json`, and all 16 policy model/optim/extra-state rank shards.
- HF export completed at `2026-07-21 15:37:35 KST`:
  - `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/exports/global_step_4/policy`
  - verified files include two `.safetensors` shards, `model.safetensors.index.json`,
    config/generation config, tokenizer files, and chat template.
- HF Hub upload failed with `401 Unauthorized` after training completed; not a training/sync failure.
- Ray stopped and logs were preserved to
  `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/ray_logs/`.
- Final Slurm accounting at `2026-07-21 15:41:59 KST`: top-level job `COMPLETED`, elapsed `00:47:55`,
  exit `0:0`. `squeue` still briefly showed `COMPLETING` while cleanup drained, but `sacct` recorded
  successful completion.

### 849721 — 20-step attempt; stall in step-2 `policy_train`
Submitted on `gh`x24 at `2026-07-21 15:48 KST` for a longer instrumented no-eval sync-stall sample.
- Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_849721.out`
- Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_rl_config.json`
- Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug20.yaml`; remote copy
  created from the 3-step debug config with `trainer.max_steps=20`.
- Actual generated Hydra args verified: `trainer.max_steps=20`, `trainer.train_batch_size=16`,
  `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`,
  `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`,
  `trainer.hf_save_interval=999999`, `trainer.placement.policy_num_nodes=16`,
  `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`,
  `generator.n_samples_per_prompt=4`, `SKYRL_WEIGHTSYNC_DEBUG=1`.
- Slurm allocation started immediately on 24 nodes; as of `2026-07-21 15:49:24 KST`, state `RUNNING`,
  elapsed `1:09`, still in expected Ray startup cleanup (`Cleaning up existing Ray instances...`).
- Live status at `2026-07-21 16:40:54 KST`: Slurm job still `RUNNING`, elapsed `00:52:39`;
  `squeue -u lukedhlee` shows this is the only queued/running user job. Earlier Ray status had 24
  active Ray nodes, all 24/24 GPUs reserved, no pending Ray resource demands/failures.
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
- **Suspected hard stall:** no log advance after file timestamp `2026-07-21 16:18:23 KST`; last
  progress line is `Policy Train epoch [1/1]: 15/16 (94%)` at `2026-07-21 16:17:53 KST`. There is no
  `16/16`, no `Finished: 'policy_train'`, and no step-2 `sync_weights` start.
- Process/thread sample at `2026-07-21 16:31 KST`: sampled policy worker GPUs remain at 100%
  utilization; policy main threads are in `ep_poll`; `pt_autograd_0` threads are running at ~81% CPU.
  This supports an in-training policy/FSDP/autograd/collective stall rather than
  eval/checkpoint/generation/fwd-logprob/sync.
- Interpretation: 849721 is not stalled in eval/checkpoint/generation/fwd-logprob/sync. It is wedged
  inside step-2 policy-side training, likely around the final mini-batch/optimizer/FSDP collective.
  Do not `scancel` without explicit user approval.
- Cancelled on user approval at `2026-07-21 16:43 KST`.
- Final Slurm accounting: `CANCELLED by 910114`, elapsed `00:55:12`, exit `0:0`.

### 850044 — 10 steps clean; post-training save/export memory crash
Submitted on `gh`x24 at `2026-07-21 16:47 KST` for the next targeted policy-train debug run.
- Job name: `vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/logs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_850044.out`
- Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/configs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_rl_config.json`
- Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_policydebug10.yaml`
- Remote SkyRL instrumentation active in
  `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/worker.py`:
  - `POLICY_TRAIN_DEBUG` logs for ppo_train enter, each mini-batch before/after `training_step`, status
    all-reduce before/after, final policy barrier, and exit.
  - `POLICY_STEP_DEBUG` logs for per-rank `training_step` enter, `to_device`, forward, policy loss,
    backward, router replay teardown, optimizer step, entropy all-reduce, and return.
  - `faulthandler.dump_traceback_later(..., repeat=True)` enabled during policy_train with
    `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`, so a stall should dump Python stacks into the Slurm log.
  - Existing `WEIGHT_SYNC_DEBUG` instrumentation in `fsdp_worker.py` remains active.
- Validated with `py_compile` and `git diff --check` on Vista.
- Generated Hydra args verified: `trainer.max_steps=10`, `trainer.train_batch_size=16`,
  `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`,
  `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`,
  `trainer.placement.policy_num_nodes=16`, `generator.num_inference_engines=8`.
- Sbatch env verified: `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`,
  `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
- Initial Slurm status at `2026-07-21 16:47:26 KST`: `RUNNING`, elapsed `0:14`, 24 nodes allocated; log
  showed `Starting Ray cluster with 24 nodes, 1 GPUs/node` and `Cleaning up existing Ray instances...`.
- Final status checked at `2026-07-21 18:52 KST`: job no longer in `squeue`; `sacct` top-level state
  `FAILED`, elapsed `01:17:22`, exit `1:0`.
- Important outcome: the suspected training stall did **not** reproduce. The run reached
  `Training Batches Processed: 10/10 (100%)` at `2026-07-21 17:56:32 KST`; steps 1-10 all completed
  generate, fwd/logprobs, policy_train, and sync_weights.
  - Step 10 timings: generate `47.45s`, fwd/logprobs/reward `8.58s`, policy_train `70.75s`,
    sync_weights `153.67s`, full step wall `280.50s`.
  - Training throughput over steps 1-10: `2836s` for 10 steps = about `12.7` training steps/hour.
  - Step-10 reward metrics: `reward/avg_raw_reward=0.578125`, `reward/avg_pass_at_4=0.6875`.
- Final checkpoint and export were created despite the failed Slurm state:
  - checkpoint: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints/global_step_11`
  - export: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/exports/global_step_11/policy`
- Failure mode: after training and after `Successfully saved model weights` at
  `2026-07-21 18:00:26 KST`, Ray raised `WorkerCrashedError` at `2026-07-21 18:02:33 KST`. The Raylet
  reported worker `skyrl_entrypoint` PID `1232885` on IP `129.114.17.45` exited with connection error
  code 2 / EOF.
- Memory evidence near the crash: one policy worker rank on IP `129.114.18.199` reported node memory
  pressure during save/export, rising from `85.2%` to `91.9%` node memory used (`18.3` to `17.2 GiB`
  available), with monitor advice to reduce concurrent trials / parallel generation workers. This
  points to a post-training save/export memory crash, not the earlier policy-train hang.
- Current Slurm queue at `2026-07-21 18:52:17 KST`: no jobs running or queued for `lukedhlee`.

### 850782 — resume from step 11; stall at step-13 `sync_weights`, TIMEOUT
Submitted on `gh`x24 at `2026-07-21 18:57 KST` for an unattended resumed long debug run.
- Job name: `vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60`
- Purpose: continue from `850044`'s healthy `global_step_11` checkpoint while keeping policy-train and
  weight-sync instrumentation active; add validation/checkpoint cadence every 10 steps.
- Local/remote config:
  `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_resume11_eval10_policydebug60.yaml`
- Generated config: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/configs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_rl_config.json`
- Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/logs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_850782.out`
- Verified generated Hydra args:
  - `trainer.max_steps=60`
  - `trainer.resume_mode=latest`
  - `trainer.ckpt_path=/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints`
  - `trainer.export_path=/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/exports`
  - `trainer.eval_interval=10`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=10`,
    `trainer.hf_save_interval=999999`
  - `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`
  - `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`,
    `generator.num_inference_engines=8`
- Verified sbatch env includes `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`, and
  `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
- Initial Slurm state at `2026-07-21 18:57:18 KST`: `PENDING`, reason `(Priority)`, 24 nodes requested,
  12h wall limit.
- Queue detail at `2026-07-21 18:59:45 KST`: still `PENDING (Priority)`. `scontrol` showed estimated
  start `2026-07-22 00:28:23 KST` and scheduled nodes listed; this is an estimate and may move.
- Live check at `2026-07-22 00:26-00:27 KST`: job is `RUNNING`, elapsed `~4:10`, but appears
  hard-stalled. Log mtime is `2026-07-21 20:46:38 KST` (shown by `stat` as
  `2026-07-21 06:46:38 -0500`), so no log progress for ~3h40.
- Resume worked: checkpoint global step 11 loaded, and training progressed.
- Step 11 post-resume sync completed at `2026-07-21 20:37:34 KST`, cost `138.40s`, progress
  `Training Batches Processed: 11/60`.
- Step 12 completed at `2026-07-21 20:42:04 KST`:
  - policy_train `77.93s`, sync_weights `130.31s`, step wall `270.57s`
  - reward `avg_raw_reward=0.5625`, train `avg_pass_at_4=0.6875`
- Step 13 reached:
  - generate/fwd completed; `policy_train` started at `2026-07-21 20:43:00 KST` and finished at
    `20:44:17 KST`, cost `77.19s`.
  - Current stall is in the following `sync_weights`, not policy_train. Last log lines are
    `WEIGHT_SYNC_DEBUG rank=0 phase=chunk_update_task_await_before chunk=18666 name=model.layers.47.mlp.experts.62.down_proj.weight`
    at `2026-07-21 20:46:27 KST`; no corresponding `await_after`, no final barrier, no
    `Finished: 'sync_weights'`.
- Interpretation update: stall timing is not deterministic. It has now appeared at prior step-2
  policy/fwd/sync phases and at resumed step 13 sync; short runs can complete without reproducing it.
- Final: **TIMEOUT** at `2026-07-21T18:17:31` CDT after wall `12:00:21`; never advanced past the
  step-13 sync hang. No new checkpoint beyond parent `global_step_11`.

### 854962 — NCCL_DEBUG=INFO capture attempt (failed to capture)
Submitted on `gh`×24 at `2026-07-22 16:01 KST` for NCCL_DEBUG=INFO sync-stall capture.
- Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo`
- Config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug20.yaml` (`max_steps=20`,
  no eval, `NCCL_DEBUG: INFO`, `SKYRL_WEIGHTSYNC_DEBUG=1`)
- Log: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo_854962.out`
- Sbatch verified: `export NCCL_DEBUG="INFO"` wins after an earlier WARN default; still has
  `SKYRL_WEIGHTSYNC_DEBUG=1`.
- Cancelled at user request `2026-07-22 04:02 CDT` after ~1h18m / **~11–12/20** clean steps (no stall).
- **NCCL_DEBUG=INFO verdict:** env was exported, but Slurm `.out` had **no real NCCL INFO library
  lines** (no Bootstrap/Channel/IB dumps). Log bloat (~235MB) was almost entirely
  `WEIGHT_SYNC_DEBUG`. Only useful side notes: P2P/SHM disabled (expected 1GPU/node); torch
  deprecation warns (`NCCL_BLOCKING_WAIT`, `TORCH_NCCL_TRACE_BUFFER_SIZE`); DTensor fsdp×ep double
  all-reduce warn. **No new stall root-cause from INFO.**

### 855070 — cancelled before start
Cancelled before start (user: fix NCCL capture first).

### NCCL debug fix (verified on gh-dev smoke 855073)
- MarinSkyRL `prepare_runtime_environment` now forwards `NCCL_DEBUG_FILE` + mkdir parent (Vista
  checkout `lukedhlee/fix-ray-255-pg-strategy`).
- Set `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,COLL`,
  `NCCL_DEBUG_FILE=/scratch/11584/lukedhlee/nccl_debug/nccl_%h_%p.log`.
- Smoke produced real files e.g. `nccl_c642-002_*.log` with **~44k `NCCL INFO` lines** each
  (NCCL 2.28.9). Evidence kept in `$SCRATCH/nccl_debug/smoke_855073/`.

### 855080 — 50 steps, eval every 5, file-backed NCCL INFO (last Vista submission)
Submitted `2026-07-22 04:18 CDT`.
- Job: `vista_moe30b_gsm8k_grpo_24n_50step_eval5` (exp dir `..._50step_eval5_2` due to name collision)
- Config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_50step_eval5.yaml`
- `max_steps=50`, `eval_interval=5`, `eval_before_train=true`, `ckpt_interval=5`,
  `train_batch_size=16`
- Log: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_50step_eval5_2/logs/`
- NCCL files: `$SCRATCH/nccl_debug/nccl_%h_%p.log`; FR: `$SCRATCH/nccl_fr/nccl_trace_`
