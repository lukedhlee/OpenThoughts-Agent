# Jupiter Megatron bring-up — validation record, checkpoint/resume, speed vs FSDP2

The full operational record of standing Megatron up on Jupiter (aarch64 GH200): the env that works, the
smoke jobs that passed, exactly which checkpoint/resume modes are validated and which are NOT, the
launcher fixes it required, and the failed attempts worth remembering.
Read this before running Megatron on Jupiter, before changing checkpoint strategy, or before assuming a
Megatron checkpoint can be resumed onto a different topology (**it cannot**).

Carved out of `handoff.md` 2026-07-29 (append-only from here). Why we switched at all:
[[megatron_vs_fsdp2]]. Full session log: `ai_memory/logs/2026-07-26_jupiter_megatron_6n_gsm8k.md`.
FSDP2 side: [[jupiter_bringup_and_throughput]].

---

## Env

- Native env: `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl-megatron`.
- **aarch64 blocker cleared:** built/installed Megatron Core 0.18.0, Megatron Bridge 0.5.0,
  TransformerEngine 2.11.0, torch 2.11.0+cu128, transformers 5.8.x, vLLM 0.22.0 in `envs/rl-megatron`.
- **SkyRL Megatron path is green on Jupiter** (`distributed/megatron/`, `workers/megatron/`,
  `config/megatron_config/`) using that native env.

## Configs and launcher fixes

Added smoke configs:
- `hpc/skyrl_yaml/jupiter/2node_qwen3_0p6b_gsm8k_grpo_megatron_smoke.yaml`
- `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_megatron_smoke.yaml`
- launcher wrapper `hpc/skyrl_standard/jupiter/run_gsm8k_megatron_grpo.sh`

Launcher fixes:
- `hpc/sbatch_rl/universal_rl.sbatch` emits RL env overrides before activation, so
  `RL_ENV_DIR=.../envs/rl-megatron` takes effect.
- `hpc/rl_config_utils.py` treats Megatron `model_config_kwargs` / `transformer_config_kwargs` as Hydra
  `++` optional overrides.
- `hpc/rl_config_utils.py` respects YAML `hf_hub_repo_id: null` instead of auto-defaulting Hub upload.

## Verified jobs (2026-07-25)

- `1043401`: 2-node Qwen3-0.6B Megatron smoke completed, exit `0:0`.
- `1043464`: 6-node Qwen3-30B-A3B trained both steps but failed in final Megatron optimizer checkpoint
  save (`gloo ... Connection closed by peer`).
- `1043650`: 6-node Qwen3-30B-A3B no-save smoke completed, exit `0:0`, elapsed `00:39:11`; train loop
  reached `Training done!`.

## Update 2026-07-26 — checkpoint save/resume validated

- Faster 6-node no-save Megatron smoke completed: job `1045827`, exit `0:0`, elapsed `00:30:38`.
  - Shape: 4 vLLM rollout GPUs + 16 Megatron policy/ref GPUs (`20/24` Ray GPUs), batch size 16,
    `MAX_STEPS=1`.
  - Key markers: `InferenceEngineClient initialized with 4 engines`, `trainer/global_step: 1`,
    `Training done!`.
- Model-only checkpoint save smoke completed after save-planning patch: job `1045872`, exit `0:0`,
  elapsed `00:30:46`; saved `global_step_1` plus end checkpoint `global_step_2`.
- Resume smoke completed after disabling Megatron `FullyParallelLoadStrategyWrapper` on load: job
  `1045992`, exit `0:0`, elapsed `00:32:22`; loaded from `global_step_1` in `50.96s`, ran resumed step
  2, saved `global_step_2` plus end checkpoint `global_step_3`.
- Full Megatron distributed optimizer-state checkpoint save/load is now validated for same-topology
  resume using Megatron `dp_reshardable` optimizer checkpoint metadata.
  - Save: job `1046376`, exit `0:0`, elapsed `00:33:51`; saved full policy checkpoints at
    `global_step_1` and `global_step_2`, each with `.metadata` and 16 `distcp` shards.
  - Resume: job `1046602`, exit `0:0`, elapsed `00:33:40`; loaded model + optimizer + LR scheduler from
    `global_step_2`, ran resumed step 3, and finished `Training done!`.

## Speed compare against FSDP2 6n fast

Megatron job `1047736`, 6 nodes, 16 policy GPUs + 8 vLLM engines, batch 32, no eval/ckpt,
`MAX_STEPS=3`, completed `0:0` in `00:46:16`.
- Megatron steps 1-3 avg: `step=313.6s`, `generate=101.9s`, `fwd=22.8s`, `policy_train=66.6s`,
  `sync_weights=122.1s`.
- Megatron warm steps 2-3 avg: `step=301.9s`, `policy_train=59.1s`, `sync_weights=122.5s`.
- FSDP2 baseline job `1042857` same 6n fast geometry avg all available train rows: `step=425.6s`,
  `generate=105.1s`, `fwd=25.3s`, `policy_train=110.7s`, `sync_weights=184.1s`.
- Interpretation: Megatron is about `1.36x` faster on full step average (`425.6/313.6`) and `1.66x`
  faster on policy train (`110.7/66.6`) for GSM8K; warm-step comparison is about `1.41x` step and
  `1.87x` policy train. Startup was heavy but amortized per-step timings are clearly better.

Earlier 6-node no-save smoke per-step numbers (job `1043650`, 24 GH200 GPUs, exit `0:0`, elapsed
`00:39:11`):
- Step 1: generate `110.69s`, fwd/logprobs `29.13s`, policy_train `78.94s`, sync_weights `115.47s`,
  step `334.36s`.
- Step 2: generate `119.08s`, fwd/logprobs `14.39s`, policy_train `58.47s`, sync_weights `117.00s`,
  step `309.01s`.

## Failed attempts worth remembering

- `1045839` hit NCCL error during checkpoint save planning.
- `1045911` timed out in policy checkpoint load using the old fully-parallel load wrapper.
- `1046102` hit Megatron's default `fully_sharded_model_space` flattened-range unsupported error.
- `1046202` showed `fully_reshardable` optimizer save is not stable on Jupiter/Ray because its Gloo DP
  gather closed by peer.
- `1046522` loaded model then OOM'd during optimizer restore until the load path started releasing
  loaded state before restoring the next component.

## Operational lessons and caveats

- **30B Megatron can be quiet for 10+ minutes** after vLLM `Model loading took ...` due to FlashInfer
  MoE autotuning and worker bring-up; **do not call it wedged too early.**
- **Megatron optimizer checkpoint portability across topology/TP/PP/EP changes is NOT validated.** The
  supported full-state path is same-topology resume with `dp_reshardable`.
- 30B Megatron full checkpoint/resume works on the validated same-topology 6-node geometry with
  `SAVE_OPTIMIZER_STATES=true` for save and `LOAD_OPTIMIZER_STATES=true` for resume. Cross-topology
  optimizer resharding remains unsupported/unvalidated; **do not use `fully_reshardable` on
  Jupiter/Ray** unless specifically debugging it.
- Apex is still absent; smoke configs disable `gradient_accumulation_fusion` to avoid the Apex fused
  grad path.
- Earlier smoke configs set `ckpt_interval: -1` and `hf_save_interval: -1` so train-loop validation
  does not trip the checkpoint path.
