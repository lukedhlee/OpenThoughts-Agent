# Jupiter FSDP2 GRPO — bring-up ladder, r10–r13, and the 4n/6n/8n throughput results

The complete Jupiter FSDP2 story: the dependency ladder that got the env working, the r10–r12 failures,
and the measured 4-node → 6-node → 8-node scaling results (including the two negative results:
`cpu_offload` is the whole game, and more inference nodes do NOT speed eval).
Read this before choosing a Jupiter FSDP2 geometry, before assuming more GPUs help, or before trying to
resume a checkpoint into a different policy world size.

Carved out of `handoff.md` 2026-07-29 (append-only from here). Durable cluster facts:
[[jupiter_cluster]]. Env landmines: [[gotchas]]. Megatron alternative:
[[jupiter_megatron_bringup]], [[megatron_vs_fsdp2]].

⚠ All `pass_at_1 ≈ 0.44–0.47` numbers below are a FORMAT artifact, not math ability — see
[[gsm8k_format_artifact]].

---

## Jupiter setup (2026-07-22) — bring-up ladder

- Fixed ladder: proxychains → upath → wandb offline → libcudart LD path → drop routed_experts kwarg →
  TP=1 (IPv6) → numpy≤2.2 → **FA2 restored**.
- **flash-attn fix:** CuTe ≠ FA2. FA2 kernels OK; `flash_attn/cute` disabled vs cutlass-dsl 4.6.1.
  Script: `hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh`.
  Log: `ai_memory/logs/2026-07-22_jupiter_flash_attn_cute.md`.

## Away mission (2026-07-22) — Jupiter GRPO 50-step, r10–r12 failures

User away: **autonomously** run Qwen3-30B-A3B gsm8k GRPO → **50 steps**, **eval every 5**, track eval
reward up.
- Config: `hpc/skyrl_yaml/jupiter/4node_qwen3_30b_a3b_gsm8k_grpo.yaml` (`max_steps=50`,
  `eval_interval=5`, `flash_attn=true`).
- For this mission only: fix+relaunch OK; `scancel` OK if job is clearly wedged/dead and blocks the
  pipeline.
- Goal evidence: step-0 baseline + evals at 5/10/…/50 with `pass_at_1` (or equiv) trend.
- **r10 1012192 FAILED** (~7m): missing `torchtitan` → installed `@a1fdd7e` + tyro.
- **r11 1012240 FAILED** (~7m): `_StridedShard` import from `_dtensor_spec` (gone in torch 2.9). Fixed
  in MarinSkyRL `fsdp_utils.py` → `placement_types` (local branch
  `lukedhlee/fix-stridedshard-torch29` @ `1f0c5b0`; same patch on Jupiter clone).
- **r12 1012293 FAILED** (~7m): stock `vllm==0.13` missing `model_loader.reload` (MoE weight-sync).
  feuer1 SIF unreadable.
- **Blocker at the time:** lab vLLM fork build still compiling on **login** (`MAX_JOBS=2`, `ninja` on
  `_vllm_fa3_C`…). Mid-build stock vLLM is uninstalled (`import vllm` fails until install finishes).
  Log: `$SCRATCH/logs/build_vllm_fork_login_maxjobs2.log`. On `reload OK` → submit **r13**
  (no auto-PR).
- MarinSkyRL `_StridedShard` patch was **cluster-local only**; rushed PR #93 was closed by user —
  proper PR only when asked.
- Do **not** set `SKYRL_W13_RELOAD_BRACKET=0` on MoE (token salad).

## Jupiter r13 (2026-07-25) — env ready, 50-step run live

- **Blocker cleared:** lab vLLM fork built into `$DCFT/envs/rl` → `vllm 0.16.0 | reload OK`;
  `flash_attn 2.8.3` (cute disabled); `torchtitan`+`tyro 1.0.15` OK; `_StridedShard` patch present in
  `MarinSkyRL/.../distributed/fsdp_utils.py` (clone on `main` @ `36fdbc0`).

### 4-node (1041623)
Submitted `2026-07-25 ~09:29 CEST`, **RUNNING** on `booster` (open, no reservation), 4 nodes
`jpbo-060-[17,29-31]`, wall `11:59:00`.
- Job name: `jupiter_moe30b_gsm8k_grpo_4n_50step_eval5`
- Config: `hpc/skyrl_yaml/jupiter/4node_qwen3_30b_a3b_gsm8k_grpo.yaml`; verified `max_steps=50`,
  `eval_interval=5`, `eval_before_train=true`, `ckpt_interval=5`.
- Exp dir: `$SCRATCH/experiments/jupiter_moe30b_gsm8k_grpo_4n_50step_eval5/`; log under `.../logs/`.
- WandB offline (`jupiter-moe-gsm8k-grpo`, entity `lukeleeai`) → sync from login after.
- Watch: step-0 baseline `pass_at_1`, first `sync_weights` completing (r12's crash point), then eval
  curve at 0/5/10/…/50. Known risk: intermittent Vista-style early `sync_weights` hang.
- **Nodes = open `booster`** (`jpbo-060-[17,29-31]`), NOT dev. `develbooster` is a reservation (8 nodes
  `jpbo-101-*`, 4-node/120-min cap) — same GH200 hardware, just short-wall policy; a 50-step run needs
  open booster's 12h wall. Fresh bigger run also goes on open booster.
- **PROGRESS (2026-07-25):** step-0 baseline `pass_at_1=0.4556`; initial `sync_weights` 204.86s clean;
  steps 1–5 done (~880s/train-step, ~15min incl. amortized); **Vista-style hang did NOT reproduce** on
  4-node Jupiter topology. `global_step_5` checkpoint written. `policy_train` ~525s/step (slow).
- **Slow-step cause:** policy on only **8 GPUs** (EP4×FSDP2) + `cpu_offload=true` +
  `gradient_checkpointing` + `micro_train_batch=1` — all memory workarounds forced by the old 4-node
  (develbooster) cap. On open booster we can scale policy to 16 GPUs and drop `cpu_offload`.
- **CHECKPOINT GOTCHA (verified in `fsdp_strategy.py`):** SkyRL FSDP2 ckpt is per-rank sharded
  `model_world_size_{WS}_rank_{R}.pt` via plain `torch.save`, loaded by exact `{WS}`+`{R}` name —
  **NOT DCP, cannot reshard.** Resume ONLY works into the SAME policy world_size. Cannot carry an
  8-GPU ckpt into a 16-GPU mesh (→ FileNotFoundError). So a bigger run must start **from scratch**
  (step-5 weights are scientifically negligible anyway: 5 steps @ lr 1e-6).
- **PLAN (executed):** 4-node was the sanity check → then scaled out on open booster.
- **4-node eval curve (1041623):** step0 `0.4556` → step5 `0.4625` → step10 `0.4420`.
  Noisy/non-monotonic, within Vista's 0.44–0.47 band; too few points for a trend.
  **Cancelled `1041623` on user OK (2026-07-25)** once 6-node was healthy (redundant, 2× slower).

### 6-node FAST (1042857) — the cpu_offload win
`hpc/skyrl_yaml/jupiter/6node_..._fast.yaml`, 16 policy (EP4×FSDP4) + 8 infer TP=1,
**`cpu_offload=false`**, 50 steps/eval5.
- **cpu_offload OFF WIN: policy_train 533s→113s (4.7×), step wall 891s→432s (2.06×), no OOM.**
- Baseline `pass_at_1=0.4511`.
- **Bottleneck shifted:** now `sync_weights` 184s (43%) + `generate` 103s (24%); `policy_train` only
  26%. Eval (~2191s/~36min every 5) is now the biggest wall-clock cost → eval-every-5 won't quite reach
  step 50 in 12h. **More POLICY GPUs (naive 12-node) ≈ no gain.**

### 8-node EVAL-OPT (1043206) — NEGATIVE RESULT
`hpc/skyrl_yaml/jupiter/8node_..._fast.yaml`, 16 policy + **16 infer** (TP=1), else same. Launched
2026-07-25, RUNNING on `jpbo-123-*`.
- **NEGATIVE RESULT — doubling inference did NOT speed eval.** 8n eval ~49–55s/unit vs 6n ~52s/unit
  (~6%, projected ~2050s vs 2191s).
- **Eval is latency/decode-bound, not throughput-bound:** `eval_batch_size=32` doesn't even saturate 8
  engines, so +8 more sit idle.
- **Real eval levers = bigger `eval_batch_size`, shorter eval `max_generate_length`, fewer eval
  samples, or `eval_interval=10` — NOT more inference nodes.**
- 8n is redundant/over-provisioned vs 6n; kill candidate (no scancel w/o user OK).

### Vista vs Jupiter (matched 24-GPU / 16pol+8inf)
Vista step ~280s @ bs16; Jupiter 6n 432s @ bs32. Normalized per-sample: generate identical (~2× =
batch), policy_train Jupiter ~20% FASTER/sample (NVLink EP locality), sync_weights Jupiter ~1.2× slower
(fixed cost). Net: **Jupiter on par / better per-sample.**

### Node limits
booster ~5679 nodes (4071 idle at check); QOS `normal` + reformo/laionize assoc have NO hard per-user
node/TRES cap — real governor is project node-hour budget + fair-share. 6–8 nodes trivial. Guardrail
≤6 running RL jobs.

### Last verified job states (2026-07-25 ~17:40 CEST)
`1042857` (6n fast) RUNNING; `1043206` (8n eval-opt) RUNNING; `1041623` (4n) cancelled.
Re-check queue before acting.
