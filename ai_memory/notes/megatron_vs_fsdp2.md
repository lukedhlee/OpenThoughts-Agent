# Megatron vs FSDP2 — PI decision to switch RL defaults (2026-07-25)

The PI's benchmark that made Megatron the RL default (~9× total compute on long sequences), and the
crucial caveat that GSM8K understates the win because our policy_train is already small there.
Read before choosing a backend or before quoting a Megatron speedup number.
Operational bring-up record: [[jupiter_megatron_bringup]].

**PI (Ben Feuer / penfever) verdict:** "megatron curb stomps FSDP2 ... guess we're switching our RL defaults!"
Megatron is **merged in MarinSkyRL** (took several PRs; **PR #7 (closed)** carries the actual results).

## Benchmark (PI's numbers, dataset = TaskTrove `pymethods2test-large`, his standard hparam dataset)
Long-sequence / compute-bound workload (NOT gsm8k). Stage seconds:

| stage (s)            | megatron_ep4 | fsdp2_cpu_offload | fsdp2_no_offload |
|----------------------|--------------|-------------------|------------------|
| policy_train         | **600.8**    | 3699.6            | 6020.7           |
| fwd_logprobs         | **92.7**     | 739.7             | 1320.5           |
| sync_weights         | **141.8**    | 500.8             | 240.4            |
| save_checkpoints     | —            | 18.3              | 102.7            |
| total_compute_ex_gen | **835.2**    | 4940.1            | 7581.6           |

- Megatron total compute ~**9× faster** than fsdp2_no_offload, ~**6× faster** than fsdp2_cpu_offload.
- policy_train ~**10×**, fwd_logprobs ~**14×**, sync_weights also faster.

## Why the win is huge there but small on gsm8k
Megatron speeds up **training compute** (policy_train + fwd_logprobs). On `pymethods2test-large` (long seqs) that's the dominant cost → 10× win. On **gsm8k** (short 512+1024) our fsdp2 policy_train is only ~113s (26% of step); bottleneck already shifted to sync_weights/generate/eval. So **gsm8k UNDERSTATES Megatron's advantage** — a fair comparison should also use a long-sequence dataset.

## Setup facts (Jupiter, `$DCFT/envs/rl-megatron`)
- SkyRL Megatron path present: `distributed/megatron/megatron_strategy.py`, `workers/megatron/megatron_worker.py`, `config/megatron_config/{policy,ref}.yaml`.
- Native aarch64 env is built and validated with Megatron Core 0.18.0, Megatron Bridge 0.5.0, TransformerEngine 2.11.0, torch 2.11.0+cu128, transformers 5.8.x, and vLLM 0.22.0.
- Apex is still absent; smoke configs disable `gradient_accumulation_fusion` to avoid the Apex fused grad path.
- FSDP2 EP note: `fsdp_strategy.py:116` asserts `ep_size>1` requires fsdp2 — Megatron uses its OWN native MoE EP instead.

## Plan
Keep FSDP2 runs going; stand up Megatron; compare speed head-to-head (ideally on gsm8k AND a long-seq dataset). See [[handoff]].

## Jupiter bring-up result (2026-07-25)
- Megatron stack works on Jupiter in `$DCFT/envs/rl-megatron` with native venv, Megatron Core/Bridge, TransformerEngine, SkyRL, Ray, and vLLM.
- 2-node Qwen3-0.6B smoke completed: job `1043401`, one train step, exit `0:0`.
- 6-node Qwen3-30B-A3B Megatron no-save smoke completed: job `1043650`, 24 GH200 GPUs, exit `0:0`, elapsed `00:39:11`.
  - Step 1: generate `110.69s`, fwd/logprobs `29.13s`, policy_train `78.94s`, sync_weights `115.47s`, step `334.36s`.
  - Step 2: generate `119.08s`, fwd/logprobs `14.39s`, policy_train `58.47s`, sync_weights `117.00s`, step `309.01s`.
- Checkpoint caveat: 30B job `1043464` completed both train steps but failed during final Megatron optimizer checkpoint gather (`gloo ... Connection closed by peer`). Smoke configs now set `ckpt_interval: -1` and `hf_save_interval: -1` so train-loop validation does not trip the checkpoint path.
- Remaining production work: choose/fix Megatron checkpointing strategy before long runs that need resume; likely model-only checkpoint/export or debugging Megatron distributed optimizer state save on Jupiter.
