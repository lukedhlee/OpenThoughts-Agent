# Vista — Qwen3-30B-A3B MoE GPU sizing

**PI target model:** `Qwen/Qwen3-30B-A3B` (general MoE — NOT Coder).
Also measured: `Qwen/Qwen3-Coder-30B-A3B-Instruct` for TPS compare only.
Both: plain `qwen3_moe`, ~30B total / ~3B active, 128 experts top-8, bf16 ≈ **60 GB**.

Vista: **1× GH200 96GB / node**. Jupiter MoE configs assume 4 GPU/node → multiply node count ×4 for same GPU total.

## Inference / load (first milestone)

| Setup | GPUs (= nodes) | Fits `gh-dev`? | Notes |
|-------|----------------|----------------|-------|
| vLLM TP=1 load + short gen | **1** | YES | ~60 GB weights → ~36 GB HBM left. Start `max_model_len=8192` (full-attn MoE; KV heavier than hybrid 35B-A3B). |
| vLLM TP=2 | **2** | YES | More KV headroom / longer ctx. Multi-node TP (1 GPU/node). |
| Jupiter RL inference slice | 8 (4× TP=2) | YES (≤8) | Matches `24GPU_qwen3_coder_30b_a3b` gen side only. |

## Train / full GRPO (Jupiter parity reference)

From `hpc/skyrl_yaml/jupiter/extra/24GPU_qwen3_coder_30b_a3b.yaml` (`colocate_all: false`):

| Role | GPUs | Vista nodes |
|------|------|-------------|
| vLLM gen (4 engines × TP=2) | 8 | 8 |
| Policy+Ref FSDP (EP=4 × FSDP=4) | 16 | 16 |
| **Total** | **24** | **24** |

Adam+grads+master ≈ 540 GB sharded → ~34 GB/GPU over 16 with `cpu_offload` (uses GH200 LPDDR).

**24 nodes > `gh-dev` MaxNodes=8** → full train must use **`gh`/`gg`** (qgh MaxNodes=64).

## `gh-dev` train smoke options (≤8 nodes)

Hard MoE divisibility: EP | 128, and (128/EP) % FSDP == 0, and EP×FSDP = policy GPUs.

| Geometry | Nodes | Valid? |
|----------|-------|--------|
| Load-only TP=1 | 1 | YES — do this first |
| Colocate train on 8: EP=4 × FSDP=2 | 8 | YES (128/4=32, 32%2=0) |
| Disagg 4 policy (EP=4×FSDP=1) + 4 inf (4× TP=1) | 8 | YES |
| Disagg 4 policy + 2 inf (1× TP=2) | 6 | YES |
| Jupiter-parity 24 | 24 | **NO on gh-dev** |

## Recommended ladder

1. ~~vLLM load + TPS~~ **DONE** (general ~266 out tok/s; Coder ~292).
2. ~~Branch + YAML~~ **DONE** — `lukedhlee/vista-moe-grpo-30b`, `hpc/skyrl_yaml/vista/8node_qwen3_30b_a3b_gsm8k_grpo.yaml` (colocate EP=4×FSDP=2, `max_steps=10`, gsm8k verifier-only).
3. **Run gsm8k GRPO** via `hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh` — look for verifier reward / val accuracy lift @ steps 0→5→10.
4. Agentic Harbor tiny-task smoke → then PI benches.
5. Full 24-node train on `gh`/`gg`.

## Measured vLLM TPS (2026-07-15, same setup)

Hardware: 1× GH200, TP=1, bf16, `max_model_len=8192`, penfever `otagent` fork-vLLM, FlashInfer CUTLASS MoE.
Bench: `vllm bench throughput` random, **1024 in / 128 out**, 16 prompts.

| Model | Job | Output tok/s | Total tok/s | Req/s |
|-------|-----|--------------|-------------|-------|
| **Qwen/Qwen3-30B-A3B** (general, PI target) | 833585 | **265.7** | 2392 | 2.08 |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 833426 | **291.6** | 2624 | 2.28 |

Gotchas: otagent `bin` on `PATH` (ninja); `gcc/14.2.0` not 15.1; `FLASHINFER_WORKSPACE_BASE` → prebuilt `fused_moe_90.so`; never JIT into `/home1`.
