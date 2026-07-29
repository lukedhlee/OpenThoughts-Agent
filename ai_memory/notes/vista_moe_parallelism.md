# Vista MoE — quantization, multi-node stack, expert residency

**Model:** `Qwen/Qwen3-30B-A3B` — 30B total / ~3B active, 128 experts top-8, **bf16 (no quant)**.

## MarinSkyRL vs SkyRL

- **SkyRL** = upstream RL framework (NovaSky lineage).
- **MarinSkyRL** = lab fork / SoT: `marin-community/MarinSkyRL` branch `penfever/working`.
- In chat we often say “SkyRL” loosely; **code we run is MarinSkyRL**. Old `penfever/SkyRL` is obsolete.

## Quantization

**None.** Weights stay bf16 (~60 GB). Train uses full-fidelity Adam (fp32 master/grads/m,v). Memory pressure handled by **sharding (EP+FSDP)** + **`cpu_offload`** to GH200 LPDDR — not INT4/FP8/AWQ.

## Framework (multi-node)

| Layer | Role |
|-------|------|
| **OT-Agent `hpc.launch`** | SLURM job + YAML |
| **MarinSkyRL** | GRPO loop, Ray placement groups |
| **vLLM** | Rollout / generation engines |
| **FSDP2 + Expert Parallel (EP)** | Policy/ref train mesh (torchtitan-style EP all-to-all) |
| **NCCL** | Cross-node collectives (Vista = 1 GPU/node → every EP/FSDP/TP edge can be multi-node). **otagent:** NCCL **2.28.9** (`nvidia-nccl-cu12`, `+cuda12.9`) with torch `2.11.0+cu128`. |
| **NCCL debug (Ray)** | Use file-backed logs: `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET,COLL` + `NCCL_DEBUG_FILE=/scratch/11584/lukedhlee/nccl_debug/nccl_%h_%p.log`. MarinSkyRL now forwards `NCCL_DEBUG_FILE` (Vista checkout). Verified smoke **855073**: ~44k `NCCL INFO` lines/file. FR (`$SCRATCH/nccl_fr/nccl_trace_`) only dumps on torch PG timeout — Ray await hangs still need a sync-await timeout. |

## Colocate vs disagg

- **Colocate** (`colocate_all: true`): same GPUs time-share vLLM gen + FSDP train (pause one, run the other).
- **Disagg** (`colocate_all: false`): separate GPU pools — some nodes only generate, some only train; weights sync between them.

## FSDP = N

`fsdp_size: N` = shard each param/grad/optimizer state across **N ranks**. N=2 → each rank holds ~1/2 of sharded state (plus its EP expert slice). Must satisfy `(128/EP) % FSDP == 0`.

## Prefill / expert activation

Per **token**: still top-8 only (~3B compute). Across a **batch of many tokens** with different routes: many/all of the 128 experts can receive work in the same step — weights stay resident; compute fans out. Not “one token activates all 128.”

## Vista meaning of “train”

Vista = **1× GH200 / node**. Jupiter “24 GPU / 6 nodes” → Vista **24 nodes**. Each parallel rank = one node.

Recommended smoke (`gh-dev` ≤8): e.g. **colocate EP=4 × FSDP=2 = 8 policy GPUs** (= 8 nodes), or disagg 4 policy + 4 infer.

## Parallelism (what we use)

- **Train (policy/ref):** `expert_model_parallel_size` (EP) × `fsdp_size` (FSDP). Hard rule: `(128/EP) % FSDP == 0` and `EP×FSDP = #policy GPUs`.
- **Infer (vLLM):** **TP** (often 2); lab MoE configs keep **`inference_engine_expert_parallel_size: 1`** (experts already fit via TP shard).
- **Not** expert offload/swapping as the primary strategy.

## Are experts always on GPU?

**Sharded residency, sparse compute — not expert swapping.**

- All 128 experts exist as parameters, **partitioned** across EP (then FSDP) ranks. Each GPU holds its shard full-time (modulo FSDP `cpu_offload` of inactive shards to LPDDR).
- Per token, router picks **top-8**; only those experts’ **compute** runs. Tokens dispatch via EP all-to-all to the ranks that own those experts.
- This is **not** “load expert from disk/CPU when selected.” Swapping would be a different (unsupported/unproven here) offload mode.

TPS smoke we ran: single-GPU vLLM held the full ~60 GB bf16 MoE on one GH200.
