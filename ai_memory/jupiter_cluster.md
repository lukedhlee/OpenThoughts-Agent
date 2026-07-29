# Jupiter — operator cheat sheet (`lee27`)

Live-checked 2026-07-22.

## Hardware
- **4× GH200 / node** (`gres=gpu:gh200:4`), **96 GB HBM3** each → **384 GB/node**
- **aarch64** Grace CPU (72 cores/superchip, 288 cores/node)
- Cross-node: InfiniBand (not NVLink fabric between nodes)

## Queue (what you can use)
Same GH200 nodes; difference is **job-size / wall policy**, not hardware type.

| Partition | State | Policy | Notes |
|-----------|--------|--------|--------|
| **`booster`** | up | max **12h**, max **2048 nodes**/job | Normal work — **use this** |
| **`largebooster`** | **down** (as of 2026-07-22) | larger jobs (up to full ~5884), no 12h QOS cap in Slurm | Default partition (`*`) when up; for very large runs |

No separate `devel` **partition**. There is a **`develbooster` reservation**: 8 nodes (`jpbo-101-[01-08]`), on `booster`, active ~1y.
Smoke: `--partition=booster --reservation=develbooster --account=reformo`.
**Limits:** `max_nodes_per_job=4`, **`max_walltime=120min`**. For >2h (e.g. 50-step+eval) use open `booster` with `RESERVATION=none`.

## Your access
- Compute accounts: **`reformo`**, **`laionize`** (both → `booster`/`largebooster`)
- Lab OT-Agent default account: **`reformo`** (do not use suspended `jureap59` QOS)
- Groups include `jupiter_booster`, `reformo`, `laionize`, …

## Paths (Jupiter = `/e/...`, not `/p/...`)
| Use | Path |
|-----|------|
| Tiny `$HOME` | `/e/home/jusers/lee27/jupiter` |
| Project (durable, inode-sensitive) | `/e/project1/{reformo,laionize}/…` |
| Scratch (heavy/churn) | `/e/scratch/{reformo,laionize}/…` |
| Fast scratch | `/e/fscratch/…` |
| Shared data | `/e/data1/…` |

**Inode quotas bind before bytes** — watch `jutil project dataquota -p reformo` / `laionize`. Never unpack millions of small files on project.

## Network / software reality
- **Login:** internet OK (HF/git downloads)
- **Compute:** **no internet** — stage models/datasets on login first
- Containers via Apptainer/SIF common for RL; wheels must be **aarch64**

## Vista contrast
- Vista `gh`: **1 GPU/node** → Jupiter same GPU count needs **÷4 nodes**
- Vista has `gh-dev`; Jupiter does **not**

## flash-attn (RL venv) — important
- **CuTe** = CUTLASS Cuda-Tensors DSL (`nvidia-cutlass-dsl` / `cutlass.cute`). Optional newer kernel path under `flash_attn/cute/`. **Not** the FA2 we train with.
- Torch: `2.9.0+cu130`. Wheel: `2.8.3+cu130torch2.9` (mjun0812 v0.7.16 aarch64).
- Bug: FA’s cute code still uses deprecated `cute.core.ThrMma`; cutlass-dsl 4.6.1 moved it to `cute.ThrMma` → vLLM FA backend dies if cute is importable. FA2 itself is fine.
- Fix: install FA `--no-deps`, move `flash_attn/cute` → `cute.DISABLED_cutlass461`. Script: `hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh`. Log: `ai_memory/logs/2026-07-22_jupiter_flash_attn_cute.md`.
- Re-run script after any FA reinstall. YAML: `trainer.flash_attn: true`.

## MoE EP deps (RL venv)
- MoE FSDP+EP needs **`torchtitan@a1fdd7e`** + **`tyro>=0.9`** (Vista used same pin).
- Install into `$DCFT/envs/rl`:
  `uv pip install --python $DCFT/envs/rl/bin/python 'tyro>=0.9' 'torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e'`
- Missing this → `ModuleNotFoundError: torchtitan` at `FSDPPolicyWorkerBase.init_model`.
- **vLLM must be lab fork**, not stock 0.13: needs `vllm.model_executor.model_loader.reload` for MoE weight sync. Pin torch-2.9-compatible **`mlfoundations/vllm@084aa19f0`**. Stock pip → crash at `skyrl_begin_weight_reload`. Do not disable `SKYRL_W13_RELOAD_BRACKET` on MoE.
## ⚠ Two env landmines found 2026-07-28/29 (both cost a job each)

Discovered while running the GSM8K probe ([[gsm8k_format_artifact]]); **both apply to any vLLM workload
on Jupiter, including the r2egym RL launch.**

### 1. `envs/rl` CANNOT import vLLM — CUDA 12 vs 13 break
```
ImportError: libcudart.so.12: cannot open shared object file
```
`envs/rl`'s `vllm 0.13.0` extensions were built **Jul 22 against CUDA 12**; torch in that venv was
**replaced Jul 27 15:31 with 2.9.0+cu130** (nvidia/ → `cu13`), removing `libcudart.so.12`.
**Jupiter has NO CUDA 12 module**, so nothing can satisfy the link — no `LD_LIBRARY_PATH` fixes it.
The last good run (job `1042857`, Jul 25→26) predates the torch swap, which is why it worked then.

| venv | vLLM | links | state |
|---|---|---|---|
| `envs/rl` | 0.13.0 | `libcudart.so.12` | **BROKEN** |
| `envs/rl-megatron` | 0.22.0 | `libcudart.so.13` | ✅ imports, has pandas/pyarrow/transformers |
| `src/vllm_fork` | source (`084aa19f`) | `libcudart.so.13` | built, not installed into either venv |

⚠ **`hpc/rl_launch_utils.py:780` still defaults `RL_ENV_DIR` to `$WORKDIR/envs/rl`**, and the last
rendered RL sbatch pins `RL_PYTHON` + `LD_LIBRARY_PATH` there ⇒ **an FSDP2 launch today reproduces this
on a 6-node reservation.** Fix options: (a) point RL at `rl-megatron` (different torch/vLLM than the
step-45 run), (b) **rebuild vLLM in `envs/rl` from `src/vllm_fork` against cu130 — recommended, one
coherent env**, (c) reinstall cu12 torch (re-breaks whatever the Jul 27 upgrade was for).

### 2. `moe_backend=auto` JIT-builds 183 CUTLASS kernels and can hang the engine
On Qwen3-30B-A3B, vLLM 0.22 auto-selects a **FlashInfer CUTLASS MoE** path that compiles 183 nvcc units
at engine start (`--threads=1`). On job `1085876` that ninja build **failed** (`FAILED: [code=9]` ×72,
died at 182/183), the worker went down, and the engine sat in `shm_broadcast` **90 min until walltime**.
Looks like a hang/slow job; is actually a failed kernel build.

**Fix: `moe_backend="triton"`** (no runtime CUTLASS build). Verified: engine up in ~4 min,
`torch.compile` 64s, KV cache 2.7M tokens, 1319 prompts generated in 941s, whole job **21 min**.
Valid values live in `vllm/config/kernel.py:122` (`MoEBackend` Literal): `auto, triton, deep_gemm,
cutlass, flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl, marlin, humming, ...`.
`moe_backend` is a real `EngineArgs` field (`arg_utils.py:483`), so it can be passed to `LLM(...)`.

### 3. HF cache needs FORCED offline
Compute nodes have no internet, and both RL venvs have `include-system-site-packages = true`, so they
import **`huggingface_hub` 1.24.0 from `miniforge3`**, which raises
`RuntimeError: Cannot send a request, as the client has been closed` instead of falling back to cache —
so an **already-cached** model still fails to load. Set `HF_HUB_OFFLINE=1` (+ `TRANSFORMERS_OFFLINE=1`).
Cache is at `/e/scratch/reformo/lee27/cache/hf`; Qwen3-30B-A3B snapshot `ad44e777…` is complete.

### 4. `sinfo` "idle" hides maintenance
During the Jul 28 outage `sinfo` showed 5655 "idle" nodes while **5542 were in `maint`** with zero jobs
running cluster-wide. Full-machine maintenance set via node state, **not** a reservation — so
`scontrol show res` shows nothing and Slurm gives no start estimate. Check
`sinfo -p booster -o "%.14T %.7D"` for real state.
