# Jupiter — why stock vLLM dies on MoE weight sync (`model_loader.reload`)

Audience: operator who wants the full picture, not just the traceback.

## One-sentence summary

RL with MoE needs a **lab-only vLLM API** (`initialize/finalize_layerwise_reload`) so that after every training step, the **updated policy weights are correctly applied into the inference engines**. Stock `vllm==0.13` from pip does not ship that API → job dies at the first weight sync.

---

## 1. What the training loop actually is

SkyRL GRPO is **not** “one process that both trains and generates.”

It is two cooperating systems:

| Role | Process | What it owns |
|------|---------|--------------|
| **Policy (trainer)** | FSDP workers (PyTorch) | Gradients, optimizer, updated weights after each step |
| **Generator (rollout)** | vLLM engines (Ray actors) | Fast sampling for prompts → completions → rewards |

Each training step roughly:

1. **Generate** with vLLM (current weights)
2. Score completions (reward / `pass@k`)
3. **Train** the FSDP policy (weights change)
4. **`sync_weights`**: copy the new policy tensors into every vLLM engine
5. Repeat

Step 4 is mandatory. If you skip it, the next generate still uses **old** weights → you’re not doing on-policy RL.

---

## 2. Why MoE makes step 4 hard

Dense models: push tensors into vLLM, mostly done.

MoE (Qwen3-30B-A3B) is worse for two reasons:

1. **Many expert tensors** (~tens of thousands of named weights). Sync is chunked.
2. **Fused MoE kernels** (FlashInfer / CUTLASS) expect a specific **layout** of gate/up projections (`w13`). After a raw tensor dump, vLLM must run a post-load pass that **re-applies** that layout / kernel prep.

If you overwrite weights without that post-pass, the MoE kernel reads transposed / half-swapped halves → **token salad** (garbage tokens, rewards ~0). That is what `SKYRL_W13_RELOAD_BRACKET=1` exists to prevent. **Do not turn it off on MoE.**

---

## 3. The “reload bracket” (the API we need)

SkyRL wraps each multi-chunk sync like this:

```
begin_weight_reload()          # prepare vLLM for a layerwise update
  update_named_weights(...)*N  # stream chunks of tensors
finish_weight_reload()         # run process_weights_after_loading ONCE
```

In code (`vllm_engine.py` → `WorkerWrap`):

- `skyrl_begin_weight_reload` imports:
  `from vllm.model_executor.model_loader.reload import initialize_layerwise_reload`
- `skyrl_finish_weight_reload` imports:
  `from vllm.model_executor.model_loader.reload import finalize_layerwise_reload`

That `vllm.model_executor.model_loader.reload` module is **not in upstream / stock pip vLLM 0.13**.
It lives in the lab fork: `mlfoundations/vllm` (pin for torch 2.9: `084aa19f0`).

Stock vLLM on Jupiter today:

- package: `vllm==0.13.0` wheel
- `model_loader/` has the usual loaders
- **no `reload.py` / `reload/` package**

So the first sync after init raises:

`ModuleNotFoundError: No module named 'vllm.model_executor.model_loader.reload'`

That is r12’s failure. Not mysterious GPU death — a missing Python module the lab patch added.

---

## 4. Why we can’t just “pip install a newer vLLM”

Options that look tempting and why they fail here:

| Option | Why not |
|--------|---------|
| Newer stock PyPI vLLM | Still may lack the lab `reload` API; also must match torch 2.9 + aarch64 + CUDA 13 |
| Copy only `reload.py` into site-packages | Fragile; fork also has other SkyRL hooks around weight update |
| Use feuer1’s SIF/venv that already has the fork | Not readable from `lee27` scratch |
| Disable `SKYRL_W13_RELOAD_BRACKET` | Avoids the import… and breaks MoE (token salad). Forbidden for this recipe |

Correct fix: **build/install lab fork into `$DCFT/envs/rl`**.

---

## 5. Why the build itself is painful on Jupiter

vLLM is not a pure-Python package. Installing the fork means compiling CUDA extensions (`nvcc` / `ninja`). Constraints:

- **Login node:** has internet (git/pypi) but limited RAM → `MAX_JOBS=16` OOM’d (`cicc` killed). Retry with `MAX_JOBS=2`.
- **Compute node (`develbooster`):** has GPUs/RAM, but **no internet** → can’t fetch deps mid-build; also module/`git` quirks.
- Architecture is **aarch64** — must compile here, not copy an x86 wheel from a laptop.

So the “issue” you’re waiting on is not training logic — it’s getting a compiled lab vLLM into the RL venv so the reload import succeeds.

Success criterion in the build log: `reload OK` (import `initialize_layerwise_reload` works). Then relaunch **r13**.

---

## 6. How this fits the earlier failure ladder

Each Jupiter failure was one missing piece of the Vista stack:

1. **flash-attn / CuTe** — FA2 kernels OK; broken optional cute import path
2. **torchtitan** — needed for FSDP+Expert Parallelism on MoE
3. **`_StridedShard`** — torch 2.9 moved an internal import; MarinSkyRL patch
4. **`model_loader.reload`** ← **you are here** — lab vLLM API for MoE weight sync

After (4), the next real work is the scientific run: 50 steps, eval every 5, watch `pass_at_1`.

---

## 7. How you could have diagnosed this yourself

1. Open the Slurm `.out` for r12; find the traceback ending at `skyrl_begin_weight_reload`.
2. Note the import: `vllm.model_executor.model_loader.reload`.
3. On the login node, in the RL venv:

   ```bash
   python -c "import vllm, pathlib; print(vllm.__version__, vllm.__file__); \
     print(list((pathlib.Path(vllm.__file__).parent/'model_executor/model_loader').iterdir()))"
   ```

4. Confirm `reload` is missing; check lab docs (`.claude/projects/marinskyrl`, `jupiter_cluster.md`) for the fork pin.
5. Build fork or point at a known-good env that already has it.

---

## Caveats of the fix

- Building the fork is slow and can break if torch/CUDA drift; pin commit `084aa19f0` for this torch 2.9 stack.
- After install, re-verify FA cute disable (`fix_flash_attn_cute.sh`) — a vLLM reinstall can pull deps that disturb flash-attn.
- Leaving W13 bracket on is required for MoE correctness; it is not optional “perf knobs.”
