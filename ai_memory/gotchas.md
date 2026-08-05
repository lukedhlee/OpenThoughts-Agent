# Gotchas (append-only)

**symptom → cause → fix**, one entry per failure that cost more than ten minutes. Dated by discovery.
**Never edit an entry** — a correction is a new dated entry below the old one.

Reconstructed 2026-07-29 from `handoff.md` and the topic notes during the memory refactor.

---

## Jupiter — environment

### 2026-07-28 · `ImportError: libcudart.so.12` when importing vLLM from `envs/rl`
**Symptom:** any vLLM import in `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl` dies with
`libcudart.so.12: cannot open shared object file`.
**Cause:** that venv's `vllm 0.13.0` extensions were built **Jul 22 against CUDA 12**, but torch in the
venv was **replaced Jul 27 15:31 with 2.9.0+cu130** (nvidia/ → `cu13`), removing `libcudart.so.12`.
**Jupiter has NO CUDA 12 module**, so nothing can satisfy the link — no `LD_LIBRARY_PATH` fixes it. The
last good run (job `1042857`, Jul 25→26) predates the torch swap.

| venv | vLLM | links | state |
|---|---|---|---|
| `envs/rl` | 0.13.0 | `libcudart.so.12` | **BROKEN** |
| `envs/rl-megatron` | 0.22.0 | `libcudart.so.13` | ✅ imports, has pandas/pyarrow/transformers |
| `src/vllm_fork` | source (`084aa19f`) | `libcudart.so.13` | built, not installed into either venv |

**Fix / open trap:** `hpc/rl_launch_utils.py:780` still defaults `RL_ENV_DIR` to `$WORKDIR/envs/rl`, and
the last rendered RL sbatch pins `RL_PYTHON` + `LD_LIBRARY_PATH` there ⇒ **an FSDP2 launch today
reproduces this.** Options: (a) point RL at `rl-megatron` (different torch/vLLM than the step-45 run),
(b) **rebuild vLLM in `envs/rl` from `src/vllm_fork` against cu130 — recommended, one coherent env**,
(c) reinstall cu12 torch (re-breaks whatever the Jul 27 upgrade was for).

### 2026-07-29 · Engine appears hung for 90 min; is actually a failed CUTLASS kernel build
**Symptom:** on job `1085876` the vLLM engine sat in `shm_broadcast` until walltime. Looks like a hang.
**Cause:** on Qwen3-30B-A3B, vLLM 0.22 `moe_backend=auto` selects a **FlashInfer CUTLASS MoE** path
that JIT-compiles **183 nvcc units** at engine start (`--threads=1`). The ninja build FAILED
(`FAILED: [code=9]` ×72, died at 182/183) and the worker went down.
**Fix:** `moe_backend="triton"` (no runtime CUTLASS build). Verified: engine up in ~4 min,
`torch.compile` 64s, KV cache 2.7M tokens, 1319 prompts in 941s, whole job **21 min**. Valid values are
in `vllm/config/kernel.py:122` (`MoEBackend` Literal): `auto, triton, deep_gemm, cutlass,
flashinfer_trtllm, flashinfer_cutlass, flashinfer_cutedsl, marlin, humming, …`. `moe_backend` is a real
`EngineArgs` field (`arg_utils.py:483`) so it can be passed to `LLM(...)`.

### 2026-07-29 · `import vllm` succeeds in `envs/rl` but the env is still broken
**Symptom:** `python -c "import vllm"` prints a version and looks healthy, so `envs/rl` seems fixed.
**Cause:** the top-level `vllm` import is lazy. The CUDA platform extension is only pulled in later, and
`import vllm._C` is what actually fails. Measured with `objdump -p`:

| env | vLLM | `_C.so` NEEDED | env ships | verdict |
|---|---|---|---|---|
| `envs/rl` | 0.13.0 | `libcudart.so.12` | no `nvidia/cuda_runtime/` at all | ❌ broken |
| `envs/rl-megatron` | 0.22.0 | `libcudart.so.13` | CUDA/13 module | ✅ works |

Jupiter has **only CUDA/13**, so nothing can satisfy `envs/rl`. The confusing bit: it's
`rl-megatron` that ships a stray `nvidia/cuda_runtime/lib/libcudart.so.12` (which it does not need),
while `rl` — which does need it — ships none. Don't infer env health from a directory listing.
**Fix:** always gate on `import vllm._C`, never `import vllm`. Use `envs/rl-megatron`; `module load
GCC/14.3.0 nvidia-compilers/25.9-CUDA-13` **alone** is sufficient (it puts CUDA/13 `lib64` on
`LD_LIBRARY_PATH`; no manual export needed), and `hpc.py:862` already sets those modules for Jupiter.
⚠ `rl-megatron` has **no `flash_attn`** and the known aarch64 wheel targets torch 2.9/cu130 while this
env is 2.11/cu128 ⇒ use `trainer.flash_attn: false` (SDPA). "megatron" is only the directory name;
`strategy: fsdp2` works there (torch 2.11 `fully_shard` + `_StridedShard` both import clean).

### 2026-07-29 · ★ Daytona is UNREACHABLE from Jupiter compute nodes — agentic RL cannot use it as-is
**Symptom:** an agentic RL run executes flawlessly end-to-end (weight sync, generation, fwd_logprobs,
advantages, policy train) and reports **`avg_raw_reward: 0.0`, `zero_rewards: 1.0`** over the whole batch.
Looks like the model is simply incapable.
**Cause:** every rollout failed with
`Failed to create snapshot: Cannot connect to host app.daytona.io:443 [Network is unreachable]
(type=DaytonaConnectionError)`. **Daytona is a CLOUD API and JSC compute nodes have no internet.**
`hpc.py` sets `needs_ssh_tunnel=True` for jupiter, but the lee27-local edit set
`proxychains_binary=""`, and `hpc.py:752` gates the whole proxy-setup block on that being non-empty ⇒
compute nodes get **no outbound path at all**.
⚠ Because `mask_exceptions` includes `DaytonaError`/`NetworkError` with `default_error_treatment: zero`,
these failures are **absorbed into zero rewards instead of crashing** — the run would have burned all 50
steps and 3 nodes reporting nothing wrong. Under GRPO, all-zero rewards ⇒ zero advantage variance ⇒
**no gradient**. This is the most dangerous silent-failure mode found so far.
**Diagnostic:** always grep the RL log for `DaytonaConnectionError` / `Network is unreachable` before
concluding anything from a low reward. `avg_raw_reward: 0.0` is far more often infra than capability.

**MEASURED network facts (2026-07-29, from a real probe job on `jpbo-040-12`):**

| from a Jupiter COMPUTE node | result |
|---|---|
| `app.daytona.io:443`, `huggingface.co:443` | BLOCKED |
| JURECA (`jureca.fz-juelich.de:22`, `134.94.1.131:22`, `jrlogin04i:9920`) | BLOCKED |
| Jupiter login via **public** IP (`134.94.0.132:22`) | BLOCKED |
| Jupiter login via **internal** `10.128.1.2` (ib0) — HTTP **and** `:22` | **REACHABLE** |
| Jupiter login via `10.99.0.2` — HTTP and `:22` | **REACHABLE** |
| `10.201.15.132` (gpfs iface) | no |

⚠ **Never probe reachability with public IPs/hostnames** — that gives a false "totally isolated" reading.
Compute↔login works over `10.128.x`/`10.99.x` (same /16 as compute), which is why Ray works. A relay on
a login node is therefore viable, and is the basis of every fix.

### 2026-07-29 · r2egym env Dockerfiles must be BUILT (not pulled), and r2egym is multi-arch
**Two facts that together decide which cluster r2egym sandboxes belong on:**
1. r2egym `environment/Dockerfile` files are **build recipes** — `FROM python:3.6-slim-buster` plus
   `apt-get install` and `pip install` — NOT published images. So `apptainer pull` cannot make a SIF;
   they need `apptainer build`. This is the opposite of SWE-bench, where every task has a *published*
   `swebench/sweb.eval.x86_64.*` that pulls directly (which is why the SWE-bench bridge was easy).
2. `python:3.6-slim-buster` is a **multi-arch official image with arm64**, so r2egym runs **natively on
   Jupiter's aarch64 GH200 nodes**. Only SWE-bench (x86_64-only) actually needs JURECA.
⇒ **r2egym sandboxes belong on a JUPITER-LOCAL apptainer bridge; JURECA is for SWE-bench only.**
**`--fakeroot`:** fails on JSC **login** nodes ("No user namespaces available") but the reference
implementation runs `apptainer build --fakeroot` on **COMPUTE** nodes (`--nodes=4 --partition=devel`),
so namespaces appear to be available there. This resolves the "untested on compute?" question in
[[apptainer_bridge_handoff]]. Build also needs internet (docker pull + apt + pip), so either pre-pull
base layers into `APPTAINER_CACHEDIR` from a login node, or proxy the compute nodes.

### 2026-07-29 · `flash_attn: false` needs TWO more settings, one of which fails LOUD and one SILENT
**Symptom (loud):** `AssertionError: Flash attention 2 should be used for use_sample_packing` at
`FSDPPolicyWorkerBase.init_model` → `model_wrapper.py:397`.
**Cause:** `trainer.use_sample_packing` defaults to **true** (`ppo_base_config.yaml:358`) and packing is
implemented only on the flash-attn-2 varlen path, so it is mutually exclusive with any non-FA backend.
**Fix:** `trainer.use_sample_packing: false`. Cost is ~zero when
`micro_train_batch_size_per_gpu: 1` — there is nothing to pack together anyway.

**Symptom (SILENT, and the one that actually matters):** the run trains, but absurdly slowly / OOMs at
long context.
**Cause:** `resolve_attn_implementation()` (`model_wrapper.py:96`) maps
`attn_backend="auto"` (the default, `ppo_base_config.yaml:372`) + `use_flash_attention_2=False` →
**`"eager"`**, *not* `sdpa`. Turning flash-attn off does NOT fall back to SDPA. At 28,672-token prompts
eager attention is quadratic-memory and glacial — a run that "works" but is unusable, with no error.
**Fix:** set **`trainer.attn_backend: sdpa`** explicitly. Valid values are
`{auto, flash_attention_2, sdpa, flex}`.

**Rule of thumb:** `flash_attn: false` is never a one-line change — it is the trio
`flash_attn: false` + `attn_backend: sdpa` + `use_sample_packing: false`.
(Aside: `context_parallel_size > 1` additionally *requires* `sdpa` or `flex`; flash-attn varlen is
rejected under CP.)

### 2026-07-29 · `colocate_all: false` silently forces the FULLY-ASYNC trainer (and its constraint set)
**Symptom:** after all 8 vLLM engines load successfully, the run dies at trainer construction with
`AssertionError: batched is not supported for fully async training.`
**Cause:** `examples/terminal_bench/entrypoints/main_tbench.py` `get_trainer()` selects
`FullyAsyncRayPPOTrainer if cfg.trainer.placement.colocate_all is False else RayPPOTrainer` — keyed on
that **single flag**, with no separate opt-in. Because the 2026-07-26 decision fixes
**disaggregated-only (`colocate_all: false`)** for this model/hardware, the fully-async trainer is
**not optional**, and its constraints are hard requirements:

| constraint (`fully_async_trainer.py` ~316–356) | our value |
|---|---|
| `train_batch_size == policy_mini_batch_size` | 32 == 32 ✓ |
| `algorithm.dynamic_sampling.type is None` | default ✓ |
| **`not generator.batched`** | was `true` ✗ → set **false** |
| `generator.async_engine` | true ✓ |
| `not colocate_all` | ✓ |
| `policy_mini_batch_size <= fully_async.num_parallel_generation_workers` | 32 <= 32 ✓ |

⚠ **`hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_r2egym_grpo.yaml` still pairs `batched: true` with
`colocate_all: false`** and will fail this identical assert — fix it before launching the 30B.

**Second trap in the same config block:** `fully_async.num_parallel_generation_workers` defaults to
**768**. Generation concurrency is actually capped by the engine working set
(`num_inference_engines * max_num_seqs / n_samples_per_prompt`; here `8*8/4 = 16` groups), NOT by worker
count — so a large pool adds **no** throughput and instead accumulates a stale backlog that pins
head-node memory (the documented 80B head-plasma/RAM overflow). Set it to the floor,
`== policy_mini_batch_size`.

**Semantics worth stating out loud:** fully-async training is **off-policy by up to
`max_staleness_steps` (default 4)**. That is a change in what the experiment *is*, not a tuning detail.
`use_tis: true` is the correct mitigation and should stay on; lower `max_staleness_steps` if a run needs
to be closer to on-policy.

### 2026-07-29 · `ImportError: cannot import name 'normalize_message'` — harbor is too OLD for MarinSkyRL
**Symptom:** the agentic entrypoint dies immediately at
`from harbor.utils.traces_utils import normalize_message` (via
`examples/terminal_bench/terminal_bench_generator.py:44`).
**Cause:** both Jupiter envs pinned harbor to **`laude-institute/harbor` @ `penfever/temp-override`
(`757e3ec0`, reported as 0.1.45)** — a branch that diverged before `normalize_message()` was added to
**`marin-community/harbor`** on 2026-07-19 (commit `1228252d`, PR #14). MarinSkyRL `36fdbc0` imports it
**directly**, NOT through `examples/terminal_bench/_harbor_compat.py`, so the compat shim does not save you.
⚠ This is NOT env-specific — `envs/rl` fails identically. Together with the daytona-SDK gap above, the
agentic r2egym path was **doubly blocked**; "vLLM/cu130" was only part of the story.
**Fix:** upgrade harbor, don't move MarinSkyRL (it carries load-bearing Jupiter-local edits to
`fsdp_utils`, `trainer`, `megatron_strategy`, `worker` that a checkout would disturb):
```bash
pip install "harbor @ git+https://github.com/marin-community/harbor.git@725fc069"
```
0.1.45 → **0.8.1**. Dry-run first (it showed no removals). Because that is a large jump, **gate on a real
import, not on the symbol**:
```python
from examples.terminal_bench.terminal_bench_generator import TerminalBenchGenerator
import examples.terminal_bench.entrypoints.main_tbench          # both must pass
import vllm._C, daytona                                          # confirm nothing regressed
```
Note `traces_utils` also has a similarly-named `normalize_message_content()` — having that one is NOT
evidence you have `normalize_message()`.

### 2026-07-29 · `RL_ENV_DIR` is right but `LD_LIBRARY_PATH` still points at the BROKEN env
**Symptom:** the rendered sbatch correctly says
`export RL_ENV_DIR=.../envs/rl-megatron` and `Activating RL environment: .../envs/rl-megatron`, and the
job looks properly configured — but it is not.
**Cause:** Jupiter's **uncommitted machine-local** `hpc/hpc.py` edit adds a hardcoded
`LD_LIBRARY_PATH` to the `jupiter` HPC `env_vars`, pinned to `envs/rl`, and it is emitted with **no
`${LD_LIBRARY_PATH}` suffix**. Two independent fatal effects when the RL env is `rl-megatron`:
1. `envs/rl/.../torch/lib` lands ahead of rl-megatron's. Torch uses **DT_RUNPATH**, which the linker
   searches *after* `LD_LIBRARY_PATH` — so a torch-2.11/cu128 interpreter loads 2.9/cu130 `.so`s.
2. It **wipes the CUDA/13 module paths**, which is the only source of `libcudart.so.13` — the one thing
   that made `rl-megatron`'s `vllm._C` importable in the first place.

The Ray `srun` command inherits it: the env list contains the hardcoded value AND ends with
`LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}`, so every worker gets it too.
**Fix:** override `LD_LIBRARY_PATH` in the YAML's `container.extra_env` (emitted *after* the hpc
`env_vars` line, so it wins; and because `env` takes last-assignment-wins, the trailing
`${LD_LIBRARY_PATH:-}` propagates the corrected value to Ray workers). Point it at rl-megatron's
`nvidia/*/lib` + `torch/lib` and append `${CUDA_HOME}/lib64`. Do NOT rely on the cluster-local
`hpc.py` edit being fixed — it is a divergence that should be reconciled separately.
**Check before every launch:** `grep -c "envs/rl/lib" <rendered sbatch>` and confirm the LAST
`export LD_LIBRARY_PATH` is the rl-megatron one. A correct `RL_ENV_DIR` is NOT sufficient evidence.

### 2026-07-29 · The two Jupiter RL envs are COMPLEMENTARY — neither is complete for agentic RL
**Symptom:** you pick `envs/rl-megatron` because it's the only env whose vLLM works, and the launcher
then dies with `ModuleNotFoundError: No module named 'daytona'` in `_default_daytona_factory`.
**Cause:** measured 2026-07-29 —

| | `envs/rl` | `envs/rl-megatron` |
|---|---|---|
| vLLM | 0.13.0 — **broken** (`libcudart.so.12`) | 0.22.0 ✅ |
| daytona SDK | 0.201.0 ✅ | **absent** |
| harbor | 0.1.45 | 0.1.45 |
| torch | 2.9.0+cu130 | 2.11.0+cu128 |
| transformers | 4.57.3 | **5.8.1** (major bump) |
| flash_attn | present | **absent** |

Agentic RL needs BOTH a working vLLM and the daytona SDK (prebuild *and* runtime rollouts), so neither
env works out of the box. This is a likely reason the agentic r2egym run was never launched.
**Fix:** `pip install daytona==0.201.0` into `envs/rl-megatron` — purely additive (dry-run showed no
removals and no torch/vLLM/ray/transformers downgrades). Then re-verify `vllm._C`, `ray`, `harbor`,
`daytona` together under the sbatch's `module load`, not bare.
⚠ Note `transformers` differs by a MAJOR version between the envs; don't assume a recipe validated on
`envs/rl` transfers.

### 2026-07-29 · `train_data` as a local parquet silently yields ZERO tasks
**Symptom:** the launcher prints `Resolved train_data: [...]` and then
`No task directories or environments found; skipping` during the Daytona prebuild. It still submits.
The job would burn a multi-node allocation training on nothing.
**Cause:** `hpc/rl_launch_utils.py` `resolve_rl_train_data()` extracts **only** when
`is_hf_dataset_path(data_path)` is true; a local path is passed through **verbatim**. SkyRL's
`TerminalBenchTaskDataset` wants a dir of task subdirs each holding `instruction.md`, not a parquet.
Passing the HF repo id instead is no good either — that path imports `data.commons` → `rapidfuzz`,
absent in `envs/rl-megatron`.
**Fix:** extract explicitly, then pass the extracted dir:
```bash
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet <cached snapshot dir> --output_dir $SCRATCH/tasks/<name>
```
r2egym: 3,328 task dirs in ~20s. **The confirmation you want in the log is
`Found 3328 task(s) with 7 unique environment(s)`** — the "7 unique environments" line is the real
proof the tasks are wired; a skipped prebuild means they are not.

### 2026-07-29 · Setting `DCFT_RL_ENV` silently activates the BROKEN env anyway
**Symptom:** you export `DCFT_RL_ENV=.../envs/rl-megatron`, the job still runs the broken `envs/rl`.
**Cause:** `hpc/rl_launch_utils.py:780` is `RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"` followed by a
bare `-d` test. `envs/rl` **exists** (it's merely unusable), so the `-d` succeeds and the
`elif [[ -n "${DCFT_RL_ENV:-}" ]]` branch is **never reached**. A directory-exists test cannot detect a
broken env.
**Fix:** set **`RL_ENV_DIR`** explicitly, not (only) `DCFT_RL_ENV`. Put it in the YAML's
`container.extra_env` — `universal_rl.sbatch:105` emits `{rl_container_env}` *before* activation for
exactly this reason (see its comment at line 102).

### 2026-07-29 · JURECA→Jupiter has NO route; the tunnel must be initiated from Jupiter
**Symptom:** planning a "vLLM on Jupiter, sandboxes on JURECA `dc-cpu`" split, the obvious direction
(workers dial the model) cannot be made to work.
**Cause:** JSC inter-system routing is **asymmetric**. Measured with a real `python3 -m http.server`
listener on Jupiter login `134.94.0.132:9931`:
- JURECA login → Jupiter: `No route to host` on the public IP, timeout on all internal IPs
  (`10.201.15.132`, `10.128.1.2`, `10.99.0.2`), and `:22` **refused**. Dead in that direction.
- Jupiter → JURECA `:22`: **OPEN** (also JUDAC `:22` open).

**Fix:** initiate from Jupiter — `ssh -R` out to a JURECA login node, where `dc-cpu` workers can reach it
(already proven: the bridge served `jrlogin04i:9920`). This is why Marianna has `tunnel_monitor.sh` +
`vllm_router.py` on a login node — the router is the relay, not a nicety. Two prerequisites:
(a) `~/.ssh/` on Jupiter was **empty**, so a keypair must be registered in JuDoor for JURECA with a
`from=` clause covering **all ten** Jupiter logins `134.94.0.131–140`; (b) `ssh -R` binding `0.0.0.0`
needs `GatewayPorts yes` on JURECA, which usually defaults to `no` (loopback only) ⇒ expect to need a
login-node relay listening on `0.0.0.0` in front of the loopback tunnel endpoint.
**Corollary:** the old handoff gate "serve 30B on JURECA `dc-gpu` at TP=4 against 40 GB A100s" is
**moot** — no model runs on JURECA at all.

### 2026-07-28 · An already-cached HF model still fails to load on compute nodes
**Symptom:** `RuntimeError: Cannot send a request, as the client has been closed` instead of a cache
fallback.
**Cause:** compute nodes have no internet, and both RL venvs set `include-system-site-packages = true`,
so they import **`huggingface_hub` 1.24.0 from `miniforge3`**, which raises instead of falling back.
**Fix:** set `HF_HUB_OFFLINE=1` (+ `TRANSFORMERS_OFFLINE=1`). Cache is at
`/e/scratch/reformo/lee27/cache/hf`; the Qwen3-30B-A3B snapshot `ad44e777…` is complete.

### 2026-07-28 · `sinfo` shows thousands of "idle" nodes during a full-machine outage
**Symptom:** `sinfo` showed 5655 "idle" nodes while nothing cluster-wide was running, and Slurm gave no
start estimate.
**Cause:** **5542 nodes were in `maint`.** Full-machine maintenance is set via node state, **not** a
reservation, so `scontrol show res` shows nothing.
**Fix:** check real state with `sinfo -p booster -o "%.14T %.7D"`.

### 2026-07-22 · vLLM's FA backend dies even though FA2 itself is fine
**Symptom:** vLLM FlashAttention backend crashes on import.
**Cause:** **CuTe ≠ FA2.** FlashAttention's optional `flash_attn/cute/` path still uses deprecated
`cute.core.ThrMma`; cutlass-dsl 4.6.1 moved it to `cute.ThrMma`. If `cute` is importable, vLLM's FA
backend dies. Torch `2.9.0+cu130`, wheel `2.8.3+cu130torch2.9` (mjun0812 v0.7.16 aarch64).
**Fix:** install FA `--no-deps`, then move `flash_attn/cute` → `cute.DISABLED_cutlass461`. Script:
`hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh`. **Re-run after any FA reinstall.** YAML:
`trainer.flash_attn: true`. Full writeup: `ai_memory/logs/2026-07-22_jupiter_flash_attn_cute.md`.

### 2026-07-22 · `ModuleNotFoundError: torchtitan` at `FSDPPolicyWorkerBase.init_model`
**Cause:** MoE FSDP+EP needs `torchtitan@a1fdd7e` + `tyro>=0.9`, which aren't in the base env.
**Fix:**
`uv pip install --python $DCFT/envs/rl/bin/python 'tyro>=0.9' 'torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e'`
(job r10 `1012192` died on this.)

### 2026-07-22 · Crash at `skyrl_begin_weight_reload` with stock vLLM
**Cause:** stock `vllm==0.13` lacks `vllm.model_executor.model_loader.reload`, which MoE weight sync
needs. (job r12 `1012293`.)
**Fix:** use the lab fork; pin the torch-2.9-compatible `mlfoundations/vllm@084aa19f0`.
**Do NOT** work around it by disabling `SKYRL_W13_RELOAD_BRACKET` on MoE — that produces token salad.

### 2026-07-22 · `_StridedShard` ImportError from `_dtensor_spec` under torch 2.9
**Cause:** the symbol moved out of `_dtensor_spec` in torch 2.9. (job r11 `1012240`.)
**Fix:** MarinSkyRL `fsdp_utils.py` → import from `placement_types`. Patch was cluster-local only
(branch `lukedhlee/fix-stridedshard-torch29` @ `1f0c5b0`).

### 2026-07-26 · A 30B Megatron job is silent for 10+ minutes and looks wedged
**Cause:** FlashInfer MoE autotuning plus worker bring-up after vLLM's `Model loading took ...` line.
**Fix:** nothing — **do not call it wedged too early.**

## Vista

### 2026-07-19 · Ray fails to start: `--gres=gpu:1` invalid
**Cause:** Vista is 1 GPU/node and does not accept gres requests; you request **nodes**.
**Fix:** drop the gres flag. (job `842804`.)

### 2026-07-19 · `python` after `conda activate` is 3.9 and has no pydantic
**Cause:** `/bin/python` won the PATH after activation. (job `842745`.)
**Fix:** use the env's full interpreter path.

### 2026-07-19 · `ImportError: PlacementGroupSchedulingStrategy`
**Cause:** Ray 2.55 moved it. (job `842924`.)
**Fix:** updated import (branch `lukedhlee/fix-ray-255-pg-strategy`).

### 2026-07-21 · SkyRL config validation: `train_batch_size (8)` rejected
**Cause:** `train_batch_size` must be at least the policy DP LCM (`policy_dp_size=16`,
`lcm_dp_size=16`). No training step runs. (job `849329`.)
**Fix:** raise `train_batch_size`/`policy_mini_batch_size` to ≥ the DP LCM.

### 2026-07-21 · `UnboundLocalError: cannot access local variable 'os'` in instrumented `fsdp_worker.py`
**Cause:** a later *local* `import os` inside a function shadowed the module-level `os`. This was our
own debug instrumentation, **not** evidence of the stall being investigated. (job `849358`.)
**Fix:** remove the inner import.

### 2026-07-21 · An end-of-run checkpoint appears despite `ckpt_interval=999999`
**Cause:** SkyRL always writes a final checkpoint + HF export at clean termination. (job `849442` wrote
`global_step_4`.)
**Fix:** none needed — just don't be surprised, and account for the disk.

### 2026-07-21 · `401 Unauthorized` on HF Hub upload after a successful run
**Cause:** missing/invalid HF token in the job env. Not a training or sync failure.
**Fix:** set `hf_hub_repo_id: null` if you don't want the auto-push, or provide a valid token.

### 2026-07-21 · Ray `WorkerCrashedError` right after "Successfully saved model weights"
**Cause:** node **memory pressure during save/export** — a policy rank reported node memory rising
85.2% → 91.9% (18.3 → 17.2 GiB available). This is a post-training save/export crash, not the
policy-train hang. (job `850044`, Slurm `FAILED` exit `1:0` even though 10/10 steps completed and both
the checkpoint and export were written.)
**Fix:** reduce concurrent trials / parallel generation workers around save; check the artifacts before
concluding the run lost work.

### 2026-07-21 · vLLM engine hits `OSError: [Errno 116] Stale file handle` during FlashInfer JIT
**Cause:** FileLock on a shared FS during JIT. **Retryable** — vLLM recovered on its own. (job
`849442`.)
**Fix:** none; do not treat as fatal.

### 2026-07-13 · Per-step eval jsonl dumps are unreliable for an A/B
**Cause:** both A/B arms wrote the same `~/exports/dumped_evals/global_step_*` paths concurrently →
last-writer-wins.
**Fix:** WandB is the source of truth for the comparison.

## JSC — access, containers, filesystems

### 2026-07-28 · `id -Gn` does not list `container` even though apptainer works
**Cause:** the apptainer authorization check hits LDAP directly while `id` is served from a stale local
cache.
**Fix:** **never gate a check on `id -Gn`** — it gave us a false negative. Test with
`apptainer --version`.

### 2026-07-28 · "User lee27 is not in group container" and no way to bypass it
**Cause:** `/usr/bin/apptainer` is a stripped ELF with the group check compiled in (not a wrapper
script), and `podman` / `docker` / `enroot` / `udocker` are **all absent** on JURECA. The group is the
only path.
**Fix (exact JuDoor path):** Software → *Request access to restricted software* → **Access to other
restricted software** → **Container Runtime Engine** → **`Get Access`** → accept the SLD. Accepted
2026-07-28; one signature covers ALL JSC systems.
**Two traps that cost time:** the card looks inert until clicked (`Get Access` is a *second* reveal),
and JSC's error text says "Container Service Level Description" while the UI labels it "Container
Runtime Engine" — so searching JuDoor for the document by name finds nothing. ("Usage Agreement for
Access to JUWELS" is a different, irrelevant document.) Propagation takes hours.

### 2026-07-28 · `apptainer pull` blows the home quota
**Cause:** default cache is `~/.apptainer/cache`; SWE-bench images are 1–3 GB each.
**Fix:** always set the cache to scratch **before any pull**:
```bash
export APPTAINER_CACHEDIR=/p/scratch/synthlaion/lee27/apptainer_cache
export APPTAINER_TMPDIR=/p/scratch/synthlaion/lee27/apptainer_tmp
```
(Benign warning, ignore: `'nodev' mount option set on /p/scratch`.)

### 2026-07-28 · `--fakeroot` fails: "No user namespaces available"
**Cause:** no user namespaces on JSC login nodes ⇒ no `apptainer build` from a def file. Consistent with
the docs: building images on JURECA requires root. `apptainer pull` (converting a *published* image)
works fine.
**Fix:** bind-mount static binaries instead of building an overlay — see [[decisions]] 2026-07-28.
Whether compute nodes have namespaces is still **untested**.

### 2026-07-28 · `asciinema requires ASCII or UTF-8 character encoding`
**Cause:** the static asciinema 3 binary needs a UTF-8 locale; with Apptainer `--cleanenv`, `LANG=C`
fails.
**Fix:** set `LANG=C.UTF-8` and `LC_ALL=C.UTF-8` (the measured SWE-bench image provides `C.utf8`).

### 2026-07-28 · Binaries staged via `--env PATH=...` vanish in later `exec instance://` calls
**Cause:** the PATH passed to `instance start` was not present in subsequent
`apptainer exec instance://...` invocations.
**Fix:** use direct file binds instead:
```bash
--bind "$TOOLS/bin/tmux:/usr/local/bin/tmux:ro" \
--bind "$TOOLS/bin/asciinema:/usr/local/bin/asciinema:ro"
```

### 2026-07-28 · OpenCode fails with `EROFS: read-only file system`
**Cause:** OpenCode writes state even for CLI startup, and the SIF is immutable. Overriding `HOME` on a
later `apptainer exec instance://...` is rejected by Apptainer.
**Fix:** bind per-instance writable dirs: `/root/.local`, `/root/.cache`, `/root/.config` (rw).

### 2026-07-28 · `opencode-linux-x64-baseline-musl` won't run despite the name
**Cause:** it is dynamically linked to `/lib/ld-musl-x86_64.so.1`, absent on JURECA and not assumable
across SWE-bench images.
**Fix:** use the **baseline glibc** build (1.18.8 verified on the login node and inside the measured
SIF).

### 2026-07-28 · A 500-image apptainer prebuild is CPU-bound, not network-bound
**Symptom:** `time apptainer pull` of a 999 MiB SWE-bench image → `real 2m46.9s, user 6m51.4s`.
**Cause:** time goes to parallel blob decompression + SIF assembly, not download.
**Consequences:** (a) more parallelism helps only up to the core count; (b) **do not run a 500-image
4-way-parallel prebuild on a shared login node** — hours of heavy CPU on a node others need, and JSC
does police it. Genuine tension: login nodes have internet but shouldn't host the load; `dc-cpu` nodes
have cores but are air-gapped and need the SOCKS proxy.

### 2026-07-29 · GPFS inode exhaustion + Apptainer futex failures during prebuild
**Cause:** unpacking/building directly on GPFS scratch. Two cancelled builds left temp trees that took
`/p/scratch` to **100% inode use**.
**Fix:** unpack on node-local `${LOCALSCRATCH:-${TMPDIR:-/tmp}}` and atomically copy only completed
SIFs to GPFS. Remove leftover temp trees **single-threaded**. After cleanup: 44% inode use, 2,476,273
free inodes.

### Standing filesystem hazards (JSC)
- **Never `find`/`du` on Jupiter GPFS** — inode quota is the binding constraint; exhaustion can get you
  banned. Same caution on JURECA scratch.
- Inode/file-count caps on project dirs are ~**3M files for ALL users** per compute project. Hitting it
  **locks the project for everyone**. Watch `jutil project dataquota -p <ACCOUNT>`.
- Put `ulimit -c 0` in bashrc/sbatch so core dumps don't explode file counts.
- Conda/mamba installs many files → envs on **scratch** or shared/module/container, never a private
  conda forest on `/p/project1`.
- Full map: [[jsc_paths_hazards]], [[jureca_what_goes_where]].

## Harbor / OpenCode

### 2026-08-03 · ★ `opencode_config.compaction.reserved` is DEAD CONFIG — OpenCode never reads it
**Symptom:** agentic rollouts 400 with
`prompt contains at least 28673 input tokens ... 4096 output tokens ... 32769 total`
**even with** `compaction.auto: true` **and** `compaction.reserved: 16384` present and verified in the
materialized Harbor TrialConfig (`agent.kwargs.opencode_config.compaction.reserved = 16384`).
**Cause:** OpenCode 1.18.8 accepts the key, Harbor `_deep_merge`s it, and it is written to
`~/.config/opencode/opencode.json` — **but OpenCode never reads it.** The proactive compaction
threshold stays at its default `limit.context - limit.output`. With Harbor's
`_resolve_model_limit` giving `context = window - output - 1024 = 27,648` and `output = 4,096`,
that threshold is **23,552**. A prompt sitting *below* it (measured: 19,892) does not compact, and
one large tool observation (measured: 44,027 bytes → **11,056 tokens**) then blows past the server's
28,672 prompt ceiling. **Same class as the `strict_json_parser` bug: accepted, written, ignored.**
Verifying a key reached the trial config proves NOTHING about whether the consumer honors it.
**Fix:** move the threshold with the knob that actually drives it — the window ADVERTISED to the
client. OpenThoughts `4fb4a158` adds `context_budget.client_window_tokens` (optional, defaults to
the served window) and materializes THAT into `model_info.max_input_tokens` instead of
`request_window_tokens`. At 20,480 → `context = 15,360`, compaction at **11,264**, leaving 17,408
tokens of slack. Server contract unchanged (32,768 / 28,672 / 4,096). Preflight and the canary now
**reject** `reserved` outright so it cannot silently return.
**Evidence:** canary `r2egym-v1-00005__Bvnfhmh` (fail) vs `r2egym-v1-00005__oAMqKmL` (pass).

### 2026-08-03 · ★ A big prompt drop is NOT proof compaction worked — it may be REACTIVE recovery
**Symptom:** trajectory shows a large prompt-token drop (measured: 19,892 → 2,951, a 16,941-token
fall) that looks exactly like healthy proactive compaction — but the run actually breached the
context ceiling first.
**Cause:** when OpenCode gets the 400, it recovers by summarizing ("create an anchored summary from
the conversation history") and continues. That recovery produces the same signature as proactive
compaction. A `has_compaction_drop()`-style check **cannot distinguish them**, so if OpenCode then
exits 0 the run passes on a ceiling breach. Under GRPO the wasted request is a real rollout defect.
**Fix:** always check for `ContextOverflowError` / `28673` / `32769` / `prompt contains at least` in
the OpenCode event stream **before** interpreting a prompt drop. `4fb4a158` makes the canary fail on
any such occurrence, and asserts `threshold + largest measured tool jump < server prompt ceiling`.
**Corollary:** grepping a trial tree for success markers is unsound — the rendered trial config
contains the instruction text, so marker strings match before the agent has done anything.

### 2026-08-03 · ★★ SkyRL's RL extraction is hard-coupled to Terminus2 — OpenCode rollouts are unusable
**Symptom:** every OpenCode RL rollout dies with
`INFRASTRUCTURE FAILURE [TypeError]: Trajectory ... failed outside the model` and
`fail_on_infrastructure_error: true` aborts the **whole generation batch** — including trials that
already earned a real verifier reward of 1.0. Underlying error:
`'NoneType' object is not subscriptable`.
**Cause:** `terminal_bench_generator._process_trial_result` (MarinSkyRL ~line 1483) reads
**unconditionally**:
```python
chat_history       = result.agent_result.metadata["all_messages"]
summarization_count = result.agent_result.metadata["summarization_count"]
```
`terminus_2.py:2157-2160` is the **only** place in Harbor that populates these. Harbor's `opencode.py`
never set `AgentContext.metadata` at all, so it is `None`. **Operator-required change A (switch
training rollouts Terminus2 → OpenCode) silently severed the training data path.** It stayed hidden
for days because every earlier run died sooner — Daytona → streaming → tool-choice → context
overflow. `store_all_messages: true` was already being passed to OpenCode and silently ignored:
**the third instance of accepted-but-unimplemented in one day**, after `strict_json_parser` and
`compaction.reserved`.
**Fix:** Harbor `179b31e9` on `lukedhlee/apptainer-opencode-bridge` — `_build_all_messages()`
reconstructs an OpenAI-style chat history from the stdout events `_convert_events_to_trajectory`
already parses; the instruction is captured in `run()` (it is a CLI arg, not an event). Populated
**before** the trajectory-conversion early returns, so a conversion failure cannot abort a batch.
Gated on `store_all_messages` ⇒ eval paths byte-identical. Validated against all 7 retained
`1218813` trajectories: all reconstruct as `user/assistant/tool` alternation.
⚠ **`rollout_details=None` is NOT the blocker** — SkyRL is None-safe there and falls back to the
3-tuple path (TIS degrades, no crash). Only `metadata` is fatal.
⚠ **FIDELITY LIMIT — gate before promoting past a smoke.** OpenCode records prose and tool calls as
SEPARATE STRUCTURED EVENTS and does not retain raw completion text, so the model's literal Qwen3 XML
tool-call markup is unrecoverable. The reconstruction is **faithful in structure, approximate in
tokens** — fine for pipeline validation, NOT sufficient to trust TIS importance ratios. The exact
path is the literal recording proxy (`literal.jsonl` → `_parse_literal_proxy_log` → per-turn
`prompt_token_ids`/`completion_token_ids`); it was NOT enabled on `1218813`. **Enable it before the
50-step promotion.**
**Deployment trap:** harbor is a **copied install** in `envs/rl-megatron/lib/python3.12/site-packages/harbor`,
NOT editable. The fix only takes effect because the rendered RL sbatch puts
`harbor-apptainer-bridge/src` FIRST on `PYTHONPATH`, shadowing it. **Always verify a harbor change
with the job's real `PYTHONPATH`** — a bare venv-python import resolves to the stale copy and will
tell you the fix is absent.

### 2026-08-03 · ★ SkyRL `all_messages` roles must be ONLY `user`/`assistant` — `tool` aborts the batch
**Symptom:** `INFRASTRUCTURE FAILURE [ValueError]: ... failed during result processing: Expected
message role to be 'user' or 'assistant', got tool`, which fail-loud escalates into aborting the
whole generation.
**Cause:** `skyrl_train/generators/utils.py:1943` accepts only those two roles. An OpenAI-style
`role: "tool"` message — the natural way to represent an observation — is rejected.
**Fix:** deliver tool observations as **user** turns (what terminus_2 does); fold the
`tool_call_id` into the message text. Harbor `1f38665f`.

### 2026-08-03 · ★★ A non-zero OpenCode exit is the NORM — fail-loud on it aborts every batch
**Symptom:** `INFRASTRUCTURE FAILURE [NonZeroAgentExitCodeError]: ... Aborting rollout generation`,
killing all 64 trajectories.
**Cause:** two compounding facts. (1) `_raise_if_fail_loud` aborts on ANY exception classified as
infrastructure — i.e. the **`mask`** treatment — when `fail_on_infrastructure_error: true`. So
adding an exception to `mask_exceptions` does **NOT** make it survivable; it makes it fatal. Only
`passthrough_exceptions` is survivable. (2) Retained results from `1219434` show
`NonZeroAgentExitCodeError` on **nearly every trial, including ones scoring reward 1.0** — OpenCode
routinely exits non-zero. One context overflow at 11:23:19 aborted the batch six seconds later.
**Fix:** put `NonZeroAgentExitCodeError` in **`passthrough_exceptions`** next to
`ContextLengthExceededError`, of which it is the downstream manifestation. The trial keeps its real
verifier reward. Genuine infra (bridge/verifier/network) stays fail-loud via `mask_exceptions`.
**Remember the taxonomy:** `mask` = infrastructure ⇒ **ABORTS** under fail-loud · `zero` = counted
as reward 0 in the baseline · `passthrough` = keep the verifier reward, no abort.

### 2026-08-03 · ★★ Qwen3.6-35B-A3B on raw r2egym is near-DETERMINISTIC per task ⇒ no GRPO gradient
**Symptom:** rewards look healthy across tasks (5 groups all-1.0, 8 groups all-0.0) but **0 of 15
groups had any WITHIN-group reward variance** at `n_samples_per_prompt: 2` — every group was `[1,1]`
or `[0,0]`.
**Why it matters:** GRPO advantage is computed *within* a prompt group. All-zero groups are dropped
by `rloo_n_filter_zero_reward_groups: true`; all-one groups carry zero advantage. **No group can
contribute gradient**, so the optimizer would have had nothing to consume even with rollouts and
extraction working perfectly. Cross-task spread is NOT a substitute.
**Evidence strength:** if per-task pass probability were mid-range (~0.7), the chance of all 13
observed pairs agreeing is ~0.05%. The near-determinism is real, not sampling noise — though n=2
per group is weak on its own.
**Action taken:** canary rebalanced to **16 groups × 4 samples = 64 trajectories** (identical cost,
reallocated toward within-group variance). 16 is the hard valid minimum under FSDP4xEP4 fully-async
GRPO. If 4 samples still show no variance, that is strong evidence the **model-specific learnable
band** (`0 < pass@k < 1`, PARKED by operator 2026-07-29) is REQUIRED, not optional — the co-lead
reached the same conclusion independently. → [[r2egym_apptainer_reference_impl]]
**Do not** read a healthy-looking `avg_raw_reward` as proof GRPO can learn; check within-group
variance explicitly.

### 2026-08-03 · ★★ Slow model load is the FILESYSTEM: `/e/scratch` gives 104 MB/s, `/e/fscratch` 2.3 GB/s
**Symptom:** the 67.0 GiB / 26-shard Qwen3.6 checkpoint takes **8–15 min per engine group** to load
(measured 7:38, 9:49, 11:40, 15:21 across runs), ~18–30 s per 2.6 GiB shard.
**Cause:** raw read bandwidth on `/e/scratch`. Measured with `dd ... iflag=direct` (O_DIRECT, 2 GiB)
from a Jupiter login node:

| path | throughput |
|---|---|
| `/e/scratch` read | **104 MB/s** |
| `/e/fscratch` write | 283 MB/s |
| `/e/fscratch` read | **2.3 GB/s (22×)** |

67 GiB ÷ 104 MB/s ≈ 11 min — matches every observation. **It is not vLLM and not the prefetch
setting.** vLLM does log
`Auto-prefetch is disabled because the filesystem (GPFS) is not a recognized network FS (NFS/Lustre)`,
which looks like the culprit, but job `1217881` already forced
`--safetensors-load-strategy=prefetch` and it was NO faster (153 s populating page cache, then
33–58 s/shard vs ~31 s baseline). You cannot prefetch your way out of a 104 MB/s pipe.
**Fix:** stage the checkpoint on **`/e/fscratch`** (flash). Same 8M inode limit, only 1% used, and
17.9 GB of a 42.9 TB data limit — a 67 GiB model is trivial. Projected load ~30 s instead of ~11 min.
⚠ Caveats: measured from a LOGIN node (verify on compute), and **fscratch retention/purge policy is
undocumented** — treat it as transient. Keep checkpoints/HF exports on `/e/scratch`; stage only the
read-only model and the high-churn `trace_jobs` on fscratch.
**Corollary:** this likely also explains the unexplained **110-min** shards→policy-init interval on
`1221005` (vs ~28 min on `1219434`, same geometry) — GPFS contention across the 26 users sharing
`reformo`, not a code regression.

### 2026-08-03 · Filesystem choice on Jupiter — inode headroom and freshness differ a lot
`jutil project dataquota -p reformo` (**never `find`/`du` on GPFS**):

| filesystem | data | inodes | accounting freshness |
|---|---|---|---|
| `/e/scratch/reformo` (exa) | 79.9 TB / 214.7 TB | **6.14M / 8M (77%)** | stale (2026-07-24) |
| `/e/project1/reformo` (exa) | 7.8 TB / 21.5 TB | **3.93M / 4M (98%)** | stale — **AVOID** |
| `/p/scratch/reformo` (JUST) | 12.0 TB / 96.6 TB | 1.00M / 4M (25%) | **fresh (08-03)** |
| `/e/fscratch/reformo` (exa flash) | 17.9 GB / 42.9 TB | **80k / 8M (1%)** | 2026-07-30 |

`/e/scratch/reformo` is shared by **26 users**, so the inode cap is a shared resource — our own
footprint was only ~470k of 6.14M (~8%), meaning tidying our debug traces frees almost nothing
(today's three runs = 156 trial dirs ≈ 1.9k inodes). `/p/scratch` is the older cross-system JUST
filesystem (our r2egym SIFs already live at `/p/scratch/synthlaion/lee27/r2egym_sif`); it has ~3M
free inodes but is not Jupiter-native, so expect no bandwidth win. **`/e/fscratch` wins on both
inodes and speed.**
**Real deletion win found:** `envs/rl` was **77,743 inodes** of a venv that can NEVER work on Jupiter
(its `vllm._C` needs `libcudart.so.12`; Jupiter ships only CUDA/13). Deleted after verifying the
ImportError and that `envs/rl-megatron` still imports `vllm._C` (vllm 0.22.0, torch 2.11.0+cu128)
**with `module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13`** — without those modules even the
good env fails on `libcudart.so.13`, so never judge a venv from a bare login shell.

### 2026-08-03 · ★ `OSError: [Errno 122] Disk quota exceeded` mid-rollout = INODE exhaustion, not space
**Symptom:** many trials abort simultaneously with
`INFRASTRUCTURE FAILURE [OSError]` / `[Errno 122] Disk quota exceeded: '/e/scratch/...'`.
`OSError` is not in the exception config, so it takes `default_error_treatment: mask` ⇒ classified
infrastructure ⇒ `_raise_if_fail_loud` aborts the whole batch (killed Jupiter `1221005` at 14:03).
**Cause:** **inodes, not bytes.** `jutil project dataquota -p reformo` showed `/e/scratch/reformo` at
37% of its data limit but **77% of its 8M inode soft limit**, and `/e/project1/reformo` at **98%**.
Every RL trial directory is many small files, so a few dozen 64-trajectory runs move the inode
counter fast. The quota counters are refreshed only every few days, so a healthy-looking number can
be badly stale (ours was 10 days old).
**Diagnose with `jutil project dataquota -p <ACCOUNT>` — NEVER `find`/`du` on GPFS**; the metadata
scan is itself the hazard the handoff warns about, and hitting a project file-count cap **locks the
project for ALL users**.
**Fix:** delete superseded experiment trace trees (`experiments/<run>/.../trace_jobs`) after
extracting their findings, and treat post-run trace cleanup as part of finishing a run rather than an
afterthought. Extract the metrics you need FIRST — the reward/variance analysis only needs
`result.json` per trial.

### 2026-08-03 · Pinned FlashInfer AOT cache is x86-64; Jupiter GH200 is aarch64
**Symptom:** `tvm_ffi.load_module` reports the `fused_moe_90` file as not found even though the file
exists with the expected SHA-256.
**Cause:** the loader's incompatible-ELF error surfaces as "file not found". The archive from
`flashinfer-jit-cache` is an ELF **x86-64** shared library; Jupiter compute reports **aarch64**.
Hash, package version, `is_aot`, and path checks all passed because **none of them called `dlopen`** —
so the 90-second AOT smoke `1216854` was a **false positive**.
**Fix:** never enable `59e661e0`'s artifact on Jupiter. Any AOT hook must validate ELF CPU
architecture and perform a real `tvm_ffi.load_module` before model load. Use cold node-local JIT.

### 2026-08-03 · Unbounded fused-MoE JIT can OOM a GH200 node during cold start
**Symptom:** ~300–350 concurrent `nvcc`/`cicc`/`cc1plus` processes saturate node memory, NVCC targets
die with exit code 9, and Ninja fails. No literal CUDA-OOM line is printed (job `1217866`).
**Fix:** `MAX_JOBS=24` bounds Ninja concurrency. Measured on `1217900`: JIT completed in ~19 min with
~1 concurrent compiler process and 660 GiB of 857 free — vs ~15.5 min unbounded, so the cost is small.
Note the multi-engine case is milder (6-node `_17` peaked at 146 processes across 4 engines/node and
succeeded unbounded); the single-engine-per-node case is the one that blew up.

## Cross-cluster

### 2026-07-22 · `NCCL_DEBUG=INFO` produces no NCCL lines in the Slurm `.out`
**Cause:** Ray worker stdout does not carry the NCCL library's INFO output; only our own debug prints
land there (235 MB of them on job `854962`).
**Fix:** file-backed logging — `NCCL_DEBUG_FILE=.../nccl_%h_%p.log` plus
`NCCL_DEBUG_SUBSYS=INIT,NET,COLL`. MarinSkyRL `prepare_runtime_environment` now forwards
`NCCL_DEBUG_FILE` and mkdirs its parent. Verified on smoke `855073` (~44k INFO lines/file).
Flight-recorder (`nccl_fr/nccl_trace_`) only dumps on a **torch PG timeout** — Ray await hangs still
need a sync-await timeout to be caught.

### 2026-07-14 · `uv sync --extra vllm` dies on `flash-attn==2.6.3`
**Cause:** `match-runtime=true` and no static metadata; MarinSkyRL's vllm extra pins torch
2.11.0+cu128 and source-builds flash-attn + transformer-engine. Production avoids this with a prebuilt
wheel cache (the gpu-rl Docker image).
**Fix:** borrow an env that already has a compiled `flash_attn`, or build from prebuilt wheels. Details:
[[jureca_agentic_daytona_plan]].

### 2026-07-14 · `examples.terminal_bench...` entrypoint won't resolve in a borrowed env
**Cause:** a rogue `examples` **regular** package in the borrowed env's site-packages shadows
MarinSkyRL's **namespace** `examples/` (which has no `__init__.py`). The env is read-only so it can't be
stripped.
**Fix:** a PYTHONPATH shim — a regular `examples/__init__.py` package that out-ranks the rogue one and
redirects `__path__` to MarinSkyRL's real examples tree. Full recipe:
[[jureca_agentic_daytona_plan]].

### 2026-07-14 · `uv run --isolated --extra vllm` needs internet, compute nodes have none
**Fix:** `uv sync --extra vllm` on the **login** node once, then in the job run
`uv run --offline --no-sync -m ...` (or activate the pre-built venv). Point uv at scratch
(`UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`) so the venv isn't an inode bomb on project1.

### 2026-07-13 · A shared SkyRL checkout's git remote embeds a leaked PAT
**Symptom:** `/p/project1/laionize/dcagent-shared/SkyRL` remote contains an EtashGuha PAT.
**Fix:** clone from the **public** repo instead (pin the same commit); flag the key for rotation. A
committed key is a leak — rotate, don't fix-forward.

### 2026-07-31 · FlashQLA 0.1.2 backward asserts `dg should be fp32` on a BF16 smoke gate
**Symptom:** the GH200 smoke compiles and executes every FlashQLA forward/backward TileLang kernel, then
fails in `chunk_gated_delta_rule_bwd` at `assert dg.dtype == torch.float32`.
**Cause:** the first smoke constructed `g = logsigmoid(g_logits)` from BF16 logits, so FlashQLA received a
BF16 decay gate. The real Transformers Qwen3.5/3.6 path explicitly computes
`g = -A_log.float().exp() * softplus(a.float() + dt_bias)`, so its gate is FP32 even when activations are
BF16. This was a smoke-harness mismatch, not the model integration.
**Fix:** construct the smoke gate from `g_logits.float()` while retaining a BF16 leaf to verify gradient
flow back through the cast. Jupiter job `1139108` then passed BF16 forward/backward on GH200 and wrote
`gpu_sm90_smoke.ok`.

### 2026-07-31 · Pinned FlashQLA wheel validation rejects valid underscore metadata names
**Symptom:** an isolated install of the five exact wheels succeeds, but its equality gate reports
`flash_qla` and `torch_c_dlpack_ext` as unexpected instead of the expected hyphenated package names.
**Cause:** wheel `Name` metadata is not guaranteed to use the same separator spelling as requirement
names. The validator lowercased names but did not canonicalize underscores to hyphens.
**Fix:** normalize `dist.metadata["Name"]` with `.lower().replace("_", "-")` in both the installer and
GRPO preflight. Do not weaken version equality.

### 2026-07-31 · Qwen3.6 login-node preflight cannot import `vllm._C` before CUDA modules load
**Symptom:** the GRPO wrapper fails before submission with `ImportError: libcudart.so.13` even though the
rendered Slurm job would load the correct modules.
**Cause:** the wrapper deliberately imports `vllm._C` on the Jupiter login node as a fail-fast check, but
that happens before generated sbatch setup. CUDA 13 is supplied by JSC modules, not the venv alone.
**Fix:** load `GCC/14.3.0` and `nvidia-compilers/25.9-CUDA-13` in the wrapper before runtime imports.
With the modules loaded, the full preflight passes using torch 2.11.0+cu128, Transformers 5.8.1, and
vLLM 0.22.0.

### 2026-07-31 · Cluster dotenv values can silently overwrite explicit launch paths
**Symptom:** a generated Jupiter job starts in `/e/scratch/jureap59/feuer1/...` or activates that
user's Miniforge even though the caller supplied owned `DCFT`, `SCRATCH`, and RL environment paths.
**Cause:** `hpc.set_environment` and the generated sbatch sourced cluster dotenv values after the
caller's environment, treating site defaults as overrides.
**Fix:** dotenv entries are defaults only; preserve explicit caller values. Jupiter dotenv path
assignments use `${NAME:-default}`, and RL jobs skip the generic cluster Conda activation before
activating their dedicated venv. Regression-test the generated sbatch, not only the Python config.

### 2026-07-31 · Offline W&B can still hang when distributed actors force `mode="shared"`
**Symptom:** the driver says `WANDB_MODE=offline`, then model construction stalls for 90 seconds while
per-node W&B actors attempt an online shared run.
**Cause:** MarinSkyRL's distributed tracking path hard-coded `mode="shared"`; it ignored the driver's
offline setting.
**Fix:** derive W&B mode once. In offline/disabled mode initialize only the local run and do not create
distributed per-node W&B actors. Jupiter job `1141941` proved the corrected path entered an offline
run without network access.

### 2026-07-31 · Qwen MoE policy construction needs the `ep` extra, not just base SkyRL
**Symptom:** all eight vLLM engines load Qwen3.6 successfully, then FSDP policy initialization dies in
`skyrl_train.models.layers.moe` with `ModuleNotFoundError: No module named 'torchtitan'`.
**Cause:** the Qwen3.6 config enables grouped-GEMM MoE with the Torch expert-parallel backend. That
path requires TorchTitan pinned at `a1fdd7e`, declared under SkyRL's `ep` extra, but the borrowed
`envs/rl-megatron` runtime had only the base/vLLM dependencies.
**Fix:** install exact TorchTitan commit `a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41` plus its missing
locked runtime dependencies (`tomli`, `tyro`) into the actual RL venv. Gate with imports of both
`torchtitan.distributed.expert_parallel` and `skyrl_train.models.layers.moe` before submitting; a
plain `import skyrl_train` is insufficient.

### 2026-08-01 · Directly resubmitting a rendered RL sbatch can lose its venv
**Symptom:** the batch driver reaches RayCluster, but every Ray head/worker step exits 127. Earlier in
the log, `Python executable:` is empty and the script says the checkout-local `envs/rl` is absent.
**Cause:** the wrapper exported an absolute `RL_ENV_DIR` in its submission environment, but the
rendered sbatch did not encode it before activation. A later YAML/container export has the correct
path but runs too late. Direct `sbatch old_rendered.sbatch` therefore falls back to
`$WORKDIR/envs/rl`, warns and continues. The driver uses an absolute Python path, while RayCluster
uses bare `ray`, producing the misleading late exit 127.
**Fix:** launch through the wrapper or explicitly export the absolute `RL_ENV_DIR`. Durable fix: emit
the resolved venv before activation and hard-fail unless both `bin/python` and `bin/ray` work. Never
allow a missing RL environment to be only a warning. Job `1144362` is the reproducer; it did not test
Ray, TorchTitan, GPUs, or JURECA.

### 2026-08-03 · FlashInfer AOT metadata/path checks can accept the wrong CPU architecture
**Symptom:** an AOT smoke verifies the exact archive/SO hashes, package version, `is_aot`, and
`aot_path`, but the real Qwen engine fails only after a 67 GiB model load when
`tvm_ffi.load_module()` reports that the verified `.so` cannot be opened.
**Cause:** the staged official FlashInfer cache wheel was x86-64, while Jupiter GH200 compute nodes
are aarch64. FlashInfer's JitSpec construction does not load the shared library; all old smoke
assertions were metadata-only. The loader's incompatible-ELF failure looked like a missing file even
though the file remained present with the expected hash.
**Fix:** never infer binary compatibility from wheel name/hash/path. On the target compute node,
compare the ELF `e_machine` field with `platform.machine()` and actually call
`tvm_ffi.load_module()` before model load. Disable the x86 artifact in Jupiter configs and use the
node-local cold JIT unless an exact aarch64 artifact is obtained. Jobs `1216854` (false-positive
smoke) and `1217562` (real late-load failure) are the paired reproducer.

### 2026-08-03 · Forced safetensors prefetch is slower for this 67-GiB Qwen checkpoint on Jupiter GPFS
**Symptom:** vLLM warns that GPFS is not a recognized network filesystem and suggests
`--safetensors-load-strategy=prefetch`, making it look like an obvious cold-start acceleration.
**Cause:** the forced path spent 153 seconds reading all 26 shards into page cache with eight
background threads while the loader also progressed slowly. Its observed shard times were roughly
33--58 seconds, versus about 31 seconds on average for the ordinary loader; it had reached only
12/26 shards after about 8.5 minutes of actual loading.
**Fix:** keep the default safetensors loader for this exact model/runtime/GPFS path. Do not treat
vLLM's generic filesystem warning as a site-specific performance result. Bounded comparator job
`1217881` was cancelled once it had falsified the optimization.

### 2026-08-03 · Unbounded FlashInfer fused-MoE JIT can exhaust a GH200 node during exact-model startup
**Symptom:** after all 26 Qwen shards load, fused-MoE JIT launches roughly 300--350 compiler
processes, node memory approaches saturation, many unrelated NVCC targets fail together with exit
code 9, and Ninja reports `subcommand failed` / `Ninja build failed`. There may be no literal
`CUDA OOM` or kernel-OOM line in the application log.
**Cause:** FlashInfer honors `MAX_JOBS`, but without it Ninja derives concurrency from the 64-CPU
allocation. Individual `cicc` processes reached several GiB RSS; the broad simultaneous SIGKILL-
like failures are compiler-memory pressure, not a bad CUDA source or model weight.
**Fix:** bound the exact diagnostic and future cold-JIT launches with `MAX_JOBS` (backup `1217900`
uses 24) and verify the rendered/runtime environment carries it before model load. Job `1217866`
is the reproducer: model load succeeded, then the unbounded build failed after roughly 20 minutes
of post-load profiling/JIT.

---

## 2026-08-04 · Qwen3.6 canary — four measurements that overturned prior conclusions

### 2026-08-04 · Flat GRPO reward variance because the agent was killed at 600s, not 1800s
**Symptom:** 0 of 22 prompt groups had ANY within-group reward variance across `1219434` (32×2) and
`1221005` (16×4). Complete n=4 groups were perfectly uniform: `[1,1,1,1]`, `[0,0,0,0]`. Attributed to
"Qwen3.6 is essentially deterministic per r2egym task" and to raw r2egym being untrainable.
**Cause:** the agent was silently capped at **600 seconds**, not the configured 1800. On `1221005`,
**19 of 25 trials** ended at exactly 600–601s, and **19 of 19** exceptions carried the literal string
`Command timed out after 600s` with `return_code -1`. The chain:

| layer | value |
|---|---|
| `trial.py:_compute_agent_timeout_sec()` | 1800 (from `agent.override_timeout_sec`) |
| `trial.py:_run_agent_phase()` | `asyncio.wait_for(agent.run(), timeout=1800)` — **OUTER** guard |
| `opencode.py:871` | issues `opencode ... run` via `exec_as_agent()` with **no `timeout_sec`** ← the gap |
| `apptainer.py:397` | `effective_timeout = timeout_sec or 600` ← the real budget |
| `worker.py:1117,1143` | kills at that deadline → `return_code -1` |

The outer 1800s guard can only fire if it is **shorter** than the exec fallback, so the worker always
won. `agent.override_timeout_sec = 1800.0` was present in the materialized TrialConfig the whole time.
**This is the FOURTH instance of the signature bug** (cf. `strict_json_parser`, `compaction.reserved`,
`store_all_messages`): a key accepted, deep-merged, written to disk, and ignored by its consumer.
**Fix:** harbor `f44a1170` adds `BRIDGE_EXEC_TIMEOUT` (default unchanged at 600). Set it **ABOVE** the
agent budget — 2100 against 1800 — so the outer guard wins: `AgentTimeoutError` is `passthrough` and
preserves the trajectory, whereas an exec kill surfaces as `NonZeroAgentExitCodeError`. OT-Agent
`2b6defd1` wires it in.
⚠ **Corollary:** `NonZeroAgentExitCodeError` is NOT "the norm for OpenCode" as
[[decisions]] records — it was the timeout kill. Reclassifying it to `passthrough` was still correct,
but the *diagnosis* was wrong, and it hid the real defect for a day.

### 2026-08-04 · Never infer filesystem bandwidth from `cp` — readahead flatters it 30×
**Symptom:** a 4-stream `cp` of the 67 GiB checkpoint off `/e/scratch` ran at **1.67 GB/s**, appearing
to refute the documented "104 MB/s" and the case for staging on `/e/fscratch`. I reported that the
22× claim "doesn't survive" and that the loader, not the tier, was the bottleneck. **Both were wrong.**
**Cause:** `cp` reads whole files sequentially through the page cache with GPFS readahead. `O_DIRECT`
strips readahead out and measures the un-prefetched rate. Measured `O_DIRECT` on compute
`jpbo-018-14` (job `1224678`):

| streams | `/e/scratch` | `/e/fscratch` |
|---|---|---|
| 1 | **0.056 GiB/s (57 MB/s)** | **2.51 GiB/s** |
| 26 | 1.586 GiB/s | 56.3 GiB/s |

`/e/scratch` is ~57–62 MB/s **per stream** and scales roughly linearly with concurrency; fscratch is
**45× faster single-stream**.
**Fix:** benchmark with `iflag=direct`, from a **compute** node, at the concurrency the real consumer
uses. A login-node `cp` answers a different question than the one you are asking.

### 2026-08-04 · vLLM model load is PER-STREAM bound, so more aggregate bandwidth buys nothing
**Symptom:** `Loading weights took 701.12 seconds` for 67 GiB (job `1217900`) = 98 MB/s, and no amount
of node/CPU headroom changed it.
**Cause:** vLLM loads the 26 safetensors shards **sequentially in one stream** (~26.9 s/shard), so it
sits exactly on `/e/scratch`'s per-stream floor. `/e/scratch` *does* scale to 1.59 GiB/s at 26 streams
— irrelevant, because vLLM only ever uses one.
**Fix:** stage the read-only checkpoint on `/e/fscratch`. Measured on the same 1-GPU canary,
end to end: **701.12s → 38.03s (job `1224804`), 18.4×**, ~11 min saved per engine load. `MODEL_PATH`
is the load-bearing knob: it reaches the launcher as `--model_path` and **overrides the YAML**, so
editing the YAML alone is silently reverted. Keep checkpoints/exports on `/e/scratch` — fscratch
retention/purge policy is UNDOCUMENTED.
⚠ This explains phase A of the load ONLY. The **110-minute** shards-loaded→policy-init regression on
`1221005` (vs ~28 min on `1219434`) happens after the bytes are in memory and is STILL UNEXPLAINED.
Loader thread-count is a second, independent lever that no filesystem change touches.

### 2026-08-04 · `ssh jureca` from a Jupiter login node fails on hostname resolution, not a dead tunnel
**Symptom:** `ssh jureca` → `Could not resolve hostname jureca`, and `ssh -O check jureca` →
`No ControlPath specified`, both of which look like the reverse tunnel has died.
**Cause:** there is **no `~/.ssh/config` on Jupiter**. The ControlMaster is reachable only through its
explicit socket.
**Fix:** `ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de "<cmd>"`. Check liveness with
`ps -p <pid>` on the mux process, not with `-O check` sans socket. The master carries the reverse
tunnel as a `-R` forward; restoring it needs one interactive TOTP that only the operator can supply,
so **do not tear it down to "retry"**.

### 2026-08-04 · `Session open refused by peer` is MaxSessions, and retrying burns TOTP attempts
**Symptom:** `mux_client_request_session: session request failed: Session open refused by peer`,
then ssh falls back to a fresh connection and prompts for TOTP, which in a non-interactive context
fails three times with `TOTP token length is wrong`.
**Cause:** too many concurrent background SSH sessions over the Mac→Jupiter ControlMaster exhaust
`MaxSessions`. It is **not** an auth expiry, despite looking exactly like one.
**Fix:** let long-running background sessions finish before issuing more; do not retry in a loop —
repeated failed keyboard-interactive attempts risk locking the account. Also: piping a remote script
through `tail` buffers its output until EOF, so `flush=True` in the remote script buys nothing; write
to a remote file and read it separately if you need progress.

### 2026-08-04 (later) · CORRECTION: it is ONE session channel, not "too many sessions"
**Supersedes the `Session open refused by peer` entry above**, which blamed generic `MaxSessions`
exhaustion from *many* concurrent background SSH sessions. Measured: with **exactly one** long-running
background `ssh jupiter` alive (a 703s remote script), `ssh -O check jupiter` reported
`Master running (pid=55008)` while every new `ssh jupiter` bypassed the mux and fell through to
`keyboard-interactive`, failing 3× with `TOTP token length is wrong`.
**Cause:** JSC's sshd allows ~**one** session channel per multiplexed connection. Everything else in a
session only works because the calls are **sequential**.
**Fix / rule:** **never run a long-lived background `ssh` to Jupiter while other work needs the
connection** — it locks out the connection for its entire duration, and every locked-out attempt burns
a TOTP attempt (repeated failures risk an account lock). For anything long: `nohup` it **remotely**,
write to a remote file, and poll with short sequential `ssh` calls. Probe with
`ssh -o BatchMode=yes` so a refusal fails cleanly instead of prompting.
⚠ Do NOT kill a long background `ssh` just to reclaim the channel if the remote script owns a bridge
env: SIGHUP can skip its cleanup `finally` and orphan an Apptainer env on a JURECA worker, and if the
env-id was only printed to a buffered pipe you have no clean way to reclaim it.

### 2026-08-04 · ~19 min of startup is vLLM profiling a VISION TOWER we never use
**Symptom:** `init engine (profile, create kv cache, warmup model) took 1145.55 s` (19 min) on job
`1217900`, unchanged by node/CPU headroom, and not explained by weight loading.
**Cause:** the exact timeline shows the gap is the **memory-profiling forward pass**, and it is
profiling multimodal inputs:

```
06:37:07  Model loading took 65.52 GiB memory and 703.18 seconds
06:37:08  Encoder cache will be initialized with a budget of 16384 tokens,
          and profiled with 1 image items of the maximum feature size
          <18m59s of SILENCE>
06:56:07  Available KV cache memory: 13.1 GiB
06:56:07  GPU KV cache size: 560,505 tokens
06:56:07  Maximum concurrency for 32,768 tokens per request: 17.11x
06:56:07  flashinfer autotune starts -> ends 06:56:12 (only 5s)
06:56:13  init engine ... took 1145.55 s
06:56:30  Multi-modal warmup completed in 15.008s
```

Qwen3.6 is a multimodal **shell**. SkyRL unwraps it to the text tower
(`SKYRL_QWEN3_5_VLM_UNWRAP=1`), the vision tower is parked and receives no gradients, and no rollout
ever sends an image — but vLLM still profiles it with one max-feature-size image, in **eager mode**
(`enforce_eager: true`), through a 35B MoE.
**Fix (staged, not yet measured):** `MM_LIMIT=0` on the canary sbatch adds
`--limit-mm-per-prompt '{"image":0,"video":0}'` (OT-Agent `8ac43278`, default unchanged). Should also
**free KV cache**, because the profile is what sizes it.
⚠ **This only became the dominant cost after fscratch.** At 701s load + 1145s init the load looked like
the problem; at **38s** load the profile is ~95% of startup. Re-derive where time goes after every win
— the bottleneck moves.

### 2026-08-04 · `max_num_seqs: 8` leaves measured KV headroom of 17–36× on the floor
**Symptom:** rollout concurrency capped at 32 sequences (4 engines × `max_num_seqs: 8`), matching the
32 JURECA sandbox workers, and adding sandbox nodes alone did nothing.
**Cause:** `1217900` measured `GPU KV cache size: 560,505 tokens` and
`Maximum concurrency for 32,768 tokens per request: 17.11x`. So KV supports ~17 concurrent
**full-length** sequences per engine while we allow 8. At the real prompt length — client window 20480
⇒ Harbor `limit.context` 15,360 — headroom is `560505 / 15360 ≈ 36×`.
**Also:** the agentic workload is **prefill-dominated**: measured **150,223 input vs 3,377 output**
tokens per trial on `1221005` (44:1), because each turn re-sends the growing conversation. At
`max_num_batched_tokens: 4096` a ~15k prompt costs ~4 chunked-prefill passes.
**Fix (staged, not yet measured):** `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` are env-overridable on
the canary sbatch (`8ac43278`). ⚠ Scaling sequences only helps if sandbox workers scale WITH it —
64 sequences needs 4 JURECA nodes, not 2. Balance both sides or one just queues on the other.
⚠ Prefix caching is **enabled** server-side, but vLLM warns
`Prefix caching in Mamba cache 'align' mode is currently enabled. Its support for Mamba layers is
experimental` for `Qwen3_5MoeForConditionalGeneration`. Do NOT assume it is effective for the GDN
layers — and do NOT read the agent's `n_cache_tokens: 0` as proof either way; that field is simply
not populated (measured 0 across all 25 trials while the server had APC on).

### 2026-08-04 · `/e/scratch` cannot create ANY new file — and it breaks `git fetch` too
**Symptom:** `OSError: [Errno 122] Disk quota exceeded: '/e/scratch/.../experiments/...'` mid-rollout,
killing `1221005` and then `1229343` (the latter at 58 min, after producing 18 honest rewards). Then
`git fetch` on a cluster checkout died with `error: unable to create temporary file: Disk quota
exceeded` / `fatal: unpack-objects failed`, so **nothing could even be deployed.**
**Cause:** INODES, not bytes. The soft limit is a **project-wide 8M shared by 26 users** and it is
exhausted. Precisely characterised:

| operation | result |
|---|---|
| `touch` a NEW file on `/e/scratch` | **Errno 122** |
| overwrite an EXISTING file | **OK** |
| `touch` on `/p/scratch` or `/e/fscratch` | OK |

⚠ Because overwrites still work, the filesystem looks healthy until something needs a *new* inode —
which is why it presents as a mid-run crash rather than a clean startup failure. And per the 08-03
measurement our own footprint is only ~8% of the 6.1M, so **deleting our own trees may not fix it.**
**Fix:** put the experiments tree AND the execution checkout on **`/p/scratch`** (OT-Agent `45572f93`).
Three reasons it beats `/e/fscratch` for this tree: ~3.0M free inodes (25% of 4M); **accounting is
FRESH** (`/e/scratch`'s counter was 11 days stale and read 77% while the filesystem was actually
full — trustworthy accounting is itself the feature); and **retention is DOCUMENTED**, which matters
because this tree holds the checkpoint and HF export.
Keep the read-only 67 GiB model on `/e/fscratch`: that is a per-stream BANDWIDTH problem and this is
an INODE problem, and they want different filesystems. The model copy is reproducible in ~43 s so
fscratch's undocumented purge costs nothing there.

### 2026-08-04 · `ReqNodeNotAvail, Reserved for maintenance` while 3,848 nodes sit idle
**Symptom:** a 5-node job pends with `Reason=ReqNodeNotAvail,_Reserved_for_maintenance` and
`StartTime=Unknown`, while `sinfo -p booster` shows **3,848 idle** nodes and `scontrol show res`
exposes no cluster-wide or MAINT reservation (only `develbooster`, `parastation-test`,
`ahf_image_test`, all node-set scoped).
**Cause:** a maintenance reservation we cannot see, and the job's **WALLTIME cannot finish before it
starts**. Slurm therefore defers the job past the whole window instead of running it.
**Fix / diagnostic:** ask the scheduler with `sbatch --test-only -t <T> <sbatch>` — free, no submit,
and it prints the estimated start. Measured:

| walltime | estimated start |
|---|---|
| 06:00:00 | 2026-08-05T12:00:00 (next day, after the window) |
| 04:00:00 and below | 2026-08-04T20:53:47 (same evening) |

Resubmitting at 4 h changed the pending reason from `ReqNodeNotAvail` to plain `(Priority)`.
**Always probe with `--test-only` before assuming a pend is queue congestion** — and note that every
walltime at or below the threshold gave the SAME start, so shortening further buys nothing.
⚠ **Coupling worth remembering:** when a Jupiter job is deferred, the JURECA sandbox fleet may expire
BEFORE it starts. `15494122` would have died ~4 h before a 20:53 CEST start. Submit a fresh 24 h fleet
whenever a run slips, or the job allocates into a world with no sandboxes and every rollout fails.

### 2026-08-04 · `pkill -f <pattern>` over SSH kills your own shell
**Symptom:** a compound remote command silently did nothing — no output from the `scancel` and `echo`
that followed the `pkill`, and the file it was meant to write never appeared.
**Cause:** `pkill -f "retarget.sh"` matches on the FULL command line, and the `ssh` invocation's own
command string contained `retarget.sh`, so it killed the very shell running it.
**Fix:** kill by pid (`pgrep` then filter out `$$`), or match a pattern that cannot appear in your own
argv. Better still: don't kill watchers at all — write them to exit on their own when the job they
poll leaves the queue.

### 2026-08-04 (run `1229488`) · The Apptainer bridge keeps NO persistent log — its history is tmux scrollback
**Symptom:** `/status` reports a lifetime `jobs_errors: 176` and the run ledger attributes it to the two
aborted runs, but that attribution cannot be checked. The only bridge log file on disk,
`/e/scratch/reformo/lee27/apptainer_bridge/server_9920_20260729.log`, is **169 bytes, last written
Aug 2 14:36** — by an EARLIER process.
**Cause:** the live server (pid `950979`, started Aug 2 17:57) has `fd1`/`fd2` → `/dev/pts/2`, i.e. a
tmux pane in session `r2egym_bridge_9920`. Its entire error history is volatile scrollback that dies
with the pane or the login node. The 169-byte file belongs to the tmux wrapper's original python,
which was replaced.
**Consequence:** any post-hoc question about bridge failures during a run is UNANSWERABLE, and every
`jobs_errors` attribution in the ledger is unfalsifiable. Verify with
`readlink /proc/<pid>/fd/1`, not by assuming the logfile in the tmux command line is still being written.
**Fix:** at the next restart with no sandboxes attached, redirect to an append-only file on
`/e/fscratch`. Do NOT bounce the bridge while envs are attached — it kills the live run.
**Read the counter as a DELTA, not a level.** `jobs_submitted == jobs_completed + jobs_errors` exactly,
so a flat `errors` across a run means zero new infrastructure failures regardless of how large it is.

### 2026-08-04 (run `1229488`) · vLLM `/metrics` does NOT exist under SkyRL — the documented way to settle prefix caching is a dead end
**Symptom:** `curl http://<head>:8000/metrics` → `{"detail":"Not Found"}` (404), and `/v1/models` 404
too, while `/health` returns 200. Grepping the job log for `Prefix cache hit rate` / `Avg prompt
throughput` returns NOTHING despite `generator.vllm_stats_interval=1`.
**Cause:** `:8000` is the **SkyRL** HTTP endpoint (`enable_http_endpoint: true`), not a vLLM OpenAI
server. SkyRL drives `AsyncLLM` **in-process** with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so no vLLM
HTTP server is ever started and there is no Prometheus route to scrape. vLLM's periodic stats logger
is also not emitting, and `generator.enable_ray_prometheus_stats=false`.
**Consequence:** [[handoff]] Open #3/#4 propose "read vLLM `/metrics` during rollouts" to settle whether
prefix caching works under Mamba `align` mode. **That method is not executable here** and the item needs
a different approach (explicitly enable the vLLM stat logger, or instrument SkyRL directly).
The only available signal remains OpenCode's `n_cache_tokens`, measured `sum=0` over 7 trials averaging
~132k input tokens — CONSISTENT WITH inactive, but as the docs already warn, not proof: it could be an
accounting gap rather than real non-reuse.

### 2026-08-04 (run `1229488`) · Fixing the 600s cap RE-EXPOSED the 32,768 context ceiling
**Symptom:** six `Input-overflow rejected by vLLM serving (returning 400, non-retryable)` /
`This model's maximum context length is 32768 tokens`, after `1221005` recorded **zero** overflows
following the compaction fix `4fb4a158`.
**Cause (hypothesis, not established):** the overflows are a CONSEQUENCE of our own timeout fix.
`1221005` killed agents at 600s, truncating trajectories before their context could grow; with the
budget genuinely restored to 1800s, agents take more turns, accumulate more context, and re-cross the
32,768 ceiling that compaction at `context_budget.client_window_tokens: 20480` does not fully contain.
**Testable prediction:** overflow rate should scale with agent runtime — checkable per-trial on this
run's completed traces. Confirm before treating it as settled.
**Lesson worth generalizing:** removing a limiter does not just improve a measurement, it moves load
onto whatever the limiter was hiding. Re-check the ceilings downstream of any budget you raise.

### 2026-08-04 (run `1229488`) · MarinSkyRL crashes on a null-content partial response
**Symptom:** `TypeError: can only concatenate str (not "NoneType") to str` at
`inference_engines/inference_engine_client.py:1190`, in
`_parse_partial_response_and_inplace_update_accum` (`accum.content += new_content`), reached from
`_chat_completion_with_retry` → the `/chat/completions` handler.
**Cause:** the retry/accumulate path assumes a partial response always carries string content. When
vLLM rejects an over-long prompt with a **non-retryable 400**, `content` is `None` and the accumulator
raises, so the SkyRL HTTP endpoint returns a 500 to OpenCode instead of a clean error. It is triggered
BY the overflow above, so the two defects are coupled.
**Severity:** non-fatal as observed (1 occurrence, batch did not abort) but it converts a clean
rejection into an opaque 500.
**Fix:** guard with `new_content or ""`. ⚠ **Do NOT deploy mid-run** — the live job reads the cluster
checkout, so editing it under a running job risks newly-spawned workers loading different code than
their peers. Deploy in the gap between runs.

### 2026-08-04 (later) · ⛔ `/p/scratch` IS NOT MOUNTED ON JUPITER COMPUTE — the inode fix had to be re-done
**Symptom:** after `45572f93` moved the experiments tree to `/p/scratch` to escape the `/e/scratch`
inode wall, jobs could not use it from compute nodes at all.
**Cause:** `/p/scratch` (the JUST filesystem) is reachable from Jupiter **login** nodes but **is not
mounted on Jupiter compute nodes**. A login-node `ls` therefore "proves" the path exists while every
compute-node write fails — the same exists-vs-works trap as the route gate and `workers_alive`.
**Fix:** `ae180c37` re-pointed the experiments tree at **`/e/fscratch`**, which is inode-cheap
(80k/8M), fast (2.51 GiB/s single-stream `O_DIRECT`), and mounted on compute. `31cd646d` additionally
moved `WANDB_DIR` off `/e/scratch`, because offline W&B creates a new run directory per job and so
burns inodes on every launch.
⚠ **THE DOC TABLES IN [[handoff]] AND `NEXT_SESSION.md` WERE STALE FOR A WHILE**, telling the next
session to launch with `DCFT=/p/scratch/...`. That path fails on compute. The correct values are:

| what | correct location |
|---|---|
| execution checkout (`DCFT`) | `/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next` |
| experiments tree, ckpt, HF exports, `WANDB_DIR` | `/e/fscratch/reformo/lee27/experiments/<job>` |
| read-only 67 GiB model | `/e/fscratch/reformo/lee27/models/Qwen3.6-35B-A3B/995ad96e…` |

**Verified on run `1229488`:** all of the above on `/e/fscratch`, 26 shards × 4 engines loaded in ~65s,
31 min from allocation to weight-sync-ready.
⚠ `/e/fscratch` retention/purge policy remains **UNDOCUMENTED**, so the durability of a checkpoint or
HF export there is unguaranteed — the hub upload (`hf_save_interval: 1`, `hf_upload_mode: all`) is the
real durability, not the on-disk copy.
**Generalizable:** verify a filesystem from a **compute** node before trusting it, exactly as with
bandwidth. Login-node visibility is not mount coverage.

### 2026-08-04 (later) · FIFTH accepted-but-ignored-key bug: `hf_upload_mode` is dead without `hf_hub_repo_id`
**Symptom:** run `1229488` carries `trainer.hf_upload_mode=all`, `++trainer.hf_hub_private=false`,
`++trainer.hf_hub_revision=main` and `trainer.hf_save_interval=1`, which reads like "export every step
and push it to the Hub." No upload happens, and nothing warns.
**Cause:** `callbacks/builtin.py:977` gates the entire `HFHubUploadCallback` on
`hf_hub_repo_id = getattr(cfg.trainer, "hf_hub_repo_id", None)`:

```python
hf_hub_repo_id = getattr(cfg.trainer, "hf_hub_repo_id", None)
if hf_hub_repo_id and hf_save_interval > 0:   # <- both required
```

`hf_hub_repo_id` is **absent from our hydra args AND absent from `ppo_base_config.yaml` entirely**, so
it resolves to `None`, the callback is never registered, and the three `hf_*` keys above are inert.
`HFModelSaveCallback` (gated only on `hf_save_interval > 0`) still registers, so the **local** export to
`trainer.export_path` IS written — the artifact exists, it just never leaves the cluster.
**This is instance #5** of the signature class (cf. `strict_json_parser`, `compaction.reserved`,
`store_all_messages`, `override_timeout_sec`). New wrinkle worth generalizing: here the key is not
ignored by its consumer — the consumer is **never constructed**, because a DIFFERENT, unset key gates
it. Grep for the `if <other_key> and ...` guard, not just for the key you set.
**Silver lining:** on Jupiter this is better than working. Compute nodes are air-gapped
(`huggingface.co:443` BLOCKED, `proxychains_binary: None`), so a registered upload callback would have
failed anyway — though harmlessly, since `_ensure_repo_exists` and `upload_folder` are both
try/except-guarded and only log.
**Consequence — the one that matters:** a checkpoint/HF export produced on Jupiter lives ONLY on
`/e/fscratch`, whose retention/purge policy is UNDOCUMENTED. Do not assume a hub copy exists. To
publish, upload from a **login** node (which does have internet) as a deliberate follow-up step, and
verify the export STRUCTURALLY first (`config.json` architecture, safetensors index, shard presence,
unwrapped text-tower key set with the vision tower absent).

### 2026-08-04 · ⛔ THE FAIL-LOUD FOOTGUN MATERIALIZED: one unenumerated exception killed a 1h46m run
**Symptom:** `1229488` died `FAILED` (exit `1:0`) at 1h46m holding **68 honest trials and 3 complete
uniform n=8 groups**, with `ray.exceptions.WorkerCrashedError` at
`main_tbench.py:134 ray.get(skyrl_entrypoint.remote(cfg))`. The Ray traceback is a RED HERRING — the
raylet lines (`the leased worker is killed because the job finished`, `Core worker … unavailable`) are
teardown CONSEQUENCE, not cause.
**Cause** — three log lines, one second apart, are the whole story:

```
05:29:11 DEBUG    _classify_exception:1296  - Exception AddTestsDirError not in config,
                                             using default treatment: MASK
05:29:11 WARNING  _process_trial_result    - ... failed with Harbor exception:
                                             Failed to add tests directory to <env>
05:29:11 CRITICAL _raise_if_fail_loud:1247 - INFRASTRUCTURE FAILURE [AddTestsDirError]
```

`AddTestsDirError` (the r2egym verifier fixture could not be staged into the sandbox) matched NEITHER
`mask_exceptions` NOR `passthrough_exceptions`, fell through to `default_error_treatment: mask`, and
`mask` = infrastructure ⇒ `_raise_if_fail_loud` aborted the WHOLE batch under
`fail_on_infrastructure_error: true`. It hit **two trials on two different coordinators within one
second** (`04114` rep 0, `06360` rep 2) — a correlated JURECA-side blip, so every run was exposed.
**This is exactly what [[handoff]] § "Remaining structural hazards" item 3 predicted**, verbatim:
*"`mask` = infrastructure ⇒ ABORTS the whole batch… any new exception subclass must be enumerated or
it silently falls through."* Predicting a hazard is not mitigating it.
**Fix (OT-Agent `c15ac55f`):** `fail_on_infrastructure_error: false` + `AddTestsDirError` enumerated in
`mask_exceptions`. Rationale for `false`: the reason it was `true` (stop infra failures becoming
training signal) is ALREADY served by `default_error_treatment: mask` — a masked trial is excluded from
the RLOO baseline and gets advantage 0, so it cannot pollute the gradient. All `true` added was making
one bad trial fatal to 128.
⚠ **Why NOT `passthrough_exceptions`:** passthrough scores the trial as a real trajectory (0). On a task
that otherwise scores 1 that **FABRICATES within-group variance**, making a zero-gradient run look like
it had learning signal. Never passthrough an infrastructure error on a binary-reward task.
⚠ **Cost of `false`:** infra failures now DEGRADE quietly. Watch per-run exception counts — if enough
trials mask out, groups fall under `rloo_n_min_group_size` and contribute zero advantage while still
presenting as a valid batch.

### 2026-08-04 · The bridge's `cleanup_loop` explains the `jobs_errors` bursts AND auto-reaps orphans
**Two mysteries, one function** (`apptainer/server.py:518`, runs every 30s):
1. **`jobs_errors` rising in bursts** (+50, +36, … ~+184 over `1229488`) while **every** trial still
   scored honestly with zero infrastructure exceptions. The loop marks a running job `error` and
   increments `_stats["jobs_errors"]` when `workers_dead` or `elapsed > timeout*2 + 120`. The bursts are
   **reaper bookkeeping**, not rollout failures — which is why they coincided with `active` dropping
   (env teardown) and never reached the trainer.
2. **Orphaned envs after a crash.** ENV_READY envs unused for `BRIDGE_STALE_READY_SEC` (default **900s**)
   get a STOP queued, capped at `BRIDGE_REAP_BATCH_CAP` (50) per cycle. After `1229488` crashed leaving
   14 ready envs / 13 active jobs, they drained 14→12→10→7→0 with NO manual intervention. This is what
   "55 orphaned sandboxes reclaimed across two sweeps" actually was.
**Operational consequence:** after a crash you must WAIT ~15 min before relaunching, because the
Qwen3.6 preflight hard-requires an idle bridge:
`ERROR: dedicated smoke requires an idle bridge; stale work found: envs.ready=7, active_jobs=6`.
**Do NOT set `REQUIRE_CLEAN_BRIDGE=0`** to get past it — that flag is for intentional fleet sharing, and
launching onto stale envs steals worker slots from your own run. Let the reaper finish.
**Read `jobs_errors` as a DELTA and always cross-check `exception_info` in the trial results** — the
counter alone systematically overstates harm.

### 2026-08-04 (evening) · `--test-only` flipped from maintenance-fit to NODE STARVATION — check `sinfo`, not just the estimate
**Symptom:** at 05:38 CEST every walltime from **30 minutes to 3h** returned the SAME estimated start,
`2026-08-04T23:19`, while `>= 3h30m` returned `2026-08-05T12:00`. Earlier the same day a ≤4h wall
started in 91 seconds (`1229488`: submitted 03:42:11, running 03:43:42).
**Cause:** this is NOT the "walltime cannot finish before a hidden maintenance window" pattern recorded
earlier — if it were, SHORTER walltimes would start sooner. A flat estimate across a 6× walltime range
means **there are no free nodes at all**. `sinfo -p booster` confirmed: **147 nodes `drain*`, 8 `down*`,
1 `idle*`**.
**Diagnostic rule:** when `--test-only` defers a job, probe at least TWO very different walltimes.
- estimate VARIES with walltime ⇒ window-fit problem, shorten the wall
- estimate FLAT across walltimes ⇒ resource starvation, shortening buys NOTHING; check `sinfo` states
**Consequence for planning:** the largest wall that started the same night was **3h**, which does not
fit 16×8 (2h51m needed, 9 min margin). Hence `be949008` dropping the canary to 16×4 (~1h48m).
⚠ **Slurm estimates are conservative** — backfill often starts a queued job earlier — so QUEUE the job
rather than waiting for a better estimate.
⚠ **Re-verify JURECA fleet TTL against the DEFERRED start**, not against submission time. This is the
Open #2 trap: a fleet alive now can easily be dead by the time a deferred job allocates.

### 2026-08-04 · ⛔ `/e/scratch` KILLED A RUN AGAIN — via Ray's object-spill dir, which the migration missed
**Symptom:** `1229643` died `FAILED` after **1m51s**, before Ray finished starting. The visible error was
`RuntimeError: Ray head process exited prematurely with code 1` — and it pointed at a log path that
**does not exist** (`$DCFT/experiments/logs/ray_head_<node>.log`), while the real log went to
`/tmp/ray_logs/` on the head node and was separately preserved to the experiment dir. The actual cause
was three directories away from the error message:

```
OSError: [Errno 122] Disk quota exceeded:
  '/e/scratch/reformo/lee27/ray_spill/ray_spilled_objects_e1802434...'
  ray/_private/node.py:1792 validate_external_storage -> external_storage.py:310 os.makedirs
```

**Cause:** `RAY_object_spilling_config` still pointed at `/e/scratch`, which is inode-exhausted
project-wide and cannot create ANY new file. Ray mints a **NEW session-hashed leaf directory per run**,
so every single launch needs a fresh inode. `os.makedirs(..., exist_ok=True)` does not save you: the
leaf name is unique per session, so it must always be created.
**Why it looked intermittent:** `1229488` started fine at 03:43 the same morning. The quota is shared by
26 users and churns, so the previous run simply won the inode lottery and this one lost it. An
intermittent `/e/scratch` failure is not a flake — it is the wall, sampled.
**Fix (OT-Agent `aab498d7`):** spill dir → `/e/fscratch/reformo/lee27/ray_spill`, in BOTH the 5-node
canary and the 6-node production YAML, plus the test that pinned the old path (now asserts the new path
AND that the old one is gone AND that it is still not another user's tree).
**The lesson that generalizes:** the `/e/scratch` → `/e/fscratch` migration moved the experiments tree,
execution checkout, model, checkpoints, exports and `WANDB_DIR` — and still missed a path, because
`ray_spill` is set in a JSON blob (`RAY_object_spilling_config`) rather than as a plain path key, so it
did not match a grep for the obvious names. **After any filesystem migration, grep for the OLD path
string across the whole repo, not for the things you remember configuring.** Survivors at time of
writing: the `rl-megatron` venv, harbor, MarinSkyRL, FlashQLA and `TILELANG_CACHE_DIR` are all still on
`/e/scratch` — reads work, any write can fail.
⚠ **Consequence already biting:** the MarinSkyRL checkout is on `/e/scratch`, so `git fetch` there fails
with `unable to create temporary file` / `Disk quota exceeded` — meaning **fork fixes cannot be deployed
to MarinSkyRL at all** until that checkout is relocated to `/e/fscratch`. (Blocked: `fafab77`, the
null-content partial-response fix.)

### 2026-08-04 · Two SSH self-inflicted wounds, both already documented, both walked into anyway
1. **`pkill -f "<pattern>"` over SSH kills your own shell** when the pattern appears in your own remote
   command string. Ran `pkill -f "retarget_g.sh"` inside an ssh command that literally contained
   `retarget_g.sh` ⇒ the shell died mid-command and the `setsid nohup ./retarget_job.sh` that followed
   it never ran. It presented as a **silent success** (no output at all), which is the dangerous part.
   **Verify the thing you tried to start is actually running** — `ps -eo pid,etime,args | grep -E
   "retarget_job\.sh 1229649" | grep -v grep`, i.e. a pattern that cannot match your own argv.
2. **`nohup … &` over SSH does NOT detach the channel.** The ssh call hangs holding the login node's one
   session channel for as long as the remote job lives, which locks out every other call to that node.
   `setsid` at least makes the child survive when you kill the local ssh (verified: watcher pid 1309824
   lived through a `TaskStop` of its parent ssh). **The reliable pattern is `tmux new-session -d`** —
   which is exactly how the long-lived bridge is run.
   ⚠ Corollary that saved us here: keep TWO login masters (login01 + login02) and dedicate one to
   polling and one to interactive work. Both lockouts this session were self-contention.

### 2026-08-04 · The tunnel watcher BROKE THE THING IT GUARDS — order load-bearing actions first
**Symptom:** on `1229649` the automated retarget watcher allocated correctly, then printed
`⛔ WARNING: NO running JURECA fleet outlives this job's walltime`, then
`Control socket connect(~/.ssh/cm_jureca/qwen36): Connection refused` → `FATAL: could not add forward`
and exited **without retargeting the tunnel** — the exact omission that killed `1225422`, reintroduced
by the automation meant to prevent it.
**Cause — two independent bugs, both mine, both introduced in the same "improvement":**
1. **Self-inflicted channel exhaustion.** The fleet-sufficiency check (added to close [[handoff]] Open #2)
   opened one ssh session **per fleet** through the JURECA ControlMaster in a tight loop. The master was
   completely healthy — `ssh -O check` reported `Master running (pid=185890)` (up 2d3h) and the bridge
   showed `workers_alive: true` with `worker_polls` climbing 145k→215k throughout — but the rapid session
   churn exhausted its channels, so the `-O forward` that followed got `Connection refused`.
   **A control-socket refusal on a live master is TRANSIENT CONTENTION, not death.** Do not treat it as
   fatal, and do not tear the master down to "retry" (restoring it needs an interactive TOTP only the
   operator can supply).
2. **Fail-wrong fleet check.** It treated an EMPTY query result as "no fleet is sufficient" and printed a
   red alarm. In reality `15495516` had **11h55m** left against a 2h52m job — the query had merely
   failed. Silence is not data; distinguish "query returned nothing" from "nothing qualifies".
**Fix (`retarget_job.sh`, redeployed):** the **retarget now runs BEFORE any diagnostics**, with 5 retries
× 20s on refusal; the fleet check is ONE session parsed by awk, and an empty result reports "cannot tell"
instead of asserting insufficiency.
**The generalizable rule:** in a guard script, do the **load-bearing action first** and diagnose after.
A guard whose diagnostics can prevent its own action is worse than the gap it was written to close.
⚠ **And the reason this was caught at all:** the watcher's *result* was verified rather than its
existence. Same habit that caught the silent `pkill` self-kill an hour earlier. `exit 1` in a detached
watcher is invisible unless you go and read its log.

## The 32-way rollout ceiling was never a JURECA limit (2026-08-04)

For weeks the working assumption was that sandbox supply capped us at 32 concurrent trials. It did not.
Three independent limits were stacked, and only one of them was real:

| limit | value | what it actually is |
|---|---|---|
| `generator.max_num_seqs` × `num_inference_engines` | 8 × 4 = **32** | the real ceiling — vLLM KV budget on the 35B |
| `harbor.n_concurrent_trials` | 32 | sized *downstream* to match the above |
| `worker.py --num-workers` | **default 16**/node, fleet was 2 nodes = 32 | a CLI default, not a limit |

`max_num_seqs: 8` on Qwen3.6-35B-A3B is not timid tuning: ~70 GB of weights on a 96 GB GH200 at
`gpu_memory_utilization: 0.85` leaves almost nothing for KV at `max_model_len: 32768`. So for the **35B**,
adding JURECA nodes buys nothing — Jupiter is the wall. For an **8B** (~16 GB bf16, ~65 GB left for KV),
`max_num_seqs` of 64+ is fine and the sandbox fleet becomes the binding constraint instead.

**Measured on an isolated bridge (port 9921) against a 4→36-node `dc-cpu` fleet at `WORKERS_PER_NODE=48`:**

| concurrent env creates | reached `ready` | p50 time-to-ready | p90 |
|---|---|---|---|
| 8 | 8/8 | 8s | 8s |
| 48 | 48/48 | 22s | 24s |
| 192 | 192/192 | 56s | 60s |
| 384 | 384/384 | 68s | 82s |
| 768 | 766/768 | 102s | 164s |

Sub-linear, and the two 768 failures were `stopped`, 0.26%. `BRIDGE_START_CONCURRENCY` (default 8/node)
throttles *starts*, not *running* instances — the code documents `starter-suid exit 255` above ~16
concurrent starts, so it paces ramp-up and nothing else.

### The bridge scales better than its architecture suggests
It is a stdlib `ThreadedHTTPServer` with one global `_lock`, which sounds like a bottleneck at scale. It
isn't, because **each node runs ONE dispatcher that batch-polls** and local workers consume its queue
("reduces SSH tunnel traffic from N workers polling to 1 dispatcher polling"). At 32 nodes the bridge sees
32 pollers, not 1,536.

### `workers_alive: false` is a bookkeeping artifact, not a dead fleet
`_last_worker_poll` is updated only by the **singular** `/worker/get_job` endpoint. Dispatchers use the
**batch** `/worker/get_jobs`, which updates neither it nor `worker_polls`. A fully healthy 32-node fleet
therefore reports `workers_alive: false` and `worker_polls: 0` while idle. Verify by creating an env and
watching it reach `ready` — never by trusting that flag. Same family as the fleet-sufficiency bug: a
liveness field doing duty it was never wired for.

### Reverse-forward backlog depends on how you bind
The long-lived bridge forward showed `LISTEN 0 5 0.0.0.0:9920` — backlog **5**, and bound to `0.0.0.0`
against the internal-interface rule. Adding a forward with an explicit internal bind address
(`-R 10.14.0.46:9923:...`) yields `LISTEN 0 128 10.14.0.46:9923`. Bind explicitly: it is both the correct
security posture and a 25× deeper accept queue.

## The milestone finally failed INSIDE the update, not in scaffolding (1229649, 2026-08-04)

`1229649` is the first run to get generation all the way done and enter the optimizer update. It then
died in the backward pass:

```
17:22:51  Finished 'wait_for_generation_buffer'   (16/16 groups, 86 trials scored)
17:22:51  reward/avg_pass_at_4: 0.375
17:26:17  Finished 'fwd_logprobs_values_reward'   206s
17:26:17  Finished 'compute_advantages_and_returns'
          ray::FSDPPolicyWorkerBase.ppo_train() -> training_step
          -> fsdp_strategy.backward -> torch.autograd
torch.OutOfMemoryError: Tried to allocate 30.57 GiB.
GPU 0 has 95.00 GiB total capacity, 3.97 GiB free.
```

**⚠ The "Finished: 'policy_train', time cost: 17.14s" line is NOT success.** The timing context manager
emits `Finished` from `__exit__`, which runs while the exception propagates. Three such lines
(`policy_train`, `train_critic_and_policy`, `run_training`) all appear AFTER the OOM was already raised.
Reading them as success is the mistake to avoid — the only trustworthy signal is the absence of a
subsequent traceback, or a `Saved checkpoint` line.

**Why 30.57 GiB with `micro_train_batch_size_per_gpu: 1`.** It is not the batch; it is the vocabulary.
One sequence at `max_prompt_length: 28672` against a ~151k-token vocab is ~8.6 GiB of logits in bf16, and
the backward holds logits + grad + softmax intermediates simultaneously. Gradient checkpointing does not
help here: it trims transformer-block activations, not the final projection. Candidate fixes, cheapest
first: chunked/fused cross-entropy over the vocab dimension, a shorter training-time sequence cap, or
sequence/tensor parallelism across the FSDP group.

**Why this failure is different from the previous nine.** Every earlier failure was scaffolding — a
filesystem quota, an unenumerated exception, a tunnel, an inode wall. This one is arithmetic in the
training step itself, reproducible without any of the agentic machinery, and testable on a single node.
The pipeline up to and including advantage computation is now demonstrated end to end with this model.

## A band probe can run "healthily" and produce zero reward — the quietest reward-integrity bug yet (2026-08-04)

First launch of the 8 band-probe shards: all 8 RUNNING, trial counts climbing (14/10/1/8/12/9/18), bridge
showing 170 active and 332 ready envs. Every single one of the 74 completed trials had
`verifier_result: null` and therefore **no reward at all**. Cause, identical in all 74:

```
BridgeOperationError: Command '['apptainer','overlay','create','--size','4096',
  '/p/scratch/.../apptainer_staging/apt_env-<id>/overlay.img']' timed out after 60 seconds
```

**Why it was silent.** `BridgeOperationError` is in `mask_exceptions`, and `default_error_treatment: mask`
with `fail_on_infrastructure_error: false` means a masked trial is excluded and the batch continues. For
*training* that is the right trade (it stops one bad trial killing 128). For a *probe* it is corrupting:
a band built from that data would have said **every task is unsolvable**, and the run's own progress
counters would have looked healthy the whole time. `scored=N` counts `result.json` files, not rewards.

**Why the overlay create times out.** Both `--size 4096` and `timeout=60` are HARDCODED in
`worker.py` (~line 536) — there is no env knob. Measured on an almost-idle 36-node fleet:

| overlay create target | time |
|---|---|
| `/p/scratch` (shared GPFS) | **18.06 s** |
| node-local tmpfs (`/tmp`) | **3.57 s** |

At ~1,024 concurrent sandbox starts the shared path blows 60 s outright. Note the earlier load test
reached 768 concurrent envs successfully — because it *only* created envs. Once agents are also doing
real I/O against the same filesystem, the create path loses.

**Fix, no code change:** the fleet's `STAGING_BASE` is an env var. Point it at `/tmp/apptainer_staging`
(overlays are per-sandbox and node-local by nature — nothing shared needs them) and drop
`WORKERS_PER_NODE` to 16 so tmpfs cannot starve the node: 16 × 4 GB overlay + 16 × 4 GB sandbox memory
≈ 128 GB of a 512 GB node.

**Standing lesson:** for any probe whose output is a dataset property, verify a **non-null reward** on the
first few trials before trusting trial counts. `ls result.json | wc -l` is a liveness metric, not a data
metric — the same "existence is not function" trap as the route gate and `workers_alive`.

## Address the rollout endpoint by IP: curl and Python resolve it differently inside the sandbox (2026-08-04)

512 band-probe trials died as `APIConnectionError` ("Connection error.") with null rewards. Every
host-level check passed — `/health` 200 from the Jupiter login node, 200 from a JURECA **compute** node,
a real `/v1/chat/completions` with the correct served model name returning a completion, and 64 parallel
connections through the tunnel all succeeding. The failure was only *inside* the Apptainer sandbox, and
only for the agent.

Measured in a single sandbox, seconds apart, via the bridge's `/env/exec`:

| probe | result |
|---|---|
| `curl http://jrlogin05i:PORT/health` | **200** |
| `python urllib http://jrlogin05i:PORT/health` | **[Errno 111] Connection refused** |
| `python urllib http://10.14.0.46:PORT/health` | **200** |
| `getent hosts jrlogin05i` | `10.14.0.46 jrlogin05i.jureca` |
| `getent ahostsv6 jrlogin05i` | *empty* |

So it is **not** IPv6-first. Python's `getaddrinfo` selects a different A record than curl does under
`search jureca`, and the `ssh -R` forward binds **only** `10.14.0.46`.

**Why it never bit the 35B canary:** OpenCode is Node.js and its resolver picks the working address.
terminus-2 is Python/httpx and does not. **Swapping the agent can silently break the rollout transport
while every existing gate passes** — the route gate tests the host, not the agent's HTTP stack, and not
from inside the container.

**Fix:** set `SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST` to the IP literal `10.14.0.46`, never `jrlogin05i`.

**New gate to run before trusting any agentic run through the tunnel:** exec, *inside a real sandbox*,
the agent's own interpreter against the endpoint —
`/testbed/.venv/bin/python3 -c "import urllib.request; print(urllib.request.urlopen('<api_base>/health').status)"`.
Tooling: `/e/fscratch/reformo/lee27/insandbox.py <port>` creates one env, runs DNS/curl/python probes
inside it, and stops it. `APIConnectionError` in `exception_info` is the signature; note the message is
just "Connection error." and names no URL, so the traces alone will not tell you this.

## Thinking mode makes every agent turn exceed terminus-2's request timeout (2026-08-04)

With the endpoint-by-IP fix in place, connections succeeded and the failure mode changed from
`APIConnectionError` ("Connection error.") to **"Request timed out."** on every trial. Measured through
the tunnel against the 8B while shards were live:

| request | result |
|---|---|
| 256 max_tokens | HTTP 200 in **5.0 s** |
| 4096 max_tokens (thinking) | HTTP 200 in **59.8 s** for 2,611 completion tokens (~44 tok/s) |

So a full 4,096-token thinking turn is ~93 s, and terminus-2's per-request timeout fires first. The
endpoint was never unhealthy — a 256-token completion answered in 5 s throughout.

`timeout` IS a passthrough kwarg in `harbor/llms/lite_llm.py` (two allow-lists, ~L149 and ~L186), so
raising it is plausible — but adding an unverified key is the accepted-but-ignored trap that has already
produced five bugs, so the durable fix is to stop generating thousands of tokens per turn:

```yaml
interleaved_thinking: false
extra_body: { chat_template_kwargs: { enable_thinking: false } }
```

A non-thinking terminus-2 turn only has to emit a JSON tool call — tens of tokens — so turns drop from
~60–90 s to a few seconds, which also makes a 1,664-trial shard fit a 3 h wall.

**⚠ The band is a property of the (model, agent, config) triple, not of the dataset.** Measuring with
thinking off measures the band for thinking-off `g1_8b`. A thinking model solves more tasks, so the
always-solved bucket grows and the band shifts. Ask the co-lead whether her band run had thinking enabled
before comparing fractions to her ~35%.

### The three-deep chain this sat behind
Each fix revealed the next, and all three produced **null rewards while every progress counter looked
healthy**:
1. `apptainer overlay create` timing out on shared `/p/scratch` → `BridgeOperationError`
2. `jrlogin05i` resolving differently for Python than curl inside the sandbox → `APIConnectionError`
3. thinking-mode turns exceeding the request timeout → `Request timed out.`

The lesson is not any one of them: it is that **`scored=N` counts files, not rewards**, and a probe must
gate on a non-null reward before it is allowed to consume a wall.

## The one-step no-op had TWO independent causes, and the generator kept restoring one (2026-08-05)

Wiring the `bare_json` tool parser was necessary but **not sufficient**. `g1_diverse_tezos_100k_8b` ends
every episode after one step for two unrelated reasons, and fixing either alone leaves the symptom intact:

| cause | what the model does | fix |
|---|---|---|
| thinking **ON** | emits a ~300-token `<think>` as plain TEXT and never produces a tool call at all | `interleaved_thinking: false` + `extra_body.chat_template_kwargs.enable_thinking: false` |
| `qwen3_coder` parser | *does* emit a bare-JSON call, but the XML-only parser never matches and never logs, so OpenCode sees assistant text | `tool_call_parser: bare_json` + `tool_parser_plugin: <path>` |

Note the thinking case is **not** a 4096-token truncation — the output is short. That misled an earlier session.

### Why the thinking setting kept coming back — fix the file the GENERATOR reads
`base8b.yaml` ships thinking **ON**. The history is a loop:
1. `a42199f5` disabled it — by editing the **shard yamls**.
2. `ca3d56f7` replaced the whole harbor block from `canary_harbor.yaml`, which carried **neither key** → thinking defaulted back on.
3. `b361f179` fixed it again — by hand-editing **one shard yaml**.

Every one of those fixes lived **downstream of `gen_band_yaml.py`**, so a freshly *generated* config regressed
each time, silently. Fixed now in `canary_harbor.yaml` itself (`ef861140`), which makes the generator's output
correct by construction. **Standing lesson: when a generator overwrites a config block wholesale, a fix applied
to its output is not a fix — it is a fix with an expiry date.**

## vLLM 0.22: `@ToolParserManager.register_module` fills `lazy_parsers`, NOT `tool_parsers` (2026-08-05)

My first parser-plumbing gate reported **FAIL: plugin did not register** — and the gate was wrong, not the
plugin. In vLLM 0.22 the decorator form registers **lazily**:

```python
for n in names:
    cls.lazy_parsers[n] = (module_path, class_name)   # no import, no tool_parsers entry
```

So `'bare_json' in ToolParserManager.tool_parsers` is **False even after a fully successful registration**.
`tool_parsers` only fills when something resolves the name. The only trustworthy check is to call
**`ToolParserManager.get_tool_parser(name)`** and confirm it returns the class.

Worse, `import_tool_parser` **swallows every exception** (`logger.exception` then returns), so a genuinely
broken plugin also leaves the registry untouched — indistinguishable from this false negative if you check
the wrong dict.

Verified on the cluster (`/e/fscratch/reformo/lee27/parsergate.py`, vLLM 0.22.0): `lazy_parsers['bare_json']
== ('bare_json_tool_parser', 'BareJsonToolParser')`, `get_tool_parser` resolves it, and it parses bare JSON,
`</think>`-prefixed, `parameters`-keyed and prose-wrapped calls while returning 0 calls for plain text.

The lazy entry's `module_path` is `obj.__module__` = the plugin's basename, resolvable only because
`import_from_path` leaves it in `sys.modules`. Registration and resolution must therefore happen **in the same
process** — fine here, since `pop_openai_kwargs` is called from `_create_engine` inside each vLLM engine actor.

## "33% pass" is NOT a band: across-group pass rate vs WITHIN-group variance (2026-08-05)

The takeover note's headline — *`8×0.0, 4×1.0 → 33% pass`, **33% ≈ Marianna's ~35%**, therefore "a learnable
band EXISTS" and "the band-may-not-exist risk is RETIRED"* — **does not follow from that evidence**, and the
fuller data contradicts it.

Re-measured over the 35B canary's whole `trace_jobs` tree (212 results, `band.py`, all-time window):

```
rewards        : {0.0: 67, None: 114, 1.0: 31}
trajectory     : n=98 median=18.0 max=35        -> multi-turn, genuinely working
trial pass rate: 31.6% (31/98)                  <- reproduces the reported ~33%
groups fully sampled at p@4                     -> 16
  always-solved  :  5 (31.2%)
  never-solved   : 11 (68.8%)
  IN BAND (0<k<4):  0 ( 0.0%)                   <- ZERO
```

**A 31.6% trial pass rate here is produced entirely by a bimodal all-or-nothing split across groups**: 5
tasks the model always solves, 11 it never solves, and **not one group where some samples pass and some
fail**. GRPO's advantage is computed *within* a group, so a group whose 4 samples all agree contributes
**exactly zero gradient**. 0 of 16 groups in band means the measured gradient signal on this subset is zero —
the opposite of the conclusion drawn.

If the true band fraction were 35%, observing 0/16 has probability `0.65^16 ≈ 0.1%`.

**The confusion is a units error and it is easy to repeat:** "reward spread" across a batch is not variance
within a group. The takeover note's own table row says *"reward spread: 0 of 46 groups varied → 8×0.0, 4×1.0"*
— the before-column counts GROUPS and the after-column counts TRIALS. Compared like that, any bimodal task
mix looks like a band.

**Caveats, stated so this is not over-read in the other direction:** 114 of 212 trials were
`BridgeOperationTimeoutError` with a null reward (early wave + the scancel tail), so scoring coverage is
partial; and this is the **35B**, not the 8B `g1_diverse_tezos_100k_8b` that Marianna's ~35% was measured on.
It does **not** show the band is empty for the 8B. What it shows is that **the band question is still OPEN**,
that the retirement of that risk was premature, and that the right measurement is exactly the 8B subset run.

**What the rollout-path fix DID retire** (this part stands): rollouts now produce real multi-turn trajectories
(median 18–21 steps) with non-null rewards. That was the actual blocker. It just is not the same claim.

⇒ Always report the band as **fraction of fully-sampled groups with `0 < passes < n`**, never as a pass rate.
`band.py <trace_jobs> [window_min]` on Jupiter does this, and flags suspected free-pass groups
(`k == n` with median steps ≤ 2) separately.

## `extra_body` (and every harbor-level LLM knob) NEVER reaches OpenCode (2026-08-05)

`interleaved_thinking: false` and `extra_body.chat_template_kwargs.enable_thinking: false` were BOTH set in
the band YAML and both verified present in the materialized config — and the 8B still thought on every turn.

**Cause:** `extra_body` is implemented **only** in these harbor agents —
```
harbor/agents/terminus_2/terminus_2.py          # extra_body -> llm_call_kwargs
harbor/agents/installed/openhands_sdk_runner.py # LITELLM_EXTRA_BODY
harbor/agents/installed/mini_swe_agent.py       # -c model.model_kwargs.extra_body...
```
There is **no OpenCode path**. OpenCode is an external Node process that builds its own OpenAI requests
against the endpoint, so *nothing* harbor puts in its own LLM-client kwargs can reach it. The key is accepted,
stored, echoed in the config dump, and ignored — **the 7th accepted-but-ignored key in this project.**

⚠ The generic lesson is broader than thinking: **for an EXTERNAL agent (OpenCode), harbor-level model/sampling
settings are inert.** Only two layers can affect it — the agent's own config, or the **vLLM endpoint itself**.

### The fix: force it server-side with `default_chat_template_kwargs`
vLLM 0.22 exposes a frontend default that the renderer merges **UNDER** any request-level
`chat_template_kwargs` (so it is a default, not an override), and **both** `OpenAIServingRender` and
`OpenAIServingChat` accept it — which is why it can ride MarinSkyRL's existing `openai_kwargs` splat:
```yaml
generator.engine_init_kwargs:
  default_chat_template_kwargs: {enable_thinking: false}
```
Wired by **MarinSkyRL `9904058`** (`pop_openai_kwargs` forwards it) + **OT-Agent `aae638eb`**
(`BAND_SERVER_NO_THINK=1` in `gen_band_yaml.py`). This reaches every request on the endpoint regardless of
which agent issued it — the property the harbor-level setting could never have.

### What thinking-on actually costs (measured on 1246344, 8B, conc 64)
| | value |
|---|---|
| trials completed in 38 min at 64 concurrent | **5** |
| completion rate | **0.17 trials/min** ⇒ full band ≈ **1,300+ h** |
| trials still in flight at 38 min | **63 of 64** |
| duration of the few that DID finish | median **3.2 min** |
| zero-tool-call trajectories | **40%** |

The bimodality is the tell: the trials that finish are the degenerate ones (1–3 steps), while the rest grind
4096-token thinking turns. For comparison, the same endpoint with thinking off answered a tool-call request in
**1 s / 21 completion tokens**.

### The failure is NOT the parser — check the JSON SHAPE before blaming it
With `bare_json` provably active (a sibling trial made 2 clean tool calls), the zero-tool trajectories failed
because the **model** emitted non-conforming JSON after thinking:
```
{... "action": "explore_repo"}                     # no name/arguments keys at all
{"parameters": {"$schema": ..., "properties": ...}} # a JSON SCHEMA, not a call
{"name":"bash","arguments":{...,"workdir":"/"}      # CORRECT shape, TRUNCATED mid-object
```
So "0 tool calls" has at least three distinct causes beyond the parser: wrong schema, hallucinated schema, and
truncation. **Read the trajectory's raw message tail before concluding a parser bug** — and note that the
truncated case means a *longer* `max_new_tokens_per_turn` is not always the fix; not thinking is.

## NEVER probe the vLLM endpoint during startup — a request during weight sync KILLS an EngineCore (2026-08-05)

I wedged `1246702` by running route/no-think gates against the endpoint while the job was still in
`init_weight_sync_state`. Cost: one full bring-up (~15 min) plus the fleet time it held.

**Mechanism.** The layerwise weight-reload bracket restores every layer's params/buffers onto the **meta**
device. An inference request arriving inside that window runs a real forward over meta tensors, and vLLM's `_C`
custom ops are registered for `torch::kCUDA` ONLY:
```
NotImplementedError: _C::rotary_embedding: attempted to run this operator with Meta tensors,
but there was no fake impl or Meta kernel registered
```
The EngineCore dies, and the driver then **hangs forever** — log frozen at the same line count, `Initialized
weight sync state` printed, zero trace dirs, job still `RUNNING` and burning wall.

`ensure_norm_meta_fakes_registered()` already immunizes `_C::rms_norm` and `_C::fused_add_rms_norm` for exactly
this reason (it was written after the same crash killed 3 engines on the 30B-A3B CP4 cell). **`rotary_embedding`
has no such fake**, so it is the next domino.

**The controlled comparison, which is what settled it:**

| run | probed during weight sync? | `_C::rotary_embedding` meta errors | outcome |
|---|---|---|---|
| `1246344` | no | **0** | generated normally, 3 batches |
| `1246702` | **yes** (12:08–12:11 vs sync 12:08:30–47) | **1** | EngineCore dead, driver hung |

The smoking gun: the crash came back **as the HTTP response to my own probe** —
`{"error": {"message": "EngineCore encountered an issue...", "code": 500}}`.

**Also why the probe results looked insane.** Identical temperature-0 requests returned 43 tokens / no
`<think>`, then 376, then 1024-with-`<think>`, then a 500. Those were not model nondeterminism and not
per-engine config drift (I guessed both, wrongly) — the engine was **mid-weight-reload** and its weights were
partly on meta. **Any measurement taken during startup is garbage, including one that looks like a pass.**

### Rules
1. **Gate ONLY after generation has started** — wait for `Starting batch generation for N trials` or the first
   `trace_jobs/*/` dir. `curl` returning 200 (or a 404 on `/v1/models`) means a socket is open, NOT that the
   model is loaded and stable.
2. A `default_chat_template_kwargs` / thinking-off probe that passes during startup proves **nothing**. My
   "NO-THINK GATE: PASS" at 12:05 was taken on an engine that was about to be reloaded.
3. Worth fixing defensively: register a Meta/fake for `_C::rotary_embedding` (and audit other `_C` ops the
   reload trace can touch) so a stray request degrades to an error instead of killing the engine. Not done —
   needs the exact C++ schema arity and a GPU check, and the measurement was the priority.

## Changing `PORTBASE` between relaunches silently orphans the reverse forward (2026-08-05)

I burned ~15 min of a 128-concurrency run on this. The rule "never reuse a port" (`841a496f`) pushes you to bump
`PORTBASE` on every relaunch — but the forward is added **by hand**, so bumping the port without re-adding it
leaves the agents with **no route to the model at all**:

```
job 1246702 -> port 18210, forward 10.14.0.46:18210 -> 10.128.16.62:8000   (added)
job 1246853 -> port 18220, forward ...                                     (NEVER ADDED)
```

**Symptoms, which look nothing like a networking fault:**
- all 128 sandboxes reach `ready`, bridge `active_jobs: 128`, `workers_alive: true`
- every `trial.log` stops after `[bridge] Env ... started: {'state': 'done', ...}`
- **`agent/` stays completely EMPTY** — no `trajectory.json`, no steps, for 14+ min
- zero results, zero exceptions, zero engine errors; job happily `RUNNING`
- the failure only surfaces at `BRIDGE_EXEC_TIMEOUT` (2100 s), i.e. **35 min of wall per trial wasted**

So the tell for "no route" is **an empty `agent/` dir with a healthy bridge**, not any error message.

**TWO things change per relaunch and BOTH must be re-pointed:**
1. `SKYRL_ROLLOUT_HTTP_ENDPOINT_PORT` (the JURECA-side listen port) — changes because we bump `PORTBASE`
2. the **head node IP** (the Jupiter-side target) — changes because Slurm allocates new nodes

```bash
S=~/.ssh/cm_jureca/qwen36; H=jureca05.fz-juelich.de
ssh -S $S -O cancel  -R 10.14.0.46:<OLDPORT>:<OLDHEAD>:8000 $H
ssh -S $S -O forward -R 10.14.0.46:<NEWPORT>:<NEWHEAD>:8000 $H
ssh -S $S $H 'ss -ltn | grep -E "<NEWPORT>|9923"'     # expect exactly 9923 + NEWPORT
```
Get the head IP from the FIRST node of `squeue -j <id> -h -o '%N'` (`getent hosts <node>`); the endpoint binds
`0.0.0.0`, so any routable IP of that node works.

**Checklist ordering that avoids all three stalls seen today:**
1. wait for `Starting batch generation` / first `trace_jobs/*/` dir — **never probe during weight sync**, it
   kills an EngineCore and hangs the driver
2. re-point the forward to the NEW port AND the NEW head IP
3. only then run the compute-node route gate
4. confirm a non-empty `agent/` dir within ~2 min — that is the real proof the route works

---

## The JURECA ControlMaster lives on the JUPITER LOGIN NODE, not on the Mac (2026-08-05)

`~/.ssh/cm_jureca/qwen36` is a path on **jupiter login02**, not on the laptop. Running the forward commands
from the Mac fails with `Control socket connect(...): No such file or directory` + `Host key verification
failed`, which reads like a broken tunnel but is just the wrong host.

It *has* to be there: a reverse forward `-R 10.14.0.46:<PORT>:<head>:8000` requires the ssh **client** to be
able to reach `<head>:8000`, and only a Jupiter node can. So every forward command is double-hopped:

```bash
ssh -o BatchMode=yes jupiter 'ssh -o BatchMode=yes -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de "<cmd>"'
```

Also: `ssh -o BatchMode=yes jureca ...` **from the Mac** is expected to fail (`Permission denied
(keyboard-interactive)`) — there is no JuDoor key path for JURECA from here. That is not a fleet outage.

Corollary — the port-re-pointing can be **armed unattended** so a PENDING job's forward lands without a human
in the loop. Write a script on the Jupiter login node that polls `squeue -j <id> -h -o %T`, and on `RUNNING`
resolves `scontrol show hostnames "$(squeue -j <id> -h -o '%N')" | head -1` → `getent hosts` → installs the
forward, then `setsid nohup` it. Cancels can be spec'd loosely (`-O cancel -R 10.14.0.46:<port>:0.0.0.0:8000`)
because OpenSSH matches a remote-forward cancel on the **listen** side only.

## Resubmitting the JURECA worker fleet is self-serviceable (2026-08-05)

Previously logged as "only the operator can do this". It is not. `dc-cpu` is usually near-empty (368 idle of
566 on 2026-08-05), so a fresh 32-node / 24 h fleet goes from `sbatch` to `RUNNING` in under a minute, and two
fleets can overlap harmlessly — workers only make **outbound** calls to the bridge, so there is no port
conflict and capacity simply adds (32 nodes x 16 workers = 512 slots each).

```bash
ssh -o BatchMode=yes jupiter 'ssh -o BatchMode=yes -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de \
  "cd /p/project1/synthlaion/lee27/harbor/src/harbor/environments/apptainer && \
   HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src \
   BRIDGE_LOGIN=jrlogin05i BRIDGE_PORT=9923 \
   STAGING_BASE=/tmp/apptainer_staging WORKERS_PER_NODE=16 \
   sbatch --nodes=32 --time=24:00:00 jureca_workers.sbatch"'
```

`BRIDGE_LOGIN`/`BRIDGE_PORT` must match the forward actually installed on the ControlMaster (`jrlogin05i:9923`,
i.e. the `10.14.0.46:9923` listener) — the sbatch's defaults (`jrlogin03i:9920`) are stale and it will abort
after 30 bridge probes. Confirm with `grep -c "starting 16 workers"` on the new job's `.out` (expect = nodes).

Never let the fleet become the schedule constraint: a Jupiter GRPO job that is `PENDING (Priority)` can sit for
hours, and `scontrol update TimeLimit=` **does not help** when the reason is `Priority` rather than `Resources`
(measured: shrinking 3:00:00 -> 1:30:00 moved the estimated start *later*, not earlier).

---

## TP>1 is a hard bring-up failure on this fork, and SkyRL mislabels it "port collision" (2026-08-05)

Job `1247578` (8 nodes, `inference_engine_tensor_parallel_size: 4`, 7 engines) never brought up a single
engine. Every `EngineCore` died with:

```
AssertionError: The env var, __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR, is not permitted
because it is reserved for the internal use.
  ... ray/_private/runtime_env/setup_hook.py:73 in export_setup_func_module
```

**Mechanism.** With TP>1 vLLM builds a distributed executor, which creates nested Ray workers with a
`runtime_env` whose `env_vars` are copied from the parent actor's `os.environ`. That environ contains Ray's own
`__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR`, and Ray asserts against seeing it in a user-supplied `env_vars`.
TP=1 never builds the distributed executor, so the path is never taken — which is why every prior run was fine.

**The log lies (rule 3 again).** `vllm_engine:_create_engine` reports this as
`Engine init hit a port collision (EADDRINUSE / engine-core init) on attempt N/5`. It is not a port collision;
`Failed core proc(s): {}` is empty and the real cause is the assertion above it. Retries cannot clear a
deterministic assertion — 5 attempts x 120 s just burns 10 minutes of allocation. **Gate on
`grep -c RAY_WORKER_PROCESS_SETUP_HOOK` in the job log, not on the EADDRINUSE string.**

**Fix: use TP=1.** For an 8B on 96 GB GH200 this is not a compromise, it is better — 4 engines/node instead of
1, so 8 nodes give 28 engines rather than 7, with no cross-GPU collectives and more aggregate KV. TP=4 in the
parity config was cargo-culted from Marianna's script (she was configuring a *training* run, likely on a larger
model); nothing about the band measurement needs it.

**Consequence for `main_tbench_generate`.** The band generator carries a comment retiring the pure-rollout
entrypoint because smoke `1235333` saw "AsyncVLLMInferenceEngine actors crash-looped inside
`create_ray_wrapped_inference_engines`, **GPUs held 4/4 in placement groups**, no traceback". 4/4 means that
smoke was TP=4 as well, and the symptom is identical to the confirmed TP=4 bug. **`main_tbench_generate` was
probably never broken — it was convicted for TP=4's crime.** Worth re-testing at TP=1, because it is the
correct path for a band: no policy, no ref, no optimizer, no FSDP, no weight sync (so the meta-tensor
probe hazard disappears), and a single `generate()` over all prompts instead of `ceil(N/train_batch_size)`
sequential steps. Do not change TP and the entrypoint in the same run.

**Why a trainer node at all, on `main_tbench`.** `placement.policy_num_nodes=1` +
`colocate_policy_ref=true` + `colocate_all=false` spends one whole node on policy+ref FSDP shards, and
`lr=0.0` is what demotes that trainer to a probe: updates still fire (which flushes the generation buffer so
the run walks the whole shard) but the weights never move, so all `n_samples_per_prompt=4` rollouts come from
the base policy. The cost is a fixed 1 node regardless of scale. On `main_tbench_generate` the node is not
freed so much as repurposed — the driver orchestrating 128 concurrent HTTP trials is genuinely CPU-hungry.

---

## r2egym scores a STOCK LIBRARY, not the agent: /testbed ships empty and the clone needs internet (2026-08-05)

**This invalidates every r2egym reward number this project has produced.** Read it before trusting any band,
pass rate, or "zero variance" claim.

### The symptom that finally exposed it

Job `1248713` (8B `g1_diverse_tezos_100k_8b`, OpenCode, pass@4, TP=1, 128 tasks):

```
results     : 356      rewards: {0.0: 139, 1.0: 217}       trial pass rate : 61.0%
groups: 96 seen, 75 fully sampled (>=4)     IN BAND (0<k<4): 0  (0.0%)
```

A 61% pass rate looks like a working pipeline. It is the tell. **At a 61% pass rate, a 4-sample group is
out-of-band only if all four pass (`0.61^4 = 13.8%`) or all four fail (`0.39^4 = 2.3%`), so ~84% of groups
should be IN band.** Observing 0 of 75 has probability ~`0.16^75`, i.e. never. So the reward cannot be a
function of the model — **it is a constant per task.** This arithmetic needs no external baseline and is the
cheapest possible check; run it before believing any pass rate.

Corroborating signals, all pointing the same way:
- a trial scored **1.0 with ZERO agent steps**
- of trials that passed, only **24% ever edited a file**; of trials that failed, **49%** did — editing
  *lowers* your score
- median steps 3 with `max_turns: 50`, i.e. trials end early, not by exhausting turns

### The cause

Every task ships an **EMPTY `/testbed`**, and `instruction.md` opens with:

```
## Environment Setup (complete these steps first)
cd /testbed
git clone https://github.com/<owner>/<repo>.git . && git checkout <base_commit>
pip install --no-build-isolation -e . ...
cp -r /setup_files/r2e_tests /testbed/r2e_tests
```

The agent is *supposed* to fetch the repo. But sandboxes have no outbound network
(`ENABLE_WORKER_PROXY` defaults to `0`, with the comment *"r2egym does not need outbound access"* — **that
comment is false for this dataset**), so the clone dies:

```
Cloning into '.'...
fatal: unable to access 'https://github.com/numpy/numpy.git/':
  Failed to connect to github.com port 443 after 7126 ms: Couldn't connect to server
```

`/testbed` stays empty, the agent burns its 4096-token budget looping `<think>` blocks (`reason: "length"`),
and the verifier then tests the **stock library the Dockerfile pip-installed** (`pip install "aiohttp==0.22.5"`
etc.) instead of a repo checkout. Whether that stock version satisfies a given task's tests is a fixed
property of the task ⇒ constant reward ⇒ 0% band by construction.

Note what was NOT wrong: the model, the `bare_json` parser, the reverse tunnels, the band metric, the
concurrency. All verified working. The environment simply never handed the agent the code.

### Why the dataset is like this, and the correct fix

`DCAgent/r2egym-patched-full-oracle` (3,328 tasks, HF) has Dockerfiles of the form `FROM python:3.9-bookworm`
that only `mkdir /testbed`. **0 of 200 sampled Dockerfiles populate it.** Yet each task's
`setup_files/test_info.json` still carries the right answer:

```json
{"docker_image": "namanjain12/aiohttp_final:04de8885...", "github_repo": "aio-libs/aiohttp",
 "base_commit": "04de8885...", "python_version": "3.9"}
```

`marianna13/harbor` @ `marianna/beam` shows the upstream-correct pattern —
`adapters/swebench/template/Dockerfile` is literally `FROM {docker_image}`, i.e. use the prebuilt benchmark
image that already contains the repo at the buggy commit. Whoever built this dataset substituted a generic
python base plus "clone it yourself" instructions.

**That pattern does not scale for us: 3,328 tasks = 3,328 DISTINCT image tags at ~1-2 GB each (TBs).**

**Our fix — offline mirrors.** Those 3,328 tasks draw from only **13 repos** (sympy 1286, Pillow 574, pandas
269, moto 220, pyramid 167, tornado 161, numpy 152, scrapy 123, datalad 97, aiohttp 80, coveragepy 78,
matplotlib 73, orange3 48). Bare-mirror all 13 once = **4.3 GB**, bind read-only into the sandbox, and rewrite
the GitHub prefix so the agent's own clone command works verbatim and offline:

```bash
bash src/harbor/environments/apptainer/mirror_r2egym_repos.sh   # harbor d64180d
# layout MUST mirror the URL path so ONE prefix rewrite covers all 13:
#   https://github.com/numpy/numpy.git -> <mirrors>/numpy/numpy.git
```

harbor `d64180d` binds `$BRIDGE_GIT_MIRRORS` at `/git_mirrors` and writes `/etc/gitconfig`:

```ini
[url "/git_mirrors/"]
	insteadOf = https://github.com/
[safe]
	directory = *
```

`/etc/gitconfig`, not an env var, because it must apply to whatever user/env the agent's own shell runs as.
`safe.directory = *` is REQUIRED or git refuses a mirror owned by another uid ("dubious ownership").
Bare mirrors keep full history, so `git checkout <base_commit>` still resolves. `instruction.md` untouched.

**A fleet restart is required** for a worker.py change to take effect, and old fleets MUST be cancelled first —
otherwise some trials are served by unpatched workers and the run is a silent mix.

### Open risk, not yet measured
After cloning, the instruction runs `pip install --no-build-isolation -e .` on numpy/pandas/sympy. Compiling
those in a **1 CPU / 1 GB** sandbox (the Marianna-parity sizing) may be too slow or OOM. If so, sandbox
resources must go up and parity on that axis is lost. Correctness first.

---

## RESOLUTION of the constant-reward bug: use the UNPATCHED dataset + her prebuilt SIFs (2026-08-06)

Supersedes the mirror/parent-checkout workaround in the entry above — that fix was correct but covered only the
**45%** of patched tasks whose `test.sh` reads `/testbed`. It is now inert; keep it for reference only.

**The patched dataset cannot be repaired.** For the "hard repos" bucket (numpy, pandas, orange3, matplotlib,
sympy = **1,828 of 3,328 = 55%**) `tests/test.sh` sets `TESTS_DIR=/r2e_tests`, deliberately OUTSIDE `/testbed`,
so pytest's rootdir walk cannot put the repo on `sys.path` and shadow the pip-installed wheel. Verified by
reading two task `test.sh` files: numpy gets `/r2e_tests`, aiohttp gets `/testbed/r2e_tests`. No amount of
fixing the clone makes those tasks respond to the agent.

**`data/r2egym/PATCHING.md` documented the whole thing all along**, including the giveaway:

> `python:3.9-bookworm` (scientific) | numpy, pandas, orange3, matplotlib, sympy | Tasks **skip
> `pip install -e .`** and pytest runs from `/tmp` so the source tree at `/testbed` doesn't shadow the
> pre-installed package. Trade-off: oracle verifies "**pre-installed wheel satisfies expected outcomes**"
> rather than "this exact commit's source compiles and passes"

The flattening served Daytona's `auto_snapshot` cache (8,101 snapshots → 3). **We are on Apptainer SIFs, so that
constraint no longer applies to us** — the justification for breaking the dataset expired and nobody noticed.

**What to use instead.** Marianna's unpatched dataset AND her prebuilt SIFs are readable on shared scratch:

```
tasks : /p/scratch/transfernetx/nezhurina1/r2egym_apptainer_dataset   4,578 tasks, FROM namanjain12/<repo>_final:<sha>
SIFs  : /p/scratch/transfernetx/nezhurina1/sif_cache/build_r2egym-*.sif   4,568 built, ~830MB each
```

4,578 is **exactly** the pool behind her "~1.6k of 4.5k ≈ 36%" band, so a number measured here is directly
comparable to hers. Cost of adopting it: **zero Docker pulls, zero SIF builds.** The Docker Hub rate limit that
looked like the blocker (~100 anonymous pulls / 6h vs 3,328 needed) is irrelevant.

**Two mechanical facts that make it work:**

1. **A SIF is keyed `build_${task_name}-${sha256(Dockerfile):12}.sif`.** So copy her tree with byte-preserving
   `shutil.copyfile` and KEEP her task dir names (`r2egym-0000`, ...). Verified 4,568/4,578 resolve; the 10
   misses are ones she never built. Rename a dir or touch a Dockerfile and every task misses the cache and
   tries to pull. Symlink her SIFs into our own writable cache (`ln -s`) rather than copying 3.7 TB — our dir
   stays writable so the fleet can still build anything missing.
2. **Her tasks have no `instruction.md`.** The prompt lives in
   `environment/workspace/metadata.json:problem_statement` (alongside `docker_image`, `base_commit`,
   `expected_output_json`), which her harbor fork reads and ours does not — our patcher is what moves it into
   `setup_files/` and writes `instruction.md`. `build_raw.py` generates `instruction.md` from that field, strips
   the `[ISSUE]` wrapper, and states the repo is already at `/testbed` so no agent tries to clone.

Her `tests/test.sh` depends on the IMAGE's contents (`/testbed/.venv`, `/r2e_tests` at root) — correct for
prebuilt images. It runs `uv pip install chardet` under `set -e`; she pre-baked chardet into the SIFs
(`rebuild_r2egym_sifs.sh`) so that succeeds offline. If the SIFs we symlink predate that rebuild, trials abort
BEFORE grading and show **null** rewards — distinguishable from a zero band, so do not pre-emptively patch it.

**Her R2E-Gym path contains no `git clone` and no `git checkout` at all** — the buggy state comes entirely from
the image's baked `/testbed`. So `base_commit~1` was OUR hypothesis and no upstream code validates it; adopting
the raw dataset retires the question instead of betting on the answer.

## `ssh -O cancel -R` matches the CONNECT address, not just the listen port (2026-08-06)

A cancel spec must reproduce the original forward exactly:

```bash
ssh -S $CM -O cancel -R 10.14.0.46:<PORT>:<ORIGINAL_HEAD_IP>:8000 $H   # works
ssh -S $CM -O cancel -R 10.14.0.46:<PORT>:0.0.0.0:8000        $H       # silently NO-OPS, exit 0
```

A wildcard or guessed IP **returns success while changing nothing**, so the port stays bound to a dead head
node. The symptom is an empty `agent/` dir with a perfectly healthy bridge, surfacing only 35 min later at
`BRIDGE_EXEC_TIMEOUT`. This happened three times on 08-05 (and killed `1246853`) because nothing recorded which
IP was installed. Recover the real one:

```bash
N=$(sacct -j <jobid> -X -o NodeList%40 -n | tr -d ' ')
H=$(scontrol show hostnames "$N" | head -1); IP=$(getent hosts "$H" | cut -d' ' -f1)
```

**Fixed structurally:** `hpc/skyrl_standard/jupiter/arm_rollout_forward.sh` appends every install to
`~/.rollout_forwards` and cancels from that file on startup; it refuses to run if the port is bound by something
it cannot account for. Also `hpc/skyrl_standard/jupiter/jup.sh` now serialises all cluster ssh through a
lockfile and forces `BatchMode=yes` — three SSH rules had been written down twice each and violated three times
in one day. Prose does not enforce invariants.

## Anchor your own grep patterns against the config echo (2026-08-06, third instance)

The job log echoes the ENTIRE hydra command line, including `mask_exceptions=[...]` and
`exclude_exceptions=[...]`. A monitor grepping `FileNotFoundError` matched
`RewardFileNotFoundError` inside those lists and reported phantom errors on a healthy run — twice, in a filter
written an hour after re-reading the `grad_norm=1.0` / `max_grad_norm=1.0` entry warning about exactly this.
Exclude the echo line (`grep -v mask_exceptions`) or anchor on a line prefix. Applies to your own monitors, not
just to inherited log patterns.
