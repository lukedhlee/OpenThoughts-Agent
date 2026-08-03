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
