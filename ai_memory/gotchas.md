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
