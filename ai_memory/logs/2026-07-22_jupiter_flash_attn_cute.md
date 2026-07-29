# 2026-07-22 — Jupiter flash-attn / CuTe fix (full picture)

## 0. What problem FlashAttention solves (attention, briefly)

Transformers spend a lot of time on **attention**:
for every query token, look at keys/values and mix them.

Naive attention materializes a giant `seq × seq` score matrix in GPU memory.
That matrix is huge → memory-bound, slow.

**FlashAttention (FA)** is an algorithm + GPU implementation that:
- tiles the work so the full score matrix never lives in slow HBM
- fuses matmul + softmax + matmul into fewer memory round-trips
- same math (up to numerics), much less memory traffic → faster + can fit longer context

When people say “we need flash attention on GH200,” they mean:
use this optimized attention path instead of a slower/generic one (eager / some SDPA paths).

---

## 1. Kernel vs Python package (critical distinction)

Two different layers get confused as “flash attention”:

| Layer | What it is | Analogy |
|---|---|---|
| **GPU kernel** | Compiled CUDA code that actually runs on the GPU | The engine |
| **Python package `flash_attn`** | A pip-installable library that *ships* those kernels *plus* Python glue *plus* optional extra code | The car + toolbox in the trunk |

Installing `flash_attn` puts a directory under `site-packages/flash_attn/` containing:
1. compiled `.so` extensions → `flash_attn_func`, `flash_attn_varlen_func` (**FA2 kernels** — the thing we train/serve with)
2. Python helpers like `bert_padding` (pad/unpad for variable-length sequences)
3. **optional** Python subtree `flash_attn/cute/` → experimental newer kernels written with NVIDIA’s **CuTe DSL**

Our bug lived in (3). (1) and (2) were fine.

---

## 2. What is CUTLASS? What is CuTe?

**CUTLASS** = NVIDIA’s library of high-performance GPU building blocks for GEMM / tensor-core work (mostly C++ historically).

**CuTe** = **Cu**da **Te**nsors — the layout/tiling algebra inside modern CUTLASS.
It is a *language for describing*:
- how data is laid out in memory
- how threads/warps map onto tiles
- which MMA (matrix-multiply-accumulate) “atoms” to use

**CuTe DSL** (package `nvidia-cutlass-dsl`, import `cutlass.cute`) = a **Python** interface to write those kernels without raw C++ templates.

So:
- CuTe = a *tool for writing* GPU kernels
- FlashAttention-2 = a specific *attention algorithm implementation*
- You *can* implement FA2 *using* CuTe — but production FA2 in the wheel is mostly **already-compiled CUDA**, not “must import cute every time”

**Analogy:** CuTe is like LLVM IR / a shader language for tensor cores.
FA2 is like “the shipping video codec.” Someone also started rewriting the codec in a new shader language (`flash_attn/cute`). That rewrite is optional.

---

## 3. What is `ThrMma`? Why did it break?

Inside CuTe, types like `ThrMma` describe “thread × MMA atom” structure — plumbing for writing kernels.

In older cutlass-dsl, code said something like:
```python
cutlass.cute.core.ThrMma   # old location
```
In **4.6.1**, that symbol moved to:
```python
cutlass.cute.ThrMma        # new location
```
(and some helpers like `cutlass.utils.ampere_helpers` disappeared).

FA 2.8.3’s `flash_attn/cute/*.py` still uses the **old** names.
cutlass-dsl 4.6.1 only has the **new** names.

So:
```text
import flash_attn.cute  →  AttributeError: cute.core has no attribute ThrMma
```

This is classic **API drift / version skew**:
two packages that talk to each other were released assuming different API shapes.

Important: **FA2 compiled kernels do not need ThrMma.**
Only the Python cute subtree does.

---

## 4. Why did this kill our job? (SkyRL + vLLM stack)

Our GRPO job has two big consumers of attention:

```text
┌─────────────────────────────────────────────┐
│ SkyRL trainer (FSDP policy / ref)           │
│  trainer.flash_attn=true                    │
│  needs: FA2 + bert_padding                  │
│  path: transformers attn_implementation=    │
│        "flash_attention_2"                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ vLLM inference engines (rollout / eval)     │
│  attention backend picker                   │
│  if standalone flash_attn wheel is present, │
│  it may import flash_attn.cute / related    │
│  paths during backend load                  │
└─────────────────────────────────────────────┘
```

Failure ladder we hit:
1. Install FA wheel → FA2 imports OK
2. vLLM tries to load its FA backend → touches cute path → **dies on ThrMma**
3. Earlier workaround: uninstall FA entirely → vLLM lives, but SkyRL `flash_attn=true` dies (no FA2 / bert_padding)
4. Temporary YAML `flash_attn: false` → train with non-FA attention → works but slower / not what we want on GH200
5. Real fix: keep FA2, hide only cute → both sides happy

So the paradox was:
- **Having FA installed** broke vLLM (via cute)
- **Not having FA** broke trainer FA
- Solution: install FA but make `flash_attn.cute` **unimportable**

---

## 5. Exact fix (mechanics)

On Jupiter RL venv:

1. Reinstall the matching aarch64 wheel **with `--no-deps`**
   - Wheel: `flash_attn-2.8.3+cu130torch2.9` (cp312, manylinux aarch64)
   - `--no-deps` so pip doesn’t “helpfully” downgrade/upgrade torch/numpy

2. Rename the broken subtree:
   ```text
   .../site-packages/flash_attn/cute
     → .../site-packages/flash_attn/cute.DISABLED_cutlass461
   ```
   Python package discovery looks for a directory/module named `cute`.
   After rename, `import flash_attn.cute` fails cleanly as “not found,”
   and code that only needs FA2 never goes there.

3. YAML: `trainer.flash_attn: true` again.

Script: `hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh`
(re-run after any FA reinstall — rename is not durable across wipe/reinstall).

Verified:
- `flash_attn_func` CUDA smoke OK
- `transformers.is_flash_attn_2_available() == True`
- SkyRL `_HAS_FLASH == True`
- `import vllm.v1.attention.backends.flash_attn` OK
- `find_spec("flash_attn.cute") is None`

---

## 6. What we lose / don’t lose

**Keep (important):**
- FlashAttention-2 kernels for training (FSDP / transformers)
- FA2 path for vLLM’s normal FA backend (once cute isn’t poisoning import)
- Speed/memory benefits of FA2 on GH200 for our recipe

**Lose (usually irrelevant for us):**
- Experimental `flash_attn.cute` DSL kernels (newer Hopper/Blackwell-ish experiments)
- Any code path that *strictly requires* `import flash_attn.cute`

**Not a correctness change** for standard FA2 attention math.
We did not “fake” flash attention; we removed a broken optional sidecar.

**Caveats:**
- Local/env fix, not upstream. Reinstall FA → cute comes back → re-run script.
- If a future vLLM/FA release *requires* cute for the only good Hopper path, we’d need a real version pin (FA cute ↔ cutlass-dsl) or upstream patch — not our situation today.
- `--no-deps` means you must manage torch/numpy yourself (we already pin numpy≤2.2 etc.).

---

## 7. How you could have fixed this yourself (DIY diagnostic)

1. **Read the traceback bottom-up.** Find the first line that is *our* failure:
   `AttributeError: ... ThrMma` inside `flash_attn/cute/...` importing `cutlass.cute`.

2. **Bisect imports on a login GPU node:**
   ```bash
   python -c "import flash_attn; print(flash_attn.__version__)"
   python -c "from flash_attn import flash_attn_func; print('FA2 OK')"
   python -c "import flash_attn.cute"   # this is the one that explodes
   python -c "import cutlass.cute; print(dir(cutlass.cute)); print(hasattr(cutlass.cute,'ThrMma'), hasattr(cutlass.cute.core,'ThrMma'))"
   ```

3. **Identify skew:** FA cute wants old API; installed cutlass-dsl has new API.

4. **Choose among options:**
   | Option | Pros | Cons |
   |---|---|---|
   | Disable `flash_attn/cute` | Keeps FA2; minimal change | Lose experimental cute path; re-apply after reinstall |
   | Pin older cutlass-dsl that still has `cute.core.ThrMma` | Might make cute work | May break other deps that need 4.6.1 |
   | Upgrade FA to a release whose cute matches 4.6.1 | “Proper” | May not exist yet for our torch/cuda/aarch64 wheel |
   | Patch FA cute imports (`core.ThrMma` → `ThrMma`) | Educational | Fragile; more files; ampere_helpers etc. still missing |
   | `flash_attn: false` forever | Easy | Give up FA2 training speed/memory — bad on GH200 |

5. **Verify both consumers** (trainer FA flag + vLLM FA backend import), not just one.

We picked the first option because FA2 is what we need; cute is optional and broken.

---

## 8. One-sentence summary

**FlashAttention-2 (compiled kernels) is fine; the optional CuTe-DSL rewrite shipped beside it is incompatible with cutlass-dsl 4.6.1; we deleted that optional folder from the import path so vLLM and SkyRL can both use FA2.**
