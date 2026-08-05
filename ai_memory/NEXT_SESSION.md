# Takeover — 2026-08-05 (supersedes the 05 Aug 10:30 KST note)

Report times in **KST**. Read `ai_memory/handoff.md` bottom-up (newest sections first), then `gotchas.md`.

## Mission
Agentic RL (GRPO) on `Qwen/Qwen3.6-35B-A3B` over r2egym. Jupiter = training + vLLM; JURECA = Apptainer
sandboxes reached via a reverse SSH forward from a Jupiter login node.

**Two goals, in order:**
1. **The milestone** — ✅ **THE MACHINERY LANDED on `1243377`.** For the first time: an optimizer update
   ran, `checkpoints/global_step_1` (**235 GB**) was written, and `exports/global_step_1` (**65 GB**,
   HF-format) was produced. The OOM fix worked in production — the log printed
   `chunked gathered log-softmax ACTIVE (chunk=1024, vocab=248320, ...)` and backward completed with
   `OOM=0`, execution continuing past `policy_train` into `train_critic_and_policy` / `run_training`.
   ⛔ **BUT THE UPDATE IS NUMERICALLY VACUOUS — `grad_norm=0`.** All 64 rollouts died with
   `BridgeOperationTimeoutError` (600s), so rewards were null→0, every group was uniform, and the
   advantage was 0. The exported weights are effectively the base model. **A meaningful update still
   requires a working rollout route (below).** Not my code: the chunked backward was verified to return
   correct non-zero gradients (a zero-returning impl would have FAILED the scaled-tolerance parity test).
2. **The learnable band** — p@4 over 3,328 tasks. **Root cause of the zero-variance wall is now known**
   (tool-call format mismatch, below). The fix is written and validated but **not yet wired on the cluster.**

## Establish reality first — do not trust the numbers below
```bash
ssh jupiter "squeue --me -o '%.10i %.9T %.6M %.6L %.4D %N'"
ssh jupiter "sacct -j 1243351 --format=JobID,State,Elapsed -n | head -3"
D=/e/fscratch/reformo/lee27/experiments/jupiter_qwen36_35b_r2egym_grpo_canary_n4t1800_20260804h
N=jupiter_qwen36_35b_r2egym_grpo_canary_n4t1800_20260804h
ssh jupiter "ls -d $D/$N/checkpoints/global_step_* 2>/dev/null; ls $D/$N/exports 2>/dev/null"   # THE MILESTONE
ssh jupiter "grep -ac 'chunked gathered log-softmax ACTIVE' $D/logs/${N}_1243351.out"           # fix engaged?
ssh jupiter "ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de 'squeue --me'"                 # bare `ssh jureca` FAILS
```
⚠ **Judge the milestone by ARTIFACTS on disk, never by a log line.** `Finished: 'policy_train'` is logged
from `__exit__` as an exception propagates. Success = a `checkpoints/global_step_*` dir **and** a
non-empty `exports/`.

## Milestone: the OOM is fixed (MarinSkyRL `637a764`, already on the cluster clone)
`_log_softmax_backward_data` in `logprobs_from_logits_v2`'s bf16 branch. That function bounds memory by
looping the **BATCH** dim — a no-op at `micro_train_batch_size_per_gpu=1` — so it ran one autocast-promoted
fp32 `log_softmax` over the full `[S, V]`. **Real vocab is 248320** (nested in `config.json → text_config`;
the old note's 151936 was the 8B's) ⇒ `30.57 GiB / 4 B / 248320 = S≈33046`, matching exactly.
Fixed by recomputing softmax per chunk in backward (saving only logits), mirroring `_EntropyFromLogits`.
Sequence-chunking alone does **not** work — `log_softmax` saves its output.
GPU-validated: chunked **33.41 GiB** vs stock **OOM at the identical "30.57 GiB"**; training-path parity
9.5e-07; and vs fp64 the chunked path is **2.4x more accurate** than the stock one.

**Off by default.** `SKYRL_CHUNKED_LOGPROBS=1`. Verify by the `ACTIVE` log line, which only fires during a
TRAINING step (~2h in, after generation) — `FIX=0` early is expected, not a failure.

**If it still OOMs:** lower `SKYRL_CHUNKED_LOGPROBS_CHUNK` (default 1024) or cap training seq len.
The op itself is proven to fit; anything left is the surrounding activation budget.

⚠ **The in-run HF push will probably FAIL, and that does NOT invalidate the milestone.**
`HF_HUB_CACHE=/e/data1/datasets/playground/ot-baf/hf_hub` is **feuer1's** dir — `lee27` gets
`Permission denied` — and the sbatch derives `HF_HOME` from it (line ~346). `1229649` carried the same
setting and simply never reached the export. There is **no `HF_TOKEN`** in the env and none under
`HF_HOME`; the valid token (user `lukeleeai`) lives at `$HOME/.cache/huggingface/token`.
The checkpoint and the LOCAL export land on `/e/fscratch` paths lee27 owns, so they are unaffected — do the
push manually from the login node per `rl-agentic-job-cleanup`, with
`HF_HOME=$HOME/.cache/huggingface` (or `HF_TOKEN=$(cat $HOME/.cache/huggingface/token)`).
For a future run, set `HF_HUB_CACHE` to a lee27-writable path before submitting.

## Band: the zero-variance wall is a TOOL-CALL FORMAT MISMATCH
The checkpoint emits **bare JSON** — `{"name":"bash","arguments":{...}}` — with no `<tool_call>`
delimiter. Census on `1242066`: **182 bare JSON, 0 XML.** vLLM ran `--tool-call-parser qwen3_coder`, which
parses only XML, so it matched nothing and **logged nothing**; OpenCode saw text, stopped at step 1; the
verifier graded an untouched repo ⇒ reward is a function of the task ⇒ **0 of 46 groups had variance.**
Thinking was never the cause (`reasoning: 0`; the `<think>` tags are literal trained-in text).

**To finish this (≈30 min):**
1. `cd /e/scratch/reformo/lee27/MarinSkyRL-apptainer-bridge && git fetch fork <branch> && git reset --hard`
   — picks up `d8bdc79`, which forwards `tool_parser_plugin`. **NOT yet synced** (deliberately: a job was
   running and Ray can spawn workers against changed code).
2. Add to the RL yaml / overrides:
   `engine_init_kwargs.tool_parser_plugin: <repo>/rl/tool_parsers/bare_json_tool_parser.py`
   and `tool_call_parser: bare_json` (OT-Agent `4add607c`).
3. Re-run one shard and check `trajcheck.py` gives **`median_steps > 1`**.

⚠ **NEW — exclude the free-pass tasks.** 10 of 46 groups score **1.0 while doing nothing**: those r2egym
verifiers pass on an unmodified repo. They put a ~22% floor under any band number and can never yield
gradient. Drop them from the denominator.

## ⛔ BLOCKER #1 — the JURECA ControlMaster is channel-exhausted; only a human can fix it
Every rollout times out because the sandboxes cannot reach vLLM. Diagnosed hop by hop:

| hop | result |
|---|---|
| vLLM listening on head node | `0.0.0.0:8000` on `10.128.50.117` ✅ |
| Jupiter login → vLLM directly | `http=200` ✅ |
| JURECA listener on `10.14.0.46:18000` | present, `TCP_OPEN` ✅ |
| **end-to-end through the tunnel** | **`Connection reset by peer` after the request** ⛔ |

The listener accepts, then the forwarded channel is reset — the master (`pid=185890`, up 2+ days) has
exhausted its channels. Same failure as the `Session open refused by peer` / spurious-TOTP errors, and the
same root cause the handoff blames for the watcher breaking on `1229649`. Re-adding the forward does not
help; **the ControlMaster must be restarted, which needs an interactive TOTP** — an agent cannot do it.

**After restarting it:** re-add `-R 10.14.0.46:18000:<head-ip>:8000`, then **run the compute-node route
gate** (a real `/v1/chat/completions` from a fleet node) — do NOT settle for `ss | grep 18000`, which is
what I did and it passed while the route was dead.

## Traps that each cost a job or an attempt today
1. **Two MarinSkyRL clones.** A bare `python` imports `/e/scratch/.../MarinSkyRL`; the RL job prepends
   **`MarinSkyRL-apptainer-bridge/skyrl-train`** to `PYTHONPATH`. Patch the BRIDGE clone; give standalone
   tests the same `PYTHONPATH` or you silently test the wrong code.
2. **`WORKDIR` is `/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next`** — NOT
   `/e/scratch/.../OpenThoughts-Agent` (which lacks `hpc/shell_utils/flashinfer_aot_cache.sh`). Submit as
   `cd $W && export DCFT=$W && sbatch --chdir=$W --time=… --export=ALL,DCFT=$W,… <sbatch>`.
   Wrong dir ⇒ instant FAIL (`1243248`, `1243289`). `--time=`/`--export=` override without editing the file.
3. **`retarget_job.sh`'s `OLD_TARGETS` is STALE** (lacks `10.128.18.209`), so a dead job's forward still
   held port 18000 and the watcher burned all 5 retries → FATAL. Cancel explicitly:
   `ssh -S ~/.ssh/cm_jureca/qwen36 -O cancel -R 10.14.0.46:18000:<oldIP>:8000 jureca.fz-juelich.de`
   **Always cancel a job's forward when it dies, or the port is burned.** Then re-add to the new head IP
   and confirm `ss -ltn | grep 18000`. **Verify a watcher's RESULT, not that you launched it.**
4. **Never run Jupiter ssh calls in parallel** — the ControlMaster refuses sessions and you get a spurious
   TOTP prompt. Same channel exhaustion that broke the watcher on `1229649`. Serialize.

## Standing constraints
Jupiter <=16 nodes / JURECA <=32 · never `scancel` another user's job · never `find`/`du` on GPFS or JURECA
scratch · bind listeners to internal interfaces, never `0.0.0.0` · `enable_db_registration: false` · local
clones are ground truth: edit locally → push to `fork` → `git fetch` + hard reset on the cluster, never
hand-edit · no `Co-Authored-By` in commits.

## Do not re-derive
- The **`0/197` band figure is RETRACTED** (one-step-trajectory artifact), and so is the old **"151k vocab
  × 28,672 tokens ≈ 8.6 GiB"** arithmetic — wrong model's vocab. Both are left in `handoff.md` deliberately.
- **Booster is no longer starved**: ~2349 idle nodes, jobs allocate in seconds, longer walls available.
  The 3h geometry that constrained `1229649` is no longer forced.
- Six accepted-but-ignored config keys so far — verify any key by **behaviour**, never by its presence in
  the materialized config.
