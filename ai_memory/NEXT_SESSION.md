# NEXT SESSION — takeover note

Written **2026-08-06 ~03:00 KST**. Supersedes all earlier versions
(previous one archived at `ai_memory/logs/2026-08-05_NEXT_SESSION_superseded.md`).
Report times in **KST** (cluster clocks are CEST = KST − 7h).

---

## 0. Read this first: the headline finding

**For this project's entire history, the r2egym reward was a constant per task and never measured the model.**
Every pass rate, every "zero variance" conclusion, every band number was measuring the container. The model,
the `bare_json` tool parser, the reverse tunnels, the band metric and the concurrency were all verified
working. None of them were ever the problem.

**How it was proven** (job `1248713`, 8B + OpenCode + pass@4, 128 tasks):

```
results 356   rewards {0.0: 139, 1.0: 217}   trial pass rate 61.0%
groups 96 seen, 75 fully sampled     IN BAND (0<k<4): 0   (0.0%)
```

At a 61% pass rate a 4-sample group is out-of-band only if all four pass (`0.61⁴ = 13.8%`) or all four fail
(`0.39⁴ = 2.3%`), so **~84% of groups should be IN band**. Observing 0 of 75 has probability ~`0.16⁷⁵`. This
arithmetic needs no external baseline — **run it before believing any pass rate.** Corroborating: a trial
scored 1.0 with **zero** agent steps; among passing trials only 24% ever edited a file versus 49% of failing
trials — editing *lowered* your score.

**Root cause.** `DCAgent/r2egym-patched-full-oracle` was deliberately flattened — documented in our own
`data/r2egym/PATCHING.md` — to collapse 8,101 Daytona snapshots into 3, by replacing each task's prebuilt image
`FROM namanjain12/<repo>_final:<sha>` (whose `/testbed` holds the repo at the buggy commit) with a generic
`python:X.Y` base plus an instruction preamble telling the **agent** to `git clone` the repo itself. Then three
things independently guarantee a constant reward:

1. sandboxes have **no outbound network** (`ENABLE_WORKER_PROXY=0`, with a comment claiming "r2egym does not
   need outbound access" — false for this dataset) → the clone dies with
   `fatal: unable to access 'https://github.com/...': Failed to connect` → `/testbed` stays **empty** → the
   verifier grades the stock pip-installed wheel the Dockerfile baked in.
2. the preamble checked out `base_commit`, which **is the fix commit**. Proven per-file: every file under
   `solution/patched_files` is byte-identical to the repo content at `base_commit`, and `solve.sh` "solves" a
   task by copying those files in — which would be a no-op if `/testbed` were already at `base_commit`. So even
   a working clone hands the agent the answer. The intended buggy state is `base_commit~1`.
3. for the "hard repos" bucket (numpy, pandas, orange3, matplotlib, sympy — **55%** of that dataset) `test.sh`
   sets `TESTS_DIR=/r2e_tests`, **outside** `/testbed`, deliberately so the repo cannot shadow the installed
   wheel. Those tasks can *never* respond to the agent, no matter what else is fixed.

The flattening existed to serve Daytona's snapshot cache. **We are on Apptainer SIFs now, so the constraint
that justified breaking the dataset no longer applies to us.**

## 1. The fix that is now deployed

**Stop using the patched dataset. Use Marianna's unpatched one and her prebuilt SIFs.** Both are readable on
JSC shared scratch. Zero Docker pulls, zero SIF builds.

| what | where | state |
|---|---|---|
| her task dataset | `/p/scratch/transfernetx/nezhurina1/r2egym_apptainer_dataset` | 4,578 tasks, read-only |
| our copy | `/e/fscratch/reformo/lee27/tasks/r2egym-raw` | **4,578 built** |
| her SIFs | `/p/scratch/transfernetx/nezhurina1/sif_cache/build_r2egym-*.sif` | ~830 MB each |
| our cache (symlinks) | `/p/scratch/synthlaion/lee27/r2egym_sif` | **4,575 linked, 4,568 resolve** |
| task lists | `band_raw_all_names.txt` (4,568) · `band_raw_32_names.txt` (32, repo-diverse) | `/e/fscratch/reformo/lee27/` |

**4,578 is exactly the pool behind Marianna's "~1.6k of 4.5k" band**, so our number becomes directly
comparable to hers rather than approximately.

**THE INVARIANT — do not break it.** A SIF is keyed `build_${task_name}-${sha256(Dockerfile):12}.sif`. Our tree
was copied with `shutil.copyfile` and keeps her task dir names, so hashes match and her cache resolves
(verified 4,568/4,578; the 10 misses are ones **she** never built). **Rename a task dir or touch a Dockerfile
and every task misses the cache and tries to pull from Docker Hub** — where we have no credentials and would
hit anonymous rate limits (~100 per 6h).

Rebuild the tree with `build_raw.py` (`hpc/skyrl_yaml/jupiter/band/`, also at
`/e/fscratch/reformo/lee27/build_raw.py`). It exists because her tasks carry **no `instruction.md`** — the
prompt lives in `environment/workspace/metadata.json:problem_statement`, which her harbor fork reads and ours
does not. It generates `instruction.md` from that field, strips the `[ISSUE]` wrapper, and states that the repo
is already at `/testbed` so no agent tries to clone. ~55 min to run (cross-cluster copy of ~27k files) — put it
in `tmux` on the cluster, not a held ssh session.

Repo mix — note **no sympy / moto / matplotlib**, unlike the patched pool which was 39% sympy:
`pandas 1442, numpy 776, pillow 619, orange3 482, aiohttp 299, tornado 259, scrapy 215, pyramid 189,
datalad 179, coveragepy 108`.

## 2. Live state at handover

| id | what | state |
|---|---|---|
| `1251403` | **the validation run** — 32 repo-diverse tasks × pass@4, 2 nodes, TP=1, 1:30 wall | RUNNING, engines starting, forward `18300 → 10.128.59.161` installed |
| `15500584` | JURECA sandbox fleet, 32 nodes × 16 workers | RUNNING, ~22h left, `SIF_CACHE` already correct |

**THE GATE — this is the whole question: does any group contain both a `0.0` and a `1.0`?**

```bash
D=/e/fscratch/reformo/lee27/experiments/jupiter_band_8b_raw_v1/jupiter_band_8b_raw_v1/trace_jobs
python3 /e/fscratch/reformo/lee27/mirrorgate.py $D   # clone → /testbed → DID PASSES EDIT A FILE → band
python3 /e/fscratch/reformo/lee27/band.py     $D 0   # full gate ladder + throughput projection
```

- **PASS** → the environment finally measures the model. Scale to all 4,568 and compare to her ~36%.
- **FAIL, rewards all null** → suspect `test.sh`'s `uv pip install chardet` under `set -e` (§4.1).
- **FAIL, rewards are 0/1 but no variance** → there is a **fourth** free-pass mechanism. Do not tune. Find it.

Cancelled deliberately: `1248713` (had proved the constant reward), `1250164` (tested the now-obsolete
clone-mirror workaround), `1247578` (TP=4, never brought up an engine).

## 3. Tooling built this session — use it, don't re-derive it

| tool | what it enforces |
|---|---|
| `hpc/skyrl_standard/jupiter/jup.sh` | `source` it. `jup`/`jrc` serialise every cluster call through a lockfile and force `BatchMode=yes`; `jup_bg`/`jrc_bg` put long jobs in `tmux` **on** the cluster. Three SSH rules were documented twice each and violated three times on 08-05. |
| `hpc/skyrl_standard/jupiter/arm_rollout_forward.sh` | records every reverse-forward it installs in `~/.rollout_forwards` so the next cancel uses the **exact** spec; refuses to start if the port is bound by something unaccounted for. |
| `hpc/skyrl_standard/jupiter/band_preflight.sh` | hard pre-submit gate: TP=1, derived context fields, dead keys, `n_samples>=2`, sane `max_turns`, ControlMaster, **both** forwards, fleet outlives walltime+30min, bridge capacity. Non-zero exit = do not submit. |
| `band.py` / `band_report.py` | the gate ladder + band. **Never eyeball a band.** |
| `mirrorgate.py` | clone → `/testbed` populated → **did passing trials edit a file** → band. |
| `ai_memory/DEAD_KEYS.md` | the 7 accepted-but-ignored config keys. A key is not set until a **log line** proves the consumer acted on it. |

## 4. Known risks, unmeasured

1. **`uv pip install chardet` under `set -e`** in her `test.sh`. She baked chardet into the SIFs so it succeeds
   offline; if the SIFs we symlinked predate that rebuild, every trial aborts *before* grading. Shows as
   **null** rewards, not a zero band — distinguishable, so do not pre-emptively patch it.
2. **We depend on another user's read-only directory.** Stable since March, but if this becomes the main pool,
   copy the 4,568 SIFs into our scratch (~3.7 TB against 36 TB free on `/p/scratch`).
3. **Sandbox sizing** is 2 CPU / 4 GB / 8 GB, raised from her 1/1/1 for the abandoned clone-based workaround.
   Nothing clones or compiles now, so it can likely come back down — that is free throughput.
4. **10 tasks have no SIF** — already excluded from the name lists.
5. **Throughput unmeasured on the raw path.** On the patched path: 10.6 min/trial median, 5.97 trials/min at
   concurrency 91 ⇒ ~37h for a full pass. Peak concurrency 91 against **1,024** free fleet slots, so the cap is
   ours: raising `n_concurrent_trials` toward ~512 projects ~6.5h. Trials should now be faster (no clone, no
   build). **Do not tune throughput until the §2 gate passes.**

## 5. Retractions — do not re-derive these

- **"358 of 4,578 ≈ 8%" is WRONG.** Marianna, asked directly: the band is **~1.6k of 4.5k ≈ 36%**, and her
  filtering pass cost `18k rollouts` (= 4.5k × 4, confirming pass@4 over a 4.5k pool). Where 358 came from is
  unknown. This flipped a verdict once: 0-in-band reads as "consistent with 8%" (p≈0.47) but is ~1-in-800,000
  against 36%.
- **"33% pass ≈ Marianna's 35%, so a learnable band EXISTS"** — compared GROUPS against TRIALS. A pass rate is
  not a band.
- **"Concurrency is the bottleneck ⇒ a ~6× bump with no new nodes"** — we hit the configured cap instantly at
  32/64/128. What actually made trials slow: `max_turns: 999999` (she caps 50) plus heavy sandboxes.
- **"`1244916` died to a node failure"** — `sacct`: `CANCELLED by 34902`. Always `sacct -j <id> -X` before
  writing any cause of death.
- **"`main_tbench_generate` is broken"** — that smoke was TP=4, a *separate* confirmed bug (§6). The
  pure-rollout entrypoint was probably convicted for TP=4's crime and is worth re-testing at TP=1: no policy,
  no ref, no optimizer, no FSDP, no weight sync, and a single `generate()` over all prompts instead of
  `ceil(N/train_batch_size)` sequential steps.
- **"Our JuDoor key is rejected / go fix JuDoor"** — false, and it was told to Luke as an action item. A
  55-minute held ssh session was consuming the connection slot; access returned the moment it closed. Two
  correlated symptoms, one confounder, no waiting.

## 6. Hard-won operational rules

- **TP=1 ONLY.** `inference_engine_tensor_parallel_size > 1` makes vLLM build a distributed executor that
  copies the parent actor's `os.environ` into a nested Ray `runtime_env`; Ray then asserts on its own
  `__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR` and **no engine ever starts**. SkyRL mislabels it
  `port collision (EADDRINUSE)` and burns 5×120s on a deterministic assertion. Gate on
  `grep -c RAY_WORKER_PROCESS_SETUP_HOOK <joblog>`. Cost: the entire 8-node allocation of `1247578`. TP=1 is
  also simply better for an 8B on 96 GB GH200 — 4 engines/node instead of 1, no collectives.
- **Route-gate at `max_tokens=4096`**, the real per-turn budget. At 256 the 8B loops `<think>` blocks and
  returns `finish_reason=length` with zero `tool_calls`, which reads exactly like a dead parser. A too-small
  gate manufactures the failure it is testing for.
- **Never probe the endpoint during weight sync** — params sit on the meta device and an inbound request kills
  the EngineCore and hangs the driver (cost: `1246702`). Wait for `Starting batch generation` or the first
  `trace_jobs/*/`.
- **Anchor your own grep patterns.** Three times a filter matched a substring of the config echo:
  `grad_norm=1.0` inside `max_grad_norm=1.0`, and `FileNotFoundError` inside
  `mask_exceptions=[...,"RewardFileNotFoundError",...]` (twice, in a monitor I wrote an hour after re-reading
  the entry warning about it). Exclude the echo line, or match on a line prefix.
- **`/e/fscratch` for anything a job touches.** `/e/scratch` has an inode cap shared by 26 users and has killed
  runs mid-rollout. `/p/scratch` is **not mounted on Jupiter compute** but IS visible from the Jupiter **login**
  node and from JURECA — the login node is the only place that sees both filesystems, which is why cross-cluster
  copies run there.
- **The JURECA ControlMaster lives on the Jupiter login node**, not on the laptop. Every forward command is
  double-hopped. `ssh jureca` from a Mac failing is expected, not an outage.
- **`ssh -O cancel -R` matches the FULL spec including the connect address.** A wildcard or guessed IP silently
  no-ops and returns success. Recover the real one with
  `sacct -j <id> -X -o NodeList%40 -n` → `scontrol show hostnames` → `getent hosts`.

## 7. Open items for Luke — he deferred these, do not chase

1. Ask Marianna for her **~1.6k band task IDs** + band-generation script, as a cross-check once our
   environment is proven. Her band code is likely in `/e/project1/jureap59/marianna/ot/dc-agent` (JSC-local,
   not on GitHub; `marianna13/dc-agent` is 404).
2. Decide whether to copy her SIF cache into our scratch for durability.
3. **Agent choice:** Luke chose **OpenCode**; her band used **terminus-structured**. Our absolute band % may
   legitimately differ. The pass/fail criterion is therefore "is there real within-group variance at all", not
   "does it equal 36%".

## 8. Launch recipe

```bash
source hpc/skyrl_standard/jupiter/jup.sh          # then use jup / jrc, never bare ssh

W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next ; D=$W
export BAND_MAX_TASKS=0 BAND_CONC=64 BAND_BRIDGE_PORT=9920 BAND_HARBOR_THINK=1
export BAND_MAX_EPISODES=50 BAND_CPUS=2 BAND_MEM_MB=4096 BAND_STORAGE_MB=8192
export BAND_MAX_MODEL_LEN=40960 BAND_MAX_NUM_SEQS=1024 BAND_GMU=0.92
export BAND_TOOL_PARSER=bare_json BAND_TOOL_PARSER_PLUGIN=$D/rl/tool_parsers/bare_json_tool_parser.py
# do NOT set BAND_TP / BAND_ENGINES -- TP=1 is mandatory (§6)
python3 $D/hpc/skyrl_yaml/jupiter/band/gen_band_yaml.py \
  $D/hpc/skyrl_yaml/jupiter/band \
  /e/fscratch/reformo/lee27/band_raw_32_names.txt \
  /e/fscratch/reformo/lee27/tasks/r2egym-raw \
  1 <NODES> 18300

bash hpc/skyrl_standard/jupiter/band_preflight.sh \
  $D/hpc/skyrl_yaml/jupiter/band/band_probe_8b_p4_shard00of01.yaml <WALL> <FLEET_JOBID>
# non-zero exit => DO NOT SUBMIT

source /e/scratch/reformo/lee27/keys/secrets.env   # NOT ~/secrets.env
export SHARD=band_probe_8b_p4_shard00of01 NUM_NODES=<N> TIME_LIMIT=<WALL>
export APPTAINER_BRIDGE_URL=http://10.128.1.2:9920 SKYRL_CHUNKED_LOGPROBS=1
export JOB_NAME=<name>
bash hpc/skyrl_standard/jupiter/run_r2egym_band_probe_8b.sh

# immediately after submit, arm the forward (it records its own IP for the next cancel):
jup "tmux new-session -d -s armfwd 'bash $W/hpc/skyrl_standard/jupiter/arm_rollout_forward.sh <JOBID> 18300'"
```

Keep `SKYRL_CHUNKED_LOGPROBS=1` — it fixes the large-vocab backward OOM (33.41 GiB peak vs stock OOM at the
identical 30.57 GiB, and 2.4× more accurate than the stock path vs fp64). Confirm via the log line
`chunked gathered log-softmax ACTIVE`, which only fires during a training step.

`TIME_LIMIT` ≤ 04:00:00 — a 6h job can be deferred a day by a hidden maintenance window. Note that
`scontrol update TimeLimit=` does **not** help a job whose reason is `Priority` rather than `Resources`
(measured: shrinking 3:00:00 → 1:30:00 moved the estimated start *later*).

## 9. Branch / repo map

| repo | branch | holds |
|---|---|---|
| `OpenThoughts-Agent` fork `lukedhlee` | `lukedhlee/vista-moe-grpo-30b` | `ai_memory/` — this file, `gotchas.md`, `DEAD_KEYS.md` |
| `OpenThoughts-Agent` worktree `~/ota-band` | `lukedhlee/band-8b-subset` | `band_preflight.sh`, `jup.sh`, `arm_rollout_forward.sh`, `gen_band_yaml.py`, `build_raw.py`, `fix_r2egym_checkout_to_parent.py` |
| `harbor` fork `lukedhlee` (local clone `~/harbor`) | `lukedhlee/apptainer-opencode-bridge` | `d64180d` offline git mirrors + `mirror_r2egym_repos.sh` — **inert on the raw path**, keep for reference |

`origin` on `OpenThoughts-Agent` is `open-thoughts/` and we have **no push rights** — push to `fork`.
