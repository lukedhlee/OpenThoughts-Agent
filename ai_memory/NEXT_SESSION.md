# NEXT SESSION — takeover note

Written **2026-08-06 ~02:40 KST**. Supersedes all earlier versions
(previous one archived at `ai_memory/logs/2026-08-06_NEXT_SESSION_superseded.md`).
Report times in **KST** (cluster clocks are CEST = KST − 7h).

---

## 0. Read this first: the finding has MOVED

Yesterday's headline was "the r2egym reward is a constant per task and never measures the model." That was
correct, and **it is now fixed and verified.** The reward measures code state. Do not re-litigate it.

**★★ THE GATE PASSED at 02:30 KST on 2026-08-06. A group now contains both a `0.0` and a `1.0`.**
This is the first genuine within-group reward variance in the project's history, and it is causally clean:

```
task r2egym-2514, two of four samples (task_name confirmed in BOTH result.json files)
  dCmGsKH  reward 1.0   "3 passed"                4x edit tool calls + read/ls/bash
  uGPuwvX  reward 0.0   "2 failed, 1 passed"      bash + glob only -- NO edits
```

The sample that **edited files** turned two failing tests into passing; the sample that did not, did not.
That is the exact **inverse** of the old pathology, in which passing trials edited *less* often than failing
ones (24% vs 49%). **The environment measures the model. Stop re-verifying this and go get a band number.**

⚠ **Be precise about which claim this supports, or you will repeat the pass-rate-vs-band error.**
The gate as posed — *"does any group contain both a `0.0` and a `1.0`?"* — is **MET**. But `band.py`'s
`IN BAND` counter only considers **fully-sampled** groups (≥4 trials), and `r2egym-2514` had 2 of 4 samples
in when this was observed, so `band.py` still reports `0 fully sampled` and cannot yet print a band %.
**Within-group variance is PROVEN; the band PERCENTAGE is not yet measured.** Those are different claims —
do not upgrade one into the other. The rest of the ladder passes cleanly:

```
[1] non-null reward : 7/7 -> PASS
[2] steps           : median 3  max 8   -> PASS (multi-turn)
[2b] TOOL CALLS     : median 2  max 11  | ZERO-tool trajectories 0 (0%) -> PASS
[3] trial duration  : median 3.4 min  p90 13.9 -> PASS (real work)
[4] groups: 6 seen, 0 fully sampled
```

**Throughput is the next real constraint, and it is severe:** 0.29 trials/min measured, peak concurrency 5
(completed trials only), ⇒ the full band of 13,312 trials projects to **~759 h** at this rate. That number is
understated-in-our-favour (the run lost its first 20 min) but it is the right order of magnitude to plan
against. Raising the edit rate (§0.3) and concurrency both feed this; scaling is now sanctioned, because the
gate that gated it has passed.

The remaining problem is a **rate** problem, not a correctness one: the agent succeeds too rarely
(1 of 7 trials made any edit), for two identified and fixable reasons — see §0.3. Those are now throughput /
quality levers on a working pipeline, no longer blockers.

### 0.1 The environment is VALIDATED — model-independently

All three documented free-pass mechanisms are refuted by direct measurement on Marianna's SIFs, **with no
model in the loop**. Do not redo these.

| check | method | result |
|---|---|---|
| `/testbed` populated? | `apptainer exec <sif> ls /testbed` | **yes** — real repo (Orange3) inside the SIF, no clone, no network |
| at the BUGGY state? | `rev-parse HEAD` vs `rev-parse <base_commit>^` | **yes, HEAD == `base_commit^` exactly** |
| do agent edits reach the tests? | `Orange.__file__` + `Orange3.egg-link` / `easy-install.pth` | **yes** — develop/editable install pointing at `/testbed` |
| `/testbed` writable by the agent? | write test on fleet node `jrc0553` | **`WRITE_OK`** (fakeroot is unavailable on JURECA compute and is NOT needed) |
| does `chardet` abort `test.sh`? | real SIF + writable overlay | **no** — `Audited 1 package`, exit 0. **§4.1 RETIRED** |

The third mechanism deserves emphasis because it was thought to doom 55% of the pool: `TESTS_DIR=/r2e_tests`
lives *outside* `/testbed`, but the **code under test** still resolves to `/testbed` through the egg-link. Tests
living outside the repo is fine — arguably desirable, since the agent cannot edit them.

### 0.2 The reward is now a real measurement — proven

Three completed trials on `1251403`, three different repos. Each ran a genuine pytest with **exactly one
failing test — the specific bug named in that task** — and everything else passing:

```
r2egym-0000  1 failed,  9 passed   reward=0.0
r2egym-2040  1 failed, 12 passed   reward=0.0     (scrapy HttpCompression)
r2egym-1742  1 failed, 13 passed   reward=0.0
```

Reward `0.0` is **correct**: the agent never fixed the bug. Flipping that one test to PASSED yields `1.0`.
This is the inverse of the old failure, where the reward was ~61% `1.0` no matter what the model did.

### 0.3 The rate problem: the agent succeeds too rarely (measured)

`agentstall.py` (`~/ota-band/hpc/skyrl_standard/jupiter/agentstall.py`, deployed at
`/e/fscratch/reformo/lee27/agentstall.py`) over the completed trials:

```
trials analysed                        : 7
with a raw-JSON tool call left AS TEXT : 5   (71%)
with a ripgrep tool error              : 7   (100%)
that made >=1 successful edit/write    : 1   (14%)  <-- and that one scored 1.0
ended with reason=stop                 : 7
median steps                           : 2
```

**The 1-in-7 edit rate is the number to move.** It is also roughly what a band needs: with pass@4, a
per-trial success probability anywhere in the mid range puts most groups in band, so lifting the edit rate is
the direct lever on the band %.

**Two independent causes, both concrete:**

1. **`rg` (ripgrep) is not in the sandbox.** OpenCode's `glob` *and* `grep` tools shell out to ripgrep, so the
   agent's two primary code-search tools fail on its very first action, in **100%** of trials
   (`"error":"ripgrep execution failed"`). The tool map in harbor
   `src/harbor/environments/apptainer/worker.py` (~line 91) is a **hardcoded dict** — `opencode`, `tmux`,
   `asciinema`, `uv` — and each entry is bound from `$BRIDGE_AGENT_TOOLS/bin/<name>` to
   `/usr/local/bin/<name>`. `$BRIDGE_AGENT_TOOLS` is `/p/scratch/synthlaion/lee27/agent_tools`, which
   contains exactly `asciinema grading-python311 opencode tmux uv`. Fix = stage a **static x86_64 musl**
   `rg` into `.../agent_tools/bin/rg` **and** add `"rg"` to that dict. Needs a **fleet restart** to take
   effect (workers are long-running), so it is not free.
2. **Thinking is ON, and the 8B then emits unparseable tool calls.** Verified by BEHAVIOUR from the trial
   `config.json`: `agent.kwargs.interleaved_thinking=True` **and**
   `agent.kwargs.extra_body.chat_template_kwargs.enable_thinking=True`.
   ⚠ **But `BAND_HARBOR_THINK` is NOT what sets the model's behaviour — it is INERT for OpenCode.** Those
   two keys are implemented for `terminus_2` / `openhands` / `mini_swe_agent` only. OpenCode never touches
   harbor's LLM client; it shells out (`opencode.py:876`) to
   `opencode --model=… run --format=json --thinking --auto`, where `--thinking` merely includes thinking
   blocks in the JSON output. Unknown kwargs are silently swallowed (`BaseAgent.__init__` takes `**kwargs`
   and never reads them), so the config *looks* applied and does nothing. `gen_band_yaml.py:80-84` says so
   in a comment. The model thinks anyway because Qwen3's chat template defaults `enable_thinking=true`.
   The parser is **not** broken — real `tool_use` events do appear. What happens is
   the model emits `<think>` prose plus **malformed JSON**: an unterminated string, two concatenated JSON
   objects, and a stray `</think>`. vLLM cannot parse that into `tool_calls`, so OpenCode records it as a
   `text` part and the step finishes `reason:"stop"` — the trial ends after ~2 steps with no edit.

   **Cause 2 is still the cheaper one, but use the RIGHT knob:** `BAND_SERVER_NO_THINK=1`, which sets
   `engine_init_kwargs.default_chat_template_kwargs={"enable_thinking": False}` (`gen_band_yaml.py:179-187`)
   and reaches every request regardless of agent. That is vLLM **engine-construction** time, so it needs a
   **new RL job** — but still **no fleet restart**. Then re-run `agentstall.py` and gate on behaviour
   (no `<think>` in trajectory text), never on the config echo.
   Unverified link in this chain: the MarinSkyRL `9904058` plumbing of `engine_init_kwargs` into vLLM.
   Task #1 was once marked "thinking-off (DONE)" — wrong, reopened — and then the reopened version told
   you to set `BAND_HARBOR_THINK=0`, which was **also wrong** (a no-op). Two errors on the same knob.

## 1. DO THIS FIRST — validate environments with NO model (operator's call, 08-06)

Luke's instruction: *"validate the environments first without running the models by actually submitting the
oracles and the non-oracle answers."* This is right, and it is far cheaper than a GPU allocation.

**The oracle is clean and free here:** `/testbed` sits at `base_commit^`, so `git checkout <base_commit> -- .`
**is** the gold patch. No `solve.sh`, no `solution/patched_files` needed (those belong to the patched dataset).

```bash
# on JURECA compute, via the existing fleet -- no new allocation, no GPU:
bash /p/scratch/synthlaion/lee27/envgate.sh <task> pristine   # expect reward 0.0
bash /p/scratch/synthlaion/lee27/envgate.sh <task> oracle     # expect reward 1.0
```

`envgate.sh` mirrors harbor's worker (instance start + `exec instance://`, writable ext3 overlay,
`/tests` + `/logs/verifier` + `/workspace` binds) and emits one JSON line per run. 32 repo-diverse tasks are
already staged at `/p/scratch/synthlaion/lee27/envgate/<task>/{test.sh,metadata.json}`
(`/e/fscratch` is **not** visible from JURECA — that staging step is mandatory).

**A pristine `0.0` and an oracle `1.0` on the same task is THE GATE, with no model involved.** Scale it across
tasks to get the solvable-task denominator a band number is meaningless without. Status at handover: the
single-task bracket for `r2egym-0000` was **still running** — re-run it, do not assume.

**Known trap inside `envgate.sh`:** plain `apptainer exec` on the SIF cannot `cd /root` — `chdir` returns
**EPERM** even though `/root` is owned by us and writes succeed. That is why the script uses
`instance start` + `exec instance://`, as harbor does. If you rewrite it, keep that.

## 2. Live state at handover (RE-PROBE, do not trust)

| id | what | state at 02:40 KST |
|---|---|---|
| `1251403` | 8B × 32 tasks × pass@4, 2 nodes, TP=1 | RUNNING, ends ~02:56 KST. 7 results, `{0.0: 6, 1.0: 1}`, **GATE PASSED** on `r2egym-2514` |
| `15500584` | JURECA sandbox fleet, 32 nodes × 16 workers | RUNNING, ~21h left, `SIF_CACHE` correct |

**`1251403` lost its entire first wave of 64 rollouts to a dead reverse forward** (see §3.1). The environment
and reward conclusions above are unaffected — they come from trials that ran *after* the repair, plus
model-free SIF probes.

The band from this job will be **small but non-zero** — it only had ~35 min of working route and a 1-in-7
edit rate. Do not read a low band % here as a defect; read it as the edit rate, which §0.3 tells you how to
raise. **Distinguish the failure modes by the verifier output:** constant `1.0` with no edits was the old
free-pass bug; `0.0` with exactly one failing test is a correct measurement of an idle agent; and a `1.0`
alongside 4 `edit` calls is the pipeline working.

## 3. What broke tonight, and what now prevents it

### 3.1 A dead rollout forward that every existing check called healthy
`1251403` produced nothing for 20 minutes. The job log simply **froze at 868 lines with no error**.

- `arm_rollout_forward.sh` logged a clean install, and `ss` on JURECA showed `10.14.0.46:18300` **LISTENING**.
- But `curl` from a **fleet compute node** returned **`HTTP=000` in 1.5 ms** — refused.
- Meanwhile the *bridge* forward (`9923`) was HTTP 200, which is why envs still started and the bridge looked
  perfectly healthy. **This is rule #1: one forward up, one down ⇒ healthy bridge, 100% rollout timeouts.**
- A `-O cancel` + `-O forward` of the **byte-identical spec** fixed it instantly.

`ssh -R` binds the listener whether or not the forwarded-to target is reachable, and a stale registration on
the same port is indistinguishable by `ss`. **Fixed structurally** (commit `003a7b94`):
`arm_rollout_forward.sh` now issues a real `/v1/chat/completions` **from a JURECA compute node** (via
`srun --overlap` against the fleet, arg 3) and **exits non-zero on `HTTP=000`**. Any HTTP status from the
server is a PASS — a `400 Model name mismatch: loaded model name g1_diverse_tezos_100k_8b` is the proof you
reached the real vLLM.

⚠ **Unexplained, recorded as observation not mechanism:** why an identical spec was dead and then worked.
Also `~/.rollout_forwards` **did not exist** even though the 18:25 arm log said "recorded in" it — so the
recorded-spec safety net was absent. Find out what removes it.

### 3.2 `jrc` ran HALF your command on the WRONG CLUSTER and exited 0
This one invalidates evidence, so read it. `jrc` was `jup "ssh ... \"\$@\"" _ "$@"`, and **`ssh` joins all
its arguments into ONE remote command string**. So `jrc 'hostname; uname -m'` became, on Jupiter:

```sh
ssh -S cm jureca05 "$@" _ hostname; uname -m
```

`_ hostname` ran on **JURECA** (printing `_: command not found`, eating the hostname) and `uname -m` ran on
**JUPITER**, printing `aarch64`. Any payload containing `;`, `&&`, `||`, `|` or a newline silently executed
part of itself on the wrong cluster **and still exited 0**. Fixed by base64-encoding the payload (`42bd89e9`).
**A one-word payload is the only case the broken version got right — so distrust any pre-08-06 cross-cluster
claim that was verified with a single-word `jrc` command.**

Same commit: `trap ... RETURN` is **bash-only**, and zsh rejects it (`undefined signal: RETURN`), so it
printed two noise lines per call, never installed, and left `$TMPDIR/.jup.lock.d` behind on any interrupted
call — making every later call block the full 300 s. Now an age-based stale-lock break, and `_jup_locked`
propagates the real exit code instead of `rmdir`'s.

### 3.3 The false "auth outage", third recurrence — and it was self-inflicted
`Session open refused by peer` then `lee27@login02...: Permission denied (keyboard-interactive)`.
**Cause: my own background pollers exhausted the ControlMaster's session cap** — including one inherited
monitor calling **raw `ssh jupiter`**, bypassing `jup.sh`'s lockfile entirely.

- **DO NOT kill the Jupiter master.** It was established interactively with TOTP and `ControlPersist 8h`;
  killing it risks needing Luke for a re-auth. That is exactly the documented false "JuDoor key is rejected".
- **DO** stop every background poller. Access came back on the **first retry**, in under a minute.
- **Run at most ONE cluster poller, and only through `jup`/`jrc`.** `BatchMode=yes` did its job here: the
  refusal failed cleanly instead of burning TOTP tries.

### 3.4 Marianna's SIFs are amd64 — probe them ONLY from JURECA
From a Jupiter login node you get
`FATAL: While checking container encryption: could not open image ...: the image's architecture (amd64) could
not run on the host's (arm64)`. The "container encryption" prefix reads like a corrupt SIF; it is only an
arch mismatch. Her SIFs derive from the prebuilt `namanjain12/<repo>_final` images, which are x86_64-only,
so the JURECA `dc-cpu` fleet is **the only place they can run** — not a preference. (The *patched* dataset's
`python:X.Y` base was multi-arch and did run on Jupiter aarch64; do not carry that assumption over.)

### 3.5 `mirrorgate.py`'s CLONE check is INVERTED on the raw path
It prints `[1] CLONE ... -> FAIL (clone still failing -- mirrors not bound?)` when `attempted=0`. On the raw
path **`attempted=0` is the CORRECT outcome** — nothing should clone, the repo is already in the SIF
(`testbedcheck.py` says so explicitly: "should be 0 now"). Ignore that line, or fix the gate. Likewise, at
tiny n its `[3]`/`[4]` verdicts print `FAIL` off `n/a` denominators — **`FAIL` at n=1 is not a gate reading.**

## 4. Open risks, unmeasured

1. **We depend on another user's read-only dir** (`/p/scratch/transfernetx/nezhurina1`). Stable since March;
   copy the 4,568 SIFs (~3.7 TB against 36 TB free) if this becomes the main pool.
2. **Sandbox sizing** is 2 CPU / 4 GB / 8 GB, raised for the abandoned clone workaround. Nothing clones or
   compiles now, so it can likely come down — free throughput, but **not before the §1 gate passes.**
3. **10 tasks have no SIF** — already excluded from the name lists.
4. **Throughput unmeasured on the raw path.** Trials that stop early finish in ~3 min, which tells you
   nothing about a working agent. Do not extrapolate from it.

## 5. Retractions — do not re-derive

- **"358 of 4,578 ≈ 8%" is WRONG.** Marianna: the band is **~1.6k of 4.5k ≈ 36%** at `0 < pass@4 < 1`, and
  her filtering pass cost `18k rollouts` (= 4.5k × 4). 0-in-band reads as "consistent with 8%" (p≈0.47) but
  is ~1-in-800,000 against 36%.
- **"33% pass ≈ her 35%, so a learnable band EXISTS"** — compared GROUPS against TRIALS. A pass rate is not a
  band. The band is per-group: `0 < passes < n_samples`.
- **"Concurrency is the bottleneck ⇒ ~6× with no new nodes"** — we hit the configured cap instantly at
  32/64/128. What made trials slow was `max_turns: 999999` (she caps 50) plus heavy sandboxes.
- **"`1244916` died to a node failure"** — `sacct`: `CANCELLED by 34902`. Always `sacct -j <id> -X` first.
- **"Our JuDoor key is rejected"** — false, three times now. See §3.3.
- **"The bare_json parser is validated"** — over-stated. It **does** fire, but "median 2 tool calls/trial" was
  never healthy for a SWE task; it is the same 2-step stall documented in §0.3.
- **"thinking-off (DONE)"** — false. It is ON. See §0.3.

## 6. Hard-won operational rules (unchanged, still binding)

- **TP=1 ONLY.** `inference_engine_tensor_parallel_size > 1` makes vLLM build a distributed executor that
  copies the parent actor's `os.environ` into a nested Ray `runtime_env`; Ray then asserts on its own
  `__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR` and **no engine starts**. SkyRL mislabels it
  `port collision (EADDRINUSE)` and burns 5×120 s. Gate on
  `grep -c RAY_WORKER_PROCESS_SETUP_HOOK <joblog>`. Cost: all 8 nodes of `1247578`.
- **Verify by BEHAVIOUR, never config.** Seven keys have been accepted, echoed and ignored
  (`ai_memory/DEAD_KEYS.md`); `strict_json_parser: False` is in the live config right now. A key is not set
  until a **log line or a trajectory** proves the consumer acted on it. Tonight's `interleaved_thinking=True`
  was caught this way.
- **Anchor your grep patterns.** The log echoes the whole hydra command line, so `FileNotFoundError` matches
  `RewardFileNotFoundError` inside `mask_exceptions=[...]`, and `grad_norm=1.0` matches `max_grad_norm=1.0`.
  Both produced phantom findings. Exclude the echo line or anchor on a line prefix.
- **Never probe the vLLM endpoint during weight sync** — params sit on the meta device; an inbound request
  kills an EngineCore and hangs the driver (cost: `1246702`). Wait for `Starting batch generation` or the
  first `trace_jobs/*/`. After that it is safe.
- **`/e/fscratch` for anything a job touches.** `/e/scratch` has an inode cap shared by 26 users.
  `/p/scratch` is **not mounted on Jupiter compute** but IS visible from the Jupiter **login** node and from
  JURECA — the login node is the only host that sees both, which is why every cross-cluster copy runs there.
- **The JURECA ControlMaster lives ON the Jupiter login node**, not the laptop. `ssh jureca` from a Mac
  failing is expected, not an outage.
- **`ssh -O cancel -R` matches the FULL spec including the connect address.** A wildcard or guessed IP
  silently no-ops and returns success.
- **Reward is at `verifier_result.rewards.reward`** — not top-level `reward`, which does not exist. What
  matters is per-group `len(set(rewards)) > 1`, never `avg_raw_reward`.
- **Do NOT rename a task dir or touch a Dockerfile** under `tasks/r2egym-raw` — SIFs are keyed
  `build_${task_name}-${sha256(Dockerfile):12}.sif`, so that breaks the free cache reuse (4,568/4,578 resolve).

## 7. Tooling — use it, don't re-derive it

| tool | what it does / enforces |
|---|---|
| `hpc/skyrl_standard/jupiter/jup.sh` | `source` it. `jup`/`jrc` serialise every cluster call through a lockfile and force `BatchMode=yes`. **Fixed 08-06** — see §3.2. `jup_bg`/`jrc_bg` detach into tmux ON the cluster. |
| `arm_rollout_forward.sh <job> <port> <fleet_jobid>` | arms the rollout forward and **verifies the route end-to-end from a compute node**. Pass arg 3 or it warns UNVERIFIED. |
| `band_preflight.sh <yaml> <wall> <fleet_jobid>` | hard pre-submit gate. Non-zero exit = DO NOT SUBMIT. **TODO: fold in the §3.1 end-to-end route probe.** |
| `/p/scratch/synthlaion/lee27/envgate.sh <task> <pristine\|oracle>` | **NEW** — model-free environment gate (§1). |
| `agentstall.py <trace_jobs>` | **NEW** — per-trial tools / ripgrep errors / edits / unparsed-JSON-as-text / last stop reason. This is how §0.3 was measured. |
| `band.py` / `band_report.py` | the gate ladder + band. **Never eyeball a band.** |
| `mirrorgate.py` | clone → `/testbed` → did passes edit a file → band. **Its CLONE verdict is inverted on the raw path (§3.5).** |
| `testbedcheck.py` | earliest signal: is `/testbed` non-empty, did anything try to clone. |

## 8. Launch recipe

```bash
source hpc/skyrl_standard/jupiter/jup.sh          # then use jup / jrc, never bare ssh

W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next ; D=$W
export BAND_MAX_TASKS=0 BAND_CONC=64 BAND_BRIDGE_PORT=9920
export BAND_SERVER_NO_THINK=1     # <-- THE knob that actually works for OpenCode (see §0.3).
export BAND_HARBOR_THINK=0        # <-- config honesty ONLY; this key is INERT for OpenCode.
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

# immediately after submit -- NOTE arg 3, which is what makes the route verified:
jup "tmux new-session -d -s armfwd 'bash $W/hpc/skyrl_standard/jupiter/arm_rollout_forward.sh <JOBID> 18300 15500584'"
# then CONFIRM: grep -E 'ROUTE OK|FATAL' ~/arm_forward_<JOBID>.log
```

Keep `SKYRL_CHUNKED_LOGPROBS=1` (fixes the large-vocab backward OOM; 33.41 GiB peak vs stock OOM at the
identical 30.57 GiB, and 2.4× more accurate vs fp64). Confirm via `chunked gathered log-softmax ACTIVE`,
which only fires during a training step.

`TIME_LIMIT` ≤ `04:00:00` — a 6 h job can be deferred a day by a hidden maintenance window. Note
`scontrol update TimeLimit=` does **not** help a job whose reason is `Priority` rather than `Resources`.

## 9. Open items for Luke

1. **Decide the order:** (a) `BAND_SERVER_NO_THINK=1` relaunch — needs a NEW job (vLLM engine-construction
   time) but no fleet restart; tests the
   likelier cause; or (b) stage `rg` + patch harbor's tool map — needs a **fleet restart**. §1's model-free
   oracle sweep is independent of both and should run regardless.
2. Ask Marianna for her **~1.6k band task IDs** + band script as a cross-check (likely
   `/e/project1/jureap59/marianna/ot/dc-agent`, JSC-local; `marianna13/dc-agent` is 404).
3. Whether to copy her SIF cache into our scratch for durability.
4. **Agent choice:** Luke chose OpenCode; her band used **terminus-structured**. Our absolute band % may
   legitimately differ, so the criterion is "is there real within-group variance at all", not "= 36%".
   ⚠ Worth revisiting now that OpenCode's `glob`/`grep` are known broken in 100% of trials.

## 10. Branch / repo map

| repo | branch | holds |
|---|---|---|
| `OpenThoughts-Agent` fork `lukedhlee` | `lukedhlee/vista-moe-grpo-30b` | `ai_memory/` — this file, `gotchas.md`, `DEAD_KEYS.md` |
| `OpenThoughts-Agent` worktree `~/ota-band` | `lukedhlee/band-8b-subset` | `jup.sh` (`42bd89e9`), `arm_rollout_forward.sh` (`003a7b94`), `band_preflight.sh`, `gen_band_yaml.py`, `build_raw.py` |
| `harbor` fork `lukedhlee` (clone `~/harbor`) | `lukedhlee/apptainer-opencode-bridge` | the apptainer worker. **The `rg` fix goes here** (~line 91). Offline git mirrors are **inert on the raw path**. |

`origin` on `OpenThoughts-Agent` is `open-thoughts/` and we have **no push rights** — push to `fork`.

---

# SWE-BENCH VERIFIED — staging facts settled 2026-08-06 (read before touching it)

## What exists now
`/Users/lukedhlee/swebench-verified-tasks` on the Mac: **500 task dirs, 0 failures, 21 MB total**,
generated with `cd ~/harbor/adapters/swebench && uv run swebench --all --task-dir <out>`.
Per task: `instruction.md`, `task.toml`, `tests/{test.sh,config.json}`, `solution/solve.sh`,
`environment/Dockerfile`. Small enough to rsync anywhere; regenerating takes ~8 min.

## Facts that change the plan
- **There is no "validation split".** `princeton-nlp/SWE-bench_Verified` ships **only `test`, 500
  instances**, and the dataset id is **hardcoded** at `adapters/swebench/.../adapter.py:46` — there is no
  `--dataset`/`--split` flag. "SWE-bench val" = those 500. Do not go looking for another split.
- **The oracle is free.** Each task has `solution/solve.sh` = `patch --fuzz=5 -p1` of the dataset `patch`.
  So the model-free gate is `nop` vs `oracle`, with **no git trickery** — unlike r2egym, where the gold fix
  is `git checkout <base_commit> -- .` because `/testbed` sits at `base_commit^`.
- **500 tasks -> 500 DISTINCT base images** (`swebench/sweb.eval.x86_64.<id>:latest`, `__`->`_1776_`),
  verified: `grep -h ^FROM */environment/Dockerfile | sort -u | wc -l` = 500. **No base-image reuse trick
  exists here** (contrast SweSmith: 2500 tasks -> 38 SIFs). Images are **x86_64-only**, so this can only run
  on the JURECA/JUWELS x86 fleet — never on Jupiter (GH200/aarch64).
- **`worker.py` already carries SWE-bench-specific shims** (make_test_spec's GitHub fetch monkey-patched to
  no-ops, Django `chdir` EPERM, Astropy DeprecationWarning). Somebody debugged this path before — read
  `_patch_test_sh_for_offline_pip()` before writing anything new.
- All three verifier wheels are already staged in `$BRIDGE_AGENT_TOOLS/wheels`:
  `swebench-4.0.3`, `datasets-2.16.1`, `fastcore-1.10.5`. The offline verifier path is provisioned.

## Do NOT use harbor's `prebuild_sifs.sh` unmodified for this
It pre-pulls every `FROM` into its own `base_*.sif` **and** builds a second SIF from a generated `.def` —
with 1:1 bases that is **~2x disk for zero reuse** (~1-2 TB, not ~1 TB). Its `.def` also has a `%post`
(apt-get, pip) needing root/`--fakeroot`, and it hides every build error behind `2>/dev/null`.
Its sbatch header is Marianna's JUWELS `projectnucleus` account, and its proxy block sources
`/e/project1/jureap59/marianna/...` (a **Jupiter** path — verify reachability from JURECA before relying on it).

`hpc/skyrl_standard/jupiter/swebench_build_sifs.sh` (committed) does a **pull-only** build instead: those
Dockerfiles only add `uv` and `/logs` on top of the base, and harbor already puts `uv` on PATH from
`agent_tools` and bind-mounts `/logs/verifier`. **This is a hypothesis — verify on a small sample before
committing to 500.**

## Unverified cost risks — measure on a sample of ~5-10 FIRST, then extrapolate and report
1. **Disk**: 500 x ~1-2 GB. Check the `/p/scratch/synthlaion` quota before starting.
2. **Docker Hub rate limits**: anonymous pulls are throttled per-IP. 500 pulls may need auth or serialising
   over many hours. This, not compute, is likely the wall.
3. **Whether pull-only SIFs actually run** (`uv` on PATH inside the container).
Build a sample, time it, measure the SIF size, run the gate on it — only then launch the 500.
