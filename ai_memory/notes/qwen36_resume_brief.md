# RESUME BRIEF — Qwen3.6-35B-A3B GRPO · written 2026-08-02

You are resuming an RL training bring-up. Read this whole file before running anything.

## LATEST LIVE UPDATE — 2026-08-03 14:10 CEST · ⛔ STOPPED, TWO DECISIONS NEEDED

Supersedes everything below. **Nothing is running on Jupiter (0 nodes).** JURECA workers
`15489111` still alive (20h01m of 24h ⇒ expire ~18:50 CEST). Bridge pristine: 3542/3542 jobs,
**zero errors**, all sandboxes reclaimed.

### ⛔ DECISION 1 — the learnable band is REQUIRED. This is now settled empirically.

**22 prompt groups measured across two runs. ZERO had within-group reward variance.**

| run | config | groups | with variance |
|---|---|---|---|
| `1219434` | 32 groups × **2** samples | 15 | **0** |
| `1221005` | 16 groups × **4** samples | 7 (4 complete) | **0** |

The four complete n=4 groups were perfectly uniform: `[1,1,1,1]`, `[1,1,1,1]`, `[0,0,0,0]`,
`[0,0,0,0]`. Rewards are real and honest — verifier-scored, `exception_info: null`, mixed ACROSS
tasks (solved vs unsolved) — but GRPO computes advantage **within** a group. All-zero groups are
dropped by `rloo_n_filter_zero_reward_groups`; all-one groups carry zero advantage. **No group can
contribute gradient.** If per-task pass probability were mid-range, four uniform n=4 groups has
probability ~0.02%.

**Qwen3.6-35B-A3B is essentially deterministic per r2egym task.** The pipeline is not the problem
any more; the DATA is. The co-lead's model-specific learnable band (`0 < pass@8 < 1` → 740 train +
100 held out) is **required, not optional**. This was PARKED by the operator on 2026-07-29 with
"forget the band for now, focus on enabling r2egym" — r2egym IS now enabled, and the parked risk has
materialised in our own measurements. **Operator decision needed; do not silently adopt the band.**
→ [[gotchas]], [[r2egym_apptainer_reference_impl]]

### ⛔ DECISION 2 — `/e/scratch` INODE pressure killed `1221005`; cleanup needs approval

`1221005` aborted at 14:03 with `OSError: [Errno 122] Disk quota exceeded: '/e/scratch/...'` across
many trials at once. Not a code defect. `jutil project dataquota -p reformo` (the sanctioned tool —
**never `find`/`du` on GPFS**):

| filesystem | data | **inodes** |
|---|---|---|
| `/e/scratch/reformo` | 79.9 TB / 214.7 TB (37%) | **6,136,227 / 8,000,000 soft (77%)** |
| `/e/project1/reformo` | 7.8 TB / 21.5 TB (36%) | **3,925,319 / 4,000,000 soft (98%)** |

Data is fine; this is **inodes**. Counters are stale (last updated 2026-07-24), so real usage is
higher after today. `/e/project1` at 98% of its inode soft limit is independently alarming —
the handoff notes hitting the project file-count cap **locks the project for ALL users**.

`/e/scratch/reformo/lee27/experiments` holds **103 entries**, including 17 dead
`jupiter_qwen36_35b_r2egym_grpo_smoke_*` trees plus today's three cancelled canary trees; each
trial directory is many small files. **Deleting them is a destructive action requiring explicit
approval, so nothing was deleted.** All evidence needed for the findings above has already been
extracted. Recommend approving deletion of the superseded `smoke_3..smoke_17` and `__dryrun_*`
trees, and adopting a post-run trace cleanup step.

### Fixes landed today — all pushed + deployed, no PRs

| commit | repo | defect it fixed |
|---|---|---|
| `4fb4a158` | OT-Agent | `compaction.reserved` is dead config → threshold via `client_window_tokens: 20480` |
| `179b31e9` | harbor | opencode never set `AgentContext.metadata` → extraction `TypeError` aborted every batch |
| `1f38665f` | harbor | observations must be **user** turns; `role:"tool"` rejected by `utils.py:1943` |
| `90109474` | OT-Agent | `NonZeroAgentExitCodeError` → `passthrough`; canary 32×2 → 16×4 |

**All four are CONFIRMED WORKING on `1221005`:** zero context overflows, zero
`Could not extract results`, zero `Expected message role`, zero non-zero-exit aborts. The pipeline
ran cleanly through Ray → 26/26 shards (7m38s) → JIT/KV → endpoint → policy init → 693-tensor weight
sync (301.9s) → 25 honest rollouts with real rewards. Only the disk quota stopped it.

### Unexplained timing regression — worth a look

`1221005` took **110 min** from shards-loaded to `init policy/ref/critic models done`, versus ~28 min
on `1219434` with identical geometry. That squeeze is why the run had only ~40 min of wall left when
rollouts began. Cause unknown; suspect GPFS metadata contention (consistent with the inode pressure).

### Still never executed with this model

**The optimizer update, checkpoint, and HF export.** A finite update remains the milestone. Also
open: the **token-fidelity limit** — opencode discards raw completion text, so reconstructed
`all_messages` is structurally faithful but approximate in tokens. Enable the literal recording proxy
(`literal.jsonl`) before trusting TIS importance ratios or promoting to 50 steps.

**Single point of failure:** Jupiter→JURECA ControlMaster (`~/.ssh/cm_jureca/qwen36`, pid 185890)
carries the reverse tunnel as a `-R` forward. If it dies, restoring it needs ONE interactive
`ssh jureca` + TOTP that only Luke can do.

## PRIOR LIVE UPDATE — 2026-08-03 11:35 CEST

Supersedes everything below. **Three pipeline layers were opened today; the blocker is no longer
infrastructure, it is the DATA.**

### ⚠ THE HEADLINE — raw r2egym gives Qwen3.6 NO GRPO GRADIENT

Job `1219434` produced 27 honest rollouts with real verifier rewards. Grouped by task:
**0 of 15 groups had ANY within-group reward variance** — every group was `[1,1]` or `[0,0]`
(5 groups all-solved, 8 all-unsolved, 2 singletons). GRPO computes advantage *within* a prompt
group: all-zero groups are dropped by `rloo_n_filter_zero_reward_groups`, all-one groups carry zero
advantage. **No group could contribute gradient**, regardless of how healthy the infrastructure is.
Cross-task spread is not a substitute. If per-task pass probability were mid-range (~0.7), all 13
observed pairs agreeing has probability ~0.05% — the near-determinism is real.

**This is the PARKED learnable-band decision (operator, 2026-07-29) meeting our own measurement.**
The co-lead reached the same conclusion independently. The canary is now 16 groups × 4 samples
(same 64-trajectory cost) to test it properly; **if 4 samples still show no within-group variance,
the model-specific band is REQUIRED, not optional.** That is an operator decision — surface it, do
not silently adopt the band. → [[gotchas]], [[r2egym_apptainer_reference_impl]]

### Fixes landed today (all pushed + deployed; no PRs opened)

| commit | repo | defect |
|---|---|---|
| `4fb4a158` | OT-Agent | `compaction.reserved` is dead config → move threshold via `client_window_tokens` |
| `179b31e9` | harbor | opencode never set `AgentContext.metadata` → SkyRL extraction `TypeError` aborted every batch |
| `1f38665f` | harbor | observations must be **user** turns; `role:"tool"` is rejected by `utils.py:1943` |
| `90109474` | OT-Agent | `NonZeroAgentExitCodeError` → `passthrough`; canary 32×2 → 16×4 |

Each fix exposed the next layer. Order encountered: context overflow → extraction `TypeError`
→ role `ValueError` + non-zero-exit abort → no within-group variance.

### Live state

Jupiter **`1221005`** submitted 11:32, 5 nodes, 3 h wall, PENDING. Predecessors `1218813`/`1219434`
cancelled by us after their defects were diagnosed; all orphaned sandboxes reclaimed via
`POST /env/stop {"env_id":...}`. Bridge pristine: 3033/3033 jobs, **zero errors**, `workers_alive:true`.
JURECA workers **`15489111`** expire **~18:50 CEST** — the binding clock. Combined 2/16 while pending.

**Single point of failure:** the Jupiter→JURECA ControlMaster (`~/.ssh/cm_jureca/qwen36`, pid 185890)
carries the reverse tunnel as a `-R` forward. If it dies the tunnel dies and rollouts lose their
endpoint; restoring it needs ONE interactive `ssh jureca` + TOTP, which only Luke can do.

### Still unproven past this point

The optimizer update, checkpoint, and HF export have **never executed** with this model. Reaching a
finite update remains the milestone. Also outstanding: the **token-fidelity limit** — opencode does
not retain raw completion text, so reconstructed `all_messages` is faithful in structure but
approximate in tokens. Fine for pipeline validation; **enable the literal recording proxy
(`literal.jsonl`) before trusting TIS importance ratios or promoting to 50 steps.**

## PRIOR LIVE UPDATE — 2026-08-03 08:00 CEST

This section supersedes every older live update below.

### The compaction blocker is CLOSED. Root cause: `compaction.reserved` is dead config.

The bounded-memory hedge worked: one-GPU diagnostic **`1217900`** (`MAX_JOBS=24`) loaded 26/26
shards in 11m40s and cleared the fused-MoE JIT in ~19 min with only ~1 concurrent compiler process
and 660 GiB of 857 free. The `1217866` memory blowup did NOT recur. `/health` 200 at 06:56;
HTTP 200 and a real `/v1/chat/completions` from actual JURECA compute node `jrc0545` through
`jrlogin05i:18000`, exact served revision `995ad96e…`, vLLM 0.22.0.

**Canary run 1 FAILED** (`CANARY_FAIL[CONTEXT]`, trial `r2egym-v1-00005__Bvnfhmh`) at the same exact
`28673 + 4096 = 32769` boundary — with `compaction.auto=true` AND `compaction.reserved=16384`
verified present in the materialized TrialConfig. So the allowlist trap was NOT the cause.

**Measured root cause: OpenCode 1.18.8 accepts `compaction.reserved`, Harbor deep-merges it, and it
is written to `opencode.json` — but OpenCode never reads it.** The proactive threshold stayed at its
default `limit.context - limit.output` = 27,648 − 4,096 = **23,552**. Observed prompt progression
was 8,774 → 19,892 → (overflow) → 2,951 → 7,815: request 2 at 19,892 sat *below* 23,552, so nothing
fired, and tool two's 11,056-token observation pushed the next request over the 28,672 ceiling.
This is the same class as the `strict_json_parser` bug: **a key accepted and written but silently
ignored by its consumer.** → [[gotchas]]

⚠ **The trap that nearly produced a false PASS.** OpenCode *recovered* by compacting REACTIVELY
(19,892 → 2,951, a 16,941-token drop). That looks exactly like success, and
`has_compaction_drop()` cannot distinguish it from proactive compaction. Had OpenCode exited 0, the
canary would have passed on a ceiling breach. **Never read a large prompt drop as a pass without
checking for `ContextOverflowError` first.**

**Fix = OpenThoughts `4fb4a158`** (`fix(rl): move OpenCode compaction threshold via client window`),
pushed to `lukedhlee/qwen3-6-r2egym-grpo` and deployed to the Jupiter `-next` checkout. It adds
optional `context_budget.client_window_tokens` (defaults to the served window) and advertises THAT
to Harbor's `model_info`, instead of `request_window_tokens`. At 20,480 Harbor derives
`limit.context = 20480 − 4096 − 1024 = 15,360` and OpenCode compacts at `context − output` =
**11,264**, leaving **17,408 tokens** of slack under the 28,672 ceiling — more than the largest
tool observation measured live (11,056). **The server contract is unchanged: 32,768 window /
28,672 prompt / 4,096 output.** The commit also rejects `reserved` in preflight and the canary
contract so the dead key cannot return, fails the canary on any `ContextOverflowError` in the event
stream, and asserts `threshold + largest measured tool jump < server prompt ceiling`.
49 focused tests pass; Ruff/format/YAML clean.

**Canary run 2 PASSED** (trial `r2egym-v1-00005__oAMqKmL`): prompt 8,774 → 20,024 → **2,981**
(proactive) → 7,702, **zero** `ContextOverflowError` occurrences, both 44,027-byte tool
observations sequential and untruncated, `CANARY_DONE`, exact identity, finite verifier reward
**0** with no exception. That reward is diagnostic only — it is NOT reward variance.

### Live successor: five-node job `1218813`

Submitted 07:5x CEST after an allocation-free preflight passed with the new contract materialized
(`OpenCode limit=15360+4096, proactive_compaction_at=11264`, exact model, vLLM 0.22, 3,328 tasks,
clean bridge, `workers_alive:true`). Allocated immediately on `jpbo-051-[17-18,20,23,25]`; head
`jpbo-051-17` / `10.128.33.241`; **3h wall to ~10:52 CEST**. Four policy/ref nodes + one rollout
node with four TP1 BF16 engines; 32 prompt groups × 2 samples = **64 trajectories**; one step,
checkpoint, HF export. Rendered sbatch verified: `export RL_ENV_DIR` line 107 before fallback 161,
zero `envs/rl/lib`, **zero AOT vars** (the pinned FlashInfer artifact is x86-64 and Jupiter GH200 is
aarch64 — `59e661e0`'s artifact must never be enabled on Jupiter). The private reverse listener was
retargeted from the dead `10.2.0.55:8000` to `10.128.33.241:8000`, internal bind only.

Diagnostic `1217900` was cancelled at 07:5x once its purpose was served. Combined allocation is
**7/16 nodes** (5 Jupiter + 2 JURECA). JURECA worker fleet **`15489111`** on `jrc[0545,0639]` is
healthy but **expires ~18:50 CEST** — that is the binding clock for any follow-on run. Bridge was
pristine at launch: 247/247 envs created/stopped, 2308/2308 jobs, **zero errors**.

Next gates, in order: 20/20 GPUs → 26/26 shards on four engines → fused-MoE JIT + KV cache → HTTP
endpoint → `init policy/ref/critic models done` → initial weight sync → 64 honest rollouts →
**real reward variance** → finite update → post-update sync → checkpoint → valid HF export.
The 6-node predecessor `_17` needed ~88 min from allocation to first rollout.

**Standing operator rule added 2026-08-03:** pushing to Luke's own repos/forks no longer needs a
per-turn ask; **opening a PR or merging still does.**

## PRIOR LIVE UPDATE — 2026-08-03 06:22 CEST

This section supersedes every older live update below. One-GPU exact-model diagnostic
**`1217866`** loaded all 26 checkpoint shards successfully in 805.82 seconds (65.52 GiB model
memory), proving the no-AOT BF16 model load. Its first real fused-MoE execution then launched an
unbounded Ninja build with about 300--350 compiler processes. Node memory became saturated; many
NVCC targets were killed with exit code 9, and at 06:20:24 Ninja failed. The log does not print a
literal kernel OOM line, but simultaneous code-9 kills at peak memory make compiler-memory
pressure the supported diagnosis. This is not a Qwen loader/model failure. Our invalid job was
cancelled at 06:21 after 40m31s; no endpoint, sandbox, rollout, reward, or update occurred.

The bounded-memory hedge is now primary: Jupiter job **`1217900`**, running since about 06:20 on
`jpbo-022-07` (`10.2.0.55`) with a two-hour wall. It is the same committed no-AOT, exact BF16,
Triton-GDN diagnostic, except `MAX_JOBS=24` limits FlashInfer's supported Ninja concurrency. The
private listener `10.14.0.46:18000` has been retargeted to `10.2.0.55:8000`; do not point it back
at the dead primary. When this backup's `/health` becomes 200, immediately prove health from
actual JURECA worker `jrc0545`, then run the local-only two-large-tool-output OpenCode/Harbor
canary described later in this brief. Require sequential tool observations, an observed >=4,096-
token compaction drop, `CANARY_DONE`, exact identity, and a finite verifier reward. This one reward
is diagnostic only and is not reward variance.

One bounded parallel comparator, Jupiter job **`1217881`**, started at about 05:50 CEST on
`jpbo-013-47`. It used the same committed one-GPU diagnostic payload but forced
`--safetensors-load-strategy prefetch` and leaves the GDN backend at its production-like default.
It was read-only with respect to training state and did not use the bridge. Forced prefetch spent
153 seconds populating page cache and still loaded shards at roughly 33--58 seconds each, versus
the primary's roughly 31-second average without prefetch. This is not an acceleration on the live
GPFS path. The default GDN path had already served successfully in `_17`, so the comparator's
remaining work was redundant; our own job was cancelled at 06:06 CEST after 15m40s. Do not use
forced prefetch for the successor.

The JURECA worker job remains **`15489111`** on `jrc[0545,0639]`; the ControlMaster is alive.
Bridge status is fully clean and healthy: no active/queued/ready/pending/starting/stopping work,
245 environments created and stopped, 2,284 jobs submitted and completed, zero bridge errors,
and `workers_alive:true`. Local-only OpenThoughts commit **`faba11d0`** remains clean and unpushed;
do not push it without an explicit ask in the same turn. A parallel read-only audit is preparing
the exact no-push five-node successor launch from cluster commit `59e661e0`, including skipping
the known-invalid x86 AOT hook and injecting `compaction.reserved=16384` at final materialization.
Combined allocation is currently 3/16 nodes: two JURECA workers plus diagnostic `1217900`.

The allocation-free five-node preflight and dry render then passed at about 05:55 CEST. The
unique reserved successor name is
`jupiter_qwen36_35b_r2egym_grpo_canary_compact16k_20260803a`; no real experiment directory or
job exists yet. The dry render is under the corresponding `__dryrun` experiment directory and
proves five nodes, a three-hour wall, last `RL_ENV_DIR` export above activation, no `envs/rl/lib`,
BF16, four TP1 engines, 32 groups x two samples, no colocation, one step/checkpoint/HF export, no
FP8, final reserve 16,384, and both context/nonzero-agent fail-fast exclusions. It still contains
the commit-59 x86 AOT variables, but no assignment to `FLASHINFER_AOT_MANUAL`; submit only with
`SBATCH_EXPORT=ALL` and `FLASHINFER_AOT_MANUAL=1`, then require runtime logs to show the normal
job-local `/tmp/flashinfer_lee27_<jobid>` JIT path and no AOT broadcast/activation. Do not submit
this successor before the one-node compaction canary is decisive and the bridge is reconfirmed
alive and idle.

## PRIOR LIVE UPDATE — 2026-08-03 05:39 CEST

This section **supersedes the 03:00 update immediately below**. Jupiter job `1215394` was
cancelled by Luke's agent at about 03:07 CEST after 11 deterministic context-window rejections,
with zero completed generation batches, zero rewards, and zero optimizer updates. This was our
own invalid six-node job; the healthy two-node JURECA worker fleet `15489111` was deliberately
left running for the successor. The combined allocation is draining from 8 nodes to 2.

The attempted one-token fix in OpenThoughts commit `668bf7da` was deployed correctly but its
semantic assumption was wrong. Every generated trial config did advertise
`model_info.max_input_tokens=28671` and `max_output_tokens=4096`, yet OpenCode still grew the
conversation until vLLM rejected the same 28,673 + 4,096 = 32,769 request. Bridge status remained
healthy (`workers_alive:true`, `jobs_errors:0`) and recycled the failed environments; this is not
a model-quality or JURECA-connectivity result.

Primary Harbor source at exact deployed commit `d93ca6396ea53a7213ade4320c6ba2fa1f80e158`
explains the failure. `src/harbor/agents/installed/opencode.py::_resolve_model_limit` treats
`model_info.max_input_tokens` as the model's **total context window**, subtracts output tokens and
a safety margin, and writes that result to OpenCode's context limit. More importantly, this run
explicitly set `opencode_config.compaction.auto=false`, so OpenCode did not use that limit to
compact a growing agent history. Lowering the advertised value by one therefore could not bound
the actual request. Do not relaunch `668bf7da` unchanged.

The successor fix must restore the correct Harbor/OpenCode model-info semantics (advertise the
32,768-token total window while retaining SkyRL/vLLM's 28,672 prompt and 4,096 completion
allowances) and enable OpenCode automatic compaction for this pipeline-validation smoke. Harbor's
existing calculation then gives OpenCode a conservative 27,648-token context limit, leaving its
1,024-token safety margin below the server prompt ceiling. Record this as smoke-only behavior;
the milestone remains honest rollouts, real nonzero reward variance, a finite update, post-update
weight sync, and valid HF export. The code change, tests, push/deploy, preflight, and one successor
submission are the immediate next actions.

### Corrected successor submitted — 03:14 CEST

The successor fix is OpenThoughts commit `0f04b25031109812d5fbfa429480c01ad1d2a5e0`
(`fix(rl): enforce OpenCode rollout context limit`), pushed to Luke's
`lukedhlee/qwen3-6-r2egym-grpo` branch and fetched/hard-reset onto the clean Jupiter checkout.
It removes the invalid one-token client-overhead abstraction, materializes Harbor
`model_info.max_input_tokens` from the total 32,768-token request window, retains SkyRL's
28,672-token prompt and 4,096-token output limits, and enables OpenCode automatic compaction for
this pipeline smoke. Focused contract/environment tests pass (15/15); all four context-budget
configs parse; compile, Ruff, and diff checks pass.

The live runtime preflight passed against the exact pinned model, compiled vLLM 0.22, all 3,328
tasks, and `workers_alive:true`. Dry-run `_dryrun_6` proves the frozen contract: Harbor window
32,768, output 4,096, compaction true; trainer/generator prompt 28,672; vLLM window 32,768. Its
rendered sbatch has `export RL_ENV_DIR` at line 107 before fallback line 161 and activation line
171, no `envs/rl/lib`, six nodes, and a three-hour wall.

Cancellation had orphaned 31 ready sandboxes and seven bridge commands. All 31 environment IDs
were recovered from this run's bounded trace tree and explicitly stopped through the bridge.
Before relaunch, the dedicated fleet reported ready/pending/starting/stopping all zero,
`active_jobs:0`, all 1,730 bridge jobs completed, zero bridge errors, and workers alive.

Exactly one corrected successor, Jupiter job **`1216594`**, was submitted at about 03:14 CEST into
experiment `_17` with a three-hour wall. Continue with this job only. Once allocated, resolve the
head IP, retarget the existing private JURECA reverse listener to its port 8000, and repeat the
real compute-node OpenCode tool gate. Then require a completed rollout with no transport/context
errors, nonzero raw reward variance, finite update, post-update sync, checkpoint, and valid HF
export. The JURECA worker job remains `15489111`; combined allocation will be 8/16 nodes when the
six-node successor runs.

`1216594` allocated immediately at **03:14:25 CEST** on
`jpbo-004-[33-34,37,40,47-48]`; scheduled end is 06:14:25. Its head is `jpbo-004-33` /
`10.128.16.177`. The live ControlMaster was reused to cancel the old reverse mapping and create
`10.14.0.46:18000 -> 10.128.16.177:8000`; JURECA `jrlogin05` confirms the internal-only listener.
An HTTP reset is expected until vLLM starts. The first batch log lines prove activation of the
correct `rl-megatron` Python/Ray executables, and Ray startup has begun. Combined allocation is
now exactly 8/16 nodes.

At 03:19 CEST all six Ray nodes and exactly 24/24 GPUs reported ready. SkyRL launched with the
expected exact command, including Harbor total window 32,768, output 4,096, compaction true;
trainer/generator prompt 28,672; eight BF16 TP1 inference engines; four samples per prompt; and
one training step. No startup, OOM, loader, or transport error is present. The run is proceeding
into model initialization.

The operator added a standing efficiency rule: optimize bring-up for information per node-minute.
Use the full six-node job as the final integration gate; convert each discovered boundary into a
cheap deterministic preflight or a one-engine exact-model agent canary. Promote future retries
through static/runtime preflight -> one-node cache/runtime gate -> five-node 64-trajectory canary
-> six-node final integration gate, stopping at the cheapest stage that disproves the hypothesis.
Run independent read-only/local work in parallel with a live allocation, but do not modify the
active cluster checkout underneath `1216594`.

The fast-preflight track is complete and pushed as OpenThoughts commit `f57ad641`
(`feat(rl): fail fast on Qwen smoke contracts`) on Luke's branch, but is deliberately **not
deployed/reset onto the checkout under the active job**. It adds an allocation-free launcher gate
that requires literal `workers_alive:true`, an idle dedicated bridge by default (opt-out only for
intentional sharing), pinned OpenCode 1.18.8 with compaction enabled, and the exact materialized
32,768/4,096 -> 27,648/4,096 Harbor limit plus coherent trainer/generator/vLLM limits. The main
agent reran the expanded focused suite: 24 passed; Ruff, bash syntax, compile, and diff checks pass.
Deploy `f57ad641` only after `1216594` ends or before a later launch.

The two read-only acceleration analyses are also decisive:

- Current `train_batch_size=32` counts prompt groups, so with four samples the job needs **128
  trajectories**, not 32; the 32 trace dirs previously observed were only the first Harbor
  concurrency wave. The hard valid minimum under FSDP4xEP4 fully-async GRPO is 16 groups x 2
  samples = 32 trajectories. A safer reduced canary is 32 x 2 = 64 trajectories with four TP1
  engines on one rollout node, for five Jupiter nodes total. It keeps more task diversity and has
  materially better odds of mixed binary rewards than 16 x 2. This is an intermediate canary;
  the established six-node topology remains the final integration gate.
- Every retry is forced to cold-compile because `triton_cache.sh` keys
  `FLASHINFER_WORKSPACE_BASE` by Slurm job ID. Historical `_16` timing attributes about 15.5
  minutes between weight load and KV readiness to fused-MoE JIT/profile, with only about five
  seconds of true autotuning. Do not use a shared writable GPFS JIT directory. The safe route is
  the official stack-matched `flashinfer-jit-cache` AOT wheel, extract only its three required
  `fused_moe_90` files into one checksummed ~50 MB archive, `sbcast` it to node-local storage, and
  fail-loud verify version/path/SO hash/AOT selection before Ray. This should save roughly 13--15
  minutes per retry. Do not implement or deploy it underneath active `1216594`.

The reduced intermediate profile is now implemented and pushed as OpenThoughts commit `6760caf7`
(`feat(rl): add reduced Qwen pipeline canary`), again **not deployed onto the active checkout**.
It adds a separate five-node launcher/config with four unchanged policy/ref nodes, one four-engine
rollout node, 32 prompt groups x two samples = 64 trajectories, and all existing exact-model,
BF16/no-FP8, compaction/context, sandbox, non-colocation, one-step, checkpoint, and export
contracts. It does not modify or replace the established six-node final config. Main-agent
verification passed 28 focused/environment tests plus Ruff, shell syntax, compile, and diff checks.

The successor-only AOT staging hook is implemented and pushed as OpenThoughts commit `87a56a23`
(`feat(rl): stage FlashInfer AOT cache node-locally`), also **not deployed or enabled for active
`1216594`**. It is an exact no-op unless all four archive/hash/key variables are set. When enabled,
it source-hashes one selective three-file `.tar.gz`, broadcasts it once with `sbcast`, rejects
traversal or any extra archive members, extracts and re-hashes on every node, verifies exact
FlashInfer version and `fused_moe_90` AOT path, prepends only the verified node-local root, and
keeps a separate per-job `/tmp` JIT fallback. No shared writable GPFS JIT is introduced. The
artifact is now downloaded and staged as described below. Main-agent verification passed 35 focused tests,
Ruff, shell syntax, and diff checks.

The real selective artifact is now prepared and verified locally and on Jupiter scratch. It
contains exactly the two tiny Python metadata files plus `fused_moe_90.so`, is 49,007,410 bytes,
and has archive SHA-256
`35271d0fbdb42dcb02003c10b92e092ef046b575f5122ceef70b76bff76bf22e`; the SO hash is
`976678d1c03a35358a165eb8ba9353bf1652ab297f07a1a763230f9493cadcd3`. The source official
wheel's full SHA was verified before extraction, and the 1.88 GB wheel was then deleted. Jupiter
path is `/e/scratch/reformo/lee27/cache/flashinfer-aot/fi0.6.11.post2-cu130-x86_64-manylinux2_28-sm90a-vllm0.22.0-torch2.11.0-py312/`.
OpenThoughts commit `59e661e0` enables the hash-pinned artifact in both Qwen base and reduced
canary configs, but remains undeployed while `_17` runs. A fresh successor is required to use it.

Because the current config actually requires 128 trajectories, the agent checked whether its
three-hour wall could be safely extended in place. Booster reports `MaxTime=UNLIMITED`, but Slurm
returned `Access/permission denied for job 1216594`; no job field changed and end time remains
06:14:25 CEST. Do not treat that command as a job failure. Leave healthy `_17` running and watch
the rollout-wave timing; the AOT/reduced successor exists if `_17` reaches wall before export.

Live `_17` update: by about **03:49 CEST** all eight exact BF16 vLLM replicas had completed all
26/26 checkpoint shards (roughly 15.5 minutes per replica group). There is no quantization, FP8,
loader, OOM, or NCCL failure. The run is now in the known fused-MoE JIT/KV-profile phase before
the HTTP endpoint; bridge remains clean/alive and no rollout is expected yet.

A read-only compute-node audit at 03:58 CEST proves this phase is active rather than hung: 146
`nvcc`/`cicc`/`cc1plus`/`cudafe++` processes were consuming near-full CPU while compiling
FlashInfer `fused_moe_90` for SM90a, cache mtimes and object counts were advancing, and all 24
GPUs retained the loaded model at roughly 68.4--68.8 GiB each. No fused-MoE `.so` or HTTP listener
existed yet. No traceback, CUDA OOM, NCCL failure, engine death, signal, or unexpected Ray exit was
found; the Ray metrics-exporter/object-spilling and c10d IPv6 messages remain nonfatal warnings.
The next expected markers are the linked `.so`, KV-cache initialization, and the HTTP endpoint.

The separate one-node artifact gate, Jupiter job **`1216854`**, completed successfully in **90
seconds** with exit code `0:0`. On a fresh node it broadcast the pinned 49 MB archive, reverified
the exact FlashInfer `0.6.11.post2+cu130` AOT selection and `fused_moe_90.so`, established the
separate writable JIT fallback, printed `AOT_SMOKE_OK`, and exited cleanly. This validates the
successor-only staging path without loading the model or disturbing `_17`; combined usage returned
to 8/16 nodes. The expected future cold-start saving remains roughly 13--15 minutes per retry.

At about 04:03 CEST the new successor canary launcher also passed its allocation-free runtime
preflight from the separate clean `-next` worktree at `59e661e0`, with intentional
`REQUIRE_CLEAN_BRIDGE=0` because `_17` owns the live fleet. It verified literal
`workers_alive:true`, OpenCode limit 27,648+4,096 with compaction, Torch 2.11.0+cu128,
Transformers 5.8.1, vLLM 0.22.0, the exact Qwen revision/model type, and all 3,328 tasks. It
explicitly submitted no job. Repeat with `REQUIRE_CLEAN_BRIDGE=1` before any successor launch.

Live `_17` then cleared the cold-start boundary: all FlashInfer compiles linked, the tiny runtime
autotune finished, all eight engines initialized, and the SkyRL HTTP endpoint started at
**04:04:11 CEST** with exactly one training step and all 3,328 tasks. As expected, `/v1/models`
returns application-level 404 while `/health` is supported. From actual JURECA compute node
`jrc0545`, `/health` returned HTTP 200 and a real `/v1/chat/completions` request crossed
`jrlogin05i:18000`, returned HTTP 200 from exact served model
`995ad96eacd98c81ed38be0c5b274b04031597b0`, and reported vLLM 0.22.0. This proves the current
private cross-site inference route before policy initialization. Next require the full OpenCode
tool-use gate, 16-rank policy load/conversion, initial weight sync, and honest rollouts.

At 04:07 CEST SkyRL began initializing the 16-rank policy process group. By 04:10 the aggregate
log showed policy ranks completing all 693/693 weights in about 27 seconds and entering the
grouped-GEMM MoE swap with no failure. The bridge remains intentionally idle and pristine
(`active_jobs:0`, all 1,730 historical jobs completed, zero errors) until policy initialization
and the initial weight sync finish.

The required real OpenCode gate then passed from existing worker allocation `15489111` on actual
compute node `jrc0545`, with **zero new nodes and zero bridge mutations**. Exact OpenCode 1.18.8
used routed endpoint `jrlogin05i:18000/v1`, exact served revision, context 27,648/output 4,096,
and compaction enabled. In about 12 seconds Qwen emitted a parsed bash tool call, OpenCode executed
it, the observation contained `TOOL_GATE_OK`, and the run produced seven events, two normal step
finishes, one matched tool, and zero errors. Direct OpenCode must receive `</dev/null` when invoked
inside a streamed shell script or it consumes the rest of that script; the first two diagnostic
attempts exposed only that invocation gotcha and did not mutate the bridge. This clears the live
provider/buffered-SSE/parser/tool path. Sandbox creation and the verifier remain the next boundary.

The reusable zero-additional-node sandbox/verifier gate is now committed **locally only** in the
clean ground-truth clone as OpenThoughts `5ed939c71ad306de458fd4e3b5eb0bf8c162c53a`
(`feat(rl): add live Qwen agent verifier canary`) atop `59e661e0`; it is not pushed or deployed,
because this turn did not authorize a push. Its launcher runs one exact Harbor r2egym Apptainer
trial against an already-live endpoint, appends a short marker-producing bash instruction, then
requires exact model/OpenCode identity, observed marker tool output, and a finite real verifier
reward (zero is allowed and explicitly not reward variance). It defaults to refusing a non-idle
bridge. Verification passed 47 focused/adjacent tests plus Ruff, format, py_compile, bash syntax,
and diff checks. After an authorized push/deploy, run
`bash hpc/skyrl_standard/jupiter/run_qwen36_live_canary.sh` from the active Jupiter checkout before
the multi-node integration gate.

Policy-side progress remains healthy. All 16 ranks formed the process group and loaded 693/693
tensors; four mesh-coordinate-zero leaders performed the real grouped-MoE conversion while the
other 12 waited. Leader RSS/faults advanced continuously from roughly 48--57 GiB to 134--151 GiB,
and all four reached FSDP wrapping by **04:29:44 CEST** with no failure. Global rank 0 then began
the known full CPU state snapshot and temporarily approached about 217 GiB RSS, still within node
memory. Do not call this hung or cancel it. A read-only source/timing audit found two future
policy-init A/B candidates without changing FSDP4xEP4/grouped-GEMM/BF16: construct grouped-MoE
destinations on meta then `to_empty` in BF16 before exact copy, and avoid the redundant
`detach().to(cpu, copy=True)` full-state snapshot. These are not deployed and require tensor-hash
equivalence tests before use.

The explicit policy milestone landed at **04:32:13 CEST**:
`init policy/ref/critic models done`. Process-group formation took 2m13s and the full PG-ready to
models-done interval was 22m33s; more than 70% was CPU grouped construction/remap, while the
post-conversion snapshot plus EP/FSDP streamed load, AdamW, and extractor took only about 2m29s.
There was no traceback, OOM, actor failure, or collective failure. The next gates are rollout
coordinators, initial policy-to-eight-engine weight sync, and the first 32 real sandboxes.

All four rollout coordinators started successfully between 04:33:22 and 04:36:22 with eight trial
slots each and the correct external `jrlogin05i:18000` endpoint. The nine-party NCCL weight-sync
communicator (policy rank 0 plus eight exact rollout engines) initialized in 34.37s, all engines
registered the CP>1 norm fake kernels, and the explicit initial
`sync_weights_to_inference_engines` began at **04:36:56 CEST**. No bridge jobs have launched yet;
that is correct until this initial 693-tensor transfer completes.

The initial sync completed successfully at **04:42:05 CEST** in 309.46s. The first honest
automatic rollout wave then launched: 32/32 trace directories, 32 live JURECA sandboxes, and all
32 bridge jobs active with `workers_alive:true` and zero bridge errors. Every rendered trial
config independently carries exact `vllm/995ad96e...`, routed `jrlogin05i:18000/v1`, Harbor
32,768/4,096 model info, compaction true, and bridge `10.128.1.2:9920`. Both rollout nodes show
real inference load (roughly 16--91% GPU utilization), so this is genuine agent generation.

One request at 04:43:23 still hit the exact 28,673+4,096=32,769 vLLM boundary. It is **not yet a
batch-wide regression**: as of 04:45:30 the aggregate count remains one, one sandbox stopped,
31 remain active, and the rollout GPUs are busy. Keep `_17` alive while the other agents run.
Treat the single oversized task/sample as invalid and require enough completed groups plus real
reward variance; cancel only if the overflow spreads or prevents the batch from filling. No
network-unreachable, connection, SQLite, undefined-provider, traceback, or bridge error is present.

The first honest retained rollout on this exact model landed at **04:48:05 CEST**:
`r2egym-v1-01755__hDvZDwx/result.json`. It used OpenCode 1.18.8 and exact served revision
`995ad96e...`, executed real sandbox tools, finished with `exception_info:null`, and received a
finite real verifier reward of **1.0** after 5m22s of agent execution. This is the first observed
model rollout and verifier reward, clearing the prior “no rollout ever observed” blocker. It is
not yet reward variance or an optimizer update.

The exact one-token overflow did spread to seven request occurrences by 04:47:10, but this did
not prevent the successful rollout above; the bridge remained `workers_alive:true` with zero
bridge job errors and roughly 30--31 agents still active. Continue `_17` while successful results
accumulate. The next decisive gate is a retained reward 0 alongside reward 1 (nonzero variance),
then enough completed prompt groups to enter the finite update. Do not infer model quality from
the first reward or cancel solely because some long conversations hit the server boundary.

### `_17` cancelled after retry storm; large-tool-output compaction gap identified

The “continue” instruction immediately above is now superseded. By 04:53 CEST the exact overflow
had repeated 13 times, no complete four-sample prompt group had entered the generation buffer,
and only seven of 32 bridge jobs were doing work while the other trial slots were held across
Harbor's 60/120/240-second retry backoff. Five honest, exception-free trajectories did complete,
all with real verifier reward 1.0, but they formed no complete GRPO group and therefore proved
neither within-group reward variance nor an update. There were no network errors and the bridge
still reported zero job errors. Luke's agent cancelled our own invalid Jupiter job `1216594` at
**04:54:01 CEST**. The resulting Ray `NodeDiedError` and Slurm SIGTERM are consequences of that
intentional cancellation, not OOM or hardware failure.

The exact OpenCode 1.18.8 source plus retained live trajectories reveal the real context bug.
OpenCode's proactive compaction decision uses the previous assistant request's usage before the
next tool result is appended. With the current defaults it compacts at about 23,552 tokens, only
5,120 tokens below vLLM's 28,672 prompt ceiling. Successful live trajectories contained a
single-turn prompt jump as large as 11,815 tokens from one tool result, so `compaction.auto=true`
alone cannot enforce the hard request boundary. This explains why the corrected total-window
semantics were necessary but insufficient.

The lowest-risk smoke-only successor contract is now committed locally in the clean ground-truth
clone as **`faba11d0`** (`fix(rl): harden Qwen agent compaction canary`), not pushed or deployed:
set `opencode_config.compaction.reserved: 16384` while keeping the
32,768 total window and 4,096 output unchanged. This moves proactive compaction to about 11,264
tokens and leaves about 17,408 tokens of observed tool-output jump slack. The same local change
puts `ContextLengthExceededError` and `NonZeroAgentExitCodeError` in Harbor retry exclusions so a
deterministic canary failure aborts promptly instead of consuming four attempts. Both the
five-node canary and six-node final configs carry the contract, and the allocation-free
preflight/live-canary validation now rejects missing or smaller reserve values. The focused suite
passes 50/50 plus Ruff/compile/bash/diff checks after formatting. The canary now requires two
sequential, nonparallel exact bash calls, each returning an untruncated 44,001-byte / 11,056-token
tool result, a subsequent >=4,096-token input drop proving compaction, no OpenCode error event,
final `CANARY_DONE`, a real verifier reward, and exact identity. Without the new reserve, the live
8.3--9.2k initial prompt is projected to grow to 30.4--31.3k after those two calls and exceed the
28,672 hard prompt ceiling; with the reserve, the first result crosses the 11,264 proactive
threshold and forces compaction before the post-second-result request.

To validate this boundary without spending five nodes, a one-GPU exact-model diagnostic endpoint
was submitted directly from the local committed sbatch as Jupiter job **`1217562`**. It allocated
`jpbo-042-47` / `10.3.0.95`, successfully staged and verified the pinned FlashInfer AOT artifact,
and started exact BF16 vLLM 0.22 initialization with qwen3_coder tool parsing, max model length
32,768, and exact served revision. The JURECA reverse listener was retargeted to
`10.3.0.95:8000`. Combined allocation is only three nodes (one Jupiter diagnostic plus the two
JURECA workers). Once `/health` is live, stream the locally committed canary driver as an
allocation-free diagnostic against `jrlogin05i:18000/v1`; do not launch the five-node successor
until this large-output gate passes.

The AOT statement immediately above is now **falsified**. Job `1217562` loaded all 26 shards in
14m23s, then failed during its first real fused-MoE execution at 05:34:29. The pinned artifact is an
ELF **x86-64** shared library, while Jupiter GH200 compute nodes report **aarch64**. Hash, package
version, `is_aot`, and path checks all passed because none actually called `dlopen`; the prior
90-second AOT smoke `1216854` was therefore a false positive. The apparent “file not found” from
`tvm_ffi.load_module` was the loader's incompatible-ELF error, not disappearance: the exact file
still existed, retained its expected hash, and `file` identified x86-64 while `uname -m` returned
aarch64. Do not enable commit `59e661e0`'s artifact on Jupiter.

A parallel Triton-GDN hedge `1217853` was cancelled by Luke's agent after 4m21s because it inherited
the same doomed x86 AOT artifact; no endpoint reached health. Local commit `faba11d0` now removes all
four AOT environment variables from both five- and six-node Qwen configs, makes the optional AOT
hook validate ELF CPU architecture and perform a real `tvm_ffi.load_module` before model load, and
uses cold node-local fused-MoE JIT for the one-node diagnostic. The diagnostic keeps Triton GDN
prefill to avoid compiling an unrelated second kernel and has a one-hour wall. Focused verification
now passes 58/58 plus Ruff/bash/diff. Corrected one-node diagnostic **`1217866`** allocated at about
05:40 CEST on `jpbo-011-17` / `10.1.1.241`; the private JURECA listener was retargeted to its port
8000. Combined usage is three nodes including the two JURECA workers. Wait for cold fused-MoE JIT
and HTTP health, then run the large-output canary; do not enable the x86 AOT artifact to accelerate
this job.

Cancellation orphaned 32 incomplete sandbox IDs. They were recovered only from this run's
bounded trace tree, validated against the exact `env-<12 hex>` form, and submitted to the bridge's
`/env/stop` endpoint; no other environments were targeted. Cleanup completed by **05:02:44 CEST**:
ready/pending/starting/stopping/active_jobs are all zero, cumulative envs created equals stopped
(245/245), bridge job errors remain zero, and `workers_alive:true`. The two-node JURECA worker
allocation remains running. No checkpoint or HF export exists from `_17`.

## LATEST LIVE UPDATE — 2026-08-03 03:00 CEST

This section **supersedes the stale 11:20 state below**. Keep the older material as failure
history, but resume from here.

### Current blocker, cancellation, and prepared successor

`1214769` was cancelled by Luke's agent at about 01:32 CEST after the first 32 automatic
rollouts proved a deterministic client/server context-contract failure. This was an invalid
bring-up run, not a capability result and not an infrastructure outage. The JURECA worker job
`15489111` remains alive and healthy; keep it running for the successor.

The decisive evidence was repeated vLLM rejection of the same exact boundary request:

```
prompt contains at least 28673 input tokens ... 4096 output tokens ... 32769 total
```

The configured server capacity is `max_model_len=32768`, with a 28,672-token prompt allowance
and 4,096-token completion allowance. OpenCode first trims message content to the advertised
28,672 input tokens, then Qwen's chat template adds one special token. The resulting request is
therefore one token over capacity. This repeated at least ten times while the generation buffer
remained `0/32`; the bridge continually recycled otherwise healthy environments and still
reported `jobs_errors:0`. A bounded scan of all trial logs found no network/bridge/LiteLLM
transport errors. Do not interpret any reward from this cancelled run.

The narrow fix is OpenThoughts commit `668bf7da` (`fix(rl): reserve Qwen chat template token`) on
Luke's `lukedhlee/qwen3-6-r2egym-grpo` branch, already pushed to Luke's fork. It adds an optional
`context_budget.client_prompt_overhead_tokens` (default zero) and sets it to `1` only for this
Qwen3.6 config. Thus Harbor/OpenCode advertises 28,671 input tokens, while trainer/SkyRL retain
28,672 and vLLM remains 32,768 with 4,096 output tokens. Qwen's one-token template expansion then
fits exactly. The focused contract/environment suite passes (`15 passed`), compile and diff checks
pass, and the three other context-budget configs retain overhead zero.

`1214769` is fully gone. The clean Jupiter checkout was fetched and hard-reset to exact
`668bf7da9240cdd080164bf50988088f8713fe72`; it is clean. The full runtime dry-run passed with the
exact model, compiled vLLM 0.22, 3,328 tasks, and the live bridge. The parsed and frozen contracts
both show client `max_input_tokens=28671`, trainer/generator prompt allowance 28672,
`max_model_len=32768`, output 4096, and `qwen3_coder` auto-tool parsing. The newly rendered sbatch
has `export RL_ENV_DIR` at line 107, fallback assignment at 161, activation at 171, and zero
`envs/rl/lib` occurrences.

Exactly one three-hour successor, Jupiter job **`1215394`**, was submitted at 01:34:39 CEST into
experiment `_16` and started at **01:39:02 CEST** on `jpbo-015-[03,05-08,10]`; its scheduled end
is 04:39:02 CEST. The batch head is `jpbo-015-03` / `10.128.18.163`. The persistent private
reverse listener was cleanly retargeted from cancelled head `10.128.27.65:8000` to
`10.128.18.163:8000`, and JURECA confirms only the internal bind `10.14.0.46:18000` is listening.
The first batch log lines prove activation of the correct `rl-megatron` Python and Ray executables;
the six-node Ray cluster formed with exactly 24/24 GPUs. The executed SkyRL command visibly carries
the corrected 28,671-token client budget, the unchanged 28,672/4,096/32,768 server contract, and
both Qwen auto-tool flags. SkyRL discovered all 3,328 task directories. All eight exact BF16 vLLM
replicas then loaded all 26/26 checkpoint shards in the expected two waves (about 12:22 and 15:21
per replica group); no quantization, FP8, OOM, loader, or transport failure appeared. The engines
then completed the cold FlashInfer fused-MoE compile/autotune on both rollout nodes. The HTTP
endpoint started at **02:22:48 CEST**, returns 200, and SkyRL reports exactly one training step.

The required real OpenCode 1.18.8 gate from actual compute node `jrc0545` passed end to end at
about 02:25 CEST. OpenCode selected exact served model
`995ad96eacd98c81ed38be0c5b274b04031597b0`, Qwen emitted a parsed `bash` tool call, OpenCode
executed `printf TOOL_GATE_OK`, the tool returned `TOOL_GATE_OK` with exit 0, and Qwen completed
normally. This proves the internal route, buffered SSE, vLLM 0.22 render-layer tool flags, and
`qwen3_coder` parser together on the corrected run.

All 16 policy actors loaded 693/693 tensors. The four grouped-MoE conversion leaders showed steady
RSS/minor-fault growth on all four policy nodes, completed conversion and FSDP sharding, and
`init policy/ref/critic models done` logged at **02:47:15 CEST**. All four rollout coordinators
started by 02:52:00, each with eight trials and the correct external route. Initial weight-sync
state formed and the 693-tensor policy-to-eight-vLLM sync completed at **02:58:02** in 330.95 s.

The first corrected automatic batch then launched: 32 trace directories and 32 ready/active
JURECA sandboxes, with `workers_alive:true` and `jobs_errors:0`. A bounded aggregate and all-32
`trial.log` scan found zero network/bridge/LiteLLM transport signatures and, critically, zero
`28673`, `32769`, or `prompt contains at least` recurrence. No rollout has completed and no reward
has been reported yet; this batch is live. Combined active JSC usage is 8/16. Next require completed
rollouts and nonzero reward variance, then finite update, post-update sync, checkpoint, and HF
export.

### Active jobs and node budget

| | |
|---|---|
| Jupiter GRPO | successor **`1215394` RUNNING**, 6 nodes, 3 h wall, started 01:39:02 CEST |
| Invalid predecessor | `1213168` cancelled after 44:50 when the real OpenCode probe exposed vLLM 0.22 render-layer validation |
| Jupiter experiment | `/e/scratch/reformo/lee27/experiments/jupiter_qwen36_35b_r2egym_grpo_smoke_16` |
| Jupiter log | `.../logs/jupiter_qwen36_35b_r2egym_grpo_smoke_1215394.out` |
| Jupiter nodes | `jpbo-015-[03,05-08,10]`; head `jpbo-015-03` / `10.128.18.163` |
| JURECA workers | **`15489111` RUNNING**, 2 nodes `jrc[0545,0639]`, 16 workers/node |
| Bridge | `10.128.1.2:9920`, clean and **`workers_alive: true`** |
| Combined JSC nodes | **8 active / 16** (6 Jupiter + 2 JURECA) |

Do not cancel unrelated or healthy jobs. The user explicitly authorized cancelling our own job
when it is clearly necessary for this bring-up, without asking again. The control connection uses
`~/.ssh/cm_jureca/qwen36`; do not disturb the tunnel while this run is active.

### Current vLLM 0.22 render fix and relaunch

`1213168` formed all 6 Ray nodes / 24 GPUs, loaded the exact BF16 checkpoint on all eight engines
(26/26 shards), completed the cold FlashInfer fused-MoE compile/autotune, started the HTTP endpoint
at 23:42:00, and reported exactly one training step. The JURECA compute-node health probe returned
HTTP 200. A real OpenCode 1.18.8 request then still received the exact 400:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

This was not the YAML or parser selection: the executed SkyRL command visibly contained both
`enable_auto_tool_choice=true` and `tool_call_parser=qwen3_coder`. Primary vLLM 0.22 source proved
the version-specific missing edge: request rendering and this validation moved into
`OpenAIServingRender`. SkyRL normalized the flags correctly but passed them only to
`OpenAIServingChat`, leaving the render object at its defaults. Official vLLM 0.22 construction
passes the same flags to both objects.

MarinSkyRL commit `8ee69f3366b32109c829417cbd931fa370ae4294`
(`fix(vllm): configure tools on render service`) now passes the normalized OpenAI kwargs to
`OpenAIServingRender` too. Two focused CPU tests pass; compile, F/E9 lint, and diff checks pass.
The commit was pushed to Luke's fork and the clean Jupiter Marin checkout was fetched and hard-reset
to exact `8ee69f3`. The full Qwen runtime preflight again passed with the exact model, vLLM 0.22,
3,328 tasks, and `workers_alive:true`.

Replacement `1214769` was submitted once through a PTY at 23:47 CEST into experiment `_15` and
allocated at 23:56:47 on `jpbo-038-[17,26,30-31],jpbo-081-[07,14]`. The batch head is
`jpbo-038-17` / `10.128.27.65`. Its rendered sbatch still has `RL_ENV_DIR` export line 107 before
activation line 171, no `envs/rl/lib`, and the frozen config still contains both parser flags.
The internal reverse listener was retargeted from cancelled head `10.128.35.125:8000` to
`10.128.27.65:8000`; JURECA confirms `10.14.0.46:18000` listening. Once the endpoint is ready,
repeat the real JURECA OpenCode probe. It
must clear the render-layer 400 before any automatic rollout or reward is trusted.

`1214769` formed all 6 Ray nodes / 24 GPUs, loaded the exact BF16 checkpoint on all eight engines
(26/26 shards; first wave 10:48, slower second wave 13:01), completed cold FlashInfer compile and
autotuning on both rollout nodes, and started the HTTP endpoint at 00:44:10 CEST with exactly one
training step. Health from actual compute node `jrc0545` returned HTTP 200. The decisive real
OpenCode 1.18.8 probe then **passed end to end**: vLLM accepted `tool_choice:"auto"`, Qwen emitted a
parsed `bash` tool call, OpenCode executed `printf TOOL_GATE_OK`, and the model completed with
`TOOL_GATE_OK`; the process exited 0. This proves the buffered SSE endpoint, internal route,
`qwen3_coder` parser, and vLLM 0.22 render-layer fix together. Do not repeat or second-guess this
gate. The run is now proceeding into policy initialization, initial weight sync, and the 32
automatic rollouts. Before accepting reward, grep for transport errors and require nonzero reward
variance.

At 01:07 CEST the job remains RUNNING with all six Ray nodes and 24/24 GPUs reserved; the bridge
still reports `workers_alive:true` and zero active jobs, so automatic rollouts have **not** started.
All 16 `FSDPPolicyWorkerBase` actors are alive. Every rank loaded 693/693 tensors, and the only
model warning remains the expected absent text-shell `Qwen3_5MoeVisionBlock`. The current quiet
phase is the CPU-side grouped-MoE conversion that follows the misleadingly early
`grouped-GEMM swap active` message in `HFModelWrapper.__init__`: four mesh-coordinate-zero ranks
(one per policy node) are progressively touching/materializing the full expert weights in host
RAM, while the other 12 ranks wait and the policy GPUs remain mostly empty. The four leaders'
minor-fault counts and RSS continue to increase over sampled 20-second windows, so this is real
forward progress, not yet evidence of a dead actor or NCCL failure. Do not cancel merely because
the aggregate log has been silent since 00:51:50; keep watching for RSS/fault progress, GPU
population, `init policy/ref/critic models done`, and initial weight sync. A diagnostic `pstack`
attach briefly left policy PID 2277832 ptrace-stopped; its orphaned gdb tracer was killed and the
actor was explicitly resumed with `SIGCONT`. It is now back in normal sleeping/running state with
`TracerPid: 0`. Do not use `pstack` again on this run.

The slow conversion completed on all four leaders by 01:10:09. FSDP sharding/streamed load then
populated all 16 policy GPUs and completed cleanly: the trainer logged
`init policy/ref/critic models done` at **01:11:39**, definitively clearing the old policy-init
blocker. All four rollout coordinators subsequently started (one per spread bundle, 8 concurrent
trials each) and `Generator startup complete` logged at 01:16:06. The 9-member NCCL weight-update
group (one policy sender plus eight vLLM receivers) formed, `init_weight_sync_state` finished, and
the initial `sync_weights_to_inference_engines` began at **01:16:31**. Policy GPUs are around
6.4 GiB and show active collective utilization; vLLM GPUs remain around 86--88 GiB. This is an
actual tensor transfer. The bridge remains healthy and idle until the sync completes. Next require
an explicit sync completion, then verify 32 automatic sandboxes launch and that transport remains
clean before interpreting any reward.

The initial sync **completed** at 01:21:52 in 321.02 s. The trainer immediately launched the first
honest automatic batch: 32 trace directories representing exactly 8 distinct R2E-Gym tasks x 4
GRPO samples. Bridge status reached 32 ready environments / 32 active jobs with `workers_alive:true`
and `jobs_errors:0`; all environment starts succeeded. A labeled live JURECA sample showed 16
visible OpenCode processes on `jrc0545` and 15 on `jrc0639` (the remaining slot was between agent
process states), confirming the intended near-even two-node distribution. A bounded scan of all
32 `trial.log` files found none of `Network is unreachable`, `Cannot connect`, `No route to host`,
connection refusal, bridge errors, or LiteLLM transport errors. The batch is still running and no
reward has completed yet. Keep monitoring the full/trial logs for transport errors through trial
completion; then require nonzero raw reward variance before allowing the optimizer/update path.

### Previous YAML parser fix (incomplete under vLLM 0.22)

The YAML-level OpenCode auto-tool-choice configuration described below is fixed and deployed. The exact checkpoint
chat template emits Qwen3 XML tool calls (`<tool_call><function=...><parameter=...>`), installed
vLLM 0.22 registers `qwen3_coder` for those tokens, and the repository already used that parser
for the exact Qwen3.6 datagen family. OpenThoughts commit
`aa919e02` (`fix(rl): enable Qwen3.6 auto tool parsing`) sets:

```yaml
generator:
  engine_init_kwargs:
    enable_auto_tool_choice: true
    tool_call_parser: qwen3_coder
```

The focused HPC contract/environment suite passes (15 tests). The commit was pushed to Luke's
fork and the clean Jupiter checkout was fetched and hard-reset to exact `aa919e02`. Dry-run `_5`
and submitted experiment `_14` both render the correct contract: `export RL_ENV_DIR` line 107,
fallback assignment line 161, activation line 171, no `envs/rl/lib`, 6 nodes, 3 h wall, and the
frozen runtime JSON contains
`++generator.engine_init_kwargs.enable_auto_tool_choice=true` plus
`++generator.engine_init_kwargs.tool_call_parser=qwen3_coder`.

`1213168` is the only Jupiter successor and was submitted at 21:12:44 CEST. While it was pending,
its estimate initially improved from 22:14 to about 22:01, then slipped
to 00:37:33 as the scheduler plan changed. A pending-only reduction to 2.5 h was tested after the
large slip; it did not improve the estimate, so the job was immediately restored to the safer
3 h wall. The launcher invocation immediately before it also submitted `1213165`
without displaying stdout through the non-PTY client; once the duplicate was discovered, it was
cancelled while still pending and at zero elapsed time. Do not run both. Once `1213168` allocates,
resolve its head node/IP and retarget the existing internal reverse listener away from cancelled
`10.128.42.145:8000`. Then wait for the endpoint and run a real OpenCode 1.18.8 invocation from
`jrc0545` before trusting automatic rollouts. That probe must now clear both buffered SSE and
`tool_choice:"auto"` validation.

`1213168` allocated at 22:59:42 CEST on `jpbo-059-[29,31-32],jpbo-060-[46-48]`; the batch head is
`jpbo-059-29` with internal address `10.128.35.125`. The existing control-master reverse forward
was retargeted from cancelled head `10.128.42.145:8000` to `10.128.35.125:8000`, and the JURECA
login side confirms `10.14.0.46:18000` is listening. The first live log lines prove activation of
`/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl-megatron/bin/python` and its matching Ray.
The run is currently forming its six-node Ray cluster. Combined usage is 8/16 nodes. Do not
retarget the listener again unless this job ends or its head changes.

### Previous streaming blocker (resolved)

A direct OpenCode 1.18.8 invocation on the actual JURECA
compute node `jrc0545`, using the exact configured custom-vLLM provider and routed endpoint,
returned HTTP 400 from SkyRL:

```
Streaming is not supported in SkyRL yet, please set stream to False.
```

OpenCode's AI SDK always sends `stream: true`; the earlier manual curl proof succeeded only because
it used non-streaming mode. This exactly explains the 30-second trial exits and Harbor's application
retry schedule. It is an API transport incompatibility, not model capability or reward. The invalid
running job `1211667` and its old-code pending successor `1212070` were therefore cancelled. No
honest rollout or reward occurred.

A scheduler audit then found another independent old-code successor, `1212067` (submitted at
19:12:33 into experiment `_10`, six nodes, no dependency). It too was cancelled before allocation.
The Jupiter queue is now empty and the combined node count is two.

Fix `MarinSkyRL-apptainer-bridge` at
`skyrl-train/skyrl_train/inference_engines/inference_engine_client_http_endpoint.py`: accept a
streaming client request at the HTTP boundary, submit it to the existing Ray/vLLM backend with
`stream: false`, then return the completed response as OpenAI-compatible buffered SSE. Preserve
tool calls, reasoning fields, `prompt_token_ids`, per-choice completion token IDs, logprobs, and
usage so Harbor's existing `RecordProxy` can still build TIS-critical rollout details. Add focused
CPU tests, push Luke's branch, fetch + hard-reset the Jupiter checkout, and relaunch. Before allowing
real rollouts, repeat the direct OpenCode compute-node test; curl alone is no longer a sufficient
connectivity gate.

Keep JURECA worker job `15489111` alive. Its 32 slots and bridge status are healthy. The current
reverse listener still targets the cancelled Jupiter head `10.128.16.66:8000`; retarget it only
after the replacement job allocates and exposes its new head address.

Local fix validation completed before deployment: two focused CPU endpoint tests pass; a real local
OpenCode client accepted the synthesized stream, emitted the expected response, and read the usage
chunk; Harbor's real stream accumulator recovered exact completion token IDs and aligned logprobs
from the same event shape. The conversion must remove `stream_options` as well as setting
`stream:false` on the internal backend request because vLLM explicitly rejects stream options on a
non-streaming request.

The fix is committed as MarinSkyRL `998973e67a657016c1c3c5754b2711b915dbf4da` on Luke's
`lukedhlee/apptainer-bridge-rl` branch, pushed to Luke's fork, and the clean Jupiter checkout was
fetched and hard-reset to that exact commit. OpenThoughts remains clean at
`4d2abff5e8d561132feabed8158822d19e6614bb`. The exact runtime preflight passed again with 3,328
tasks and `workers_alive:true`. The `_12` rendered sbatch has `export RL_ENV_DIR` at line 107 and
activation at line 171, no `envs/rl/lib`, the external route `jrlogin05i:18000`, and a confirmed
three-hour Slurm limit.

Replacement job `1212291` was submitted at 19:53:41 CEST and allocated at about 20:12 CEST on
`jpbo-074-[33-36,38,42]`. The Ray head is `jpbo-074-33` / `10.128.42.145`. The existing internal
reverse listener was successfully retargeted from cancelled head `10.128.16.66:8000` to
`10.128.42.145:8000`; `jrlogin05` confirms `10.14.0.46:18000` is listening. The first log lines
prove the correct `rl-megatron` Python and Ray executables and show normal cluster startup. After
the HTTP proxy becomes ready, require a real OpenCode invocation from `jrc0545` or `jrc0639` to
succeed before accepting the 32 automatic rollouts. Then enforce the usual transport-error grep
and nonzero reward-variance gate.

By 20:18, all six Ray nodes and all 24 GPUs were registered and SkyRL connected. The exact launch
command confirms BF16, eight vLLM engines, one GRPO step, no FP8, and the exact model revision. At
20:22 the Qwen3.5/3.6 VLM shell was recognized with `language_model_only=True`. All eight vLLM
actors formed in two staggered groups, and the first group began loading the 26 safetensors shards
at about 20:28; 1/26 was observed after 52 seconds. No fatal error has appeared. Metrics-exporter
RPC code 14, c10d IPv6 address-family, image-processor deprecation, and NCCL environment warnings
are the same known nonfatal initialization noise.

All eight vLLM replicas completed 26/26 shards (about 14:49 for the first group, 15:25 for the
second), the long FlashInfer autotuning phase completed, and at 21:00 SkyRL initialized the Harbor
generator and confirmed `Total training steps: 1`. The routed health endpoint returned HTTP 200
from the actual compute node `jrc0545`. A real OpenCode 1.18.8 invocation then proved the buffered
SSE fix had cleared the old 400, but vLLM returned the **next** exact application error:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

OpenCode necessarily sends tools with `tool_choice:"auto"` for an agentic R2E-Gym rollout. The
engines were launched with neither auto-tool-choice support nor a tool parser, so `1212291` could
not produce an honest agent rollout. It was reasonably cancelled at about 21:04 CEST after 52
minutes. This is not a streaming regression: the request crossed the SSE proxy and reached vLLM's
tool-choice validation. Keep JURECA `15489111` alive. Determine the Qwen3.5/3.6-appropriate parser
from the installed vLLM 0.22 primary source, add `enable_auto_tool_choice` plus that parser to the
exact YAML/launcher, test the rendered arguments, and relaunch. Do not work around this by stripping
tools or forcing tool choice off; that would not be an honest coding-agent rollout.

`1210865` started at about 16:35:43 CEST. Its first-minute log proves the correct
`rl-megatron/bin/python` and `ray` are active and the Ray head started on `10.128.18.83`; the old
venv-ordering regression has not recurred. All six Ray nodes and 24 GPUs registered and all 3,328
task directories validated. All eight vLLM engines loaded the exact revision in BF16 with
`language_model_only=True`, no FP8, and completed 26/26 shards. The apparent post-load silence was
the expected FlashInfer fused-MoE autotuning; it completed at about 17:14. The HTTP endpoint is
live on `127.0.0.1:8000`, all eight engines registered, and the trainer confirmed `Total training
steps: 1`. All 16 policy ranks loaded 693/693 tensors, enabled grouped-GEMM swapping, ignored only
the absent text-shell vision block, and completed FSDP/NCCL setup. The trainer explicitly logged
`init policy/ref/critic models done` at 17:38:03, clearing the old TorchTitan blocker. All four
rollout coordinators started, weight sync state initialized, and the initial sync completed far
enough for 32 live rollouts to launch at 17:46. The repaired Harbor fleet split them evenly: 16
OpenCode agents on each JURECA node, with separate per-instance `opencode.db`/WAL/SHM state. This
proves the SQLite isolation and dispatcher-capacity fix.

`1210865` was nevertheless **not an honest rollout**. Per-instance OpenCode logs showed every
call to the correct model/provider failing with `Cannot connect to API`; JURECA compute nodes
return `No route to host` for Jupiter's private `10.128.18.83:8000`. No reward was produced. The
job was reasonably cancelled at 17:52 CEST rather than waiting for invalid zero rewards.

Routing fix now deployed:

- MarinSkyRL `58aeaf5` adds `SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST/PORT`, affecting only the external
  coordinator/OpenCode base URL while preserving the local vLLM readiness address.
- OpenThoughts `4d2abff5` sets the exact smoke to `jrlogin05i:18000`.
- The existing Jupiter->JURECA control master now has an internal-only reverse listener at
  `10.14.0.46:18000`; retarget it to the next Jupiter head's `10.128.x.x:8000` after allocation.
- Both Jupiter checkouts were fetched and hard-reset clean to those commits. No PR was opened.

For clean relaunch state, worker `15488971` was cancelled, the bridge pane was respawned on the
same internal `10.128.1.2:9920` bind (all counters reset to zero), and replacement `15489111`
started successfully with `workers_alive:true`. The exact preflight passed against deployed
commits. New Jupiter job `1211667` was submitted into `_9`; its rendered sbatch had the 3 h wall,
the two rollout-route exports, and correct RL venv ordering. Slurm could not backfill 3 h or 2.5 h,
so the pending job was shortened in place to the original 2 h wall. It started at 18:10:18 CEST.
The Ray head is `jpbo-002-18` / `10.128.16.66`; the control-master reverse listener was retargeted
from the cancelled job's `10.128.18.83:8000` to `10.128.16.66:8000`. Before rollouts, prove from
a JURECA compute node that `/health` and a minimal `/v1/chat/completions` request both return HTTP
200 before rollouts (`/v1/models` is not implemented by the SkyRL proxy). Ray metrics-exporter RPC
code 14, c10d IPv6 address-family, and image-processor deprecation messages remain known nonfatal
noise.

At 18:14:41, Ray reported all 6 nodes, 24 GPUs, and no pending/failed nodes. The SkyRL entrypoint
launched with the exact one-step arguments. It completed its delayed import/W&B setup and at
18:19:40 recognized the exact checkpoint as the Qwen3.5/3.6 VLM shell, logging
`language_model_only=True`. All 24 GPUs are reserved as intended: 8 for vLLM and 16 for the
policy placement group. All eight `AsyncVLLMInferenceEngine` actors have started across
`jpbo-002-18` and `jpbo-002-20`. The first four began loading the 26 safetensors shards at about
18:22:33 and reached 1/26 after about 32 s; the second four started roughly 90 s later. This is
real forward progress, not a silent hang. No fatal error has appeared. The Ray metrics-exporter
RPC code 14 and c10d IPv6 address-family warnings are the same known nonfatal noise.
An in-place attempt to restore the rendered 3 h wall at 18:25 was rejected by Slurm with
`Access/permission denied for job 1211667`; the active wall remains 2 h (ends about 20:10 CEST).
At 18:40:30 all eight vLLM replicas had completed 26/26 safetensors shards in BF16. The first
four took 14:36 and the later four 15:24; observed memory is roughly 67--69 GiB per engine, as
expected. The run is now in vLLM memory profiling/KV setup and fused-MoE autotuning, which was
silent for 10+ minutes in prior successful initializations. There is still no fatal model,
CUDA, NCCL, or network error.
Both per-node cold FlashInfer fused-MoE builds completed after roughly 10--11 minutes of active
`ninja`/`nvcc` work. All eight engines finished KV profiling and autotuning; the first group
reports 14.39 GiB / 615,693-token KV cache. The 8-engine HTTP proxy started on
`127.0.0.1:8000` at 18:55:45, and the trainer recognized 3,328 tasks and exactly one training
step. SkyRL's proxy intentionally has no `/v1/models` route (confirmed from `/openapi.json`), so
that requested path returns an application-level 404 both locally and through the tunnel. The
supported `/health` route returned HTTP 200 from JURECA compute node `jrc0545`, then a real
one-token POST to `http://jrlogin05i:18000/v1/chat/completions` returned HTTP 200, exact model
`995ad96eacd98c81ed38be0c5b274b04031597b0`, and a valid completion token. This is definitive
compute-node-to-Jupiter inference proof; the reverse route fix works end to end. Policy actor
construction has now started. At 18:57:54 Ray still had all 6 nodes, no failures, and 24 GPUs
reserved (8 inference + 16 policy).

At 19:01:10 all 16 policy ranks formed the expected DP4 x SP/EP4 process group. Every rank then
loaded all 693 policy tensors from the exact checkpoint and enabled the grouped-GEMM MoE swap;
the only ignored advertised class is the absent text-shell `Qwen3_5MoeVisionBlock`, as expected.
At 19:05 the actors are still alive in `FSDPPolicyWorkerBase.init_model` while FSDP wrapping and
optimizer construction proceed, with no traceback or Ray actor failure. The bridge remains clean
and idle (`workers_alive:true`, zero jobs/errors), which is correct because rollouts have not yet
started. The exact model config has 40 mixed full/linear-attention layers; a live policy actor's
environment contains both `SKYRL_GDN_FLASHQLA=1` and `SKYRL_GDN_FLASHQLA_REQUIRED=1`. Since
`ModelWrapper.__init__` passed the required `engage_flashqla` call on all ranks without raising,
the loud FlashQLA runtime gate has cleared even though the stdlib INFO line is not forwarded into
the aggregate Ray log. Job `1211667` has about 57 minutes of walltime remaining. Do not churn it while this
initialization continues to make progress; if the 2 h wall kills otherwise-healthy work before
the one-step milestone, successor `1212070` is already queued with a verified
`afterany:1211667` dependency. Its fresh experiment is `_11`; its rendered sbatch has a 3 h wall,
`RL_ENV_DIR` exported at line 107 before activation at line 171, and the `jrlogin05i:18000`
rollout-route exports. Cancel this still-pending successor if `1211667` lands the entire milestone.
Otherwise, when it allocates, retarget the internal reverse listener to its new Ray head before
rollouts.

At 19:20:32 `1211667` completed FSDP/optimizer construction and explicitly logged
`init policy/ref/critic models done`, clearing the historical TorchTitan/model-construction
blocker on the exact checkpoint. The live rollout dispatcher explicitly selected the repaired
external endpoint `jrlogin05i:18000`. All four coordinators started by 19:24:05 with eight trial
slots each. The nine-rank policy-to-eight-engine NCCL weight-sync communicator initialized
successfully at 19:24:33; all 16 policy ranks cleared its drain barrier and the eight engines
registered the CP>1 norm fake kernels. The initial 693-tensor policy-to-vLLM sync is now underway.
There are still no rollout connection errors, bridge errors, tracebacks, CUDA OOMs, or NCCL
failures. About 45 minutes remain on the current wall.

The initial sync finished at 19:29:29 in 295.72 s and 32 sandbox trials launched. A rendered
trial config was captured before cleanup and is correct: exact served model ID,
`api_base=http://jrlogin05i:18000/v1`, and bridge `10.128.1.2:9920`. The bridge split all 32 envs
across the live workers and reports zero transport/job errors. However, every trial exited in only
about 30 s, far too quickly to be honest agent work. Harbor is therefore retrying an
application-level exception with its 60/120/240 s backoff: as of 19:34 it has completed three
full 32-env attempts (`envs_created=envs_stopped=96`, 1,152 bridge jobs, zero bridge errors). The
coordinators are alive in `run_shard`, awaiting the final retry. The first three trial directories
were removed by Harbor's `safe_rmtree` before each retry, so no per-trial result survived. The
fourth/final attempt should start around 19:38 and should retain `result.json`; inspect it before
changing code. Do **not** treat this batch as a rollout or accept any reward yet. There is no
`Network is unreachable`/connection-refused line in the aggregate log, but application-level
OpenCode/provider failure remains likely until the retained result proves otherwise.

`15488427` was explicitly authorized for cancellation after its old Python process made the
deployed fix unusable. First replacement `15488970` exited in 9 s because the noninteractive
submission omitted required `HARBOR_SRC`. `15488971` was resubmitted with explicit
`HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src`, `BRIDGE_LOGIN=jrlogin05i`, and
`BRIDGE_PORT=9922`. Both new node processes are present and `workers_alive: true` has remained
stable while `worker_polls` increases. Startup cleanup removed all stale staging directories.

The exact-model dry-run passed again. `1210865` was submitted into fresh smoke directory `_8`.
Rendered sbatch ordering is correct: `export RL_ENV_DIR=.../rl-megatron` at line 107, activation
at line 169.

### Fixes already landed and deployed

OpenThoughts branch `lukedhlee/qwen3-6-r2egym-grpo`:

- `166a03a7` — emit `container.extra_env` before venv activation. Rendered job `1210004`
  reverified `export RL_ENV_DIR=.../envs/rl-megatron` at line 107 and activation at line 169.
- `5727eaef` — honor disabled HF Hub uploads.
- `8acd11a0` — preflight the Qwen3.6 FSDP wrap policy.
- `6e8f3558` — set Harbor `upload_agent_logs: false` to avoid redundant sandbox-log upload and
  retry-backoff starvation.

MarinSkyRL branch `lukedhlee/apptainer-bridge-rl`:

- `c340a10` — ignore optional no-split transformer classes absent from the instantiated model.
- `8a9e7c0` — for OpenCode only, rewrite `hosted_vllm/<served-id>` to
  `vllm/<served-id>` while preserving the configured vLLM API base.

Both Jupiter checkouts were fetched and hard-reset to those commits and were clean. No PR was
opened. The full launcher `PREFLIGHT_ONLY=1` passed for the exact model, 3328 tasks, FSDP wrap
policy, and FlashQLA environment.

### Why `1209328` failed

`1209328` got much farther than the old resume state: model load, policy construction, initial
693-parameter weight sync (286.99 s), and 96 sandbox attempts. It then failed before a real model
response/reward because OpenCode interpreted `hosted_vllm` as an unregistered provider and tried
to parse `"undefined/chat/completions"` as a URL. The `8a9e7c0` provider-prefix fix addresses that
exact failure. The Ray pickling and worker-crash errors at teardown were fallout, not root cause.

Before `1210004`, the bridge server had 29 stale `ready` environments while both JURECA nodes had
no Apptainer instances. Only the bridge server pane was respawned; workers reconnected and the
server counters reset to zero. The tunnel and worker allocation were not touched.

### Current checkpoint in `1210004`

Verified so far:

1. Ray: all 6 nodes and all 24 GPUs healthy; no failed or pending nodes.
2. Inference: all 8 vLLM engines loaded the exact revision in BF16 with `quantization=None` and
   text-only `language_model_only=True`. Each loaded 26/26 shards and ~64.69 GiB of weights.
3. KV/profile: first four engines expose ~14.49 GiB KV cache (620,005 tokens); second four expose
   ~13.78 GiB (589,824 tokens). FlashInfer MoE autotuning completed and the HTTP endpoint is live
   on `127.0.0.1:8000`.
4. Policy: the 16-rank process group formed as the intended DP4 x SP/EP4 mesh. All ranks loaded
   693/693 tensors. Grouped-GEMM swapping is active. The patched wrap policy ignored only the
   absent `Qwen3_5MoeVisionBlock` from the text-only shell.
5. Policy/ref/critic initialization finished at 15:51:21. Four rollout coordinators then started
   successfully and rewrote the inference host to the routable head IP `10.128.18.83:8000`.
6. Weight-sync state initialized in 27.52 s. The initial 693-tensor policy-to-vLLM sync completed
   successfully at 16:00:10 in 324.29 s. This is the first Qwen3.6 job to reach live rollouts with
   the corrected OpenCode provider; it is **not** yet a completed GRPO step.
7. Thirty-two first-wave sandbox trials were created and started. Their rendered configs use
   `vllm/995ad96eacd98c81ed38be0c5b274b04031597b0` and
   `http://10.128.18.83:8000/v1`. There are no `undefined/chat/completions`, network-unreachable,
   bridge-job, reward, or trajectory results yet.
8. One trial immediately raised `NonZeroAgentExitCodeError` with OpenCode `database is locked`.
   Sixteen OpenCode processes then waited with zero-byte agent logs, while an 8-second sample
   showed all eight vLLM GPUs at 0% utilization. A separate one-token request to the same endpoint
   succeeded with HTTP 200 in 15.14 s, proving the model endpoint and cross-site route were alive.
   The blocker was shared OpenCode state caused by Apptainer rejecting the start/exec-time
   `HOME=/root` override (`Overriding HOME ... is not permitted`), so concurrent agents fell back
   to the same host-home SQLite database.
9. The worker dispatcher also overclaims work: it computes capacity from only local queue depth,
   not running worker threads. `jrc0467` claimed all 32 bridge jobs, ran 16, and held 16 locally;
   `jrc0639` remained idle. This does not explain the SQLite stall, but it halves rollout
   throughput and must be corrected before the next walltime-sensitive launch.

`1210004` ended naturally; it was not cancelled. The final rollout failure consisted of 17
`VerifierTimeoutError`s (120 s) and 12 `BridgeOperationTimeoutError`s (630 s), with zero rewards.
The fail-loud `VerifierTimeoutError` became `InfrastructureFailureError`, after which normal Ray
placement-group teardown surfaced as the outer `WorkerCrashedError`. That Ray error is fallout,
not the root cause. Any JURECA worker restart requires explicit permission because `15488427` is
still RUNNING.

### Fix prepared and deployed to the checkout

Harbor branch `lukedhlee/apptainer-opencode-bridge` now has:

- `d93ca639` — export HOME and XDG state paths inside every sandbox command shell, where
  Apptainer cannot reject them, so OpenCode uses the existing per-instance state binds; account
  for both queued and running jobs before a dispatcher claims another batch.

Focused unit tests pass (`7 passed`), and the required changed-file Ruff checks pass. The commit
was pushed only to Luke's fork. `/p/project1/synthlaion/lee27/harbor` was fetched and hard-reset to
`d93ca63` and is clean. The existing worker processes still have the old Python loaded, so the fix
does not take effect until worker job `15488427` is restarted. Do not restart it without explicit
permission.

Known nonfatal noise: Ray metrics-exporter RPC code 14; c10d IPv6 address-family warnings followed
by successful IPv4 initialization; Transformers image-processor/dtype deprecations. vLLM also
logged a nonfatal permission error while trying to save its optional model-info cache under
`/e/data1/datasets/playground/ot-baf/vllm/modelinfos`; loading and engine init continued.

### Next gates — do not call the milestone early

1. Require `init policy/ref/critic models done` and the explicit FlashQLA-required runtime gate.
2. Initial weight-sync state and the first 693-tensor sync have passed.
3. Keep monitoring the running job, but treat shared OpenCode state isolation as the leading
   blocker if the 16 waiting agents time out. Fix from a local clone; do not patch cluster files.
   Also fix the worker dispatcher's active-slot accounting before restarting the worker fleet.
4. Before trusting any reward, grep for `Network is unreachable`, connection errors,
   `BridgeOutage`, `BridgeOperation`, and tracebacks.
5. Require **nonzero reward variance**. Zero variance is stop-and-report.
6. Then require a finite update/loss/grad norm, post-update weight sync, checkpoint, and valid HF
   export. No rollout, reward, update, checkpoint, or export has yet occurred in `1210004`.

## Objective

Complete **one honest GRPO step** for `Qwen/Qwen3.6-35B-A3B` @
`995ad96eacd98c81ed38be0c5b274b04031597b0` on Jupiter, with JURECA supplying Apptainer sandboxes.

**This run is PIPELINE VALIDATION, not model quality.** The milestone is a single optimizer step
showing: model load → rollouts → **real reward variance** → finite update → weight sync → valid HF
export. A small or noisy reward is a PASS. Zero reward *variance* is a STOP-and-report, not a
promote.

## State as of 2026-08-02 11:20 CEST

| | |
|---|---|
| Jupiter queue | **empty** — nothing running or pending |
| Last smoke `1144362` | **FAILED after 51 s** on 2026-07-31T19:05 |
| JURECA workers | **gone** — queue empty, fleet is down |
| Bridge `10.128.1.2:9920` | up, but **`workers_alive: false`** |
| Node budget | 0 / 16 in use |

Attempt history: `1141386` (9 s) · `1141455` (9 s) · `1141628` (10 min) · `1141941` (26 min,
TorchTitan) · `1144362` (51 s).

**No rollout, reward, or update result has ever been observed on this model.**

## The blocker — fix this first

`1144362` did **not** fail on TorchTitan. It failed *earlier* than `1141941` did, so it is a
**regression introduced while fixing TorchTitan**, not a recurrence.

Root cause is an **ordering bug in the rendered sbatch**:

```
line 129:  RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"      # ← activation happens HERE
line 130:  if [[ -d "$RL_ENV_DIR" ]]; then source .../activate
...
line 173:  export RL_ENV_DIR=".../OpenThoughts-Agent/envs/rl-megatron"   # ← correct value, 44 lines TOO LATE
```

`$WORKDIR` is `$DCFT` = `/e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge`, which has
**no `envs/` directory at all**. So no venv is activated → `python: command not found` →
`ray start` exits 127 → "Ray head process exited prematurely with code 127".

The correct venv is `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl-megatron` (verified to
exist). Note the venv lives under `RUNTIME_ROOT`, **not** under `DCFT` — those are different
checkouts.

**Real fix:** make the emitter write `container.extra_env` *before* the activation block.
`ai_memory/gotchas.md` states `universal_rl.sbatch:105` emits `{rl_container_env}` before
activation "for exactly this reason" — in this rendered sbatch it does not. Find out why that
changed and fix it in the local clone.

**Quick unblock (acceptable to test with, not to ship):** export `RL_ENV_DIR` in the submitting
shell so `${RL_ENV_DIR:-…}` at line 129 resolves. Confirm the launcher submits with `--export=ALL`
or the variable will not survive.

**Verify before submitting:** in the rendered sbatch, the last `export RL_ENV_DIR` must appear
*above* the `source .../bin/activate` line, and `grep -c "envs/rl/lib"` should be 0. A correct
`RL_ENV_DIR` value is not sufficient evidence — check the ordering.

## Restart order

**1 · JURECA sandbox workers** (they must be up before the GRPO job, and they take time to start):

```bash
ssh jureca
sbatch --nodes=2 /p/project1/synthlaion/lee27/harbor/src/harbor/environments/apptainer/jureca_workers.sbatch
# WORKERS_PER_NODE=16 → 2 nodes = 32 slots = n_concurrent_trials. 24 h walltime. MAX_CHAIN=0.
```

**2 · Confirm the fleet is actually alive** — do not skip this:

```bash
ssh jupiter 'curl -s http://10.128.1.2:9920/status'
# REQUIRE "workers_alive": true
```

> ⚠ **Trap:** the launch wrapper's preflight only checks that `/status` returns HTTP 200. It does
> **not** check `workers_alive`. Right now the bridge answers 200 with a dead fleet, so preflight
> would pass and every rollout would then fail. Check `workers_alive` by hand.

**3 · Fix the sbatch ordering bug**, then launch:

```bash
ssh jupiter
cd /e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge
bash hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo.sh   # defaults: 6 nodes, 1 step, 2 h
# PREFLIGHT_ONLY=1 to dry-run the gates without submitting
```

## Follow the job through these checkpoints

1. Ray cluster forms, 6 nodes
2. 8 vLLM engines load (~64.69 GiB each), `quantization=None`
3. **Policy construction** — the TorchTitan/MoE gate that killed `1141941`
4. FlashQLA engages (must fail loudly if absent — `SKYRL_GDN_FLASHQLA_REQUIRED=1`)
5. Rollouts run against JURECA
6. **Reward variance ≠ 0** ← the milestone
7. Finite update, weight sync, HF export

Before believing any low reward, `grep` the log for connection errors and
`Network is unreachable`. `avg_raw_reward: 0.0` is far more often infrastructure than capability.

## Hard constraints

- **≤ 16 combined JSC nodes** active+pending. The plan is 6 Jupiter + 2 JURECA = 8.
- **Never `scancel` a RUNNING job** without explicit permission; never anyone else's.
- **Never `find`/`du` on GPFS or JURECA scratch** — inode exhaustion locks the project for all users.
- **Never `git push` / open a PR** without an explicit ask in that same turn. `lukedhlee`-owned
  repos only.
- **Local clones are ground truth.** Edit locally → push → cluster `git fetch` + hard reset. Never
  `git pull` a divergent tree, never hand-edit or rsync a patch onto a cluster.
- Listeners bind **internal interfaces only**, never `0.0.0.0`. Login-node processes stay small.
- **No FP8 anywhere.** If you find `dequantize` or FP8 in a Qwen3.6 path it is stale — delete it.
  The FP8 revision hash does not carry across repos.
- Do **not** re-litigate: the learnable band is parked; sandbox option D is settled; GRPO-vs-RLOO is
  a non-lever; colocation is falsified.

## Known traps that have already cost time

- `flash_attn: false` is a **trio** — also needs `attn_backend: sdpa` and
  `use_sample_packing: false`. Otherwise you silently get `eager` attention: works, unusably slow.
- A parquet passed as `train_data` yields **zero tasks** and still submits. Confirmation to look for
  in the log: `Found 3328 task(s) with 7 unique environment(s)`.
- Exception classification is **name-based, not `isinstance`**. A new exception subclass falls
  through silently unless enumerated in the YAML lists.
- Don't call a big job wedged too early — MoE autotuning can be silent for 10+ minutes.
- Don't "fix" MoE weight sync by disabling `SKYRL_W13_RELOAD_BRACKET` — it produces token salad.

## After the smoke passes

1. **Tunnel resilience gate** — force a bounded Jupiter↔JURECA disconnect at
   `ServerAliveInterval=1` / `ServerAliveCountMax=1`, prove failure is loud and workers recover,
   then restore conservative keepalives. Never disrupt an active rollout.
2. **Before promoting to 50 steps, do the walltime arithmetic.** At a measured step time of
   ~18 min this is 15 h; at the historical 46 min it is 38 h. `TIME_LIMIT` currently defaults to
   `02:00:00`, and JURECA workers run `MAX_CHAIN=0` (no successor queued) with a 24 h wall. So a
   promoted run needs a raised `TIME_LIMIT`, a working auto-chain (**never tested across a real
   walltime boundary**), and identical-topology restarts — FSDP2 checkpoints cannot reshard.

## Reference

- `ai_memory/handoff.md` — full state (⚠ its "1144362 is queued" line is now stale; that job failed)
- `ai_memory/gotchas.md` — read before debugging anything environment-shaped
- `ai_memory/decisions.md` — read before changing an established choice
- Config: `hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml`
- Launcher: `hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo.sh`
