# Handoff — lukedhlee · updated 2026-08-06 (17:30 KST)

> ## ⛔ THIS FILE IS TWO DAYS BEHIND. START AT `NEXT_SESSION.md` §0.0-FINAL (2026-08-06).
>
> Everything below predates the 2026-08-05/06 pivot to the **8B band-reproduction goal** and the
> 2026-08-06 **task-pool validation campaign**. The current truth, in one paragraph:
> the r2egym harness is validated model-free; the pool is measured — **allowlist v3 = 4,469 of
> 4,568 tasks solvable (97.8%)**, the 99 exclusions diagnosed (env-sensitive grading, not broken
> harness); zero free-pass tasks exist on the raw path; the edit-rate stall was root-caused to
> malformed bare-JSON tool calls (parser repair `d1e3ecb8` shipped + deployed) plus missing `rg`
> (harbor `1019a36` staged, needs the next fleet); pandas' 1,332 "dead" tasks were recovered by a
> config-strip shim (harbor `55c416cd`) + a **narrow oracle** (envgate `fa5acdd6`).
> The model-side bring-up is a copy-paste RUNBOOK in `NEXT_SESSION.md`, gated only on the operator's
> Jupiter→JURECA ControlMaster TOTP. Historical context below remains valid as history; where it
> conflicts with `NEXT_SESSION.md` or `gotchas.md` (2026-08-06 entries), the newer docs win.

> **OLD START HERE: § "Live state — 2026-08-04 10:30 KST" in this file.** It supersedes the 2026-08-03
> block below it and `notes/qwen36_resume_brief.md`, both of which are now two framings behind.
> Then read [[gotchas]] § 2026-08-04 and [[decisions]] § 2026-08-04.
>
> **Two things a returning reader must not get wrong:**
> 1. **The 600s agent timeout was real and is fixed** — but fixing it did NOT produce reward variance.
>    **The learnable band IS REQUIRED.** An earlier version of this file said the opposite; that
>    reading was based on the truncated measurement and has been reversed by `1229343`'s data.
> 2. **`/e/scratch` cannot create any new file** (project-wide inode cap, exhausted). It breaks
>    `git fetch`, so the experiments tree and the execution checkout now live on **`/p/scratch`**.
>
> ⚠ Also RETIRED: "raw r2egym collapses" (it does not — same ~45.5 p@1, 120 steps vs 60) and "Qwen3.6
> is deterministic per r2egym task" (trajectories vary widely; only OUTCOMES are invariant — which is
> the real finding, and it is why the band is needed).

## Objective

GRPO-train **`Qwen/Qwen3.6-35B-A3B`** on r2egym and measure the paired post-training change on the
validated SWE-Bench Verified 100-task set. Exact revision
**`995ad96eacd98c81ed38be0c5b274b04031597b0`**. Earlier Qwen3-30B/Qwen3-8B work is historical
bring-up evidence; Vista is concluded.

> ### ⚠ THE GOAL OF THIS RUN IS PIPELINE VALIDATION — operator, 2026-07-31
> **Prove the RL pipeline works.** Not model quality, not a competitive number. The milestone is the
> one-step GRPO smoke producing **real reward variance** and a finite update. A small or noisy
> improvement is a PASS. The paired SWE-Bench arms are the science that follows, not the thing under
> test. Anything not on the path to that gate is deferred. Read this before proposing scope.

**⚠ FP8 IS DEAD (2026-07-31).** An earlier same-day plan used `Qwen/Qwen3.6-35B-A3B-FP8` @
`95a723d0…`. The operator reversed it. **That revision hash is invalid for the plain repo** — hashes
do not carry across repos. The 262-line dequantizer and its sbatch are deleted. If you find FP8 or
`dequantize` anywhere in a Qwen3.6 path, it is stale.

### Current branches — READ THIS FIRST

| repo | worktree | branch | head | pushed? |
|---|---|---|---|---|
| OpenThoughts-Agent | `~/OpenThoughts-Agent-clean` | `lukedhlee/qwen3-6-r2egym-grpo` | `fc262339` | **yes** |
| OpenThoughts-Agent | `~/OpenThoughts-Agent-apptainer-bridge` | `lukedhlee/archive/r2egym-integration-20260731` | `647d7cc1` | **no** |
| MarinSkyRL | `~/MarinSkyRL-apptainer-bridge` | `lukedhlee/apptainer-bridge-rl` | `c0a7098` | **yes** |
| Harbor | `~/harbor-apptainer-bridge` | `lukedhlee/apptainer-opencode-bridge` | `6b25eb16` | **yes** |

**`lukedhlee/qwen3-6-r2egym-grpo` is the execution branch.** Cut fresh 2026-07-31 from
`origin/penfever/working` @ `32416f6e` (the canonical OT-Agent branch per CLAUDE.md — *not* `main`).
Fourteen commits, **37 files / +3,407 / −30**, 12 focused tests green. Contains **no `ai_memory`, no
Vista, no GSM8K**.

The archive branch preserves all 83 original files at `647d7cc1`. **Never merge it forward** — it is
based on a 07-18 commit that is now ~45 behind upstream.

⚠ **`ai_memory` is deliberately excluded from the clean branch** (operator: agents run locally, the
cluster has no use for it; grep confirms no code references it). It IS still tracked on
`lukedhlee/vista-moe-grpo-30b`, which is pushed to the **PUBLIC** fork
`github.com/lukedhlee/OpenThoughts-Agent`. Operator has accepted that exposure. Upstream
`open-thoughts` has zero `ai_memory` files.

**Cluster roles (settled 2026-07-29 evening — two earlier framings in this file were both wrong; this
one is measured from the co-lead's live scripts, not inferred):**
- **Jupiter (JSC)** — ALL model execution (RL training + vLLM), plus the **bridge server on a Jupiter
  LOGIN node**. Jupiter has **no CPU-only partition** (`sinfo`: only `booster` + `largebooster`, both
  `gpu:gh200:4`, 288 CPUs / 878 GB per node) ⇒ hosting sandboxes here burns GH200s running pytest.
- **JURECA (JSC) `dc-cpu` / `synthlaion`** — **ALL sandboxes, for r2egym AND SWE-bench.** CPU-only, free,
  x86_64. Workers dial IN to the Jupiter bridge over a reverse SSH tunnel from the JURECA login node.
- **No model is ever served on JURECA.** Jupiter compute never contacts JURECA — see § Sandbox.

**Current focus = get ONE honest GRPO step.** The pipeline runs cleanly end-to-end through Ray → 26/26
shards → fused-MoE JIT/KV → HTTP endpoint → policy/ref init → 693-tensor weight sync → honest rollouts
with real verifier rewards. Five defects have been root-caused and fixed (four on 08-03, plus the 600s
exec cap on 08-04).

⚠ **The 08-03 claim that "raw r2egym yields ZERO within-group reward variance" is not clean evidence.**
The agent was capped at 600s rather than the configured 1800s, so 76% of trials were killed mid-task.
Whether variance appears under the restored budget is the **next measurement**, and it gates whether a
learnable band is needed at all. See § Live state — 2026-08-04.

## State

- **READY LOCALLY — Qwen3.6 GRPO + paired SWE-Bench workflow, on the clean branch.** The hub
  checkpoint is staged verbatim and unwrapped in memory; there is no conversion step and no derived
  artifact. Policy, reference, rollout, and both eval arms all load the same pinned revision, so
  nothing about format can confound the measured delta. Details in § plan of record.
- **PASSED — FlashQLA 0.1.2 GH200 autograd gate.** Qwen3.6 has three GDN layers per four-layer block; the
  Transformers pure-Torch fallback is prohibitively slow at the 28,672-token training prompt length.
  The fully pinned ARM64 layer at `/e/scratch/reformo/lee27/pydeps/qwen36-flashqla-0.1.2` passed real
  BF16 forward/backward on GH200 in job `1139108` and wrote `gpu_sm90_smoke.ok`. MarinSkyRL binds it
  to `Qwen3_5MoeGatedDeltaNet` and fails loudly if the kernel is absent or unused.
- **VALIDATED LOCALLY — scientific gates.** Focused suites pass (11 OpenThoughts tests, 2 MarinSkyRL
  tests; Ruff, bash syntax, YAML parse, and diff checks pass). The expanded contamination audit parsed
  all 3,328 r2egym tasks and all pinned 100 SWE-Bench tasks: zero direct issue matches, including
  same-repository comparisons across differing commits. One unrelated SymPy pair shares repo+base
  commit (`r2egym-v1-07587` vs `sympy__sympy-20801`, similarity `0.174699`).
- **CLOSED — GSM8K:** the `pass_at_1 ≈ 0.45` plateau was a **format artifact**, not math ability
  (paired, n=1319); no headroom worth reporting. → [[gsm8k_format_artifact]]
- **CONCLUDED — Vista:** the 24-node stall was never made deterministic (it moved between
  `fwd_logprobs`, `policy_train`, `sync_weights`); work moved to Jupiter, where it didn't reproduce.
  → [[vista_24n_sync_stall]]
- **READY — Jupiter training backends:** FSDP2 6-node disagg (16 policy EP4×FSDP4 + 8 infer) at
  ~432s/step; Megatron 1.36× step / 1.66× policy_train faster, same-topology resume only.
  → [[jupiter_bringup_and_throughput]], [[jupiter_megatron_bringup]], [[megatron_vs_fsdp2]]
- **READY · Apptainer bridge on JURECA: r2egym scale gate passed.** All seven r2egym SIFs exist, a
  pristine oracle scored 1, a deliberate no-op scored 0, real Terminus2/Qwen3-8B completed with a
  genuine verifier failure, and 128/128 concurrent oracle rollouts completed in 96s with zero
  exceptions/retries/connection errors. This is the r2egym sandbox runtime, not a SWE-bench side
  project.
  → [[apptainer_bridge_handoff]] § "2026-07-29 (later)"
- **READY · pinned SWE-bench-100 oracle validation is 100/100.** `psf__requests-2317`, the former
  `httpbin.org` infrastructure exception, now uses an in-container HTTP/HTTPS httpbin service plus a
  local TCP connect-timeout tarpit. Its pristine oracle scored reward 1 on the air-gapped JURECA
  `dc-cpu` partition with no worker proxy. No oracle/model failure remains hidden behind an
  infrastructure zero. → [[apptainer_bridge_handoff]] § "Pinned SWE-bench-100 oracle"
- **SYNCED — execution repositories.** The execution branch is
  `lukedhlee/qwen3-6-r2egym-grpo` in `~/OpenThoughts-Agent-clean`; it and the Marin branch are pushed.
  The Jupiter execution checkouts were reconciled by fetch + hard reset (never `git pull`) and match
  `fc262339` / `c0a7098` / Harbor `6b25eb16`. Historical dirty trees remain historical and are not
  used for execution.
- **HISTORICAL — r2egym GRPO (Qwen3-8B) pre-bridge run, 2026-07-29.**
  Job `1087425`/`1087417` reached a full training step: 8 vLLM engines up (15.27 GiB each, **472,016
  KV tokens** per engine), harbor 0.8.1, transformers 5.8.1, FSDP2 policy init, `init_weight_sync_state`
  6.4s, `sync_weights_to_inference_engines` 5.9s, `fwd_logprobs` 5.1s, advantages, policy train.
  **But `avg_raw_reward: 0.0`, `zero_rewards: 1.0`** — 128/128 rollouts died on the then-current
  Daytona path. That historical failure was cancelled; the later JURECA Apptainer path resolved this
  sandbox blocker. See § Sandbox. → [[gotchas]]
  - **The old `envs/rl` blocker is RESOLVED but the fix was different than expected:** `envs/rl` is
    genuinely unusable (its `vllm._C` needs `libcudart.so.12`; Jupiter ships only CUDA/13), so the run
    uses **`envs/rl-megatron`** ("megatron" is only the dir name — `strategy: fsdp2`). Nine further
    launch blockers were fixed to get there — env completeness (daytona SDK), harbor too old
    (`normalize_message`), `LD_LIBRARY_PATH` pointing at the broken env, `train_data` as a parquet
    yielding ZERO tasks silently, the fully-async trainer's constraint set, and the
    `flash_attn: false` trio. All recorded in [[gotchas]]; all fixed on the branch.
- **⚠ RETIRED 2026-08-04 — "raw r2egym collapsed" was never true.** The co-lead's own result summary:
  her RAW arm reaches the **same ~45.5 p@1** on SWE-bench as the filtered band, in 120 steps instead of
  60. The band is `p@4` (not `p@8`), yields **~1.6k of 4.5k (~36%)**, is built for an **8B**, and gives
  **no performance boost** — convergence speed and ~20% compute only (60k → 48k rollouts). Our own flat
  reward curve was a **600s agent timeout**, not the data. The band's only value for a
  pipeline-validation goal is guaranteeing a nonzero gradient exists.
  → [[r2egym_apptainer_reference_impl]], [[decisions]] 2026-08-04, [[gotchas]] 2026-08-04
- **VALIDATED — the pinned SWE-bench-100 sandbox set.** The second 100-task manifest remains parked;
  do not invent a 200-task set. The confirmed first 100 now has complete oracle coverage as described
  above.

## Sandbox — former blocker, resolved 2026-07-29

**The one-line version:** agentic RL needs a sandbox per rollout; the only sandbox we had wired was
**Daytona, which is a CLOUD API**, and **JSC compute nodes have no internet** — so every rollout fails.

### Why it wasn't obvious — ⚠ CORRECTED 2026-07-31, the earlier explanation here was WRONG

The old text claimed `mask_exceptions` + `default_error_treatment: zero` absorbed connection failures
into zero rewards. **That is not what the code does.** Read
`skyrl-train/examples/terminal_bench/terminal_bench_generator.py:1215-1249`:

- `mask` means **exclude from the RLOO-N baseline** (correct treatment for infrastructure failure).
- Returning `False` is what includes a rollout in the baseline at reward 0.
- `default_error_treatment` applies **only to exceptions in none of the three lists**.

**The actual defect was line 1215:** `exception_type = type(exception).__name__` — classification is an
**exact class-name string match, not `isinstance`**. Job `1087425` raised `DaytonaConnectionError`,
which matches neither `DaytonaError` nor `ConnectionError`, so it fell through to the `zero` default
and became real training signal.

**Fixed on the branch:** `default_error_treatment: mask`, `fail_on_infrastructure_error: true`, and
named infra exceptions (`BridgeOutageError`, `BridgeOperationError`, `BridgeOperationTimeoutError`,
`VerifierInfrastructureError`). Note the matching is still name-based, so **any new exception subclass
must be enumerated or it silently falls through.**

Under GRPO, all-zero reward ⇒ zero advantage variance ⇒ **no gradient**. **Always grep an RL log for
`Network is unreachable` / connection errors before believing a low reward.**

`hpc.py` does set `needs_ssh_tunnel=True` for jupiter, but the machine-local edit set
`proxychains_binary=""`, and `hpc.py:752` gates the entire proxy-setup block on that being non-empty ⇒
compute nodes get **no outbound path at all**. For the old GSM8K (non-agentic) runs that was harmless,
which is why it went unnoticed.

### The measured network shape (this is what constrains every option)

From a real probe job on Jupiter compute `jpbo-040-12` — full table in [[gotchas]]:
- internet (`app.daytona.io:443`, `huggingface.co:443`): **BLOCKED**
- JURECA, any address/port: **BLOCKED**
- Jupiter login via **public** IP: **BLOCKED**
- Jupiter login via **internal** `10.128.1.2` / `10.99.0.2`, HTTP **and** `:22`: **REACHABLE**

⚠ **Never probe with public IPs** — doing so gives a false "totally isolated" reading. Compute↔login
works over `10.128.x`/`10.99.x` (same /16), which is why Ray works, and a **relay on a Jupiter login
node is the basis of every viable fix.**

### DECIDED — option D. (A/B/C below are historical; do not re-open them.)

**Plan of record:** RL + vLLM on Jupiter · **bridge server on a Jupiter login node** · **sandbox workers
on JURECA `dc-cpu`/`synthlaion`** · **reverse SSH tunnel dialed IN from the JURECA login node**. This is
the co-lead's production architecture with JURECA substituted for her JUWELS (**JUWELS ≈ JURECA for our
purposes — operator, 2026-07-29**).

Historical options, superseded: **A** = login-node proxy → cloud Daytona (now only a fallback if image
builds stall; needs `proxychains_binary` refilled, `hpc.py:752` gates on it). **B** = Jupiter-LOCAL
bridge, workers+SIFs on Jupiter compute — rejected, Jupiter has no CPU partition. **C** = "JURECA bridge
for r2egym", rejected at the time as WRONG CLUSTER for a reason that does not apply (below) — C was
essentially right.

### Why the old rejection of C was wrong

Reading the co-lead's scripts (not inferring from them) showed her design is **neither A nor B**: RL on
Jupiter, bridge server on a Jupiter **login** node, sandbox workers on a **separate CPU cluster**
(JUWELS `batch`, 8 nodes × 48 CPUs, no GPUs), joined by a **reverse SSH tunnel initiated from the CPU
cluster's login node**. Full measurements: [[r2egym_apptainer_reference_impl]] § CORRECTION.

- **Option C was rejected for a reason that does not apply.** Jupiter compute never talks to the remote
  cluster in this design, so "Jupiter compute can't reach JURECA" is irrelevant — the traffic goes
  Jupiter-compute → Jupiter-login-bridge (internal, measured working) and the remote login node dials
  IN. This also dissolves the routing gap flagged against the SWE-bench eval plan.
- **Option B is the worse choice.** Jupiter has **no CPU-only partition** (measured `sinfo`: only
  `booster` + `largebooster`, both `gpu:gh200:4`, 288 CPUs / 878 GB per node) ⇒ Jupiter-hosted sandboxes
  burn GH200 nodes running pytest.
- **Option D (take this):** her architecture with **JURECA `dc-cpu` / `synthlaion`** as the sandbox
  cluster — JUWELS ≈ JURECA for our purposes (operator call). Our worker sbatch is already ported and
  smoke-passed there. **Option A's Jupiter proxy is then NOT required** for the sandbox path at all; a
  proxy is still needed on the *build* side (apt/pip on air-gapped compute), which her prebuild script
  already handles.
- Must port: the **auto-chain** (worker job submits its successor) — the sandbox allocation's walltime
  is shorter than the training run.

## r2egym enablement plan — THE PLAN OF RECORD (2026-07-29 evening)

> ## ⚠ POLICY NOTE — 2026-07-29 late. Hold LIFTED by operator; proceed with guardrails.
>
> **Operator ruling (2026-07-29):** proceed. Criterion = **"fine as long as we don't sabotage other
> users."** The co-lead runs this architecture in production and is on good terms with the JSC manager.
> A support ticket is NOT being filed. **Do not re-raise this as a blocker** — it was raised twice and
> decided. Do observe the guardrails below, which operationalize the operator's own criterion.
>
> **Guardrails (these ARE the standard now):**
> - Bind every listener to an **internal interface**, never `0.0.0.0`, never a public IP.
>   **Operational trap:** the stale Jupiter scratch copy of
>   `eval/jureca/jupiter_to_jureca_tunnel.sh` still attempted a public `0.0.0.0` reverse bind on
>   2026-07-29. One new `:9930` forward was caught and cancelled before a worker/trial used it. Use the
>   reviewed internal-only copy staged at
>   `/e/scratch/reformo/lee27/tunnel_internal_20260729.sh` until the scratch checkout is synchronized.
> - Login-node processes stay **small and idle-cheap**. No bulk work on a shared login node — no
>   multi-image parallel pulls/builds there (this was already a measured hazard: image pulls are
>   CPU-bound, `user 6m51` vs `real 2m47`).
> - Route heavy build work to **compute** nodes, as the co-lead does.
> - **Private key on Jupiter: passphrase-protected, `chmod 600`, never group/world-readable.** This one is
>   not a policy question — it protects us, and a readable key is the same class of failure as the
>   co-lead's leaked Docker PAT.
> - Tear services down when not in use; don't leave listeners up across idle days.
> - Never `find`/`du` on GPFS; inode exhaustion is the one thing that genuinely does harm other users.
>
> **Ask the co-lead (one line, high value):** *did you clear the relay/tunnel with anyone specific, or is
> it just understood?* A name is worth having; it converts "she does it too" from hearsay into cover.
>
> **Still worth checking on its own merits — the JSC Container Build System** (SLD item 2, Dockerfile→
> image webservice with a login-node CLI; see [[apptainer_bridge_handoff]]'s resolution ladder). Not for
> compliance reasons now, but because it may simply be **easier** than porting a SOCKS relay for `apt`/
> `pip`. Check before building the relay path.
>
> Historical record of what was flagged and why (four items, JSC doc refs) — kept because the reasoning
> is sound even though the hold was lifted:
> 1. **SOCKS relay giving `dc-cpu` jobs `apt`/`pip` access** — JURECA docs state compute nodes are **not
>    allowed internet access**, and direct users to support when login-node staging is insufficient.
>    This is the clearest concern.
> 2. **Outbound SSH ControlMaster Jupiter→JURECA using a private key stored on Jupiter** — JSC prohibits
>    storing private keys on JSC storage except limited support-approved cases, and states **outgoing SSH
>    connections are not allowed**.
> 3. **Reverse-forwarded ports listening on a JURECA login node.**
> 4. **A persistent HTTP bridge + listening port on a shared Jupiter login node.**
>
> Refs: JURECA batchsystem + access-restrictions pages.
> **The co-lead doing all of this is NOT authorization** — she may hold approvals under her own project
> (`transfernetx`/`projectnucleus`). Ask her *which approvals she holds*, not just how it works.
>
> **Sanctioned alternative to item 1 that may moot it:** the **JSC Container Build System** (SLD item 2 —
> an external Dockerfile→image webservice with a CLI on the login nodes), already noted as the
> "sanctioned path" in [[apptainer_bridge_handoff]]'s resolution ladder. If it can build the 7 r2egym
> images, **no compute-node internet is needed at all.** Investigate this BEFORE re-proposing a relay.
>
> Unknown and worth establishing: whether the SIFs need any network **at runtime** (probably not — deps
> bake in at build). If yes, that changes the runtime design, so establish it early.

Goal: **r2egym running end-to-end on Jupiter+JURECA, for training AND validation.** Five steps, each with
a gate. State the key config, then go — the operator has pre-authorized relevant JSC jobs under the
combined 16-node ceiling.

**1 · Stand up the sandbox service.** Bridge server on a Jupiter login node; workers on JURECA
`dc-cpu`/`synthlaion`; reverse SSH tunnel dialed IN from the JURECA login node. Port her per-node
**dispatcher** model (1 dispatcher/node polls through the tunnel, N workers consume a local queue — her
header: *"Only 32 dispatchers poll through tunnel (not 512 workers)"*). Our `jureca_workers.sbatch` is
already ported + smoke-passed from the SWE-bench work.
*Gate:* a JURECA worker picks up a job from Jupiter and returns a result through the relay.

**2 · Build the 7 r2egym SIFs.** All 3,328 tasks share **7 unique env Dockerfile hashes** (measured). But
they are **recipes, not published images** — unlike SWE-bench. `worker.py:_build_sif()` converts
Dockerfile→`.def` (handles `FROM`/`RUN`/`ENV`/`WORKDIR`) then `apptainer build`, `--fakeroot` first with
non-fakeroot fallback. Build needs internet (`apt`+`pip`) on an air-gapped compute node ⇒ port the SOCKS
proxy her `prebuild_sifs.sh` raises when it detects a compute node. Her driver:
`prebuild_r2egym_sifs.sbatch` → `srun` over 4 nodes → 48-way parallel, keyed on
`r2egym_build_list.txt` (`task_name df_hash docker_image`), idempotent.
*Gate:* one task runs end-to-end in our own SIF and returns a correct reward.

**3 · Prove the REAL agent works in them. ← highest-risk step, deliberately early.** Everything validated
so far was **OpenCode** on the **newer** `harbor.environments.apptainer`; training uses **terminus-2**;
her design targets **terminus-structured** on the **older** `harbor.environments.apptainer_bridge`. Do
not assume her dispatcher/tunnel transfers unchanged.
*Gate:* one task, training's real agent, real model, correct reward — AND a deliberately-broken task
scores 0 rather than erroring.

**4 · Scale + endure.** 1 → 128 concurrent sandboxes; port the **auto-chain** (worker job submits its
successor before starting) because the CPU allocation's walltime is shorter than the RL run. Then time
it: generation was ~46 min/step against Daytona (mostly retry backoff) and 50 steps must fit 12h.
*Gate:* a full step's rollouts complete with a low failure rate; measured step × 50 fits with margin.

### Steps 1–4 completion record — PASSED 2026-07-29

- **Step 1:** internal-only bridge at `10.128.1.2:9920`; reverse listener at `10.14.0.44:9920`;
  JURECA workers picked up and returned jobs. The Jupiter→JURECA ControlMaster requires one interactive
  MFA establishment per master lifetime, not per worker or rollout.
- **Step 2:** seven SIFs under `/p/scratch/synthlaion/lee27/r2egym_sif`; standalone JUWELS oracle
  reward 1 with 26 passing tests and three expected failures. No Marianna transfer is needed.
- **Step 3:** bridge oracle reward 1; deliberate no-op reward 0 with all 29 tests collected; real
  Terminus2/Qwen3-8B negative reward 0 with 24 pass/5 fail and no connection errors. A false zero from
  a missing `/setup_files/test_info.json` was rejected and the transfer path fixed.
- **Step 4 scale:** 128/128 concurrent oracle rollouts completed in 96s, reward 1, zero exceptions,
  retries, cancellations, or connection errors. Eight JURECA nodes × 16 workers/node used exactly one
  dispatcher/node.
- ⚠ **The old claim "auto-chain preserved eight nodes and walltime" is NOT SUPPORTED.** Commits
  `b1d9dcbd`/`187ade94` taught the chain to *propagate* node count and walltime, but every recorded
  smoke ran `MAX_CHAIN=0` and the chain has **never crossed a real walltime boundary**. This is no
  longer on the critical path because the worker request now uses the measured 24h maximum.
- **Step 4 real-model parser gate:** `strict_json_parser` had been dead configuration; Harbor now
  forwards it into Terminus2 and rejects parser warnings instead of auto-fixing malformed terminal
  commands. A post-fix Qwen3-8B rollout rejected two missing-newline responses, accepted the corrected
  third response, completed in 5m56s, and produced a genuine reward 0 only after the model overwrote
  `aiohttp/web.py` and pytest failed import/collection. Bridge: 48/48 worker jobs completed, zero job
  errors, environment stopped cleanly.
- **Training preflight:** bridge, all 3,328 extracted tasks, cached Qwen3-8B, patched Harbor, and patched
  MarinSkyRL passed. The preflight explicitly exited without submitting an RL job.
- **Fail-loud isolation:** bridge job errors/timeouts are named infrastructure exceptions and are not
  subclasses of agent `TimeoutError`; air-gapped verifier dependency failures are
  `VerifierInfrastructureError`; r2egym configs abort on classified infrastructure instead of emitting
  a training reward zero. Model `AgentTimeoutError`/context-limit outcomes remain passthrough.
- **Key local commits (not pushed):** OpenThoughts integration/config through `502227ba`; Harbor
  through `9792153e`; MarinSkyRL `2adac62`.

**5 · Launch.** Config is DONE and debugged: `hpc/skyrl_yaml/jupiter/3node_qwen3_8b_r2egym_grpo.yaml` +
`run_r2egym_qwen3_8b_grpo.sh`. `ckpt_interval` small, `eval_interval: 999999`, score held-out r2egym
OFFLINE on saved checkpoints (agentic val is synchronous/blocking).
*Success:* reward leaves zero and climbs; held-out improves with it.

### Marianna dependency — CLEARED

No Marianna action is required for r2egym enablement or launch. Her build-list/cache paths were useful
references, but we built and validated our own seven SIFs in `synthlaion`. Her permission-denied
learnable-band datasets remain parked by operator decision.

## Qwen3.6-35B-A3B paired experiment — plan of record (2026-07-31 evening)

**Initialization — no conversion step exists any more**
- Exact origin: `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Stage it **verbatim** from the hub to
  `/e/scratch/reformo/lee27/models/Qwen3.6-35B-A3B/995ad96eacd98c81ed38be0c5b274b04031597b0`.
  The revision is pinned **in the directory name**; preflight asserts it. Both eval arms must resolve
  to that exact revision or the paired comparison is meaningless.
- The published checkpoint is a **multimodal shell** (`Qwen3_5MoeForConditionalGeneration`). Measured
  composition of the 67.0 GiB checkpoint: **text tower + lm_head 64.6 GiB (96.4%)**, MTP head 1.6 GiB
  (2.3%), **vision tower 0.8 GiB (1.2%)**.
- **`SKYRL_QWEN3_5_VLM_UNWRAP=1`** (SkyRL's default). SkyRL unwraps to the text tower **in memory** —
  `skyrl_train/models/qwen3_5_vlm.py` documents it as a pure reference re-point of already-loaded
  submodules, text keys mapping **1:1, 0 missing / 0 extra**. The same flag gates
  `map_text_name_to_vlm_engine`, so policy and rollout engines stay in lockstep.
  **Do not set this to 0** — that was the old design, which required an on-disk pre-unwrapped
  checkpoint that no longer exists.
- Cost of in-memory unwrap: ~**3.6%** extra read per load (~2.4 GiB of vision+MTP), ~10 loads/job.
  Rejected alternative: a second permanent 67 GB artifact plus a conversion job and script.
- **The vision tower is parked, and cannot affect learning** — it receives no gradients and is not in
  the optimizer. Consequence deferred to publishing: the export will be **text-only**, so it is not a
  drop-in replacement for the base model. Re-attaching later is ~40 lines given the 1:1 key mapping.

**GRPO**
- Six Jupiter booster nodes: four policy/reference nodes = 16 GPUs (`EP=4 × FSDP=4`), plus two
  rollout nodes = eight independent TP1 BF16 vLLM engines.
- One-step smoke first on all 3,328 r2egym tasks; 32 sampled rollouts (`8 prompts × 4`) and fail-loud
  Apptainer infrastructure handling. Require model load, real reward variance/nonzero advantages,
  finite policy update, weight sync, and valid HF save.
- Only after that gate passes, start the promoted 50-step run with a non-smoke job name and restart
  allowance. The model-specific learnable-band filter remains parked; if reward variance is zero,
  stop and report rather than silently promoting.

**Paired SWE-Bench evaluation**
- Same pinned 100 tasks and same evaluation file for both arms: OpenCode 1.18.8, one attempt,
  77,824 input + 4,096 output = 81,920 tokens, compaction off.
- Jupiter serving is identical in both arms: BF16 weights, FP8 KV cache, DP4 × TP1 on one GH200 node,
  stable served model ID, `qwen3_coder` tool parser and `qwen3` reasoning parser. JURECA runs only the
  x86_64 sandboxes.
- Run the immutable pre-GRPO checkpoint first. After training, select a valid HF export and rerun the
  same 100 tasks. The paired report must include raw resolved counts/rates, absolute percentage-point
  delta, per-task transitions, paired-bootstrap 95% interval, and exact McNemar p-value. The model
  card's SWE-Bench score is not a baseline.

**Checked-in artifacts (all on `lukedhlee/qwen3-6-r2egym-grpo`, post-rename)**
- FlashQLA: `hpc/skyrl_standard/jupiter/prepare_qwen3_6_flashqla.sh`,
  `hpc/sbatch_rl/jupiter_qwen3_6_flashqla_smoke.sbatch`.
- GRPO: `hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml`,
  `hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo.sh`.
- Evaluation/analysis: `hpc/datagen_yaml/qwen3_6_35b_a3b_swebench80k_jupiter.yaml`,
  `hpc/harbor_yaml/eval/configs/eval_opencode_apptainer_qwen3_6_swebench100_ctx80k.yaml`,
  `scripts/analysis/compare_paired_swebench.py`,
  `scripts/analysis/audit_r2egym_swebench_overlap.py`.
- **Deleted:** `dequantize_qwen3_6_fp8_for_grpo.py`, `jupiter_dequantize_qwen3_6_35b_fp8.sbatch`,
  `prepare_qwen3_6_35b_fp8_for_grpo.sh`.

**Execution order — REORDERED for the pipeline-validation goal**
1. **DONE:** push the two execution branches and reconcile Jupiter by fetch + hard reset, not pull.
2. **DONE:** raise the JURECA worker walltime to 24h (`6b25eb16`).
3. **DONE:** install the pinned FlashQLA layer, pass GH200 autograd job `1139108`, and stage the exact
   26-shard hub checkpoint verbatim.
4. **IN PROGRESS:** run the one-step GRPO smoke first. This is the milestone. It writes a separate export and does
   not modify the staged checkpoint, so it is safe to run before the baseline. Require: model load,
   real reward variance / nonzero advantages, finite policy update, weight sync, valid HF save.
5. **NEW resilience gate after smoke:** force a bounded Jupiter↔JURECA disconnect with 1-second
   keepalive detection, reconnect, prove workers and queued work recover, then restore conservative
   production values. Do not disrupt an active rollout.
6. Only if the smoke passes: promote to 50 steps, and run baseline SWE-Bench-100 in parallel.
7. Serve the selected post-GRPO export, run the matched arm, generate the paired report (raw counts,
   pp delta, per-task transitions, paired-bootstrap 95% CI, exact McNemar p).

## Operator-required changes

**A · DONE — training rollouts switched from terminus-2 to OpenCode.** Known Terminus2 issue; operator
call. The GRPO and eval configs now both use OpenCode 1.18.8. **This is not only a workaround — it removes a real confound:**
training the model against Terminus2's interaction format and then measuring it under OpenCode's is a
train/eval mismatch. Same agent on both sides is strictly better science.

The allowlist trap was fixed and contract-tested: `harbor_config.py` builds `AgentConfig(name=…, kwargs=…)`
through an **allowlist** of `FieldMapping`s. Keys absent from that table are silently dropped — this
is exactly the bug you already hit once, where `strict_json_parser` "had been dead configuration"
until Harbor was patched to forward it. Marin commits `7e938d8`, `e4a7f8a`, and `cd641b1` verify the
OpenCode fields arrive (`version`, `preinstalled`, prompt/config/model limits); Terminus-only keys are
absent.

**B · Baseline eval doubles as an eval-harness sanity check.** Compare our SWE-Bench-100 baseline
against Qwen's published SWE-Bench Verified figure — not to reproduce it, but to catch a broken
harness before attributing a low number to the model.
- It costs nothing extra: the baseline arm already produces the number.
- **It is a band, not a reproduction.** Different agent scaffold, our pinned n=100 subset vs the full
  500 (binomial SE ≈ ±5 pp at 95%), one attempt, no compaction, 81,920 ctx. A gap under ~10 pp is not
  distinguishable from sampling plus harness differences.
- **State the alarm threshold before running**, so it is a real gate and not post-hoc rationalisation.
  Suggested: if we land more than ~15 pp below the published figure, stop and audit the harness rather
  than proceed.
- Precedent for why: the GSM8K `pass_at_1 ≈ 0.45` plateau was a **format artifact**, not ability. A
  number that looks like capability is often the harness. → [[gsm8k_format_artifact]]

## Live state — 2026-08-04 10:30 KST

Supersedes the 2026-08-03 block below AND the earlier 01:00 KST version of this section, whose
"variance is the next measurement" framing has since been **answered**. Read this, then
[[gotchas]] § 2026-08-04 and [[decisions]] § 2026-08-04.

### The two headline results

**1. The 600s agent timeout was real, and is fixed.** `agent.override_timeout_sec = 1800` was
materialized but unreachable: OpenCode issues its long-running `run` via `exec_as_agent()` with no
`timeout_sec`, so it inherited `ApptainerEnvironment.exec()`'s hardcoded `timeout_sec or 600`; the
trial layer's `asyncio.wait_for(agent.run(), 1800)` is an OUTER guard that can only fire if it is
shorter. On `1221005`, 19 of 25 trials ended at exactly 600–601s and 19 of 19 exceptions carried the
literal `Command timed out after 600s`. Fourth accepted-but-ignored-key bug. Fixed in harbor
`f44a1170`; proven in isolation (`sleep 700` returned 0 after 703s) and live — **39% of trials now run
past 600s and score honestly.**

**2. ⛔ THE BAND IS REQUIRED. This REVERSES the 01:00 reading.** With the budget restored, variance
STILL did not appear. `1229343` produced 18 honest rewards: **0 of 6 groups varied**, and the same
tasks returned the same outcomes as under the cap —

| task | 600s cap (`1221005`) | 1800s (`1229343`) |
|---|---|---|
| 04114 | 1,1,1,1 | 1,1,1,1,1 |
| 01068 | 1,1,1,1 | 1,1,1 |
| 01755 | 1,1,1 | 1,1 |
| 05999 | 0,0,0,0 | 0 |
| 06360 | 0,0,0,0 | 0,0,0 |
| 07753 | — | 0,0,0,0 |

Five tasks agree perfectly across independent runs at 3× the budget; agent times now span 140–862s
with **nothing at the 1800s ceiling**, so the agent is no longer time-starved. Both things are true at
once: the old measurement WAS invalid, and its conclusion survives anyway.
⇒ **Next action is a ~384-task `p@4` probe** (1,536 rollouts, ~1.7h at 4 rollout + 8 JURECA nodes).
Caveat: 6 PARTIAL groups on the same 6 task ids both runs sampled — strong evidence about *these*
tasks, not a population estimate. That is what the probe is for.

### ⛔ `/e/scratch` IS FULL — read before touching the cluster

**It cannot create ANY new file.** Project-wide 8M inode soft limit shared by 26 users, exhausted.
`touch` → `Errno 122`; overwriting an existing file still works, which is why it presents as a
mid-run crash rather than a startup failure. It killed `1221005` AND `1229343`, and it breaks
`git fetch` (`unable to create temporary file`), so **deploys into `/e/scratch` are impossible.**

⛔ **CORRECTED 2026-08-04 (later): `/p/scratch` IS NOT MOUNTED ON JUPITER COMPUTE.** `45572f93` moved
the tree to `/p/scratch`; **`ae180c37` reverted it to `/e/fscratch`** because compute cannot see
`/p/scratch` at all (login-node visibility ≠ mount coverage — same exists-vs-works trap as the route
gate). `31cd646d` also moved `WANDB_DIR` off `/e/scratch`: offline W&B creates a new run dir per job.
Verified live on `1229488`. The table below is the CORRECTED one.

| path | now lives on | why |
|---|---|---|
| experiments tree, checkpoints, HF exports, `WANDB_DIR` | **`/e/fscratch/reformo/lee27/experiments`** | inode-cheap (80k/8M), 2.51 GiB/s, and MOUNTED ON COMPUTE |
| execution checkout (`DCFT`) | **`/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next`** | `git fetch` needs to create objects |
| read-only 67 GiB model | `/e/fscratch/reformo/lee27/models/...` | per-stream BANDWIDTH, not inodes |

⚠ `/e/fscratch` retention/purge is UNDOCUMENTED ⇒ the HF hub upload, not the on-disk copy, is what
makes a checkpoint durable.

⚠ Still on `/e/scratch` and fragile: the `rl-megatron` venv, harbor, MarinSkyRL, FlashQLA, and
`TILELANG_CACHE_DIR`. Reads work; any write through them can fail. The 08-03 dead-tree cleanup remains
UNAPPROVED and unresolved — `/p/scratch` routes around the problem rather than fixing it.

### Run ledger

| run | config | outcome |
|---|---|---|
| `1221005` | 16×4, 600s cap | quota abort; 25 rollouts, 0/7 groups varied — **invalid, truncated** |
| `1225422` | 16×8, 1800s | FAILED 3h12m, **zero rewards** — reverse tunnel still pointed at the previous head; my omission |
| `1229343` | 16×8, 1800s | FAILED 58m on the `/e/scratch` quota — but produced **18 honest rewards**, the data above |
| `1229438` | 16×8, **6h** | cancelled — 6h could not fit before a hidden maintenance window, deferred to Aug 5 12:00 CEST |
| `1229446` | 16×8, **4h** | superseded before running; relaunched as the `…20260804f` run below |
| `1229488` | 16×8, 4h, all paths on `/e/fscratch` | **FAILED at 1h46m** on an unenumerated `AddTestsDirError` — cleanest run yet, killed by the fail-loud footgun. See below. |
| `1229643` | 16×**4**, 3h | **FAILED at 1m51s** — Ray's `ray_spill` dir still on inode-dead `/e/scratch`. Fixed `aab498d7`. |
| `1229649` | 16×**4**, 3h | **the live milestone retry** — both fixes in, watcher + monitor armed |

### Run `1229488` — 2026-08-04. Best startup + best reward data yet, then aborted by one bad trial

⛔ **OUTCOME: FAILED (exit `1:0`) at 1h46m, holding 68 honest trials and 3 complete uniform n=8 groups.**
A single unenumerated exception (`AddTestsDirError`) fell through `default_error_treatment: mask` and
`fail_on_infrastructure_error: true` aborted the entire batch — the hazard § "Remaining structural
hazards" item 3 predicted word-for-word. Fixed in OT-Agent `c15ac55f`. Full autopsy in
[[gotchas]] § "THE FAIL-LOUD FOOTGUN MATERIALIZED".
⚠ **The milestone (update / checkpoint / HF export) STILL has not executed.** Generation was ~53% done
and on pace to finish with ~75 min of margin, so this was lost to a config footgun, NOT to time.

Allocated 10:43:42 KST on 5 nodes `jpbo-026-[36,40,44-45,47]`, head `jpbo-026-36` / `10.128.25.20`.
**Startup, allocation → weight-sync-ready: 31 min**, no `110-minute` regression (2nd consecutive clean run):

| phase | duration |
|---|---|
| 26 shards × 4 engines from `/e/fscratch` | **~65s** (vs `701.12s` on `/e/scratch`) |
| vision-tower profiling + KV + warmup | ~16 min ⇒ now **~95% of startup**; `MM_LIMIT=0` still UNSET/unmeasured |
| `load_checkpoints` (`resume_mode=latest`, nothing to resume) | 0.00s |
| `init_weight_sync_state` | 15.35s |
| `sync_weights_to_inference_engines` (35B MoE) | **300.73s** — the historical 5.9s was Qwen3-8B, not a regression |

✅ **Compute-node route gate PASSED** — a real `/v1/chat/completions` from a compute node returning
`"model": "995ad96e…"`. This is the step whose omission killed `1225422`; the watcher now automates it.

**⛔ REWARD RESULT — the band question is now ANSWERED, not merely evidenced.** Final state at abort:
**68 honest trials, 11+ groups, 0 with within-group variance**, and — the part that settles it — **THREE
COMPLETE n=8 groups, all perfectly uniform**, at the exact group size we train at:

| task | complete n=8 group |
|---|---|
| `01068` | `1,1,1,1,1,1,1,1` |
| `01755` | `1,1,1,1,1,1,1,1` |
| `05999` | `0,0,0,0,0,0,0,0` |

Every earlier zero-variance claim rested on PARTIAL groups of 2–5, which is why it was always arguable.
Two saturated at 1 and one at 0 — both extremes of the failure mode. Outcomes also reproduced across
three independent runs at two different agent budgets, and the invariance now spans **11 distinct tasks**
(new: `00222`, `02033`, `03150`, `05359`, `06171`) rather than the original six.
⇒ **A band (or DAPO `dynamic_sampling`) is a PRECONDITION for any real training run, not an
optimisation.** The 384-task `p@4` probe is now about sizing the in-band set, not about testing whether
one is needed.

**Other measurements from `1229488`:** 44% of trials ran past 599s (up from 31% at n=16) — the timeout
fix keeps earning; `n_cache_tokens: sum=0` across all 32/68 trials sampled at ~130k input tokens each
(consistent with prefix caching inactive, still not proof); rollout throughput **0.95–1.03 trials/min**
at 32-way concurrency; dispatch fills **8 groups at once**, not 4; input tokens saturate ~145k with one
runaway trial at **452k** (3×), which is the likely source of the 9 context overflows.
⚠ Still a small task sample — the ~384-task `p@4` probe is what turns this into a population claim.

**Bridge errors are NOT a training-impact signal.** `jobs_errors` rose +89 over baseline in two bursts,
each coinciding with `active` env count dropping (teardown), while **32/32 trials scored honestly with
ZERO infrastructure exceptions**. Only `NonZeroAgentExitCodeError` ×6 appeared, which is `passthrough`
by design. Read the counter as a DELTA and cross-check against `exception_info`, never alone.

**Other measurements:** 44% of trials ran past 599s (up from 31% at n=16) — the timeout fix keeps
earning; `n_cache_tokens: sum=0` across all 32 trials at ~130k input tokens each; rollout throughput
**0.95 trials/min** at 32-way concurrency, and dispatch fills **8 groups at once**, not 4.

### Shipped this session

| commit | repo | change |
|---|---|---|
| `f44a1170` | harbor | `BRIDGE_EXEC_TIMEOUT`; default still 600; 4 tests |
| `6b76270b` | OT-Agent | vLLM canary reads shards from a chosen filesystem |
| `2b6defd1` | OT-Agent | exec timeout 2100, n=8, fscratch model, `-next` checkout |
| `8ac43278` | OT-Agent | `MM_LIMIT` / `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` knobs, defaults unchanged |
| `45572f93` | OT-Agent | experiments tree → `/p/scratch` (the inode fix) |
| ai_memory | local only | corrections + gotchas, not pushed |

**Shipped 2026-08-04 afternoon (all pushed to `lukedhlee` forks, deployed + verified on-cluster):**

| commit | repo | change |
|---|---|---|
| `c15ac55f` | OT-Agent | `fail_on_infrastructure_error: false` + `AddTestsDirError` masked — one bad trial no longer aborts the batch |
| `be949008` | OT-Agent | canary 16×8 → **16×4** to fit the 3h wall the starved cluster allows |
| `aab498d7` | OT-Agent | `ray_spill` → `/e/fscratch` (killed `1229643`); same fix in the 6-node config; test updated |
| `fafab77` | MarinSkyRL | null-content partial response (`accum.content += None`) → **⛔ CANNOT DEPLOY**, see below |

⛔ **MarinSkyRL fork fixes cannot reach the cluster.** Its checkout is on `/e/scratch`, so `git fetch`
fails with `unable to create temporary file: Disk quota exceeded`. **Relocate
`/e/scratch/reformo/lee27/MarinSkyRL-apptainer-bridge` to `/e/fscratch` before any MarinSkyRL fix is
needed on a critical path.** Deployed MarinSkyRL is still `8ee69f3`; harbor is `f44a1170`.

**Measured:** `Loading weights` **701.12s → 38.03s** (1 GPU, 18.4×) / **61.6s** in-job (11.4×) by
staging on `/e/fscratch`. `/e/scratch` 0.056 GiB/s per stream vs fscratch 2.51 (`O_DIRECT`, compute
`jpbo-018-14`, job `1224678`); vLLM reads the 26 shards **sequentially in one stream**, so it sat on
that floor. ⚠ Startup is now ~95% **vision-tower profiling** (~920s, *"1 image items of the maximum
feature size"*) for a tower SkyRL unwraps and no rollout uses — `MM_LIMIT=0` is staged but UNMEASURED.

### Band evidence to date — READ THIS BEFORE CLAIMING ANYTHING ABOUT THE BAND

**There has been NO dedicated `pass@8` (or `p@4`) band probe.** Every number below is a *byproduct* of
GRPO training rollouts on whatever 16 prompts the dataloader served. Nobody has computed per-problem
pass@k as a filter. The 384-task probe (Open #1) has never run.

| run | geometry | groups observed | with within-group variance |
|---|---|---|---|
| `1219434` | 32 × **2** | 15 | **0** |
| `1221005` | 16 × 4, 600s cap | 7 | **0** ← INVALID (agent truncated) |
| `1229343` | 16 × 8 | 6 (18 rewards) | **0** |
| `1229488` | 16 × 8 | 11 (68 rewards) | **0** |

**Distinct problems with named reward evidence: 11** — of 3,328 tasks, i.e. **0.33%**. Zero varied.
Three of those were COMPLETE n=8 groups: `01068` all-1, `01755` all-1, `05999` all-0.
Known task ids: `00222 01068 01755 02033 03150 04114 05359 05999 06171 06360 07753`.

⚠ **Two caveats that matter more than the counts:**
1. **The sample is NOT random.** The same six ids recur across independent runs because the dataloader is
   deterministic — that recurrence evidences fixed ordering, NOT broad sampling. We know essentially
   nothing about the other 99.7% of the dataset.
2. **`1219434`'s 15 groups were n=2**, where a uniform pair is unremarkable at any p. Those 15 carry far
   less weight than the raw group count implies.

⇒ Defensible claim: **these particular tasks are saturated (0/11, incl. 3 complete n=8 groups).**
NOT defensible: any statement about the dataset's in-band fraction. That is exactly what the probe is for.

### Verified 2026-08-04: nothing filters or drops uniform groups on our config

Checked because it decides whether a zero-variance run can still reach the milestone:
- `advantage_estimator=grpo` → `compute_grpo_outcome_advantage` (`ppo_utils.py:1676`). A uniform group
  gives `(r − mean)/(std + 1e-6)` = **exactly 0**. Nothing is dropped; batch stays full size.
- `rloo_n_filter_zero_reward_groups` belongs to the **`rloo_n`** estimator — **not ours**. [[handoff]]'s
  older BLOCKER 1 text attributing the no-gradient outcome to that filter is wrong for this config.
- `handle_replace_sampling`'s `bad_uids` (zero-std groups, `trainer_utils.py:378`) is reached only via
  `handle_dynamic_sampling`, called at `trainer.py:480` **only if `dynamic_sampling.type is not None`** —
  and `fully_async_trainer.py:348` REQUIRES it to be `None`. So it never runs for us.
- Saving is gated only on step intervals (`fully_async_trainer.py:801-814`), not on gradient magnitude.

⇒ **A zero-gradient step still executes the update, checkpoint and HF export.**
⚠ **But it is a WEAK validation:** loss 0 ⇒ grads 0 ⇒ `optimizer.step()` is a numerical no-op, and the
exported checkpoint would be byte-identical to the input. It does NOT prove gradients flow through the
MoE experts, that FSDP2 reduces, or that AdamW updates anything.
⚠ **`n_samples_per_prompt: 1` is NOT a fix** — it passes every `validate_cfg` assertion and yields a
nonzero advantage (`len==1` → mean 0, std 1 ⇒ advantage = raw score), but a group of one has no
group-relative baseline, so it is **not GRPO** — it is REINFORCE with no baseline, and on binary rewards
it only ever reinforces successes. Do not report such a run as a GRPO step or reuse its checkpoint.
**Better idea on the table (operator call):** validate the update on **GSM8K**, where `p@1 ≈ 0.45` makes
a uniform n=8 group a ~1% event, so real GRPO with a real nonzero gradient — and being non-agentic it
removes the bridge, tunnel, JURECA fleet, sandbox and verifier from the failure surface entirely. Needs
a new YAML (existing GSM8K configs target Qwen3-30B, and the clean branch excludes GSM8K).

### Live resources

*(updated 2026-08-04 22:50 KST / 15:50 CEST)*

- **Jupiter:** `1229649` **RUNNING** — allocated 15:14 CEST on `jpbo-016-[01,06,09,12-13]`, head
  `jpbo-016-01` / `10.128.18.209`; 5 nodes, **3h** wall, **16×4 = 64 trajectories**. At 33 min it is
  through shard load and into RolloutCoordinator startup; `scored=0/64`, generation imminent.
  ✅ **Tunnel retargeted (MANUALLY) and the compute-node route gate PASSED** — a real
  `/v1/chat/completions` from JURECA compute node `jrc0502` returning `"model": "995ad96e…"`.
  ⚠ Slurm's `--test-only` had estimated a 23:19 start; it actually allocated in **~6 minutes** via
  backfill. Queue, don't wait.
  `1229488` (FAILED 1h46m) and `1229643` (FAILED 1m51s) are both closed — see the run ledger.
- **⛔ booster is RESOURCE-STARVED, not maintenance-blocked.** `sinfo`: **147 nodes `drain*`, 8 `down*`,
  1 `idle*`**. `sbatch --test-only` returned the SAME estimate (`2026-08-04T23:19`) for every walltime
  from 30 min to 3h, and `2026-08-05T12:00` for ≥3h30m ⇒ **3h is the largest wall that starts tonight**,
  which is why the geometry is 16×4. A FLAT estimate across walltimes means no free nodes; shortening
  buys nothing. ⚠ The estimate is conservative — `1229643` was estimated 23:19 and actually started in
  **6 minutes** via backfill, so QUEUE rather than wait.
- **JURECA (measured 15:47 CEST):** `15495516` **11h55m left** (sufficient — outlives the 2h27m job) +
  `15494122` **~1h left** (will expire mid-run). 2 nodes × 16 each. When `15494122` dies its 32 workers
  vanish but `15495516`'s 32 remain, and `n_concurrent_trials: 32` is still satisfiable, so this is
  survivable rather than fatal. ⚠ **Re-verify fleet TTL against the job's ACTUAL start, not submission
  time.** ⚠ Do NOT sort `TIME_LEFT` as a string — `1:03:58` sorts above `11:55:35`, which is how the
  route gate ended up running against the *expiring* fleet instead of the durable one (harmless there,
  but the wrong fleet).
- **⛔ THE WATCHER FAILED ON `1229649` AND I RETARGETED BY HAND.** `retarget_job.sh` allocated, then hit
  `Control socket connect(...cm_jureca/qwen36): Connection refused` → `FATAL: could not add forward` and
  exited **without retargeting the tunnel** — the exact omission that killed `1225422`, reintroduced by
  the automation meant to prevent it. **The ControlMaster was healthy throughout** (`ssh -O check` →
  `Master running (pid=185890)`, up 2d3h; bridge `worker_polls` climbing 145k→215k). The watcher's own
  fleet-check loop opened one ssh session PER FLEET and exhausted the master's channels.
  It also printed a FALSE `⛔ no fleet outlives this job` while `15495516` had **11h55m** left — it
  treated an empty query as insufficiency.
  **Both bugs fixed and redeployed** (retarget now runs BEFORE diagnostics with 5×20s retries; fleet
  check is one session and reports "cannot tell" on an empty result). Not currently running — this run's
  tunnel is already correct. → [[gotchas]] "the watcher BROKE THE THING IT GUARDS".
  **Lesson: verify a detached watcher's RESULT, not that you launched it.** `exit 1` in a background
  script is invisible until you read its log.
- **Two independent Jupiter ControlMasters** (`login02` pid 55008, `login01` pid 68552), both
  `BatchMode`-usable. Poll from **login01** and keep interactive calls on **login02** — JSC allows ~one
  session channel per mux, and sharing one node's channel caused two `Session open refused by peer`
  lockouts. The bridge at `10.128.1.2:9920` answers from BOTH login nodes (verified 200 from login01),
  so a dead login02 *mux* is recoverable; a dead login02 *node* is not (the bridge process lives there).
- **Bridge** `10.128.1.2:9920` idle (`ready: 0`) at `1229649`'s launch — the preflight HARD-REQUIRES an
  idle bridge and refused an earlier attempt (`envs.ready=7, active_jobs=6`). **Do not set
  `REQUIRE_CLEAN_BRIDGE=0`** to get past it; the bridge's own `cleanup_loop` reaps orphaned envs after
  `BRIDGE_STALE_READY_SEC` (900s), so after a crash just WAIT ~15 min.
  ⚠ `jobs_errors` is **not** a training-impact signal: it reached +184 over `1229488` while 68/68 trials
  scored honestly with ZERO infrastructure exceptions, because the same `cleanup_loop` increments it as
  bookkeeping. Read it as a DELTA and cross-check `exception_info` in the trial results.
  ⚠ The bridge keeps **NO persistent log** — its stdout is a tmux pane (`readlink /proc/950979/fd/1` →
  `/dev/pts/2`), so bridge error history is unauditable. Fix at a restart with no envs attached.
- **ControlMaster** pid 185890 alive 30h+. Reach JURECA ONLY as
  `ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de`.
- **Watcher:** `/p/scratch/reformo/lee27/retarget.sh` polls `1229446`, then on allocation retargets the
  reverse listener to the new head and runs the compute-node route gate automatically.

### Scheduling — a hidden maintenance window

`ReqNodeNotAvail,_Reserved_for_maintenance` appeared while **3,848 nodes were idle** and
`scontrol show res` showed no MAINT reservation. Cause: the walltime could not finish before a window
we cannot see. `sbatch --test-only -t <T>` is the free diagnostic: **6h → Aug 5 12:00 CEST**,
**≤4h → same evening**. Every walltime at or below 4h gave the same start, so shortening further buys
nothing.

### Open

1. **⛔ The band probe** — ~384 tasks at `p@4`. Full set would be 13,312 rollouts ≈ 14.5h at 4+8 nodes
   (~58h at today's 32-way), so it needs chunking + a done-task manifest; the probe fits one slot.
2. **Fleet↔job dependency is not robust.** `qwen36_grpo_preflight.py:34` checks `workers_alive` — a
   LIVENESS check used as a SUFFICIENCY check. It cannot see that a deferred job will outlive its
   fleet. Fix cheaply by asserting fleet `TIME_LEFT > walltime + margin`; fix durably by validating and
   enabling the **auto-chain** (`MAX_CHAIN`, `dependency=afterany:` — present in
   `jureca_workers.sbatch` but has NEVER crossed a real walltime boundary).
3. **Rollout throughput is prefill-dominated** — 150,223 in vs 3,377 out per trial (44:1).
   `max_num_seqs: 8` against measured KV headroom of 17× (32k) to ~36× (real 15k). Prefix caching is
   ON but vLLM warns it is **experimental for Mamba `align` mode**; effectiveness UNMEASURED and
   `n_cache_tokens: 0` is not evidence either way. Read vLLM `/metrics` during rollouts to settle it.
4. **Operator decision — DAPO `dynamic_sampling: filter`** targets the blocker as a flag but requires
   `colocate_all: true`, colliding with disaggregated-only. → [[decisions]] 2026-08-04.
5. **The 110-minute** shards→policy-init on `1221005` — NOT reproduced on `1229343` (~35 min).
   Plausibly GPFS metadata contention while `/e/scratch` hit the inode wall. Still unexplained.
6. **Token fidelity** (`literal.jsonl`) never enabled; required before trusting TIS ratios.
7. **The optimizer update, checkpoint and HF export have STILL never executed.** The milestone.

## Superseded live state — 2026-08-03

**Nothing is running on Jupiter (0 nodes).** JURECA sandbox workers `15489111` alive until
~18:50 CEST 08-03. Bridge pristine: 3542/3542 jobs, **zero errors**, all sandboxes reclaimed.

### ⛔ BLOCKER 1 (operator decision) — the learnable band is REQUIRED

**22 prompt groups measured across two runs; ZERO had within-group reward variance.**
`1219434` (32 groups × 2 samples): 15 groups, 0 with variance. `1221005` (16 × 4): 7 groups, 0 with
variance — the four *complete* n=4 groups were perfectly uniform (`[1,1,1,1]`, `[1,1,1,1]`,
`[0,0,0,0]`, `[0,0,0,0]`). Rewards are honest (verifier-scored, `exception_info: null`) and DO differ
across tasks, but GRPO forms advantage **within** a group: all-zero groups are dropped by
`rloo_n_filter_zero_reward_groups`, all-one groups carry zero advantage ⇒ **no gradient**. If per-task
pass probability were mid-range, four uniform n=4 groups is a ~0.02% event.
**Qwen3.6-35B-A3B is effectively deterministic per r2egym task.** The co-lead's model-specific band
(`0 < pass@8 < 1`) is required, not optional. It was PARKED 2026-07-29 ("enable r2egym first") —
r2egym IS enabled now and the parked risk has materialised in our own data. **Do not silently adopt
the band; it is an operator call.**

### ⛔ BLOCKER 2 (operator decision) — filesystem: inodes AND bandwidth

`1221005` died on `OSError: [Errno 122] Disk quota exceeded` on `/e/scratch` — **inodes, not bytes.**
`/e/scratch/reformo` is shared by **26 users** at 6.14M/8M (77%); `/e/project1/reformo` is at **98%**
(avoid). Our whole footprint is only ~470k (~8%), and today's three runs were 156 trial dirs
(~1.9k inodes) — **tidying our traces frees almost nothing.** Freed ~85k by deleting `envs/rl`
(77,743 inodes, verified permanently unusable: needs `libcudart.so.12`, Jupiter has only CUDA/13).
**Measured, and this is the big one:** `/e/scratch` reads at **104 MB/s**; `/e/fscratch` (flash) at
**2.3 GB/s — 22×**. 67 GiB ÷ 104 MB/s ≈ 11 min, which explains EVERY slow model load (7:38–15:21)
and probably the unexplained 110-min shards→policy-init on `1221005`. Not vLLM, and not the prefetch
flag (`1217881` already falsified that). **Recommend staging the read-only model + high-churn
`trace_jobs` on `/e/fscratch`** (80k/8M inodes, 17.9 GB/42.9 TB). ⚠ measured from a login node — verify
on compute; fscratch retention/purge policy is UNDOCUMENTED, so keep checkpoints/HF exports on
`/e/scratch`. `/p/scratch` (JUST) has ~3M free inodes and freshest accounting but is not
Jupiter-native ⇒ inode relief without a speed win. → [[gotchas]]

### Fixed today — all pushed to lukedhlee forks + deployed, NO PRs

| commit | repo | defect |
|---|---|---|
| `4fb4a158` | OT-Agent | `compaction.reserved` is DEAD CONFIG (opencode never reads it) → threshold via new `context_budget.client_window_tokens: 20480` |
| `179b31e9` | harbor | opencode never set `AgentContext.metadata` → SkyRL extraction `TypeError` aborted every batch |
| `1f38665f` | harbor | observations must be **user** turns; `role:"tool"` rejected by `utils.py:1943` |
| `90109474` | OT-Agent | `NonZeroAgentExitCodeError` → `passthrough`; canary 32×2 → 16×4 |

**All four CONFIRMED WORKING on `1221005`:** zero overflows, zero `Could not extract results`, zero
role errors, zero non-zero-exit aborts, 25 honest rollouts. Deployed refs: OT-Agent `-next` at
`90109474`, harbor at `1f38665f`, MarinSkyRL at `8ee69f3`.

### Remaining structural hazards

1. **TOKEN FIDELITY — gate before promoting past a smoke.** opencode records prose and tool calls as
   separate structured events and discards raw completion text, so reconstructed `all_messages` is
   **faithful in structure but approximate in tokens**. Adequate for pipeline validation; NOT adequate
   for trusting TIS importance ratios. Exact path = the literal recording proxy (`literal.jsonl` →
   `_parse_literal_proxy_log`), which has never been enabled here. **Enable before 50 steps.**
2. **The update / checkpoint / HF-export path has NEVER executed with this model.** Reaching a finite
   update is still the milestone.
3. **Fail-loud taxonomy is a footgun:** `mask` = infrastructure ⇒ **ABORTS the whole batch** under
   `fail_on_infrastructure_error`; only `passthrough` survives. Adding an exception to
   `mask_exceptions` makes it MORE fatal, not less. Classification is also name-based, not
   `isinstance`, so any new subclass must be enumerated.
4. **Cross-checkout coupling — FIXED for the canary path (`2b6defd1`).** Both
   `run_r2egym_qwen3_6_35b_grpo{,_canary}.sh` now default `DCFT` to `-next`, and the 5-node YAML's
   `prompt_template_path` points there. ⚠ Still to check: the **6-node** production YAML and
   `run_qwen36_live_canary.sh`, which may retain the stale `0f04b250` path.
5. **Forced tunnel reconnect recovery is still unproven** (bridge itself healthy: internal
   `10.128.1.2:9920`, workers polling, zero job errors).
6. ⚠ **SINGLE POINT OF FAILURE:** the Jupiter→JURECA ControlMaster (`~/.ssh/cm_jureca/qwen36`,
   pid 185890) carries the reverse tunnel as a `-R` forward. If it dies the tunnel dies and rollouts
   lose their endpoint; restoring it needs ONE interactive `ssh jureca` + TOTP that **only Luke can
   do**. Also note `MaxSessions` on the Mac→Jupiter master: too many concurrent background SSH
   sessions yields `Session open refused by peer`, which looks like an auth expiry but is not.

**Resolved former defects:** JURECA workers request 24h; execution branches pushed and Jupiter
checkouts match local ground truth; the active tunnel is the reviewed internal-only path; the
`RL_ENV_DIR`-before-activation ordering is correct in rendered sbatch (export line 107, fallback 161);
TorchTitan EP present. Do not resurrect the stale public-bind script or the historical dirty checkout.
⚠ The pinned FlashInfer AOT artifact is **x86-64 and Jupiter is aarch64** — never enable
`59e661e0`'s artifact here; all four AOT vars are removed from the Qwen configs.

## Invariants & constraints

- **Never `scancel` a RUNNING job without explicit user OK.** Standing exception: experiments *we*
  spawned on Megatron/Jupiter may be cancelled. Never cancel anyone else's job.
- **JSC jobs are pre-authorized for this work, including RL.** State the key config, then go; combined
  active/pending allocations for this work must never exceed **16 nodes**. Destructive actions still
  need explicit approval.
- **Autonomy:** on smoke/bring-up failures, keep fixing + relaunching without waiting for "go fix it".
  Report when blocked (e.g. MFA) or when a milestone lands.
- **Git (HARD):** never `git push`, open a PR, or merge without an explicit ask **in that turn**. Push
  only to `lukedhlee`-owned repos. Cluster-local patches are fine for unblocking; upstream PR is a
  separate opt-in step. An unmerged fork fix rides `--harbor-ref` / `--skyrl-ref`.
- **Local clones are ground truth; clusters never diverge.** Edit locally → push → cluster `git fetch`
  + explicit hard reset to the pushed execution ref. Never `git pull` a divergent execution tree; no
  hand-editing or patch-by-rsync. Historical dirty Jupiter trees are out of scope and never executed.
- **Never `find`/`du` on Jupiter GPFS or JURECA scratch** — inode exhaustion can get the whole project
  banned.
- **Daytona snapshot caps are HARD** (10 new/launch, 60 org) — clean stale, never raise. Always pass
  `--api-key-env DAYTONA_API_KEY` to the snapshot manager (the default is a different org).
- `enable_db_registration: false` in YAMLs. **≤6 RUNNING RL jobs per cluster.**
- **Secrets** only from `$DC_AGENT_SECRET_ENV`; never commit; never source someone else's keys. A
  committed key is a leak — rotate, don't fix-forward. → [[vista_secrets]]
- **Compute nodes have no internet** on every JSC system — stage on login/JUDAC; WandB offline → sync
  from login. → [[jupiter_wandb]]
- **Don't call a big job wedged too early**, and don't "fix" MoE weight sync by disabling
  `SKYRL_W13_RELOAD_BRACKET` (token salad). Both, plus ~30 other failure modes → [[gotchas]].
- Full working agreement (how Luke plans, PR procedure, cluster index) → `notes/operating-preferences.md`.

## Decisions in force

One line each; full context, rejected options, and consequences in [[decisions]].

- **This run's purpose is PIPELINE VALIDATION, not model quality.** The milestone is the one-step GRPO
  smoke with real reward variance. *(2026-07-31 eve)*
- **Pushing to Luke's own repos/forks needs NO per-turn ask; opening a PR or merging still DOES.**
  Standing operator grant. Still secret-scan before pushing (the OT-Agent fork is PUBLIC), and deploy
  to clusters by fetch + hard reset, never rsync or hand-edit. *(2026-08-03)*
- **`compaction.reserved` is dead config; the OpenCode compaction threshold moves via
  `context_budget.client_window_tokens`.** Advertise a SMALLER window to the client (20480) while the
  server keeps 32768/28672/4096. Rejected: trusting a key merely because it appears in the
  materialized TrialConfig — verifying a key arrived proves nothing about whether the consumer reads
  it. Third such key after `strict_json_parser` and `store_all_messages`. *(2026-08-03)*
- **Tool observations reach SkyRL as `user` turns, not `role:"tool"`.** `utils.py:1943` accepts only
  `user`/`assistant`; anything else raises and fail-loud aborts the batch. *(2026-08-03)*
- **`NonZeroAgentExitCodeError` is `passthrough`, not `mask`.** It is the NORM for OpenCode (present
  even on reward-1.0 trials), and `mask` ⇒ abort-the-batch. *(2026-08-03)*
- **Canary geometry is 16 groups × 4 samples (64 trajectories), not 32 × 2.** Same cost, reallocated
  toward within-group variance — the only kind GRPO consumes. *(2026-08-03)*
- **PENDING OPERATOR DECISIONS (do not decide unilaterally):** (a) adopt the model-specific learnable
  band, now empirically required; (b) move model + `trace_jobs` to `/e/fscratch` for 22× read speed
  and inode headroom. *(raised 2026-08-03)*
- **Model = `Qwen/Qwen3.6-35B-A3B` @ `995ad96e…`, the plain release.** FP8 reversed by operator; the
  dequantizer is deleted. The FP8 repo's `95a723d0…` hash does not carry across repos. *(2026-07-31 eve)*
- **The multimodal shell is unwrapped IN MEMORY (`SKYRL_QWEN3_5_VLM_UNWRAP=1`), not on disk.** Rejected:
  a pre-unwrapped 67 GB artifact + conversion script, to save a measured 3.6% read overhead. Vision is
  parked; it cannot affect learning. Export will be text-only — a publishing decision, deferred.
  *(2026-07-31 eve)*
- **Branch layout: a curated `lukedhlee/qwen3-6-r2egym-grpo` on current upstream, plus an archive
  branch.** `ai_memory` excluded from the clean branch; the old branch was 48% prose, 17% Vista/GSM8K
  that are both concluded. *(2026-07-31 eve)*
- **Config-file convention:** one config per (cluster × model-family × task). Never fork a config for
  something the launcher parameterizes — `--num_nodes`, `--model`, `--skyrl_override`. The four Vista
  YAMLs were 84% identical and should have been one. Retire configs when a workstream concludes.
  *(2026-07-31 eve)*
- GSM8K retired as an MoE-RL vehicle; keep the `strict` scorer if it's ever rerun. *(2026-07-29)*
- **Sandboxes = option D: bridge on a Jupiter login node, workers on JURECA `dc-cpu`, tunnel dialed in
  from JURECA. JUWELS ≈ JURECA.** A/B/C are closed. *(2026-07-29 eve)*
- **Current Qwen3.6 experiment: r2egym is the GRPO training set; the validated pinned
  SWE-Bench-100 is the paired pre/post performance evaluation.** This supersedes the 2026-07-29
  Qwen3-8B bring-up decision to use held-out r2egym as the only validation. The nonexistent second
  SWE-Bench-100 manifest remains parked; do not invent 200 tasks. *(2026-07-31)*
- **The learnable band is PARKED — operator call.** Enable r2egym first. *(2026-07-29 eve)*
  ⚠ **Its premise is retired (2026-08-04):** raw r2egym did NOT collapse for the co-lead, and the band
  gives no quality gain. Whether we need one is now an open measurement gated on the 600s-cap fix.
- For this run, report the exact model origin and **specified eval regime** (pinned SWE-Bench-100,
  81,920 total context, no compaction, DP4×TP1, 100×1) plus the paired uncertainty statistics.
  Formal broader ID benchmarks remain Mrinal's. *(updated 2026-07-31)*
- Bridge agent is **OpenCode**, not Terminus; tooling injected by **bind-mounting static binaries**.
  *(2026-07-28)*
- **Training-side decisions, all settled — re-read [[decisions]] before touching an RL config, don't
  re-litigate here:** disaggregated only (colocation falsified); `use_tis: true` and no R3;
  `cpu_offload=false` when memory allows; never add inference nodes to speed eval; larger geometry starts
  from scratch (FSDP2 ckpts can't reshard); Megatron is the PI's default but GSM8K understates it; NCCL
  debug must be file-backed; GRPO-vs-RLOO is a non-lever. *(2026-07-13 → 07-26)*

## Open questions

**Qwen3.6 paired experiment:**
- Sandbox enablement is closed: Terminus2, all seven r2egym SIFs, all 100 SWE-Bench SIFs, fail-loud
  verification, and 128-way scale are measured. Auto-chain remains unvalidated but is not required
  within the now-configured 24h worker walltime.
- **Immediate open action:** make venv activation self-contained/fail-fast, then launch a fresh smoke
  through the wrapper (not direct old-sbatch resubmission) and follow it through policy/ref load,
  FlashQLA engagement, weight sync, OpenCode/JURECA rollouts, reward variance, finite update, second
  sync, checkpoint, and HF export. Then run the bounded tunnel disconnect/reconnect test.
- The first smoke's observed reward distribution decides whether raw r2egym is trainable enough to
  promote. The learnable-band filter remains parked unless the operator changes that decision.
- **Load-bearing FlashQLA gate is closed:** version 0.1.2 passed GH200 BF16 forward/backward in job
  `1139108`. The full policy-model binding and first training backward remain part of smoke `1144362`.
- **Never recorded in `decisions.md`:** the original switch from `Qwen/Qwen3-30B-A3B` to Qwen3.6 has no
  decision entry, unlike every other pivot. Worth back-filling with its rationale.

**Parked / no longer on the critical path:** second 100-task SWE-bench manifest · the other 400
SWE-bench SIFs · 30B-A3B on A100-40GB at TP=4 (moot — no model runs on JURECA).

**Other:**
- **What fraction of r2egym failures are non-substantive** (malformed patch, out-of-budget) rather than
  capability? Must be measured before `avg_raw_reward` is trusted — this is the transferable lesson
  from the GSM8K artifact.
- **TP > 1 in the RL loop on Jupiter** — believed possible, never run. It is the only thing that could
  reopen colocation.
- **Weight-sync cost** is still ~43% of a 6-node step. Can it be reduced?
- Do unsupported Megatron checkpoint modes fail early with a clear message yet? (They should.) And are
  there regression tests for the checkpoint fixes?

## Map

**⚠ TEMPORARY:** `NEXT_SESSION.md` — takeover note for a fresh session (written 2026-08-03).
Condensed current state, the two pending operator decisions, and the traps worth not repeating.
**Delete it once absorbed into this file + `gotchas.md`.**

**How to write memory:** `memory-guide.md` — two-tier rule + triage test. Read when *writing* memory.

**Permanent, append-only:** `decisions.md` (every choice + context + rejected options — read before
re-litigating anything above) · `gotchas.md` (~30 symptom→cause→fix — **read before debugging anything
env-shaped**: CUDA/vLLM/flash-attn/apptainer/inodes/NCCL) · `logs/<date>_<topic>.md` (raw session
journals: 07-19/07-20 Vista, 07-22 Jupiter flash-attn + vLLM reload, 07-26 Megatron 6n + TIS/R3) ·
`weekly/2026-07-27.md` (plain-language Megatron week).

**Apptainer — the focus, read these first:**
- `notes/qwen3_6_35b_r2egym_grpo.md` — ⚠ PARTLY STALE (still describes the dropped FP8 conversion).
  The handoff § plan of record supersedes it. Still useful for: FlashQLA gate, six-node
  GRPO geometry, contamination result, matched SWE-Bench regime, execution order, and current blocker.
  This note currently lives in the clean integration worktree until that branch is integrated.
- `notes/apptainer_bridge_handoff.md` — operational state, exact commits, measured smokes, and the
  **§ "2026-07-29 (later)"** run scoping. **Read before touching the bridge.**
- `notes/apptainer_bridge_swebench.md` — why the bridge exists: x86_64-vs-aarch64, 1-snapshot-per-task,
  her architecture. **Read before proposing any SWE-bench val set.**
- `notes/jureca_marianna_setup_plan.md` — her stack + fork refs. **Read before messaging her.**
- `notes/jureca_ssh.md` · `notes/judac_ssh.md` · `notes/jureca_what_goes_where.md` ·
  `notes/jsc_paths_hazards.md` — JURECA access (the `from=` IP trap) and where files may/may not go.
- `notes/vista_moe_30b_sizing.md` · `notes/vista_moe_parallelism.md` — MoE sizing + EP/FSDP/TP rules.
  **Read before setting TP for the 30B serving smoke.**

**Queued RL work:** `notes/r2egym_apptainer_reference_impl.md` (**READ FIRST for anything r2egym** — the
co-lead already runs r2egym+apptainer+8B RL on Jupiter; has the harbor wiring, the SIF build path, a
leaked-credential flag, and — ⚠ **CORRECTED 2026-08-04** — the retired claim that raw r2egym collapses;
it does not, and the band is a compute lever, not a quality one) · `notes/r2egym_grpo_plan.md` (design, val-set decision, eval handoff — **read before
launching**) · `notes/jupiter_bringup_and_throughput.md` · `notes/jupiter_megatron_bringup.md` ·
`notes/megatron_vs_fsdp2.md` (**read before choosing a geometry/backend**) · `notes/moe_grpo_tis_r3.md` ·
`notes/jupiter_cluster.md` · `notes/jupiter_ssh.md` · `notes/jupiter_wandb.md` · `notes/vista_secrets.md`.

**Reference / historical:** `notes/operating-preferences.md` (working agreement) ·
`notes/gsm8k_format_artifact.md` (**read before citing any GSM8K number**) ·
`notes/vista_24n_sync_stall.md` (full Vista job record; only if Vista revives) · `notes/tacc_vista.md` ·
`notes/vista_moe_gsm8k_grpo.md` · `notes/vista_moe_milestone_plan.md` ·
`notes/jureca_grpo_vs_rloo_plan.md` (completed A/B + fairness table) ·
`notes/jureca_agentic_daytona_plan.md` · `notes/jureca_agentic_smoke_recap.md` ·
`experiments/vista_moe_gsm8k_tracker.md` · `scripts/` (`jureca_from_clause.sh` = JuDoor `from=` line,
`explore_marianna_jureca.sh`, `jupiter_monitor_r10.sh`, `vista_import_loop.py`).

---

## Overnight 2026-08-04 → 08-05 — the band probe exists and the concurrency ceiling is gone

**What changed strategically.** Marianna's band/raw comparison reframed the band from "a compute-efficiency
nicety" into **the fix for our actual blocker**. Her top-right panel: the raw pool sits at `avg_pass@8` ≈ 0.5
with wide scatter, the p@4-filtered band at ≈ 0.95. Ours has been at the tail of the raw pool where every
group is uniform — 11/11 tasks with named reward evidence, zero within-group variance, and 6 more complete
n=4 groups from `1229649` tonight (2 solved / 4 unsolved / 0 band). Zero variance is zero advantage is zero
gradient. The band is not a speed-up for us; it is the difference between a gradient and no gradient.

**Dataset accounting.** Her "4.5k" is `R2E-Gym/R2E-Gym-Subset`/`-Lite` = **4,578**. Ours is
`DCAgent/r2egym-patched-full-oracle` = **3,328** — a 73% subset of the *same* pool, keyed by V1 index
(`path` = `r2egym-v1-00005`, `00007`, …; gaps are tasks that failed patching/oracle validation). So a band
list keyed on V1 index or docker_image **intersects our pool directly**. If her 1.6k is uniformly spread,
expected overlap ≈ 1,600 × 3,328/4,578 ≈ **1,160 tasks**.

**The 32-way ceiling was never JURECA.** It was `max_num_seqs: 8` × 4 engines on the 35B, with
`n_concurrent_trials` and a 2-node × 16-worker fleet both sized downstream of it. Full curve and the three
stacked limits are in `gotchas.md`. Measured tonight on an isolated bridge: **768 concurrent sandboxes,
766/768 ready, p50 102s**, sub-linear. `--num-workers 48` works fine; the suid limit throttles *starts*, not
running instances.

**Why the probe is structurally safer than any training run we have attempted.** It uses
`examples.terminal_bench.entrypoints.main_tbench_generate` — vLLM engines plus one `generate()` call over the
shard. No policy model, no ref model, no optimizer, no FSDP, no weight sync, no checkpoint. `policy_pg` is
`None` (`policy_strict_spread_pg: false`), so no GPUs are reserved away from the engines. Critically, **there
is no batch gate**: each trial writes its own `trace_jobs/<task>/result.json`, so a shard that dies at 80%
still yields 80% usable band data. Every one of our nine training failures lost everything because the
optimizer update sat behind a 16-complete-group barrier.

**Two traps caught before they corrupted the band, both worth remembering:**
1. `BRIDGE_EXEC_TIMEOUT` defaults to **600s**, shorter than the 1800s agent cap, so the bridge kills the exec
   mid-task — measured at **76% of trials** on the 35B canary. In a *training* run that shows up as bad
   reward. In a *band probe* it is invisible and corrupting: truncated trials score 0, and tasks the model
   never got to finish get classified "too hard" and filtered out of the band.
2. `bridge_url` left as `${oc.env:APPTAINER_BRIDGE_URL,http://10.128.1.2:9920}` would silently attach the
   probe to the bridge serving the live 35B run if the env var were unset at hydra-resolution time. Hardcoded.

**Launcher bugs hit en route** (all fixed in the committed wrapper/generator, none pre-existing knowledge):
- `parse_rl_config` rejects the seven derived context fields; `3node_qwen3_8b_r2egym_grpo.yaml` predates that
  rule, so anything generated from it must declare `context_budget` instead.
- Omitting `--model_path` is an **AttributeError**, not a fallback: `construct_rl_sbatch_script` does
  `exp_args.get("model_path") or parsed.model.get(...)` and `ParsedRLConfig` has no `.model`.
- `OT_AGENT_RAY_LOG_DIR` points at `/e/data1/.../ot-baf/experiments/_ray_logs`, another project's tree;
  `ray_utils._start_node` mkdir's it and dies EACCES at 52s.
- `ssh -O cancel -R` needs the **bare `bind:port`** form to clear a previous job's forward; the full spec
  carries the old head IP and will not match, so the new forward fails with "remote port forwarding failed".

### Band evidence after 1229649 (2026-08-04) — still zero, and now bimodal

`1229649` contributes the cleanest sample yet: 16 complete groups requested, 86 trials written,
**68 with a readable reward and 18 masked (21% mask rate)**. Of the tasks reaching k>=4:

| bucket | tasks |
|---|---|
| always-solved (4/4) | 3 |
| never-solved (0/4) | 4 |
| **band (partial)** | **0** |

SkyRL's own metric agrees: `reward/avg_pass_at_4: 0.375`.

**Read this correctly.** A 0.375 pass@4 with a zero band does NOT mean "the model is too weak" — it means
the tasks are **bimodal**: each one is either reliably solved or reliably not, with nothing in between.
That is precisely the distribution a learnable-band filter is supposed to remove, and it is why the band
matters to us as a gradient prerequisite and not merely as a compute saving. A group with 4/4 or 0/4 has
zero within-group reward variance, hence advantage 0 under `grpo`, hence no gradient — no matter how many
steps we run.

Cumulative across all runs: **~18 distinct tasks with a complete group, 0 with within-group variance.**
Still a non-random sample (deterministic dataloader keeps re-drawing the same task ids), so this supports
"these tasks are saturated or impossible" and still does NOT license any estimate of the dataset's in-band
fraction. The 8B p@4 probe is what would license that.

## Live state — 2026-08-04 18:15 CEST / 05 Aug 01:15 KST

### Running: the learnable-band p@4 probe, 8 shards
| shard | job | port | nodes |
|---|---|---|---|
| 0–7 | `1235927 1235928 1235929 1235930 1235931 1235932 1235934 1235935` | 18130–18137 | 2 each (1 policy + 1 rollout) |

Started 18:11:46, **wall ends 22:11:46 CEST**. Model `g1_diverse_tezos_100k_8b` (Qwen3-8B SFT, the exact
checkpoint the co-lead's band/raw comparison used). Agent `terminus-2`. 416 tasks per shard × p@4 =
1,664 trials per shard, 13,312 total. 48 concurrent trials per shard = 384 total.
Sandboxes: JURECA fleet **15498197** — 32 × `dc-cpu`, `WORKERS_PER_NODE=16`,
`STAGING_BASE=/tmp/apptainer_staging`, on the **isolated bridge 9921** (NOT 9920).

### How to read the result
```bash
bash /e/fscratch/reformo/lee27/band_report.sh        # band fraction + per-shard trial counts
python3 /e/fscratch/reformo/lee27/rewardcheck.py     # rewards_ok / null + top exception types
# band task ids land in /e/fscratch/reformo/lee27/BAND_TASKS.txt
```
**Check `rewardcheck.py` FIRST.** Trials are durable per-trial, so the report is meaningful at any point —
but `scored=N` counts `result.json` files, not rewards. The first launch had 74 trials and **zero
rewards** (see the overlay-timeout entry in `gotchas.md`); trial counts looked perfectly healthy.

### Verified by behaviour before launch, not by existence
- vLLM serving on a shard head (`/health` 200 from the login node)
- JURECA **compute** node `jrc0462` → tunnel → shard: 200 on ports 18130/18131
- a real `/v1/chat/completions` with the exact served name `g1_diverse_tezos_100k_8b`, model generating
- 384 concurrent sandbox creates on the tmpfs fleet: **384/384 ready, p50 20s** (was p50 68s on `/p/scratch`)

### If a shard dies
Nothing special is needed — losing a shard costs only its own tasks, and completed trials are already on
disk. Relaunch one with:
```bash
cd /e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next
SHARD=band_probe_8b_p4_shard0Nof08 NUM_NODES=2 TIME_LIMIT=04:00:00 JOB_NAME=band_p4_sN \
  bash hpc/skyrl_standard/jupiter/run_r2egym_band_probe_8b.sh
```
then forward its port — **use a NEW port**, never a used one (`ssh -O cancel -R` cannot reclaim a port
left by a dead job; 18100 and 18120–18127 are burned).

### Not running / not done
- **The milestone.** `1229649` FAILED at 2:18:36 having reached the update and OOM'd in the backward
  (30.57 GiB alloc). Next step is a config fix on the logits memory, not more infrastructure work.
- `main_tbench_generate` (pure-rollout entrypoint) is broken in this fork — worth fixing in daylight,
  it would delete the policy/optimizer/FSDP/weight-sync from probe runs entirely.
- The MarinSkyRL `/e/fscratch` relocation is still undone, so `fafab77` (null-content fix) is stranded.

## Live state — CORRECTED, 2026-08-04 21:05 CEST / 05 Aug 04:05 KST

The 18:15 block above is superseded. Three relaunches happened after it, each fixing a distinct cause of
**null rewards** (full chain in `gotchas.md`): overlay-create timeouts on shared `/p/scratch` → Python vs
curl resolving the tunnel host differently inside the sandbox → thinking-mode turns exceeding
terminus-2's request timeout. terminus-2 was then abandoned for the night.

### Running now
| shard | job | port |
|---|---|---|
| 0–7 | `1236536 1236557 1236558 1236559 1236560 1236561 1236562 1236563` | 18160–18167 |

3 h wall (ends ~00:05 CEST). **Agent is OpenCode 1.18.8**, the canary's block verbatim — the only agent
configuration proven to run for hours against this bridge and tunnel. `n_concurrent_trials: 32` per shard
(what the 35B canary actually sustained), `client_window_tokens: 20480`, endpoint addressed as
`10.14.0.46` (IP, never `jrlogin05i`), bridge 9921, fleet **15498197** (32 × dc-cpu, 16 workers/node,
tmpfs staging).

### Check this FIRST, before anything else
```bash
python3 /e/fscratch/reformo/lee27/rewardcheck.py    # MUST show rewards_ok > 0
bash    /e/fscratch/reformo/lee27/band_report.sh    # band fraction, once rewards exist
```
If `rewards_ok=0` and `null>0`, read the top exception type it prints — every failure tonight was visible
there and nowhere else. `scored=N` counts files, not rewards, and looked healthy through all three
failures.

### Known-good facts to reuse rather than re-derive
- Endpoint must be the **IP** `10.14.0.46`. Python and curl resolve `jrlogin05i` differently *inside* the
  sandbox, and it varies BY NODE.
- Sandbox staging must be node-local (`STAGING_BASE=/tmp/apptainer_staging`): overlay create is 3.57 s
  there vs 18.06 s on `/p/scratch`, against a **hardcoded 60 s** timeout in `worker.py`.
- JURECA ports are **single-use**. Burned so far: 18100, 18120–18127, 18130–18137, 18140–18147,
  18150–18157.
- OpenCode requires `engine_init_kwargs: {enable_auto_tool_choice: true, tool_call_parser: qwen3_coder}`
  or vLLM rejects every agent request before generation.
- **Never `pkill -f` over SSH.** It killed my own session mid-chain tonight, which silently skipped a repo
  sync and a port bump and launched 8 shards on a stale config against poisoned ports — 16 jobs ran at
  once before I caught it. Use `tmux kill-session`, or `pgrep` to look without killing.

### Two questions for the co-lead, both cheap and both decision-changing
1. Her band task-ID list (V1 indices or docker_images). Our 3,328 is a 73% subset of her 4,578 keyed by
   the same index, so it intersects directly — one message versus ~10 h of generation.
2. Did her band run have **thinking enabled**, and which agent? The band is a property of the
   (model, agent, config) triple, so her ~35% is only comparable to ours if those match.

## FIRST REAL BAND DATA — 2026-08-04 21:40 CEST (shard 0 partial, OpenCode)

`rewards_ok=141, null=0` — the first probe trials in this project with readable rewards.

| bucket | tasks | share |
|---|---|---|
| always-solved (4/4) | 7 | 21.2% |
| never-solved (0/4) | 26 | 78.8% |
| **band (0<s<4)** | **0** | **0.0%** |

148 trials, 0 unreadable, **33 tasks with a complete k=4 group**, pass@4 = 7/33 = 21%.

**Why this is more than another zero.** Every prior zero rested on ≤11 tasks and was easy to dismiss as a
small non-random sample. At 33 tasks the contrast with the co-lead's ~35% becomes quantitative: if the
true in-band fraction were 0.35, then P(0 of 33 in band) ≈ 0.65^33 ≈ **6e-7**. So the band is genuinely
near-zero **for this (model, agent, config) triple** — 8B + OpenCode 1.18.8 + 20480 client window on
`r2egym-patched-full-oracle` — and ~35% is therefore NOT a property of r2egym alone.

**What that implies, stated carefully.** r2egym rewards are binary on the test suite, and at temperature
1.0 a task with genuine partial difficulty should sometimes vary across 4 samples. Seeing none says these
tasks are decisively easy or decisively hard *for this configuration*. Two readings remain open and the
probe cannot separate them yet:
1. the co-lead's band comes from a materially different agent/harness (her arm is terminus-structured,
   possibly with thinking and different windows), so her p@4 lands on a different difficulty curve; or
2. our harness truncates or mis-scores marginal attempts in a way that pushes them to a clean 0.

Caveat that still stands: shard 0 is a round-robin slice of a deterministic order, not a random sample.
The remaining 7 shards will take this to ~2,000+ tasks, which settles it.

## ⚠ RETRACTION AND THE REAL FINDING — 2026-08-04 22:40 CEST

**Retract the "0/197 band" number above as a statement about the dataset.** It is an artifact of the
agent, not a property of r2egym. Do not quote it.

### What the trajectories actually show
976 trials completed with **zero exceptions** and readable rewards, so the transport was finally healthy.
But:

| measure | solved | failed |
|---|---|---|
| median trial duration | 56 s | 52 s |
| **median trajectory messages** | **1** | **1** |
| median trajectory size | ~1.2 KB | ~1.2 KB |

Against a 1800 s cap, and ~32 min/trial on the 35B canary. One message per trajectory is not an attempt.
`agent/trajectory.json` shows why:

```json
"model_name": "vllm/g1_diverse_tezos_100k_8b",
"steps": [ { "step_id": 1, "message": "<think>\nLet me analyze the task..." } ]
```

The model emits reasoning as **plain text, never a tool call**. OpenCode gets no tool call, so the episode
ends after one step and the repo is never touched. The 18.8% "solved" are tasks that pass with no
meaningful action; the 81% never got a real attempt. **Zero within-group variance follows trivially** —
outcomes are decided by the task, not by any search — which is exactly why the band read as empty.

### The actual lesson: match the agent to the MODEL, not to the cluster
`g1_diverse_tezos_100k_8b` is a **DCAgent terminus-trained** checkpoint, so its output format is
terminus-structured. `tool_call_parser: qwen3_coder` + OpenCode expects Qwen3 XML tool calls it never
emits. I picked OpenCode because it was proven **on the 35B** — proven infrastructure, wrong model.
The co-lead's arm is terminus-structured for exactly this reason.

So **terminus-2 is the correct agent**, and its `Request timed out.` is the one real blocker left.

### Next step, narrow and concrete
1. Run terminus-2 again (config is in git at `a42199f5`, thinking off) and fix the timeout properly. Facts
   already established, do not re-derive: a standalone completion on the same port took **10.5 s / 434
   tokens, HTTP 200**; `/tokenize` and `/v1/models` both **404 in ~1 ms**; no short completion timeout is
   set anywhere in `harbor/llms`; `timeout` IS a passthrough kwarg in `lite_llm.py` (~L149, ~L186) —
   try threading it through the harbor agent kwargs and **verify by behaviour** that it is honoured.
2. Confirm terminus-2 produces multi-step trajectories before spending a wall: check
   `agent/trajectory.json` has **more than one step**. That is now a mandatory gate — it is the only check
   that would have caught tonight's failure, since rewards were readable and exceptions were zero.
3. JURECA fleet **15498197** (32 × dc-cpu, 16 workers/node, tmpfs staging, bridge 9921) is left RUNNING
   with ~21 h and is correctly configured — reuse it, do not resubmit.
4. Ports burned: 18100, 18120–18167. Start at 18170.

### Gate ladder for any agentic run, in order (each caught a distinct failure tonight)
1. `rewardcheck.py` → `rewards_ok > 0`, `null` small
2. `agent/trajectory.json` → **median steps > 1**
3. median trial duration in minutes, not seconds
4. only then trust the band numbers

### Last action of the night — terminus-2 + timeout=900, shard 0 only
Job **1236881**, port **18170**, 2 h wall, agent **terminus-2** (thinking off), with `timeout: 900` set on
the harbor block as a test of the `lite_llm.py` passthrough. Generation began 20:25:48 cluster time.
Fleet 15498197 still up (~21 h).

**Read the gate first, it decides everything:**
```bash
python3 /e/fscratch/reformo/lee27/trajcheck.py    # median_steps > 1  => GATE PASS
python3 /e/fscratch/reformo/lee27/rewardcheck.py
```
- **GATE PASS** → the timeout passthrough is honoured and terminus-2 works. Launch the other 7 shards:
  `bash /e/fscratch/reformo/lee27/launch_band_shards.sh 1 7 04:00:00` (bump the port base to 18180 first
  in that script; 18100–18170 are burned), then forward with
  `setsid nohup bash /e/fscratch/reformo/lee27/fwd_all.sh &`.
- **GATE FAIL with "Request timed out."** → the passthrough is NOT honoured; that is a sixth
  accepted-but-ignored key. Next candidate is threading the timeout through the generator's
  `timeout_multiplier`, or setting it inside the sandbox via litellm env (`LITELLM_REQUEST_TIMEOUT`).
- **GATE FAIL with one-step trajectories** → same failure as OpenCode; the agent is not driving the model
  at all, and the next move is to check what terminus-2's `strict_json_parser` does with this checkpoint's
  output format.

**Bridge congestion caveat for 1236881.** Cancelling 8 shards at once left the 9921 bridge with ~180 envs
in `stopping` and `active_jobs` pinned near 45 with `ready: 0`. `cleanup_loop` caps reaping per cycle by
design ("so a sudden flood of zombies doesn't overwhelm workers"), so the drain is slow and the new
shard's `/env/create` requests queue behind it — which is why 1236881 showed 0 trial dirs ~25 min after
`Starting batch generation`. It is congestion, not a new fault: `envs_created` kept climbing (7,124 →
12,076).

If the morning finds 1236881 with no trajectories, check `curl :9921/status` for a large `stopping` count
before assuming a bug. Cleanest reset when nothing is attached: restart the bridge
(`tmux kill-session -t bridge_9921`, then relaunch `server.py --host 10.128.1.2 --port 9921`, log to
`/e/fscratch/.../apptainer_bridge/server_9921.log`). **Do not restart it while a shard is running** — every
env operation goes through it.

Lesson: mass-cancelling agentic shards has a cost paid by the NEXT run. Cancel, then wait for
`stopping` to reach ~0 before launching again.

## RESULT of the terminus-2 timeout test (1236881) — the key is IGNORED

**`timeout: 900` on the harbor block is NOT honoured.** Trials still failed with:
```
exception_type: APITimeoutError
exception_message: Request timed out.
```
That is the **sixth** accepted-but-ignored key in this project (after `strict_json_parser`,
`compaction.reserved`, `store_all_messages`, `override_timeout_sec`, `hf_upload_mode`). It appears in the
materialized config and changes nothing.

**And the one-turn no-op is NOT agent-specific.** terminus-2 produced `median_steps=1, max_steps=1` exactly
like OpenCode — but the single "step" is the **prompt itself**, with no model turn recorded at all. So
under terminus-2 the trajectory is empty because the LLM call times out, and under OpenCode it was one
text response with no tool call. Two different mechanisms, same visible symptom. My earlier note saying
OpenCode was simply "the wrong agent for the model" is therefore only half the story: the timeout blocks
terminus-2 before the format question can even be asked.

### The one thing to fix first, before any more probe launches
Make terminus-2's LLM request timeout actually take effect, and **prove it by behaviour**. Established
facts, do not re-derive:
- a standalone completion on the same port, same time: **HTTP 200 in 10.5 s, 434 tokens** (thinking off)
- `/tokenize` and `/v1/models`: **404 in ~1 ms** (SkyRL implements neither; handled gracefully)
- no short completion timeout is set anywhere in `harbor/llms`
- `timeout` sits in two passthrough allow-lists in `lite_llm.py` (~L149, ~L186) yet does not reach the client

Candidates, cheapest first: (a) set it inside the sandbox as a litellm env var so it cannot be filtered by
harbor's kwarg plumbing; (b) trace where `lite_llm` builds the `AsyncOpenAI` client and see whether the
allow-listed `timeout` is dropped between the agent kwargs and the client; (c) check whether the timeout is
actually the *streaming* idle timeout rather than a total-request timeout, which would explain a 10 s call
succeeding while an agent turn dies.

~~Nothing is running now.~~ **Superseded — see below.** JURECA fleet **15498197** is up with ~21 h.

## RUNNING NOW: 1242066 — OpenCode + thinking OFF

**This is the live experiment.** Job `1242066`, port **18180**, 3 h wall, 2 nodes, agent **OpenCode
1.18.8** with thinking disabled. Tunnel forwarded to `10.128.18.17`. Bridge 9921 restarted clean
(`jobs_errors` back to 0 after clearing 164 `stopping` envs).

### Why OpenCode and not terminus-2
OpenCode's **transport was never broken**: 1,219 trials, zero exceptions, readable rewards, no timeouts.
terminus-2 cannot get a single request through (`APITimeoutError`, and `timeout: 900` is ignored). So the
proven transport is OpenCode's; the only defect was the one-turn episode.

### The cause of the one-turn episodes was MINE, not OpenCode's
Adopting the canary harbor block wholesale **discarded the thinking-off settings**, and that block carries
no `extra_body` at all — so Qwen3's template defaulted thinking **ON**. The model then spent its turn on a
~300-token `<think>` emitted as plain TEXT (not a 4096 truncation — the whole trajectory was ~1.2 KB) and
never produced a tool call, so OpenCode ended the episode after one step. terminus-2 received the
thinking-off treatment; OpenCode never did. Now fixed:

```yaml
interleaved_thinking: false
extra_body: { chat_template_kwargs: { enable_thinking: false } }
engine_init_kwargs: { enable_auto_tool_choice: true, tool_call_parser: qwen3_coder }   # still required
```

### The gate, and what to do on each outcome
```bash
python3 /e/fscratch/reformo/lee27/trajcheck.py     # now globs band_oc_s* ; median_steps > 1 => PASS
python3 /e/fscratch/reformo/lee27/rewardcheck.py   # now globs band_oc_s*
```
- **PASS** → launch the remaining 7 shards. Set the port base in
  `/e/fscratch/reformo/lee27/launch_band_shards.sh` to **18190** first (18100–18180 are burned), then
  `bash launch_band_shards.sh 1 7 04:00:00` and `setsid nohup bash fwd_all.sh &`. Wait for the bridge's
  `stopping` count to reach ~0 before launching, or the new shards queue behind the drain.
- **FAIL, still one step** → thinking was not the cause. Then the question is whether this checkpoint emits
  Qwen3 XML tool calls at all; inspect `agent/opencode.txt` for what it returned, and consider that a
  terminus-trained model may simply never produce OpenCode's format — in which case terminus-2's timeout
  becomes the only path and must be fixed.

---

## 2026-08-05 (session: gate resolved, OOM fixed) — the FAIL branch was right

### The gate FAILED, and the cause is a tool-call FORMAT mismatch, not thinking
`1242066` produced **182/182 steps with `median_steps=1`, `max_steps=1`**. Thinking-off **did** apply
(`reasoning: 0` tokens); the `<think>` tags are literal trained-in text, so that knob was never the lever.

Root cause, verified end to end:
1. The checkpoint emits tool calls as **bare JSON in the message body** —
   `{"name":"bash","arguments":{"command":"ls -la /testbed"}}`. Census over every step: **182 bare JSON,
   0 `<tool_call>`, 0 `<function=`.**
2. vLLM is served `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, which parses only the XML
   grammar. It never matches and **never logs** — the silent failure that hid this.
3. OpenCode gets one `type:"text"` part, zero tool parts, `reason:"stop"` → one step.
4. Verifier grades an untouched repo ⇒ reward is a function of the TASK, not the agent.

⚠ **The zero-variance symptom is now explained, not merely observed.** 46 task groups, **0 with
within-group variance**. The agent never acts, so reward cannot vary within a group.

⚠ **NEW: ~22% of r2egym tasks pass with zero work.** Grouped: **10/46 all-1.0, 36/46 all-0.0, 0 mixed.**
Those 10 verifiers pass on an unmodified repo. They put a free floor under any band number and can never
yield gradient — **exclude them from the band denominator.**

### The parser fix is written and validated — but NOT yet wired
`OpenThoughts-Agent/rl/tool_parsers/bare_json_tool_parser.py` (uncommitted). Validated offline against
**all 182 real captured outputs: 182 parsed, 0 missed**, 8/8 edge cases, streaming emits exactly 1 delta.
Why it was cheap — four stock parsers each miss by one detail:
`qwen3_coder`/`qwen3xml` want XML · `hermes` wants this exact JSON but wrapped in `<tool_call>` ·
`llama3_json` has the right logic but hard-requires `<|python_tag|>`, absent from the Qwen3 vocab, so it
raises at construction · `xlam` strips `</think>` and takes the same keys but demands a top-level ARRAY.
⛔ **Remaining gap:** `skyrl_train/inference_engines/vllm/utils.py::pop_openai_kwargs` forwards
`enable_auto_tool_choice` / `tool_call_parser` / `openai_sampling_params` but **NOT `tool_parser_plugin`**,
so the plugin currently has no path through. ~4 lines mirroring the existing `tool_parser` passthrough.

### MILESTONE: the backward OOM is FIXED (MarinSkyRL `637a764`)
⚠ **The old arithmetic in this doc was wrong.** It used the 8B's vocab for the 35B. The 35B's real vocab is
**248320**, nested in `config.json → text_config` (top level has only `image_token_id` etc.).
That reproduces the failure exactly: `30.57 GiB / 4 B / 248320 = S≈33046 ≈ max_seq_len`. The grad is
allocated in **fp32** because autocast promotes `log_softmax` — which is why a "bf16 branch" produced a
30 GiB fp32 tensor. Real budget for that one op: logits 15.2 + saved output 15.2 + fp32 grad 30.6 ≈ **61 GiB**.

The failing op is `_log_softmax_backward_data` / `LogSoftmaxBackward0` inside `logprobs_from_logits_v2`'s
bf16 branch. That function bounds memory by looping the **BATCH** dim — a no-op at
`micro_train_batch_size_per_gpu=1`.
⚠ **Sequence-chunking alone does NOT work**: `log_softmax` saves its output, so every chunk's output stays
live. The fix mirrors `_EntropyFromLogits`: save only logits, recompute softmax per chunk in backward.

GPU-validated on GH200 (jobs `1243216`, `1243229`):

| check | result |
|---|---|
| fp32 parity | fwd diff **0.000e+00**, grad 2.4e-07 |
| bf16 under autocast (**the training path**) | fwd **9.5e-07**, grad 3.1e-05 |
| seq not divisible by chunk | pass |
| peak @ S=33046, V=248320 | **chunked 33.41 GiB** vs stock **OOM "Tried to allocate 30.57 GiB"** (byte-identical to the original failure) |
| bf16 no-autocast "mismatch" | adjudicated vs fp64: chunked is **2.4x MORE accurate** — it was the reference's bf16 error |

**Off by default.** `SKYRL_CHUNKED_LOGPROBS=1` engages it and the call site logs
`[logprobs] chunked gathered log-softmax ACTIVE` **once** — verify by that line, never by the config.
Note the line only fires during a TRAINING step (~2h in, after generation), so `FIX=0` early is expected.

### Four operational traps that each cost a job or an attempt
1. **There are TWO MarinSkyRL clones.** A bare `python` imports
   `/e/scratch/reformo/lee27/MarinSkyRL`; the RL job prepends
   **`MarinSkyRL-apptainer-bridge/skyrl-train`** to `PYTHONPATH`. Sync/patch the BRIDGE clone, and give any
   standalone test the same `PYTHONPATH` or it silently tests the wrong code.
2. **`retarget_job.sh`'s `OLD_TARGETS` is STALE** — it lacks `10.128.18.209` (1229649's head), so that dead
   job's forward still held port 18000 and the retarget burned all 5 retries and exited FATAL. Cancel the
   previous head explicitly: `ssh -S ~/.ssh/cm_jureca/qwen36 -O cancel -R 10.14.0.46:18000:<oldIP>:8000 …`
   **Always cancel a job's forward when it dies, or the port is burned.**
3. **Submit RL from the repo root with `DCFT` set** — `1243248` FAILED instantly with
   `FATAL: WORKDIR=... is not the OpenThoughts-Agent repo root`. Use
   `cd /e/scratch/reformo/lee27/OpenThoughts-Agent && export DCFT=$PWD && sbatch --chdir=$PWD …`.
   `sbatch --time=` and `--export=ALL,VAR=v` override the generated sbatch without editing it.
4. **Do not run Jupiter ssh calls in parallel** — the ControlMaster refuses sessions
   (`session request failed`) and you get a bogus TOTP prompt. This is the same channel exhaustion that
   broke the watcher on `1229649`. Serialize.

### Booster is no longer starved
`sinfo`: **2349 idle** nodes (was 147 drain / 1 idle). Jobs now allocate in seconds and longer walls are
available — the 3h geometry that constrained `1229649` is no longer forced.

---

## 2026-08-05 (late) — THE ROLLOUT PATH WAS THE BUG THE WHOLE TIME

### The finding that reframes the project
**Two reverse forwards ride the JURECA ControlMaster, and only one was ever documented.** Every rollout in
this project's history timed out because the second one (workers → bridge) was missing or dead. With both up:

| signal | every prior run | after the fix |
|---|---|---|
| trajectory steps | `median=1 max=1` | **`median=18.5 max=26`** |
| bridge timeouts | 32/32 | **0 of last 12** |
| reward spread | 0 of 46 groups varied | **8×0.0, 4×1.0 → 33% pass** |

**33% ≈ Marianna's ~35%.** ⇒ **A learnable band EXISTS on this task set.** The zero-variance wall was an
artifact of an agent that never ran, not a property of the model or the tasks. The "band may not exist"
risk is RETIRED.

The two forwards:
- `-R 10.14.0.46:18000 → <jupiter-head>:8000` — sandboxes → vLLM (this one was known)
- `-R 10.14.0.46:9923  → 10.128.1.2:9920`     — **workers → bridge (this one was not)**

The bridge is a **python3 process on the Jupiter login node** (`10.128.1.2:9920`); workers address it as
`jrlogin05i:9923`. The port pair exists ONLY in the fleet log
(`/p/scratch/synthlaion/lee27/dc_agent_eval/logs/apptainer_workers_<fleet>.out` → "Bridge URL") —
`BRIDGE_LOGIN` defaults to `jrlogin03i:9920` in the sbatch and is overridden at submit.
⚠ `workers_alive: false` **with live worker processes** = a missing bridge forward, NOT dead workers
(`pgrep -fc worker.py` distinguishes them). Recovery is instant — `active_jobs` jumped to 181 in seconds.

### Two SSH facts that made the master un-reproducible
1. **`-4` is mandatory.** `jureca.fz-juelich.de` resolves IPv6-first; the key's JuDoor `from=` clause rejects
   IPv6, giving `Permission denied (publickey)` with TOTP never offered. With `-4` the key reaches
   `Authenticated ... partial success` and JSC *then* prompts for the TOTP.
   **The long-standing note blaming "hostname resolution" is WRONG** and sent everyone down the wrong path.
2. **Pin `jureca05.fz-juelich.de`** (single A record 134.94.1.132). The tunnel must bind `10.14.0.46`, owned
   only by jrlogin05; the round-robin alias landed us on jrlogin10 and the forward failed with `NO_LISTENER`
   and nothing holding the port — indistinguishable from a burned port, but not one.

⚠ **Capture a dying master's forwards BEFORE killing it** (`ps -u $USER -o args | grep 'ssh .*-R'`). Killing
one without doing so cost ~an hour rediscovering forward #2.

### MILESTONE: machinery LANDED (`1243377`, COMPLETED exit 0:0)
First clean end-to-end run in the project: real optimizer update → `global_step_1` checkpoint (**235 GB**) →
HF-format export (**65 GB**, `policy/`). Preserved as `MILESTONE_1243377_{checkpoints,exports}`.
⚠ Its `grad_norm` was **0** — rollouts were still timing out, so the update is numerically vacuous. It
proves the machinery, not learning. The OOM fix (`637a764`) is confirmed working in production: the log
printed `chunked gathered log-softmax ACTIVE (chunk=1024, vocab=248320)` and execution continued past
`policy_train` into `train_critic_and_policy` / `run_training` — that *continuation* is the proof, since
`Finished: 'policy_train'` alone is logged from `__exit__` on exception.

### Measured throughput (supersedes all estimates)
| measured | value |
|---|---|
| real agent trial | **median 8.9 min**, p90 13.3, max 15.2 |
| trial that times out | ~62 min p90 → **a failure costs 7× a success** in slot-time |
| peak achieved concurrency | **32** = exactly the config cap |
| sandboxes READY while only 32 ran | **110** |

Full band = 13,312 trials; 10h ⇒ 22 completions/min ⇒ **~200 concurrent needed** (295 on p90) against
**512 fleet slots**. So band-in-10h is a **~6× concurrency bump, not more nodes**. An earlier "666
concurrent" figure was wrong — it used the 1800s agent *budget* as the trial duration.

### A near-miss worth remembering
I almost reported success on `grad_norm=1.0`. The only match in the log was **`max_grad_norm=1.0`** — the
clipping threshold from the config echo — and no training step had run at all. Same family as
`scored=N counts FILES`. **Anchor log patterns; a metric name that appears as a substring of a config key
will lie to you.**

### Launch-environment archaeology (three jobs died, one each)
- `1243248` — submitted from a scratch dir ⇒ `FATAL: WORKDIR ... is not the repo root`.
- `1243289` — `DCFT` pointed at `/e/scratch/.../OpenThoughts-Agent`, which lacks
  `hpc/shell_utils/flashinfer_aot_cache.sh`. The real WORKDIR is
  **`/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next`**, recoverable only from a successful
  run's log line `Working directory: …`.
- `1243351` — reached full Ray startup then died on `AssertionError: WANDB_API_KEY is required`. The assert
  is **unconditional** (ignores `WANDB_MODE=offline`). Secrets live at **`$SCRATCH/keys/secrets.env`**, not
  `~/secrets.env` as ops.md says.
`--time=` and `--export=ALL,VAR=v` override a generated sbatch without editing it.

### The 8B's one-turn no-op — root-caused and fixed (unwired)
`g1_diverse_tezos_100k_8b` emits tool calls as **bare JSON**, 182/182 steps, zero XML, while vLLM ran
`--tool-call-parser qwen3_coder` (XML-only) — never matched, **never logged**. Parser written and validated
against all 182 real captured outputs (182 parsed, 0 missed) + 8 edge cases + streaming:
OT-Agent `4add607c`; MarinSkyRL `d8bdc79` forwards `tool_parser_plugin` (a plugin must be *imported* so its
registration decorator runs, not passed as a kwarg). **Neither is synced to the cluster.**
Also: **10 of 46 groups score 1.0 while doing nothing** — those verifiers pass on an unmodified repo; a ~22%
free floor that can never yield gradient. Exclude from the band denominator.

### ~~`1244916` died to a NODE FAILURE~~ — **RETRACTED: I `scancel`led it (see below)**
> ⚠ **This section was written by the OUTGOING session at 17:52 KST, one minute after the incoming session
> `scancel`led `1244916` at ~17:51.** Two sessions overlapped in this repo. The `NodeDiedError` it saw is what
> Ray logs *when Slurm tears the job down* — it was the scancel's own aftermath, not a hardware fault. The job
> was verifiably alive immediately before: at 17:44 KST `squeue` showed it RUNNING with 4h28m left and the log
> was writing live timestamps. **There is no evidence of node death on Jupiter.** Do not shorten walls or plan
> for hardware failure on the strength of this. Original text kept below only so the retraction is legible.
>
> Authoritative evidence — `sacct -j 1244916 --format=State,ExitCode -X`:
> ```
> 1244916   CANCELLED by 34902   0:0
> ```
> uid `34902` is `lee27`, and exit `0:0` is a clean teardown. A genuine node failure records `NODE_FAIL`.

After the tunnels were fixed it ran 1h35m with a healthy route (0 bridge timeouts, median_steps 18.5,
33% pass) and then died: `ray.exceptions.NodeDiedError`, head node `10.128.32.219` (jpbo-045-27) dead
mid-generation, Slurm CANCELLED. No `grad_norm`, no checkpoint. So the pipeline is PROVEN to generate real
multi-turn rollouts with reward variance, but a non-zero gradient has still never been observed.
⇒ **a dead job's vLLM forward must be cancelled and re-added at the new head** — the bridge forward
(`9923 → 10.128.1.2:9920`) is Jupiter-side and survives. *(That part is correct and load-bearing.)*

**Standing lesson:** a job vanishing from `squeue` plus a `NodeDiedError` in the log is **not** evidence of
node failure — every teardown path produces both. Distinguish by `sacct` state/exit code, or by whether
anyone cancelled it. Attributing an intentional cancel to hardware is how a phantom failure mode gets
designed around for weeks.

⚠ Also re-triggered the **Jupiter-side** ControlMaster channel exhaustion by running one compound command
with several nested `ssh` calls (`Session open refused by peer` → spurious TOTP prompt → `Permission
denied`). The Mac→Jupiter master stayed alive; the remote sshd had simply run out of channel slots, and it
recovers on its own. **One remote call per `ssh` invocation.**

### Direction set by Luke at end of session
**Do NOT chase the full band.** Get the band **reliably on a SUBSET**, on the **8B Marianna used**, then
**extrapolate** to answer whether the system scales to a full band in 7–10h. Inference-only for now: it
removes training-side concerns and debugs ~10× faster (~9 min/trial vs ~1.5h/step). The parser must be
wired first or the subset re-measures one-turn no-ops.

## 2026-08-05 (evening) — goal reframed to REPRODUCING Marianna's band; two headline claims retracted

### The reframe (Luke, mid-session)
Not "get a learnable band within 10h" — **build scalable, trustworthy, functioning infra by reproducing
Marianna's band work.** Success is **matching her number**, because a target number is a far stronger
correctness test than a throughput estimate, and a deadline invites a *false* band.


> ⚠ **RETRACTED 2026-08-05 23:5x KST — "358 of 4,578 ≈ 8%" IS WRONG.** Marianna, asked directly, said the
> band is **~1.6k of 4.5k ≈ 36%**, and that her filtering pass cost `18k rollouts` (= 4.5k x 4, confirming
> pass@4 over a 4.5k pool). Where 358 came from is unknown; treat every "8%" / "358" below as void. This
> matters because it flips a verdict: our measured 0-in-band was "consistent with 8%" but is
> ~1-in-800,000 against 36%. Her run also used **terminus-structured**, not OpenCode.

**Her result:** `358` learnable tasks of the **4,578**-task r2egym pool at **`0 < pass@4 < 1`** ⇒ **≈8%**.
(Her shared script says `n_samples_per_prompt=8`; that is the TRAINING config — Luke confirmed the band was
built at p@4. Our pool, `r2egym-patched-full-oracle` = 3,328, is a 73% slice of hers.)

### Retraction 1 — "33% pass ≈ 35% ⇒ a learnable band EXISTS"
A **trial pass rate is not a band.** Re-measured over the 35B canary's whole trace tree (212 results):
31.6% pass rate, and **0 of 16 fully-sampled groups in band** (5 always-solved, 11 never-solved, none mixed).
GRPO's advantage is computed *within* a group, so an all-agreeing group contributes **exactly zero gradient**.
The inherited table put **GROUPS** in the before-column and **TRIALS** in the after-column; compared that way
any bimodal task mix looks like a band. `0.65^16 ≈ 0.1%` if the true band were 35%.
Her ~8% shows the band is genuinely small, so 0/16 at p@4 is *consistent* with it — not a contradiction.
⇒ Report the band ONLY as *fraction of fully-sampled groups with `0 < passes < n`*. `band.py` does this.

### Retraction 2 — "band-in-10h needs a ~6× concurrency bump, not more nodes"
Concurrency was **never** the binding constraint: we hit the configured cap instantly on every run
(32 → 64 → 128 sandboxes live, `envs.ready` matching). The wall is **per-trial latency**, from two things:
1. **`context_budget.max_turns: 999999`** ⇒ `max_episodes` unbounded ⇒ a trial can only end by hitting the
   1800s wall. **Marianna caps `max_episodes=50`.** Measured on `1246344`: **63 of 64 trials still running at
   38 min, 5 completions, 0.17 trials/min.**
2. **Heavy sandboxes** — ours 2 CPU / 4 GB / 4 GB vs hers **1 CPU / 1 GB / 1 GB**.

I spent hours treating **thinking** as the bottleneck. It is a contributor (4096-token turns vs ~40) but not
the cause, and chasing it was optimizing the wrong thing — she runs thinking **ON**. Sizing note: ~25 engines
saturate the 512 fleet slots ⇒ **~8 Jupiter nodes, not 16**; we had been using **2** of a 16 cap.

### Tool-call parsing: wired, verified in production, and `qwen3_coder` was the wrong DIALECT
g1's template lives in a separate **`chat_template.jinja`** (`tokenizer_config.json`'s `chat_template` is
EMPTY) and instructs `<tool_call>\n{"name":…,"arguments":…}\n</tool_call>` — the **hermes** dialect.
`qwen3_coder` expects `<function=name><parameter=x>`, a different grammar, hence never matching and never
logging. Measured comparison: `hermes` parses both XML shapes but **misses bare JSON**; `bare_json` parses all
three ⇒ **`bare_json` is a strict superset**, and keeping it is right because a *missed* call ends the episode
while a spurious one only wastes a step. Verified live: `finish_reason: tool_calls` from compute nodes, and
`r2egym-v1-00300` made 2 clean tool calls in a real trajectory.

⚠ **`extra_body` / `interleaved_thinking` never reach OpenCode** (harbor implements them for `terminus_2`,
`openhands`, `mini_swe_agent` only) — the 7th accepted-but-ignored key. The only lever that reaches an external
agent is server-side: `generator.engine_init_kwargs.default_chat_template_kwargs` (**MarinSkyRL `9904058`**).
Committed, but NOT needed for parity since she runs thinking on.
Also: her agent uses **`use_fn_calling=False`** (raw-text action parsing, no vLLM tool parser at all), so this
part of the pipeline can never be identical to hers — OpenCode requires OpenAI tool-calling.

### Two self-inflicted losses worth not repeating
1. **Probing the endpoint during weight sync killed an EngineCore and hung the driver** (`1246702`). Params sit
   on the **meta** device during the reload bracket and `_C` ops are CUDA-only ⇒
   `_C::rotary_embedding … Meta tensors`. The controlled comparison settled it: `1246344` (unprobed) 0 errors
   and generated; `1246702` (probed) 1 error, dead engine — and the crash returned **as the HTTP 500 to my own
   probe**. It also means my earlier "NO-THINK GATE: PASS" was measured mid-reload and is void.
2. **Bumping `PORTBASE` without re-adding the reverse forward** (`1246853`) — 128 sandboxes `ready`, bridge
   healthy, and every `agent/` dir **empty** for 14 min, failing only at the 2100s bridge timeout. **The tell
   for "no route" is an empty `agent/` dir with a healthy bridge**, not any error. TWO things change per
   relaunch and both must be re-pointed: the endpoint **port** and the **head node IP**.

### Live at handoff
`1247578` RUNNING — the parity run: 128 tasks × p@4 = 512 trials, 8 Jupiter nodes, `max_episodes=50`,
1 CPU/1 GB sandboxes, 40960 window (g1's native; we had been serving 32768), TP=4 × 7 engines,
`max_num_seqs=1024`, thinking ON, `bare_json`, endpoint port 18300, bridge 9920.
Read it with `band.py` — it yields the band rate, the zero-tool-call %, and per-trial duration together.

**Blocked on others:** Marianna's paths are `Permission denied` for `lee27`
(`/e/project1/jureap59/marianna/...`, `/e/data1/datasets/playground/ot/hf_hub/...`). Worth requesting: the
**358-task learnable set** + `merge_split_learnable.py` (reusing it makes the 18,312-trial sweep unnecessary),
her **band-generation** script, and `qwen3_thinking_acc.jinja2`. A fresh 24h JURECA fleet is operator-only.
