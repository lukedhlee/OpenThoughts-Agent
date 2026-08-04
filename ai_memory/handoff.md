# Handoff — lukedhlee · updated 2026-08-04 (01:00 KST)

> **START HERE: § "Live state — 2026-08-04 10:30 KST" in this file.** It supersedes the 2026-08-03
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

| path | now lives on | why |
|---|---|---|
| experiments tree, checkpoints, HF exports | **`/p/scratch/reformo/lee27/experiments`** | 3.0M free inodes, FRESH accounting, DOCUMENTED retention |
| execution checkout (`DCFT`) | **`/p/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next`** | `git fetch` needs to create objects |
| read-only 67 GiB model | `/e/fscratch/reformo/lee27/models/...` | per-stream BANDWIDTH, not inodes |

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
| `1229446` | 16×8, **4h** | **PD `(Priority)`** — the 4h walltime cleared the maintenance block |

### Shipped this session

| commit | repo | change |
|---|---|---|
| `f44a1170` | harbor | `BRIDGE_EXEC_TIMEOUT`; default still 600; 4 tests |
| `6b76270b` | OT-Agent | vLLM canary reads shards from a chosen filesystem |
| `2b6defd1` | OT-Agent | exec timeout 2100, n=8, fscratch model, `-next` checkout |
| `8ac43278` | OT-Agent | `MM_LIMIT` / `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` knobs, defaults unchanged |
| `45572f93` | OT-Agent | experiments tree → `/p/scratch` (the inode fix) |
| ai_memory | local only | corrections + gotchas, not pushed |

**Measured:** `Loading weights` **701.12s → 38.03s** (1 GPU, 18.4×) / **61.6s** in-job (11.4×) by
staging on `/e/fscratch`. `/e/scratch` 0.056 GiB/s per stream vs fscratch 2.51 (`O_DIRECT`, compute
`jpbo-018-14`, job `1224678`); vLLM reads the 26 shards **sequentially in one stream**, so it sat on
that floor. ⚠ Startup is now ~95% **vision-tower profiling** (~920s, *"1 image items of the maximum
feature size"*) for a tower SkyRL unwraps and no rollout uses — `MM_LIMIT=0` is staged but UNMEASURED.

### Live resources

- **Jupiter:** `1229446` PD `(Priority)`, 5 nodes, 4h wall.
- **JURECA:** `15494122` R (~13h left) + `15495516` PD (24h), 2 nodes × 16 each. Two fleets deliberately
  overlap so any start time between now and ~10:15 KST tomorrow is covered.
- **Bridge** `10.128.1.2:9920` clean (`ready: 0`). Lifetime `jobs_errors: 176` are all from the two
  aborted runs. 55 orphaned sandboxes were reclaimed across two sweeps.
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
