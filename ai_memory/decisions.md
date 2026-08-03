# Decisions (append-only)

One entry per decision: date, what was decided, the context that forced it, what was rejected, and the
consequence. **Never edit an entry.** A reversal is a NEW dated entry that names the one it supersedes.

Reconstructed 2026-07-29 from `handoff.md` and the topic notes during the memory refactor; dates are
the dates the decisions were actually made.

---

## 2026-07-13 — Stop investing in GRPO-vs-RLOO; they are statistically tied
**Context:** clean A/B on JURECA, Qwen2.5-1.5B, gsm8k, 35 steps, only `advantage_estimator` differed
(GRPO job 15426255, RLOO 15426256). Final held-out pass@1: GRPO **77.71%** vs RLOO **78.24%**; curves
track within ±0.5pp at every checkpoint.
**Decided:** treat the estimator as a settled non-lever; use GRPO.
**Rejected:** a third "canonical RLOO without KL" arm — it would confound two variables at once.
**Consequence:** no further estimator sweeps. Details + fairness table: [[jureca_grpo_vs_rloo_plan]].

## 2026-07-19 — Disaggregated only for 30B-A3B; colocation is not a supported layout
**Context:** colocate OOMs on Qwen3-30B-A3B on 1-GPU-per-node GH200s.
**Decided:** all 30B-A3B geometries are disagg (separate policy and vLLM GPU pools).
**Consequence:** every Vista/Jupiter config since is `colocate_all: false`. Reconfirmed and hardened
2026-07-26 (below).

## 2026-07-22 — NCCL debugging must be file-backed, not stdout
**Context:** `NCCL_DEBUG=INFO` was correctly exported on job `854962` but the Slurm `.out` contained no
real NCCL library lines — only 235 MB of our own `WEIGHT_SYNC_DEBUG`.
**Decided:** always use `NCCL_DEBUG_FILE` (+ `NCCL_DEBUG_SUBSYS=INIT,NET,COLL`); MarinSkyRL
`prepare_runtime_environment` now forwards it and mkdirs the parent.
**Rejected:** more stdout-based NCCL capture attempts.
**Consequence:** verified on gh-dev smoke `855073` (~44k INFO lines/file). See
[[vista_24n_sync_stall]].

## 2026-07-25 — Megatron replaces FSDP2 as the RL default (PI call)
**Context:** PI (Ben Feuer / penfever): "megatron curb stomps FSDP2 … guess we're switching our RL
defaults!" On TaskTrove `pymethods2test-large` (long-seq) Megatron total compute is ~**9×** faster than
fsdp2_no_offload; policy_train ~**10×**.
**Decided:** stand Megatron up as the default backend; keep FSDP2 runs going in parallel.
**Caveat that limits it:** GSM8K **understates** the win (our policy_train is only ~26% of a step
there) — a fair comparison needs long-seq data. On Jupiter GSM8K the measured win is 1.36× step /
1.66× policy_train.
**Consequence:** [[megatron_vs_fsdp2]], [[jupiter_megatron_bringup]]. Note this did NOT apply to the
r2egym run (see 2026-07-27).

## 2026-07-25 — `cpu_offload=false` whenever memory allows
**Context:** 4-node policy on 8 GPUs with `cpu_offload=true` ran policy_train at 533s/step. Turning it
off on the 6-node/16-GPU layout: **policy_train 533s→113s (4.7×), step wall 891s→432s (2.06×), no
OOM.**
**Decided:** `cpu_offload` is a last-resort memory workaround, not a default.
**Consequence:** bottleneck moved to `sync_weights` (43%) and `generate` (24%).
[[jupiter_bringup_and_throughput]].

## 2026-07-25 — Do NOT add inference nodes to speed up eval
**Context:** 8-node (16 infer) vs 6-node (8 infer) eval: ~49–55s/unit vs ~52s/unit — ~6%. Eval is
latency/decode-bound; `eval_batch_size=32` doesn't even saturate 8 engines.
**Decided:** the eval levers are bigger `eval_batch_size`, shorter eval `max_generate_length`, fewer
eval samples, or `eval_interval=10`.
**Rejected:** scaling out inference for eval throughput; 8n is over-provisioned vs 6n.
**Consequence:** a genuine negative result, recorded so it isn't retried.
[[jupiter_bringup_and_throughput]].

## 2026-07-25 — A larger-geometry run starts from scratch, not from a smaller run's checkpoint
**Context:** SkyRL FSDP2 checkpoints are per-rank sharded `model_world_size_{WS}_rank_{R}.pt` written
with plain `torch.save` and loaded by exact name — **not DCP, cannot reshard** (verified in
`fsdp_strategy.py`). An 8-GPU ckpt into a 16-GPU mesh → FileNotFoundError.
**Decided:** scale-out runs start from scratch.
**Consequence:** acceptable because 5 steps @ lr 1e-6 is scientifically negligible.

## 2026-07-26 — Keep TIS on for MoE; do NOT build router-replay (R3)
**Context:** measured the real generator↔trainer gap on job `1045840`:
`tis/log_ratio_abs_mean ≈ 0.030` nats (imp_ratio ≈ 1.009), routing tail
`tis/imp_ratio_capped_fraction ≈ 0.035%`. Also established `max_staleness_steps: 4` was a red herring
(async-only knob; sync trainer hardwires staleness 0) so the runs were on-policy all along.
**Decided:** `use_tis: true` as cheap insurance for MoE; do not implement R3.
**Rejected:** R3 / router replay — would recover ~nothing here. Revisit only for long-context,
larger-MoE, or high-staleness regimes.
**Consequence:** prior `_fast` runs are trustworthy, not broken. [[moe_grpo_tis_r3]].

## 2026-07-26 — Colocation for 30B-A3B on GH200s is FALSIFIED, not just unfavorable
**Context:** four distinct colocation failures in the bake-off (TIS/batched → expandable_segments →
24-GPU expert-sharding → weight-sync OOM) vs disagg 6n running clean at ~430s/step.
**Decided:** disaggregated is the default and colocation is closed for this model/hardware.
**Rejected:** the "colocation faster / 14 steps/hr" hypothesis.
**Consequence:** TP>1 remains the only thing that could reopen it, and has not been run in the RL loop.

## 2026-07-27 — Next headline experiment = r2egym FSDP2 GRPO (FSDP2, not Megatron, for this run)
**Context:** GSM8K has no headroom worth reporting; r2egym gives a binary test-pass reward that cannot
be gamed by formatting. Megatron's checkpoint story was still same-topology-only.
**Decided:** train Qwen3-30B-A3B on `DCAgent/r2egym-patched-full-oracle` (3,328 tasks) with FSDP2 +
GRPO, disaggregated, `use_tis: true`, via the agentic Harbor+Daytona path.
**Rejected:** Megatron for this particular run; the standard-gym gsm8k path.
**Consequence:** [[r2egym_grpo_plan]].

## 2026-07-28 — Validation set = a ~200-task held-out r2egym slice, evaluated offline
**Context:** r2egym has no built-in split (single 3,328-task `train`). SWE-bench as in-loop val is
impossible: 1 snapshot per task, 500 unique env hashes, harbor never reaps snapshots, org headroom 40.
**Decided:** carve a random ~200-task held-out slice as `val_data`; train on the remaining ~3,100; run
it **offline on saved checkpoints** (`eval_interval: 999999`, `ckpt_interval: 5`), not in-loop, because
agentic val is synchronous/blocking. Live signal = `reward/avg_raw_reward`.
**Rejected:** SWE-bench as in-loop val (snapshot cap is the decider); no val set at all (loses
best-checkpoint selection).
**Consequence:** the apptainer bridge stays OFF the launch critical path.

## 2026-07-28 — Formal ID benchmarks are handed to Mrinal, with the regime specified
**Context:** a single ID benchmark needs 85–100 snapshots and exceeds our 60-cap org; it's the eval
team's infra and the DATA org. Separately, the co-lead measured that eval regime flips the answer:
`g1_diverse_tezos_100k_8b` scored **+2.7 at 40k/summ-ON** but **+10 at 81920/no-summ/TP4**.
**Decided:** our deliverable is an HF model id + size (`laion/<job>-<step>-<size>` via
`rl-agentic-job-cleanup`); we specify the regime: SWE-bench, ~81920 ctx (YARN), NO summarization, TP=4,
200×1. Also ask for base `Qwen/Qwen3-30B-A3B` on the same set so the headline is a base→RL Δ.
**Rejected:** running ID benchmarks ourselves; publishing the trainer's auto-pushed
`lukedhlee/<job>` repo (wrong nested layout).
**Consequence:** [[r2egym_grpo_plan]].

## 2026-07-28 — Bridge agent = OpenCode, not Terminus
**Context:** the RL co-lead is migrating off `terminus_structured`; our pinned harbor's `AgentName` enum
has no `terminus-structured`. OpenCode runs headlessly and needs neither tmux nor asciinema.
**Decided:** port OpenCode as the bridge's target agent.
**Rejected:** porting `terminus_structured` (a fading target).
**Consequence:** the tmux/asciinema injection work becomes compatibility evidence, not a prerequisite.
[[apptainer_bridge_handoff]].

## 2026-07-28 — Inject agent tooling by bind-mounting static binaries, not by building overlays
**Context:** JSC has no user namespaces on login nodes ⇒ no `--fakeroot`, no `apptainer build`; the
co-lead's `Bootstrap: localimage` tmux/asciinema overlay does not transfer. SWE-bench images ship
neither tmux nor asciinema.
**Decided:** stage static `tmux` / `asciinema` / `opencode` / `uv` on scratch and direct-bind them into
`/usr/local/bin`. Verified against a real 999 MiB SIF with the persistent `apptainer instance` model.
**Rejected:** `--overlay`/`--writable-tmpfs`, the JSC Container Build System, and relying on
`--env PATH=...` (that PATH did not survive later `apptainer exec instance://...` calls).
**Consequence:** no build, no namespaces, no root, no runtime network needed.

## 2026-07-28 — Push only to `lukedhlee`-owned repositories; no PRs
**Context:** standing git guard, reinforced after a rushed MarinSkyRL PR #93 was closed by the user.
**Decided:** push branches only to repos owned by `lukedhlee`. No pushes to upstream or another
person's fork; no PRs. An unmerged fork fix rides `--harbor-ref` / `--skyrl-ref`.
**Consequence:** verified with `git push --dry-run` against `marin-community/harbor` and
`marianna13/harbor` (both 403, no refs changed). Successful pushes:
`lukedhlee/OpenThoughts-Agent` and `lukedhlee/harbor`, branch `lukedhlee/apptainer-opencode-bridge`.

## 2026-07-29 — GSM8K is retired as a vehicle for demonstrating MoE RL
**Context:** paired re-scoring of one generation pass (job `1086698`, n=1319) showed
`strict@1024 = 41.6%` (reproducing the run's step-0 `0.4511`) vs `flexible@4096 = 90.67%` ≈ Qwen's
published 91.81%. The model answers in `\boxed{N}` and never emits `####`; `finish_reason=stop`.
Formatting costs +37.7 pts, truncation +11.3.
**Decided:** the 0.4511→0.5057 lift over 45 steps was most plausibly format compliance, not reasoning —
do not report it as evidence MoE GRPO improves math. Move to r2egym.
**Rejected:** switching to the `flexible` scorer as a reward (last-number extraction ⇒ gameable, rewards
luck).
**Consequence:** [[gsm8k_format_artifact]]; the transferable lesson is carried into
[[r2egym_grpo_plan]].

## 2026-07-29 — Bridge development smoke uses `Qwen/Qwen3-0.6B`, not the 30B training model
**Context:** the smoke only needs to exercise OpenCode → OpenAI-compatible vLLM → tool-call → sandbox
command end to end. Model quality is irrelevant.
**Decided:** one JURECA A100, TP=1, 32,768 max model len, `--enable-auto-tool-choice
--tool-call-parser hermes`.
**Consequence:** passed (vLLM job `15476440`, bridge worker `15476441`). The 0.6B later looped until
context exhaustion — model quality, not an integration failure.

## 2026-07-29 — Apptainer prebuild unpacks on node-local scratch, never on GPFS
**Context:** earlier attempts unpacking directly on GPFS scratch caused inode exhaustion (100% inode
use) and Apptainer futex failures.
**Decided:** use `${LOCALSCRATCH:-${TMPDIR:-/tmp}}` for unpack/build and atomically copy only completed
SIFs to GPFS.
**Consequence:** prebuild job `15476943` completed `0:0` in `00:09:14`; scratch recovered to 44% inode
use with 2,476,273 free.

## 2026-07-29 — Do NOT claim a comparable 200-task SWE-bench evaluation
**Context:** the confirmed pinned set is `DCAgent2/swebench-verified-random-100-folders` @
`a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a` (exactly 100 task dirs). No credible manifest for a second
100 was found; the public `sample100` overlaps by 18, so combining gives 182 unique tasks.
**Decided:** the pinned 100-task evaluation can be launched now; do not invent or label a 200-task
subset until the second manifest's provenance is obtained from the co-lead.
**Consequence:** the only unresolved prerequisite for a headline 200-task claim.

## 2026-07-29 — Ordinary experiment submission no longer requires approval
**Context:** approval-per-submission was slowing the bridge work with no safety benefit.
**Decided:** state the key configuration before submission, then proceed. Destructive actions still
require explicit approval, and the standing task-specific guard remains: **do not launch an RL job**
without being asked.
**Consequence:** supersedes the earlier per-submission approval practice.

## 2026-07-29 (later) — Validation is SWE-bench; the held-out r2egym slice is DROPPED
**Context:** operator call, superseding the 2026-07-28 "validation = ~200-task held-out r2egym slice"
entry. That decision was driven by the Daytona snapshot cap (SWE-bench = 1 snapshot/task, ~40 headroom
⇒ SWE-bench could not be an in-loop val). The apptainer bridge removes the cap entirely, so the
constraint that forced the r2egym slice no longer exists.
**Decided:** validate on SWE-bench only. No held-out r2egym slice is carved; all r2egym tasks that
survive learnable-band filtering stay in train.
**Rejected:** keeping both (in-distribution overfit detection + cross-benchmark transfer) — one eval
path, not two.
**Consequences:**
- **Two distinct uses of SWE-bench, do not conflate them.** (a) *Checkpoint selection / val curve* —
  any SWE-bench-verified subset; n should be as large as SIF prebuild allows, because at n=100 the 95%
  CI is ±9.0pp and best-of-N selection on that is winner's-curse territory. (b) *Headline comparable
  number vs the co-lead* — the pinned 100 @ `a2e51e9e` only. The missing second manifest blocks (b),
  NOT (a).
- The JURECA bridge + a Jupiter→JURECA route move ONTO the RL critical path. Previously validation
  needed zero new infra; now no checkpoint can be scored without the bridge.
- **Contamination check is now mandatory** and was not needed before: r2egym train tasks must be
  verified disjoint from the SWE-bench-verified val instances (repo + instance id). Overlap would make
  the val set meaningless.
- Lost: in-distribution overfit detection. Accepted as the price of a single eval path.

## 2026-07-29 (later, 2) — r2egym is BOTH the training set and the validation set
**Context:** operator call, reversing the earlier same-day "validation is SWE-bench" entry. Stated goal:
"make r2egym work for both training and validation."
**Decided:** r2egym trains AND validates. A held-out r2egym slice returns as the val set. SWE-bench and
the whole JURECA apptainer bridge come OFF the critical path — parked, not cancelled; it remains the
route to a headline cross-benchmark number later.
**Consequence — this is a large simplification, note what it deletes:**
- Validation now reuses the **same 7 Daytona snapshots** training already needs ⇒ **zero new infra for
  validation.** One sandbox path, on one cluster, instead of two.
- The **unsolved Jupiter→JURECA route** stops being load-bearing. It was the biggest unmeasured risk.
- The r2egym↔SWE-bench **contamination check is moot** (mandated by the previous entry).
- The **±9pp / n=100 selection-noise** problem is moot; held-out size is ours to choose freely.
- The missing **second 100-task manifest** now blocks nothing on the critical path.
**Carried forward unchanged:** the learnable-band question is still the gate — raw r2egym collapsed for
the co-lead, and the band is model-specific. Held-out slice must be carved from the BAND, post-filter,
or the val set won't match what training sees.
**Also dropped from tracking (operator call):** the co-lead's leaked Docker Hub token. Flagged once in
[[r2egym_apptainer_reference_impl]]; not ours, not used by us; no longer surfaced in status reporting.

## 2026-07-29 (late) — COMPLIANCE HOLD on the option-D sandbox architecture
**Context:** a second session, reading JSC's own documentation, flagged that option D's mechanics may be
prohibited: (a) JURECA states **compute nodes are not allowed internet access** and directs users to
their mentor / SC Support when login-node staging is insufficient — our SOCKS relay gives `dc-cpu` jobs
`apt`/`pip`; (b) JSC prohibits **storing private keys on JSC storage** except limited support-approved
cases and states **outgoing SSH connections are not allowed** — our Jupiter→JURECA ControlMaster does
both; (c) reverse-forwarded listeners on a JURECA login node; (d) a persistent HTTP bridge + listening
port on a shared Jupiter login node.
**Decided:** stop all enablement work. Obtain written confirmation from the JSC project mentor /
`sc@fz-juelich.de` on all four before resuming. The prior "steps 1–4 need no approval" line is REVOKED.
**Rejected:** proceeding on the grounds that the co-lead already runs exactly this architecture in
production. **Another user's working setup is not our authorization** — she may hold approvals under
`transfernetx`/`projectnucleus` that we do not. Ask her *which approvals she holds*.
**Consequences:**
- The whole plan of record is gated on an external answer with a human-scale latency. Nothing on the
  critical path routes around it.
- **Possible sanctioned escape for the biggest item:** the **JSC Container Build System** (SLD item 2,
  Dockerfile→image webservice with a login-node CLI). If it can build the 7 r2egym SIFs, compute-node
  internet is never needed and item (a) evaporates. Investigate before re-proposing any relay.
- Establish whether SIFs need network **at runtime** (likely not). If not, the exposure is build-time
  only, which is exactly what the sanctioned service addresses.
- **Process lesson, generalize it:** "the reference implementation does X" answers *feasibility*, never
  *permissibility*. Check the site policy for any mechanism that crosses a security boundary — outbound
  network, listening ports, stored credentials — BEFORE designing around it.

## 2026-07-29 (late, 2) — Compliance hold LIFTED; proceed with guardrails
**Context:** supersedes the "COMPLIANCE HOLD" entry immediately above, same day.
**Operator ruling:** proceed. Criterion = **"it's okay as long as we don't sabotage other users."** The
co-lead runs this exact architecture in production and is on good terms with the JSC manager. **No
support ticket is being filed.**
**Raised twice, decided — do NOT re-raise as a blocker.** The reasoning in the prior entry is kept
because it is sound, not because it is still actionable.
**Standing caveat, recorded once:** an informal relationship covers HER under HER project
(`transfernetx`/`projectnucleus`); it does not automatically extend to `lee27`/`synthlaion`. Mitigation
is one line in the message she is already getting: *did you clear this with anyone specific, or is it
just understood?* A name converts hearsay into cover.
**Guardrails that now ARE the standard** (these operationalize the operator's own criterion):
- listeners bound to **internal interfaces only**, never `0.0.0.0`/public;
- login-node processes stay small and idle-cheap; **no bulk pulls/builds on a shared login node**
  (measured: image pulls are CPU-bound, `user 6m51` vs `real 2m47`) — heavy work goes to compute;
- **private key on Jupiter: passphrase-protected, `chmod 600`.** NOT a policy question — a readable key
  is the same class of failure as the co-lead's leaked Docker PAT;
- tear services down when idle; never leave listeners up across idle days;
- never `find`/`du` on GPFS — inode exhaustion is the one failure that genuinely does harm other users.
**Independent of all this:** still evaluate the **JSC Container Build System** before porting a SOCKS
relay — not for compliance now, but because it may simply be less work.

## 2026-07-31 (late) — Relevant JSC jobs are pre-authorized, with a 16-node ceiling
**Context:** the Qwen3.6 pipeline-validation sequence needs one-node kernel smokes, JURECA CPU sandbox
workers, and a six-node Jupiter GRPO smoke. The earlier standing rule required a fresh explicit ask
before any RL submission.
**Decided:** the operator explicitly authorized all safe, task-relevant JSC submissions, including RL,
without further permission prompts. At no point may the combined active/pending allocations taken for
this work exceed **16 nodes**. Continue to observe JSC inode, path, listener, network, and shared-login
guardrails.
**Consequence:** the one-step Qwen3.6 GRPO smoke may be submitted as soon as its technical gates pass.
MFA or other human-only authentication remains an external action, not an approval gate.

## 2026-07-31 (late, 2) — Put human-only MFA commands on the operator's clipboard
**Context:** the Jupiter-to-JURECA ControlMaster occasionally needs an interactive passphrase/TOTP,
which the agent cannot enter. Printing the command creates avoidable copy/paste friction.
**Decided:** whenever MFA is the only blocker, copy the exact safe interactive command to the local
macOS clipboard and tell the operator to paste/run it. Never copy a secret, passphrase, or TOTP.
**Consequence:** use `pbcopy` for the command instead of only displaying it, then continue
automatically once authentication is live.

## 2026-07-31 (late, 3) — Stress and harden the Jupiter-JURECA connection after the smoke
**Context:** the RL pipeline depends on a reverse SSH forward from Jupiter to JURECA; a stale tunnel
can otherwise surface as sandbox outages or misleading zero reward.
**Decided:** after the one-step smoke is no longer using the path, run a bounded resilience test with
failure detection at one second or less, deliberately break and reconnect the ControlMaster/forward,
and verify bridge workers recover. Then restore conservative production keepalive values informed by
the test rather than leaving the aggressive stress settings enabled.
**Consequence:** connection loss/recovery is now an explicit pipeline-validation action item. Do not
disrupt an active RL rollout to perform it.

## 2026-08-03 — Optimize bring-up for information per node-minute
**Context:** sequential Qwen3.6 integration failures were discovered only after repeatedly paying for
eight inference-engine loads/JITs, a 16-rank policy load, and initial weight sync. These failures were
real, but the full six-node GRPO job was an unnecessarily expensive probe for bridge health, tool
rendering, and client context-limit semantics.
**Decided:** treat the full six-node one-step GRPO job as the final integration gate, not the default
debugger. Every failure it exposes must become a seconds-long deterministic contract/preflight test
or, where execution is essential, a one-engine exact-model OpenCode+sandbox+verifier canary. Run
independent read-only/local investigations in parallel with a live allocation. Evaluate smoke shape
and safe persistent compile caching by information gained per node-minute, while preserving honest
reward variance and the exact final model/update/export milestone.
**Consequence:** do not blindly relaunch the full topology for a boundary that can be tested below it;
promote through static contract -> one-engine agent canary -> full one-step GRPO.

## 2026-08-03 (later) — Reserve 16k for smoke compaction; deterministic agent failures fail fast
**Context:** corrected total-window semantics plus `compaction.auto=true` produced five honest
Qwen3.6 rollouts, but 14 context overflows and Harbor's 60/120/240-second retries prevented even one
complete GRPO group. Exact OpenCode 1.18.8 checks the previous assistant request before the next tool
result is appended. Its default threshold left only 5,120 prompt tokens of jump room, while live tool
results caused single-turn jumps up to 11,815 tokens.
**Decided:** for this pipeline-validation smoke, keep the server/model contract at 32,768 total and
4,096 output, but set `opencode_config.compaction.reserved: 16384`. This moves proactive compaction
to about 11,264 tokens and leaves about 17,408 tokens below the hard prompt ceiling. Put
`ContextLengthExceededError` and `NonZeroAgentExitCodeError` in Harbor retry exclusions: a
deterministic canary failure should abort promptly, not hold a scarce rollout slot for four attempts.
**Rejected:** lowering output tokens as the primary fix; it does not address a tool result appended
after the usage measurement. Also rejected another multi-node retry without an exact boundary probe.
**Consequence:** require a one-engine exact-model canary with two sequential 44,001-byte tool results,
an observed prompt-token drop proving compaction, no context/transport error, and a real verifier
result before the five-node intermediate GRPO canary.

## 2026-08-03 (later, 2) — Disable the x86 FlashInfer AOT artifact on Jupiter
**Context:** the pinned artifact passed a 90-second metadata/path/hash smoke, but its first real
`dlopen` after a 67 GiB model load failed. Jupiter GH200 compute nodes are aarch64; the official
artifact staged by commit `59e661e0` is an x86-64 ELF. The smoke never loaded the module.
**Decided:** remove all AOT variables from the Qwen five- and six-node configs and return to
job-local cold JIT. Any future AOT artifact must pass both ELF-host architecture matching and a real
`tvm_ffi.load_module()` on the target compute node before it can be enabled.
**Rejected:** trusting `is_aot`, the package version, the wheel filename, or a matching SHA as binary
compatibility evidence; all four were true for the incompatible artifact.
**Consequence:** successor startup pays the known fused-MoE JIT cost until an exact aarch64 cache
exists, but it cannot fail late on an architecture mismatch. The one-node context diagnostic uses
Triton GDN to avoid paying an unrelated second JIT while leaving final RL kernel choices unchanged.

## 2026-08-03 (later, 3) — Do not force safetensors prefetch for Qwen3.6 on Jupiter GPFS
**Context:** vLLM's generic warning suggested forced prefetch because it does not classify GPFS as
NFS/Lustre. A parallel exact-model one-node comparator provided a direct timing test without using
the bridge or training state.
**Decided:** retain the default safetensors load strategy. Comparator `1217881` spent 153 seconds
prefetching and then loaded materially more slowly than diagnostic `1217866`; it was stopped after
the result was decisive. This is a performance choice for the pinned Qwen/runtime/Jupiter path, not
a universal claim about vLLM prefetch.
**Consequence:** future successor renders must not add `--safetensors-load-strategy=prefetch` merely
to silence the warning. Optimize the larger, measured cold-JIT cost instead.

## 2026-08-03 (later, 4) — Bound FlashInfer cold-JIT concurrency with `MAX_JOBS`
**Context:** exact no-AOT diagnostic `1217866` loaded Qwen successfully, then an unbounded
fused-MoE Ninja build spawned roughly 300--350 compiler processes and failed many targets together
with code 9 at peak memory. FlashInfer's installed build helper explicitly honors `MAX_JOBS`.
**Decided:** use a bounded compiler fan-out for Jupiter cold JIT. Diagnostic backup `1217900`
starts with `MAX_JOBS=24`; promote that value only after its memory/time behavior is measured.
The eventual five-node render must explicitly carry the proven bound before accepting cold JIT as
the x86-AOT replacement.
**Consequence:** bounded compilation may take longer but must stay within node memory and the
three-hour RL wall. Do not treat an unbounded 64-CPU Ninja launch as the default fallback.
