# ⚠ TEMPORARY TAKEOVER NOTE — written 2026-08-03 ~14:40 CEST · DELETE ONCE ABSORBED

You are taking over a Qwen3.6-35B-A3B agentic-RL bring-up mid-flight. **Nothing is running on
Jupiter.** Read this file first, then `notes/qwen36_resume_brief.md` § "LATEST LIVE UPDATE —
2026-08-03 14:10", then `handoff.md`. `gotchas.md` has ~40 symptom→cause→fix entries; 8 are from
2026-08-03 and every one of them cost a multi-node allocation to learn.

## The one thing to understand before you do anything

**The infrastructure works. The DATA is the blocker. Do not "fix" the pipeline again.**

The pipeline now runs cleanly: Ray → 26/26 shards → fused-MoE JIT/KV → HTTP endpoint → policy/ref
init → 693-tensor weight sync → **honest rollouts with real verifier rewards**. Four defects were
root-caused, fixed, pushed and deployed on 08-03, and all four are confirmed working.

What blocks the milestone is that **raw r2egym produces ZERO within-group reward variance for this
model**: 22 prompt groups across two runs, none with variance; the four *complete* 4-sample groups
were perfectly uniform (`[1,1,1,1]`, `[1,1,1,1]`, `[0,0,0,0]`, `[0,0,0,0]`). GRPO forms advantage
**within** a group, so all-zero groups are filtered and all-one groups carry zero advantage ⇒ **no
gradient can exist**. Rewards ARE honest and DO differ across tasks; that is not a substitute.

⇒ **Relaunching as-is cannot succeed.** It will burn 5 nodes and reproduce the same result.

## Two decisions are the operator's. Do NOT decide them yourself.

1. **Adopt the model-specific learnable band?** (`0 < pass@k < 1`, the co-lead's approach). Now
   empirically required, not optional. It was explicitly PARKED by the operator on 2026-07-29
   ("forget the band, focus on enabling r2egym"). r2egym IS enabled now and the parked risk has
   materialised in our own measurements. **Surface it; do not silently adopt it.**
2. **Move the model + `trace_jobs` to `/e/fscratch`?** Measured: `/e/scratch` reads at **104 MB/s**,
   `/e/fscratch` at **2.3 GB/s (22×)**. Also fixes inode pressure (`/e/scratch` 77% of an 8M cap
   shared by 26 users; `/e/fscratch` 1%). Caveats: measured from a login node, and fscratch
   retention/purge is UNDOCUMENTED ⇒ stage only the read-only model and transient traces; keep
   checkpoints/HF exports on `/e/scratch`.

## Hard-won lessons you will otherwise repeat

- **A config key arriving in the materialized TrialConfig proves NOTHING about whether the consumer
  reads it.** Three OpenCode keys were accepted, deep-merged, written to disk, and silently ignored:
  `strict_json_parser`, `compaction.reserved`, `store_all_messages`. Verify by BEHAVIOUR.
- **Fail-loud taxonomy is inverted from intuition:** `mask` = infrastructure ⇒ **ABORTS the whole
  batch** under `fail_on_infrastructure_error`. Only `passthrough` survives. Adding an exception to
  `mask_exceptions` makes it MORE fatal.
- **A large prompt drop is NOT proof compaction worked** — it may be *reactive* recovery after the
  ceiling was already breached. Always check for `ContextOverflowError` first.
- **Never grep a trial tree for success markers**: the rendered config contains the instruction text,
  so markers match before the agent has done anything. I briefly reported a false pass this way.
- **Harbor is a COPIED install** in `envs/rl-megatron/.../site-packages/harbor`; the fix only lands
  because the sbatch puts `harbor-apptainer-bridge/src` first on `PYTHONPATH`. Verify harbor changes
  with the job's real `PYTHONPATH` — a bare venv import reports the fix as absent.
- **Never judge a venv from a bare login shell.** `rl-megatron` fails on `libcudart.so.13` without
  `module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13`. (`envs/rl` was genuinely broken and is
  deleted — it needed CUDA 12, which Jupiter does not have.)
- **Never `find`/`du` on GPFS.** Use `jutil project dataquota -p reformo`. For subtree estimates use
  bounded one-level `ls` counts.
- **The pinned FlashInfer AOT artifact is x86-64; Jupiter is aarch64.** Hash/version/path checks all
  passed because none called `dlopen`, so the 90-second "AOT smoke" was a FALSE POSITIVE.

## Deployed state (verify before trusting)

| repo | Jupiter path | ref |
|---|---|---|
| OT-Agent | `/e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next` | `90109474` |
| harbor | `/e/scratch/reformo/lee27/harbor-apptainer-bridge` | `1f38665f` |
| MarinSkyRL | `/e/scratch/reformo/lee27/MarinSkyRL-apptainer-bridge` | `8ee69f3` |

Local ground truth: `~/OpenThoughts-Agent-clean` (`90109474`), `~/harbor-apptainer-bridge`
(`1f38665f`). `ai_memory` lives on branch `lukedhlee/vista-moe-grpo-30b` in `~/OpenThoughts-Agent`
(commits `068f1578`→`d523d0ce`, local only).

⚠ **Cross-checkout trap:** the 6-node YAML hardcodes `prompt_template_path` into the STALE
`OpenThoughts-Agent-r2egym-bridge` (`0f04b250`), and `run_qwen36_live_canary.sh` defaults `DCFT` to
that same stale tree. Always override `DCFT` to `-next`. Collapse this before promotion.

## Live resources

- **Jupiter: 0 nodes.** Ceiling is 16 combined; never cancel anyone else's job. Cancelling *our own*
  wedged job is pre-authorised.
- **JURECA sandbox workers `15489111`** — expire **~18:50 CEST 2026-08-03**. If gone, submit a
  replacement worker job through the live ControlMaster.
- **Bridge `10.128.1.2:9920`** — was pristine at handoff: 3542/3542 jobs, zero errors,
  `workers_alive:true`, all sandboxes reclaimed. Check `/status` first thing.
- ⚠ **SINGLE POINT OF FAILURE:** the Jupiter→JURECA ControlMaster (`~/.ssh/cm_jureca/qwen36`,
  pid 185890) carries the reverse tunnel as a `-R` forward. If it dies the tunnel dies and rollouts
  lose their endpoint. Restoring needs ONE interactive `ssh jureca` + TOTP — **only Luke can do it.**
  Separately, `Session open refused by peer` on the Mac→Jupiter master is `MaxSessions` exhaustion
  from too many concurrent background SSH sessions, NOT an auth expiry.

## Useful operational recipes

- Orphaned sandboxes after a cancellation: recover IDs **only from that run's own trace tree**
  (`grep -rhoE "env-[0-9a-f]{12}"`), validate the exact form, then
  `POST /env/stop` with `{"env_id": "..."}`. (`/env/<id>/stop` is a 404 — wrong shape.)
- Allocation-free preflight: `DCFT=<next> PREFLIGHT_ONLY=1 REQUIRE_CLEAN_BRIDGE=1 bash
  hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo_canary.sh`. Expect
  `OpenCode limit=15360+4096, proactive_compaction_at=11264`.
- Within-group variance readout (the number that actually matters): group `trace_jobs/*/result.json`
  by task prefix and check `len(set(rewards)) > 1` per group. **Never** read a healthy-looking
  `avg_raw_reward` as evidence GRPO can learn.
- Route gate before trusting any rollout: prove `/health` **and** a real `/v1/chat/completions` from
  an actual JURECA compute node (`srun --overlap --jobid=<workers> --nodelist=jrc0545`) through
  `jrlogin05i:18000`, and confirm the served revision is `995ad96e…`.
- After a job allocates, retarget the reverse listener:
  `ssh -O cancel -R 10.14.0.46:18000:<old>:8000 jureca` then `-O forward` to the new head. Internal
  binds only — never `0.0.0.0`, never a public IP.

## Never proven with this model

The **optimizer update, checkpoint, and HF export** have never executed. Reaching a finite update is
still the milestone. And the **token-fidelity limit** stands: opencode discards raw completion text,
so reconstructed `all_messages` is faithful in structure but approximate in tokens — fine for
pipeline validation, NOT for trusting TIS importance ratios. Enable the literal recording proxy
(`literal.jsonl`) before promoting to 50 steps.

## Suggested first moves

1. `curl http://10.128.1.2:9920/status` and `squeue` on both clusters — establish reality.
2. Confirm the JURECA worker fleet still exists; it may have expired.
3. Ask the operator the two pending decisions above. **Do not relaunch before decision 1** — the
   variance result means a relaunch cannot produce a gradient.
4. If decision 2 is approved, the fscratch staging is cheap and independently valuable (22× load
   speed) — worth doing even while decision 1 is open.
