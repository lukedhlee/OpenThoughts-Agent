# SkyRL issue queue — upstream findings against MarinSkyRL

Bugs and gaps found in **MarinSkyRL main** (and the harbor RL path it drives) that need a fix
we don't own. One entry per finding. Entries live here whether or not they've been filed
upstream — this file is the working record; a GitHub issue is just one possible outcome of it.

> **Standing rule: never open an issue, PR, or comment on a repo Luke doesn't own without an
> explicit ask.** Draft the entry here first. Filing upstream is a separate decision, made by
> Luke, every time. (We have `pull` only on `marin-community/MarinSkyRL` — an issue filed there
> can be closed by us but never deleted.)

## What matters right now

**Two bugs, both unfixed, plus one latent gap. On an agentic arm they mean neither the `tis/` metrics
nor the sampling temperature is what it claims to be.**

They are independent — **#1 breaks a number, #2 breaks the run, #3 is armed by a missing config line.**

**#1 breaks the TIS number.** TIS exists to catch drift between the trainer's probability for a
token and vLLM's probability for that same token: it divides one by the other and expects roughly 1.
But the trainer applies temperature before computing its number and vLLM doesn't, so the division
mostly reports that difference in convention rather than the drift it was built to detect.

**#2 breaks the run itself.** The temperature in your YAML is never sent to vLLM on the Harbor path,
so rollouts were sampled at whatever the checkpoint's `generation_config.json` happened to say.
Training ran fine — it just wasn't the experiment you configured.

Fixing either one leaves the other.

What that costs you: `tis/` numbers off an agentic run are not evidence of anything, and an agentic
arm samples at whatever its checkpoint's `generation_config.json` says — even when your YAML says 1.0.

**#3 is a gap, not a live bug — today.** MarinSkyRL's TITO fix (`tito_full`) assumes the chat history
only ever grows. Terminus-2's context summarization rewrites it, and MarinSkyRL has no handling for
that (it does for OpenCode), so a summarized trajectory silently drops back to re-tokenization. Every
terminus arm we own sets `enable_summarize: false`, which is the only thing keeping it inert — the
MarinSkyRL default is `true`, so a new YAML that forgets the line is exposed.

## Index

| # | Finding | Status | Scope | Found | Upstream |
|---|---------|--------|-------|-------|----------|
| 1 | TIS ratio mixes tempered trainer vs raw vLLM logprobs | **LIVE — unfixed** | `use_tis` arms at temp ≠ 1.0 | 2026-08-21 | [#438](https://github.com/marin-community/MarinSkyRL/issues/438) (closed, no response) |
| 2 | Harbor RL path never passes sampling temperature to vLLM | **LIVE — unfixed** | every agentic arm, TIS or not | 2026-08-21 | [#438](https://github.com/marin-community/MarinSkyRL/issues/438) (closed, no response) |
| 3 | TITO-full has no terminus-2 summarization handling — summarized trajectories fall back to re-tokenization | **WORKED AROUND** (`enable_summarize: false` in every terminus YAML); gap LIVE on main | terminus-2 arms with summarization on (default `true`) | 2026-08-27 | not filed |

Both were filed as the single issue #438; they are separate defects with separate fixes and are
tracked separately here. Fixing one does not fix the other.

---

# 1 — TIS ratio mixes distributions (π^(1/T) vs π)

**Status:** LIVE on main, unfixed. Found 2026-08-21 (audit vs origin/main `babc08e`).
Upstream: [#438](https://github.com/marin-community/MarinSkyRL/issues/438), filed 2026-08-22,
closed 2026-08-28 by Luke with zero maintainer response. Code unchanged.

The TIS importance ratio compares the trainer's **tempered** logprobs against vLLM's **raw
(untempered)** logprobs, so it is wrong whenever rollout temperature ≠ 1.0. This is an arithmetic
defect in the ratio itself — it holds even if temperature were plumbed perfectly everywhere. Read
before trusting any `tis/` metric or launching a TIS arm at temp ≠ 1.0.
Related: [[moe_grpo_tis_r3]] (the 0.030-nat measured gap — see calibration below).

## Mechanism

- Trainer side is TEMPERED: worker passes `generator.sampling_params.temperature` into the forward;
  `model_wrapper.py` does `logits.div_(temperature)` (FSDP and Megatron paths both).
- vLLM side is RAW: `logprobs_mode` is never set anywhere in MarinSkyRL or OT-Agent → vLLM default
  `raw_logprobs` = logprobs BEFORE temperature/top-k/top-p. (vLLM docstring: "processed" =
  after all processors including temperature.)
- The ratio at `skyrl_train/utils/policy_losses.py:318-321` (origin/main 2026-08-21) subtracts one
  from the other: `exp(old_log_probs − rollout_logprobs)` = π^(1/T)/π, not a real IS ratio.
- Direction: T>1 → ratio systematically <1 (silently down-weights batches); T<1 → ratio >1,
  pins at `tis_imp_ratio_cap` (2.0).

## Blast radius

**Affected** (`use_tis: true` AND temp ≠ 1.0), `hpc/skyrl_yaml/jupiter/`:
`64GPU_qwen3_32b` (1.2) · `56GPU_qwen3_8b` (1.2) · `24GPU_qwen3_30b_a3b_thinking` (1.2) ·
`6node_qwen3_30b_a3b_gsm8k_grpo_fast_tis` (0.7).
**Clean *for this finding*:** TIS arms at temp 1.0 (r2egym, currease×5, base-gsm8k fsdp2 arms) and
all non-TIS arms. Note this says nothing about finding #2 — an agentic arm listed clean here can
still have sampled at the wrong temperature.

## Calibration — how bad is it in practice?

[[moe_grpo_tis_r3]] measured the gap on the gsm8k TIS arm (job 1045840, T=0.7):
`tis/log_ratio_abs_mean` ≈ 0.030 nats, imp_ratio ≈ 1.009, capped fraction 0.035% → **small there**
(peaked post-SFT distributions: tempered ≈ raw on high-confidence tokens; mismatch concentrates on
uncertain tokens). That arm is non-agentic, which is precisely why it was measurable — finding #2
doesn't touch it, so T was known to be 0.7. The **agentic arms are the unmeasured concern**: long
multi-turn rollouts where TIS matters most, and where #2 makes the true sampling T unknown. So:
bug real, gsm8k impact small, agentic impact unquantified — do not extrapolate the 0.030 nats.

## Fix

1. Set `logprobs_mode="processed_logprobs"` at vLLM engine init (flag exists since vLLM 0.10.2; we
   run 0.16.0 / 0.20.2 — both fine). MarinSkyRL; worktree → PR → main, never self-merge.

## Verification after fix

Step-0 / on-policy `tis/` ratio histogram in W&B must sit at ~1.0 (the TIS diagnostics block in
worker.py documents this invariant); heavy `tis_imp_ratio_cap` clamping = still misaligned.

---

# 2 — Harbor RL path: YAML sampling temperature is INERT

**Status:** LIVE on main, unfixed. Found 2026-08-21 (audit vs origin/main `babc08e`).
Upstream: filed as part of [#438](https://github.com/marin-community/MarinSkyRL/issues/438), closed
2026-08-28 with no maintainer response. Code unchanged.

On the agentic (Harbor) path the configured temperature never reaches vLLM at all, so rollouts are
sampled at the served checkpoint's own default instead. This is a provenance defect, not an
arithmetic one: you don't know which distribution your tokens came from. It is **independent of
TIS** and degrades any agentic run — read before launching or interpreting one.

## Mechanism

This is a missing field inside MarinSkyRL, not something we misconfigured and not a code path we
forgot to switch on. It hits anyone running agentic RL there.

Follow one temperature setting through the system:

1. You put `temperature: 1.0` in the YAML.
2. The trainer reads it and holds it correctly.
3. When it's time to run an agent trial, the trainer hands off to the Harbor runner.
4. The Harbor runner builds a small config describing that trial — which model to call, what address
   to call it at, the token cap. **Temperature is not one of the fields it copies in.**
5. The agent calls vLLM with no temperature in the request.
6. vLLM, receiving no temperature, falls back to the served model's own `generation_config.json`
   default. Qwen3 ships non-1.0 values there.

So the configured value exists and is correct inside the trainer, and simply never gets written into
the request that reaches vLLM. Net on the "1.2" agentic arms: sampling at an unknown model default
while the trainer divides logits by 1.2 — a constant matching NEITHER the sampler NOR the raw
logprobs.

**Ben's PR #412 is not a fix we declined to adopt.** He hit the same missing-field problem on the
trace-generation side and wrote `get_sampling_params_for_backend` to assemble those params properly.
The RL side has the identical gap and the equivalent line was never written there. The sibling
`mini_swe` trajectory runner does pass them through; Harbor's does not.

Anchors (verified against origin/main `b45a6ed`, 2026-08-28 — still true):

- The harbor trajectory runner reads exactly ONE field from `sampling_params`:
  `max_generate_length` (`trajectory_runners/harbor/runner.py:2029,2037`).
- The trial config (`trajectory_runners/harbor/configuration.py`) carries only model_name/api_base —
  no temperature; Terminus passes none; the HTTP endpoint injects no default.
- `get_sampling_params_for_backend` is called by `trainer.py`, `fully_async_trainer.py`,
  `evaluate.py`, the generate entrypoint, and `trajectory_runners/mini_swe/runner.py` — but not by
  `trajectory_runners/harbor/runner.py`.

## Blast radius

**Affected: every agentic (Harbor) RL arm**, regardless of `use_tis` and regardless of the
configured temperature — a YAML saying 1.0 is as inert as one saying 1.2. Top-p/top-k are dropped
by the same code path.
**Clean:** the non-agentic generator path (gsm8k, r2egym and other standard GRPO arms), which passes
`sampling_params` through normally; and the datagen/generation-only entrypoint since PR #412.

## Calibration

**Unmeasured.** Nobody has yet recorded what the agentic arms actually sampled at — it is whatever
`generation_config.json` shipped in each served checkpoint, so it may differ per arm and per
resumed chain. Quantifying this is the prerequisite for reinterpreting any historical agentic
result, including the `tis/` traces discussed in finding #1.

## Fix

1. Plumb `sampling_params.temperature` (and top-p/top-k) through the harbor runner's trial config,
   reusing the #412 `get_sampling_params_for_backend` pattern. MarinSkyRL; worktree → PR → main,
   never self-merge.

## Verification after fix

Confirm the requested temperature reaches the sampler — a served-model request echoing the
configured value rather than the checkpoint default. Separately, and regardless of the fix, read the
served checkpoints' `generation_config.json` on-cluster once to establish what the historical
agentic arms really sampled at.

---

# 3 — TITO-full vs terminus-2 summarization: a context rewrite silently drops the trajectory to re-tokenization

**Status:** WORKED AROUND (every terminus-2 RL YAML we own sets `enable_summarize: false`); the gap
itself is LIVE on main, unfixed. Found 2026-08-27 (audit vs MarinSkyRL upstream/main `b69d22c`,
harbor origin/main `725fc069`). Upstream: not filed.

MarinSkyRL's TITO fix rebuilds the training sequence from the exact per-turn token ids Harbor
captured, which only works if each turn's prompt is the previous prompt + completion + observation.
Terminus-2's context summarization breaks that: it rewrites the chat history mid-rollout. MarinSkyRL
detects the break and falls back — quietly — to the old re-tokenize + LCS path for the whole
trajectory, so the TITO guarantee is lost exactly on the longest, most valuable rollouts. Read this
before writing a new terminus-2 RL YAML or before trusting `tis/` on one that summarizes.

## Mechanism

- Terminus-2 summarizes both proactively (free tokens < `proactive_summarization_threshold`, default
  8000) and reactively on `ContextLengthExceededError`; both are gated on `enable_summarize`
  (`harbor/agents/terminus_2/terminus_2.py:301-302,1387,1468`). Harbor defaults it to `True`, and so
  does MarinSkyRL's trial config (`trajectory_runners/harbor/configuration.py:106`, `default=True`).
- On summarization the SAME `Chat` object's message list is replaced with a summary-seeded one
  (`terminus_2.py:1328`), but its token-id streams keep accumulating across the reset —
  `Chat._accumulate_rollout_details` (`harbor/llms/chat.py:139`) appends every turn and nothing
  clears it. What reaches the trainer is `all_messages = chat.messages` (`terminus_2.py:2208`) =
  post-summary turns only, paired with prompt/completion ids for ALL turns. Harbor's own docstring
  warns: "Rollout details will be incomplete if context summarization occurs" (`terminus_2.py:336`).
- `_assemble_response_ids_tito_full` (`trajectory_processing.py:1472`) requires
  `len(assistant_msgs) == n_turns` (`:1510`) and the prefix invariant (`:1522`); a summarized
  trajectory fails both → returns `None` → caller logs "Full TITO prompt-id assembly declined"
  (`:1729`) and falls through to re-tokenize + splice. The splice then pairs message index `i` with
  served-ids turn `i`, which are now different turns → ids diverge → LCS last resort (`:1654`).
- Contrast: OpenCode context resets ARE handled — `_select_opencode_literal_chain`
  (`harbor/runner.py:77`) keeps the last continuous served chain. Terminus-2 gets only a counter:
  `generate/trajectories_summarized` (`runner.py:1149`); nothing changes how the sequence is built.
- Consequence for a summarized trajectory: pre-summary turns are never trained on; post-summary
  turns train on re-tokenized text with LCS-aligned logprobs (finding #1's precision caveats apply on
  top). The blog-recommended handling (freeze the pre-rewrite context as masked prompt, train the
  post-rewrite samples as exact ids) is not implemented for terminus-2.

## Blast radius

**Affected:** any terminus-2 RL arm with summarization enabled — i.e. any YAML that omits
`enable_summarize: false`, since both harbor and MarinSkyRL default it on. Zero such YAMLs exist
today (checked 2026-08-27: all 10 terminus YAMLs under `hpc/skyrl_yaml/jupiter/` and all 6 under
MarinSkyRL `cloud/iris/configs/` set it `false`). **Clean:** every current terminus arm (summarization
off → on overflow terminus raises `ContextLengthExceededError` at 2048 free tokens and the trajectory
simply ends, `terminus_2.py:1387-1395`); every OpenCode arm (reset handled); every non-agentic arm
(no Harbor). This finding says nothing about #1/#2 — a clean arm here can still carry both.

## Calibration

**Unmeasured, and currently unmeasurable on our arms** because summarization is off everywhere. The
exposure on a hypothetical summarizing arm is `generate/trajectories_summarized / n_trajectories`
per step — that single W&B metric is the whole severity estimate. Expect it to be highest exactly
where TITO matters most (long multi-turn rollouts near the context limit).

## Fix

1. Cheapest, immediate: make `enable_summarize: false` a documented requirement for terminus-2 RL
   YAMLs (or flip MarinSkyRL's `configuration.py:106` default to `False` for the RL path).
2. Real fix: give terminus-2 the OpenCode treatment — on `summarization_count > 0`, select the last
   continuous served chain from the token-id streams (prefix-invariant walk) and train only on it,
   with the summary-seeded prefix masked as prompt. MarinSkyRL; worktree → PR → main, never self-merge.
3. Fail-loud interim: when `summarization_count > 0` and `tito_full` declines, count it under a
   dedicated metric (e.g. `generate/tis/tito_declined_summarized`) instead of folding it into
   `lcs_fallback_fraction`.

## Verification after fix

Launch a terminus-2 smoke with `enable_summarize: true` and a small context so summarization fires;
`generate/trajectories_summarized > 0` must coincide with `generate/tis/exact_match_fraction` staying
~1.0 and no "Full TITO prompt-id assembly declined" warnings. Today that same smoke would show the
warning on every summarized trial.

---

# Template for a new entry

```
# N — <short name>: <one-line thesis>

**Status:** LIVE — unfixed | FIXED upstream (<PR>) | WORKED AROUND (<where>) | WITHDRAWN
Found <date> (audit vs origin/main `<sha>`). Upstream: <issue/PR link, or "not filed">.

<2-4 line orienting paragraph: what's wrong, and when a future session must read this.>

## Mechanism
<bullets, each with a chaseable anchor — file:line, commit, job ID, PR number.>

## Blast radius
**Affected:** <arms/configs/paths>  **Clean:** <what's provably unaffected, and for which finding>

## Calibration
<measured severity, and explicitly what remains unmeasured.>

## Fix
<numbered, one sentence each.>

## Verification after fix
<the observable that proves it landed.>
```
