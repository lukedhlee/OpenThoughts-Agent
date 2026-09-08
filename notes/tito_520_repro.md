# MarinSkyRL#520, reproduced — and what each repair actually costs

Run on TACC Vista (GH200, `gh-dev`/`gh`, account CCR24067) on 2026-09-08, in the worktree
branch `lukedhlee/tito-tacc-repro`. Harness: `scripts/tito_repro/`.

## What matters

The bug is real, it is close to certain on long rollouts, and it is *two tokens wide*.
Each incident re-cuts about two token positions where the model's own sampled IDs meet a
BPE boundary — 10 of 28,860 context tokens (0.035%) across the retained arm-3 trajectory.
SkyRL's prefix check is all-or-nothing, so those two tokens throw away full-TITO assembly
for the entire trajectory, silently, with `tis/exact_match_fraction` still reading 1.0.

Three consequences follow, in the order worth acting on:

1. **A recorded capture plus its tokenizer is enough to measure this.** The isolated
   decode/re-encode test predicts the in-context outcome exactly — 24 of 24 boundaries on
   the retained Snowball trajectory, and 224 of 224 against a live vLLM (44 predicted
   breaks, 180 predicted holds, no misses, no false alarms). No GPU, no cluster, no live
   arm needed to get the rate for any model.
2. **Decline probability compounds with trajectory length**, so a fixed "~20% of
   trajectories" understates it for long agentic rollouts and overstates it for short ones.
   The per-token hazard for the Snowball serving tokenizer is ≈1.7e-4.
3. **Repair A (token-preserving serving) works and is cheap; repair B (per-turn replay)
   is exact but costs 12× the training tokens** on a 25-turn trajectory.

## The two failure modes are not the same bug

Running the probe against Qwen3 declined 100% of turn boundaries — and none of them were
the #520 mechanism. Qwen3 puts an empty reasoning block (`<think>\n\n</think>`) in the
*generation prompt*, then drops it when the same turn is re-rendered as history, so the
divergence lands four tokens *before* the completion even begins. That is a genuine
full-TITO decline, and SkyRL already has a detector for it
(`detect_qwen3_5_empty_think_prefix`), but it is a template asymmetry, not a re-tokenized
completion.

The probe now classifies every broken boundary as `template` (the sampled IDs survived
verbatim somewhere in the next prompt) or `recut` (they did not). Any measurement that
does not separate these is measuring the chat template.

## Evidence

### The mechanism, on real data

`scripts/tito_repro/hazard_census.py` against the retained arm-3 trajectory and the
serving checkpoint's own tokenizer:

| Measure | Value |
| --- | --- |
| Turn boundaries where the served history broke | 5 / 24 (20.8%) |
| Turns whose sampled IDs fail an isolated round trip | 5 / 25 (20.0%) — the same five |
| Context positions re-cut | 10 / 28,860 (0.035%) |
| Positions changed per incident | 2, in all five |

Per-token hazard from a span sweep on the same completions: 1.0% at 16 tokens, 2.0% at 64,
6.5% at 256, 24.0% at 1024 — consistent with an independent per-token rate of ≈2.6e-4, so
P(trajectory declines) ≈ 1 − exp(−1.7e-4 × sampled tokens). A 29k-token trajectory is
essentially certain to decline; a 3k-token one is about even odds.

The five incidents are all the same shape — one BPE merge moving across a punctuation
boundary:

```
sampled : Ġreturn  Ġ\\  (/  {     if  (flag
served  : Ġreturn  Ġ\\  (   /{    if  (flag
```

### The trainer's view

`scripts/tito_repro/tito_checks.py` vendors SkyRL's seven assembly checks. Replayed against
the retained trajectory it returns exactly what the trainer recorded: `prefix`, turn 9,
mismatch offset 13,387, 257 tokens into turn 8's completion, `[10122, 90]` → `[7, 9573]`.

### Live, against vLLM on a GH200

Qwen3-1.7B, `enable_thinking: false`, 32 trajectories per row, code-heavy tasks with
synthetic terminal observations fed back. `text` is Harbor's transport today; `tokens` is
repair A. Both loops run the same tasks and seeds against the same server.

| turns | boundaries | strict declines | of those, re-cut (#520) | repair A declines | repair A server-echo mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 64 (100%) | 9 (14.1%) | 0 | 0 / 128 |
| 4 | 192 | 192 (100%) | 21 (10.9%) | 0 | 0 / 256 |
| 8 | 448 | 448 (100%) | 82 (18.3%) | 0 | 0 / 512 |
| 16 | 960 | 960 (100%) | 98 (10.2%) | 0 | 0 / 1024 |

Three things to read off this. The re-cut rate — 210 of 1,664 boundaries, 12.6% — is the
same magnitude as the 20.8% measured on the real Snowball trajectory, so the mechanism
reproduces on a different model and a different tokenizer. Repair A removed every decline,
and vLLM confirmed on all 1,920 requests that it ran on exactly the IDs the client sent,
with no empty or malformed completions. And repair A is **not behaviour-neutral**: mean
completion length fell from 288–328 tokens to 201–258, because the model is now
conditioned on its own untouched history rather than a re-rendered one. Adopting it
mid-project changes the rollout distribution.

The 100% strict-decline column is the Qwen3 template asymmetry, not #520: for this model
family full TITO never succeeds, whatever the tokenizer does.

Repair A was also consistently *faster* on the same work — 28.5 s vs 48.0 s for the
16-turn row, and ahead at every turn count — and the mechanism shows up directly in vLLM's
prefix-cache counters: **74.4% hit rate under text transport against 98.0% under token
transport** (32 trajectories, 8 turns). A token-transported prompt is an exact extension of
the previous one, so the KV prefix cache hits all the way; anything that shifts the
rendered history invalidates it from the divergence onward.

Read that gap carefully on Qwen3: the template asymmetry shifts every turn's prefix on its
own, so most of the 24-point difference here is the template, not #520. The Llama-3.1 run
below isolates the re-cut's own contribution.

#### Llama-3.1-8B — the clean measurement

Llama-3.1 shares the Snowball serving checkpoint's tokenizer family (the same 128k vocab,
the same `<|start_header_id|>assistant<|end_header_id|>` generation prompt seen in the
retained capture) and its template has no reasoning-block asymmetry. So here the strict
decline rate *is* the #520 rate, with nothing else mixed in:

64 trajectories per row, both transports against the same server:

| turns | text: trajectories declined | text: boundaries re-cut | repair A declines | repair A echo mismatches | replay cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 22 / 64 (34.4%) | 26 / 192 (13.5%) | 0 | 0 / 256 | 2.6× |
| 8 | 52 / 64 (81.2%) | 85 / 448 (19.0%) | 0 | 0 / 512 | 4.6× |
| 16 | 57 / 64 (89.1%) | 163 / 960 (17.0%) | 0 | 0 / 1024 | 8.6× |

**This is the compounding, measured.** The per-boundary rate is flat at 13.5–19.0% — 274 of
1,600 boundaries, 17.1% overall, matching the 20.8% on the real Snowball trajectory. The
per-*trajectory* decline rate is what moves: 34% → 81% → 89% as turns double. Every single
failure classified as `recut`, not one as `template`, which is the check that the Qwen3
100% really was the template.

Repair A declined nothing anywhere, and vLLM confirmed on all 1,792 requests that it ran on
exactly the IDs the client sent.

The prefix-cache gap here is 1.9–2.1 points (88.8→90.9, 93.9→95.8, 97.0→97.7) rather than
Qwen3's 24, confirming that most of the Qwen3 throughput difference was the template
asymmetry and only a couple of points of it is #520 itself.

## Repairs

### A — token-preserving serving

Carry the conversation as integer IDs and never re-tokenize history: send the accumulated
prompt to `/v1/completions`, append the sampled completion IDs verbatim, and encode each
new observation against a fixed dummy conversation base so its tokens do not depend on
what came before. This is the ApexAgents TITO recipe the issue references.

Validated in the harness: the encoder reconstructs the chat template exactly, with and
without the stop token in the sampled completion, and vLLM confirms it ran on the exact
IDs the client sent.

**Feasibility in Harbor.** Harbor already has the shape of this — `llms/tinker.py` builds
`prompt_tokens` client-side through a renderer — but it calls
`renderer.build_generation_prompt(messages)` every turn, so it re-tokenizes the whole
history exactly like the LiteLLM path. Both backends have the defect; the token
accumulator in `llms/chat.py` only *records* IDs, it never feeds them forward. The serving
change also gives up server-side chat templating and tool-call parsing, and any history
edit (summarization, unwinding) becomes token surgery. vLLM's `/tokenize` endpoint means
Harbor would not need to ship a tokenizer.

### B — per-turn replay

Train each turn against its own served prefix. Exact by construction and needs no serving
change, but instead of one sequence of length `len(prompt[-1]) + len(completion[-1])` it
trains `n_turns` sequences. On the retained 25-turn trajectory that is **12.1×** the
forward/backward tokens (565,164 vs 46,877).

### C — the one the data suggests

Neither repair addresses what actually costs us today: a two-token re-cut discards the
whole trajectory's exact assembly, and nothing counts it. Deciding whether an
all-or-nothing decline is proportionate — versus recording the incident and its blast
radius — is cheaper than either repair and is a prerequisite for judging them.

## Reproducing

```bash
# offline, no GPU: rate + blast radius for any capture + tokenizer
python scripts/tito_repro/hazard_census.py --tokenizer tokenizer.json --capture capture.json

# live, on Vista (gh-dev: 2h, 1 job, no --gres, caches on scratch)
sbatch --export=ALL,DCFT=$SCRATCH/OpenThoughts-Agent-tito,MODEL=<model> \
    scripts/tito_repro/run_probe_vista.sbatch
# then, into the same allocation, without paying model load again:
srun -p gh-dev -N 1 -n 1 -t 00:40:00 --overlap --jobid=<id> bash -c \
    'bash $SCRATCH/OpenThoughts-Agent-tito/scripts/tito_repro/sweep.sh'
```
