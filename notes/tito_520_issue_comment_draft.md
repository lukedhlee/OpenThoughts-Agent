# DRAFT — not posted

A comment for [MarinSkyRL#520](https://github.com/marin-community/MarinSkyRL/issues/520),
written for Luke to review. Nothing has been posted, commented, or opened upstream.

---

Reproduced end to end against a live vLLM, plus two numbers that change what the fix
should be.

**The blast radius is two tokens, and the decline is all-or-nothing.** On the retained
arm-3 trajectory, 5 of 24 turn boundaries break, and each one re-cuts exactly two token
positions — 10 of 28,860 context tokens, 0.035%. Those ten tokens discard full-TITO
assembly for the whole 25-turn trajectory, and `tis/exact_match_fraction` still reads 1.0.
Whether an all-or-nothing decline is the proportionate response to a two-token re-cut
seems worth deciding before either repair lands.

**The rate is a function of trajectory length, not a constant.** Decoding each turn's
sampled IDs and re-encoding them in isolation predicts the in-context outcome exactly —
24 of 24 boundaries agree — so any recorded capture plus its tokenizer measures this
without a GPU. A span sweep on those completions gives a per-token hazard of ≈2.6e-4, i.e.
P(decline) ≈ 1 − exp(−1.7e-4 × sampled tokens): near-certain for a 29k-token rollout,
about even odds at 3k. A single "~20% of trajectories" figure will over- and under-state
it depending on rollout length.

**The per-boundary rate is flat; the per-trajectory rate compounds.** On Llama-3.1-8B —
same tokenizer family as our serving checkpoint, and a template with no reasoning-block
asymmetry, so the strict decline rate *is* this bug — 64 trajectories per row:

| turns | trajectories declined | boundaries re-cut |
| ---: | ---: | ---: |
| 4 | 22 / 64 (34.4%) | 26 / 192 (13.5%) |
| 8 | 52 / 64 (81.2%) | 85 / 448 (19.0%) |
| 16 | 57 / 64 (89.1%) | 163 / 960 (17.0%) |

274 of 1,600 boundaries, 17.1%, against 20.8% on the retained trajectory.

**Token-preserving serving works.** The same harness runs a second loop that carries the
conversation as integer IDs, with observations encoded against a fixed dummy base. It
declined zero across every row, with the server confirming on all 1,792 requests that it
ran on exactly the IDs the client sent, and no empty or malformed completions. It also
lifted vLLM's prefix-cache hit rate by about 2 points at each turn count, so the re-cut is
costing a little generator throughput as well.

One caveat worth stating: repair A is not behaviour-neutral. Mean completion length fell
(214→197, 206→193, 201→189 tokens), because the model now conditions on its own untouched
history rather than a re-rendered one. Adopting it mid-run changes the rollout
distribution.

**Per-turn replay costs 12×.** Training each turn against its own served prefix is exact
and needs no serving change, but on the retained 25-turn trajectory it is 565,164
forward/backward tokens against 46,877 for the single linear sequence.

**One caution for anyone measuring this.** Qwen3 declined 100% of boundaries in the same
harness, and none of it was this bug: the template puts an empty `<think>\n\n</think>`
block in the generation prompt and drops it when the turn is re-rendered as history, so
the divergence lands four tokens before the completion begins. A measurement that does not
separate "the sampled IDs came back re-cut" from "the template rendered the turn
differently" is measuring the template.

**On the Harbor side**, `llms/tinker.py` already builds prompt token IDs client-side, but
it calls `renderer.build_generation_prompt(messages)` every turn, so it re-tokenizes the
whole history exactly like the LiteLLM path. Both backends have the defect; the ID
accumulator in `llms/chat.py` records what was served but never feeds it forward.

Harness and full method: `scripts/tito_repro/` on `lukedhlee/tito-tacc-repro`.
