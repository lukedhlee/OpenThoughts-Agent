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

**Token-preserving serving works.** Two agent loops against one vLLM on a GH200, same
tasks and seeds — one resending the message list (what Harbor does today), one carrying
the conversation as integer IDs with observations encoded against a fixed dummy base.
Across 832 turn boundaries the token loop declined zero, with the server confirming it ran
on exactly the IDs the client sent; the text loop re-cut 6–20% of boundaries depending on
turn count. The token loop was also faster at every turn count (28.5 s vs 48.0 s on the
16-turn config), which is consistent with the re-cut invalidating vLLM's prefix cache — so
this may be costing generator throughput too.

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
