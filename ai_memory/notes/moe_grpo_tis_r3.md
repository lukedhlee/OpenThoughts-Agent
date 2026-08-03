# MoE GRPO — how on-policy our runs actually were (staleness / TIS / R3) + the colocation bake-off

Answers two questions we kept re-deriving: (1) were the `_fast` gsm8k GRPO runs off-policy or broken —
**no, they were on-policy and trustworthy**; (2) is colocation viable for 30B-A3B on GH200s — **no,
disaggregated wins decisively.** Also records the measured vLLM↔FSDP2 logprob gap and why R3 was killed.
Read before changing trainer config, before re-litigating colocate-vs-disagg, or before building
router-replay.

Carved out of `handoff.md` 2026-07-29 (append-only from here).
**Full writeup with all sections:** `ai_memory/logs/2026-07-26_moe_grpo_offpolicy_tis_r3.md` (§9, §10).
Decisions this produced: [[decisions]]. Related: [[jupiter_bringup_and_throughput]].

---

## Finding (2026-07-26)

The `_fast` gsm8k GRPO configs run `use_tis: false` + `moe_router_replay: false`. They inherit
`max_staleness_steps: 4` but that is a **RED HERRING** — it's a `fully_async`-trainer knob and these
runs use the sync `main_base` trainer (staleness hardwired to 0; `trainer.py:1628`), so they were
**on-policy on the data axis all along**; `max_staleness_steps: 0` is a no-op (don't bother). Runs
trustworthy, not broken.

**Key nuance:** the small `policy/log_ratio ≈ 0.02` is the *update* axis — NOT the generator-trainer
(vLLM↔FSDP2) gap, which is `tis/log_ratio` and was **unmeasured** (TIS off).

→ **MEASURED** via job `1045840` (TIS-on 6n rerun): `tis/log_ratio_abs_mean` ≈ **0.030 nats**
(imp_ratio ≈ 1.009), routing tail `tis/imp_ratio_capped_fraction` ≈ **0.035%** → mismatch small,
routing tail negligible.

**DECISION: keep TIS on for MoE (cheap insurance); DO NOT build R3** (would recover ~nothing; revisit
only for long-context / larger-MoE / high-staleness).

## Throughput bake-off (§10 of the log): DISAGGREGATED WINS decisively

Colocation is NOT viable for 30B-A3B on these GH200s (4 distinct failures: TIS/batched →
expandable_segments → 24-GPU expert-sharding → weight-sync OOM). Disagg 6n runs clean ~430s/step
(~8.3 steps/hr), reward 0.47→0.57 by step 3. The "colocation faster / Ben 14 steps/hr" idea is
**FALSIFIED** for this setup.

**Fork gotcha:** TIS needs `generator.batched: true` here (attempt 1 `1045833` crashed on
auto-set-logprobs vs the `batched:false` guard); the "proven" `batched:false` TIS configs are stale on
this fork.
