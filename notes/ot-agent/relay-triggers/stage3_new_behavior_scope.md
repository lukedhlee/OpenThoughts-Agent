# Stage 3: new behavior (spans, hand-back, several interventions)

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why:** this is what today's code can't express: `k >= 2`, handing back after a takeover, more than one takeover per
episode, general cooldowns, and `next` for any reply-time trigger.

**Change-set:**
- Router: owner comes from the controller, not the one-way `take_over` (1183); summarization and `/tokenize` route by
  the current owner (826-829, 436); student view after a hand-back follows the policy knob (`for_student`).
- Records: every intervened record gets `intervention = {id, trigger, where, span, i}` next to the legacy flags; every
  fire (acted, suppressed, shadow) goes into a `decisions` list.
- Consumers move to the new field: `readout.py` (`repair_view`, H4, takeover views), `horizon/arm_gates.py`,
  `msa/gate.py`, `select_kept` (`stalled`, `teacher_worked`), and `--student-view-after-takeover` in `render_msa.py`.

**Gate (GO/NO-GO):** scenario tests for spans and hand-back pass; the stage-2 identity replay still holds for the legacy
config; a synthetic k=2 hand-back episode renders teacher turns trained and student turns masked, and readout reads it
without false H4 failures. CPU only.

**Depends on:** stage 2.
