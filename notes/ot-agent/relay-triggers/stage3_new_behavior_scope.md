# Stage 3: new behavior (spans, hand-back, several interventions)

**Status (2026-10-16):** done on `lukedhlee/relay-triggers-s2` (router a07e2e6b, consumers 7edd2b1b), gate GO. Under
`--trigger-config` every knob of the interface acts in the live router; without it routing, records and every
consumer's output stay byte-identical. Companion: `README.md`.

**What a config can now do online:** spans of k turns that hand back, several interventions per episode with cooldown
and caps, releases (e.g. `action_error` until `action_ok`), `rewrite` or `next` for any reply-time trigger, the three
view knobs per intervention, a `teacher_note` on any intervention's first teacher turn, named endpoints
(`endpoints: {name: url | {url, model}}`; an entry's `teacher:` picks one for its teacher turns, a trigger calls one
through `Event.extra['call_endpoint']`). Summarization and `/tokenize` go to the current owner (a span counts).

**Decision, the clock:** under `--trigger-config` the student's agent clock leaves out the teacher's turns (from a
request the teacher answered to the next request: its call and its action's run), so a hand-back does not charge the
student for the teacher's work; the teacher's budget counts its turns only; the run's deadline stays wall time;
`time_budget` reads the student's clock. Records carry `teacher_clock_sec`. Without the flag the clocks are today's.

**Records (only under `--trigger-config`):** every teacher turn carries `intervention = {id, trigger, where, span, i,
teacher, start_turn, view}`; `decisions` holds every fire (acted, pending, suppressed and why, shadow, releases);
events `intervention` and `handback`; `endpoint_calls` for a trigger's own model calls.

**How to rerun the gate:** the identity replay as in stage 2 (`router_replay.py run/diff`, plain and
`--inject-retries 1`); `router_replay.py run --trigger-config <yaml>` then `router_replay.py check <out>` (every teacher
turn under its intervention, turns counted 1..k, starts acted and announced, hand-backs logged, sticky ones released);
the consumers on recorded runs, old tree vs new, with the same `PYTHONHASHSEED` (the old readout's would-fire table
follows a set's order); `tests/test_router_stage3.py` and `tests/test_stage3_consumers.py`.

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
