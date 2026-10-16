# Stage 0: interface and controller

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why first:** every later stage (replay, router wiring, judge) codes against these types. Getting the controller's
state machine right on its own, with no router in the loop, is the cheapest place to settle the semantics of
`rewrite` / `next`, spans, hand-back, cooldown, caps and retries.

**Change-set:** new `data/relay/router/interventions.py` (Event, Reply, Clock, Decision, Intervention, Trigger
protocol, Controller, YAML config loader with `module:Class` plugin import) and
`data/relay/router/tests/test_interventions.py`. No existing file changes.

**Semantics to pin in tests:**
- Owner of a request = teacher while sticky or while `turns_left > 0`; `turns_left` drops only on main, non-retry
  teacher turns (summaries and retries don't count).
- A reply-phase `rewrite` routes the same request to the teacher and counts as teacher turn 1 of the span.
- A reply-phase `next` is stored and applied at the next request-phase point; a request-phase decision is always `next`
  (validated; a request-phase `rewrite` is a config error).
- After the span ends the student resumes; triggers re-arm only after `cooldown` student turns, so a loop detector
  doesn't re-fire on the same window at once.
- `max_interventions` and `earliest_turn` are hard limits; a decision past them is logged as suppressed.
- Same request (retry key) gives the same routing.
- Several fires on one turn: config order wins, all are logged.

**Gate (GO/NO-GO):** a scenario table covering rewrite/next x span {1, 2, -1} x {cooldown, cap, retry, two triggers
firing} passes; `git diff` touches no existing file. CPU only.

**Depends on:** nothing.
