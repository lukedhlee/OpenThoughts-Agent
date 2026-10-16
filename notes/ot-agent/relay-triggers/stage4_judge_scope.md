# Stage 4: LLM judge and scripted triggers

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why:** an online judge is the main non-rule trigger coworkers want, and it is the one with a real cost: latency.

**Change-set:**
- `llm_judge` built-in: prompt template over the student's view (never grader files or teacher-only text), threshold,
  timeout (a timeout is "no decision", logged). Reply phase: called before the harness sees the reply (serial). Request
  phase: started as a task when the previous turn finishes and awaited at the next request, so it overlaps tool time.
- Router: a `judge` endpoint in the URL map, model list and health check; judge time paused on the agent clock by
  default (today only chat calls pause it, 1214 / 1335).
- `scripted` built-in: fire at given turns or from a per-task plan file (controlled experiments, oracle labels),
  after pedagepa's `maybe_inject` plan format.

**Gate (GO/NO-GO):** on one Horizon node, judge latency per turn (p50/p95) measured in both phases; no deadline or
budget regressions in a 4-task run; offline replay of the judge on 200 recorded episodes. ~2 node-h.

**Depends on:** stage 3 (online), stage 1 (offline replay).
