# Stage 2: wire the router, same behavior

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why:** the refactor must be proven harmless before any new behavior goes in, because the same router serves every
live relay run.

**Change-set:** `relay_router.py`: the request-time block (890-903, 968-999) becomes `controller.on_request(ev)`, the
reply-time decision block (1249-1273, 1116-1140) becomes `controller.on_reply(ev)`. Endings, autofix and the parse check
stay in front. Today's flags (`--assist`, `--time-takeover-frac`, context budget, trigger config, repair arm) build a
legacy trigger config, keeping today's order: environment, then context, then time, then assist; Terminus-2 checks
the other decision triggers before done-claim, MSA checks done-claim first. Legacy record fields are written exactly as
today.

**Gate (GO/NO-GO):**
- the 81 existing tests pass unchanged;
- a router replay: recorded request bodies of past runs (Terminus-2 `relay_repair`, MSA with assists and time
  takeover, a `control` run) go through the old and the new router against stub endpoints that return the recorded
  replies; owner sequence and legacy fields are identical on 100 % of turns.
CPU only.

**Depends on:** stage 0.
