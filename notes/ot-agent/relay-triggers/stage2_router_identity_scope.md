# Stage 2: wire the router, same behavior

**Status (2026-10-16):** done on `lukedhlee/relay-triggers-s2`, gate GO. The router asks one Controller per episode
on the config its flags stand for (`router_triggers.legacy_config`), or on `--trigger-config <yaml>` in their place.
Replayed through the router before and after, 8,165 recorded requests (160 episodes, 10 runs) give byte-identical
records, upstream calls and responses; the same with every request sent twice (a transport retry); and the YAML
equivalent of each run's flags routes the same. Two retry rules are kept from today: on a retry of an assisted request
the budgets are asked again before the teacher answers it again, and on Terminus-2 a retry keeps a pending done claim.
Companion: `README.md`.

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

**How to rerun the identity gate** (stage 3 must keep it): `data/relay/router/router_replay.py run --router <old
relay_router.py> ...` and `... --router <new>`, then `router_replay.py diff <a> <b>` (0 mismatches); `--inject-retries 1`
for the retry paths, `--trigger-config-from-flags --all-modes` with `diff --ignore decisions,intervention` for the YAML.

**Stage 2b (2026-10-16):** today's Terminus-2 retry rule is kept too (a retry of the request a done claim was made on
leaves it pending; the Controller alone drops it), pinned by two tests that pass on the 35ca0c95 router and fail
without it. The router fills Event.last_result from `action_results.py` (shared with replay.py), so `action_error` /
`action_ok` work live; a release that ends a takeover gives the student the episode back. Same gate, all green.

**Depends on:** stage 0.
