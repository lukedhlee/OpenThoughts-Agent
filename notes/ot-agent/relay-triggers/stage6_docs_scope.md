# Stage 6: coworker docs and examples

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why:** the point of the abstraction is that coworkers can add triggers without reading the router.

**Change-set:** a "write a trigger" section in `data/relay/README.md` (the interface, the YAML schema, the knobs table,
offline replay first, then online with `log_only`, then acting); `data/relay/triggers/examples/` with a 20-line rule
trigger and a judge config.

**Gate (GO/NO-GO):** a fresh agent given only the docs writes a toy trigger, replays it on recorded episodes and
produces the report. No compute.

**Depends on:** stage 1 for the offline part, stage 3 for the online part.
