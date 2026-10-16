# Stage 5: online smoke

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why:** the first end-to-end check that the new knobs work in a real relay run and produce trainable rows.

**Change-set:** none in code; a run config. One reply-time rule trigger in four arms (rewrite x {1, -1}, next x {1, -1})
plus one shadow arm (same trigger, `log_only`) on ~32 CalibForge tasks, Horizon, through `serve_submit.sh` and
`launch_driver.sh`.

**Gate (GO/NO-GO):** no router errors; every `intervention` record matches the controller log; spans and hand-backs
appear where the config says; `render_think_limit.py` produces rows for every arm; readout runs. Outcome differences
are not judged at this size. ~20-30 node-h.

**Depends on:** stage 3 (stage 4 optional).
