# Stage 1: offline replay harness

**Status (2026-10-15):** scoped, no code. Companion: `README.md`.

**Why now:** coworkers' first job is finding failure modes, which is offline work. With replay they can write and
score triggers on recorded episodes while the router wiring (stages 2-3) is still in progress. It needs only stage 0.

**Change-set:** new `data/relay/triggers/replay.py` and tests. Inputs: run dirs (`turns.jsonl`, `bodies/ep*.jsonl.gz`,
the census / verifier outcome), a trigger config. It rebuilds each turn's request-phase and reply-phase Event in order
(never from the finished episode), feeds the triggers, and writes:
- per episode: first fire per trigger (turn, phase, reason, score), outcome, and the labelled point of no return when a
  label exists;
- a summary: fire rate on passing vs failing episodes, lead time before failure, fire-turn histogram, per-trigger
  precision against labels.

**Scope limit:** fire-vs-outcome is only clean on student-alone episodes (`student_only` arm, solo runs). On relay
episodes the history after the first intervention contains teacher turns, so replay reports those separately.

**Gate (GO/NO-GO):** on recorded student-only runs, replaying the built-in `loop` and `no_progress_wait` reproduces the
router's logged `would_fire` turn for turn (100 %), and the report runs end to end on >= 1,000 episodes. CPU only.

**Open:** count the student-only and solo episodes with outcomes on Horizon (MSA and Terminus-2) before starting.

**Depends on:** stage 0.
