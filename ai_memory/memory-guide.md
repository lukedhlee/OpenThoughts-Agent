# `ai_memory` Guideline

How to keep agent memory cheap to read and lossless to store.
*Lives at `ai_memory/memory-guide.md`. Read it when **writing** memory, not when resuming work.*

## The one idea

Cost is not file size — it's **size × read frequency**. `handoff.md` is read in full at every
session start, so every line in it is paid for forever. Everything else is read at most once,
when actually needed.

So: keep detail *outside*, at full fidelity, and make `handoff.md` a state file with pointers
good enough that an agent knows **when** to open them.

## Two tiers

|                  | `handoff.md`                | everything else                 |
| ---------------- | --------------------------- | ------------------------------- |
| read             | every session, in full      | on demand, rarely               |
| lifetime         | mutable, rewritten          | append-only, permanent          |
| holds            | current state               | history and rationale           |
| budget           | ~150 lines / ~1.5k tokens   | unbounded                       |
| loss tolerance   | lossy by design             | zero                            |

**Core invariant:** nothing leaves `handoff.md` unless it was first appended somewhere permanent.
This is what makes aggressive pruning safe, and what makes the whole structure near-lossless.

## What goes in `handoff.md`

It must answer three questions in 30 seconds: *Where are we? What's next? What must I not break?*

1. **Objective** — 1–3 lines. What this work is for.
2. **State** — done / in progress / blocked. Terse, current only.
3. **Next action** — exactly one unambiguous step.
4. **Invariants & constraints** — things that break silently if violated (env quirks, API shapes, "never regenerate X").
5. **Decisions in force** — one line each, with a pointer to the rationale.
6. **Open questions** — what's undecided and who/what decides it.
7. **Map** — annotated pointers to the archive.

Rules:

- One line per item. If it needs a paragraph, it belongs outside with a one-line stub here.
- Only what is **currently true**. Superseded facts move out, they don't get struck through.
- Never restate what the repo already says — file trees, signatures, `git log`, test names.
  An agent can read code; it cannot reconstruct intent. Store intent, not inventory.

## What goes outside

Everything valuable but not needed *right now*: rationale, rejected alternatives, failed attempts,
run logs, benchmark tables, design explorations, debugging narratives, dead ends.

Write these at **full fidelity — no summarizing, no compression.** This is where losslessness lives,
and it's nearly free because these files are rarely read.

```
ai_memory/
  handoff.md          # hot: state + index. Rewritten each session. Hard budget.
  memory-guide.md     # this file
  decisions.md        # append-only: dated decision + context + rejected options + consequences
  gotchas.md          # append-only: symptom → cause → fix
  log/2026-07-29.md   # append-only session journal: what was tried, what happened, raw output
  notes/<topic>.md    # one topic per file, stable filename, 2-line abstract at the top
```

Every file outside `handoff.md` is **append-only**. Corrections are new dated entries that
supersede old ones, never edits. A rewritten history is a lost history.

Each `notes/` file opens with a 2-line abstract, so an agent that opens it can bail immediately
if it's the wrong file — and so the `handoff.md` pointer can be regenerated from the file itself.

## The pointer rule

A pointer is lossless only if the agent can decide **without opening it**.
Every pointer = *path + what's inside + when to read it*.

- Bad: `See notes/training.md`
- Good: `notes/grpo-moe.md — routing-mismatch debugging log + R3 config that worked. Read before changing trainer config.`

## Triage test

Ask, in order:

- Derivable from the code or `git log`? → **write it nowhere**
- Would an agent make a wrong decision in the next hour without it? → **`handoff.md`**
- Needed to reconstruct *why* something was chosen? → **`decisions.md`**, stub in handoff
- Needed only when revisiting one specific topic? → **`notes/<topic>.md`**
- A failure that cost more than 10 minutes? → **`gotchas.md`**
- Everything else that happened? → **`log/`**

Rule of thumb: if it's read in fewer than 1 in 5 sessions, it does not belong in `handoff.md`.

## End-of-session protocol

1. Append to `log/<date>.md` — raw and verbose, this is the cheapest place to be thorough.
2. Append any new decision to `decisions.md`, any new failure mode to `gotchas.md`.
3. **Rewrite `handoff.md` from scratch.** Don't patch it — patched handoffs accrete dead state.
4. Check the budget. If over, prune by *moving out*, never by deleting.
5. Verify every pointer still resolves and still describes its target.

## Anti-patterns

- `handoff.md` as a changelog — it's a snapshot, not a diary.
- Summarizing detail files "to save space." They're not on the hot path; summarizing only loses information.
- The same fact in two places. Copies drift, and then neither is trustworthy.
- Bare links with no scent — forces the agent to open everything, which is the expensive outcome.
- Deleting a stale line without appending it anywhere. That's the only way this structure loses data.