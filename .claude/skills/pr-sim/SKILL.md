---
name: pr-sim
description: Simulate a PR in chat before any branch or file exists, so Luke can review the idea as a reviewer would. Use when he says "/pr-sim", "simulate a PR for X", or "show me what the PR would look like".
---

# pr-sim: the PR as a chat message, for review first

Luke understands a change fastest by reading it the way a reviewer would, not as an explanation. So before cutting a
branch or writing a draft file, write the PR in the chat, in the exact house style of `pr-marin`, and end by asking
for review. Nothing is written to disk and nothing is pushed.

Invoked with a feature name or a row from the PR queue → write that PR. Invoked bare → write the next one on the
ranked list.

## Shape (identical to a real body, see pr-marin "House style")

1. **Title line**, bold: `[<area>] <what changes, plain words>`; the feature it belongs to is in the title.
2. **First paragraph, bold opening sentence: the impact.** Who is hurt, how, in one sentence a reviewer with no
   context understands. Then the measured effect in one or two sentences.
3. `Related: ...` links.
4. **What goes wrong.** Mechanism in plain words.
5. **The fix.** Two to four bullets.
6. **Measured.** Before/after, same setup, counts.
7. **Tests.** Files and suite count as they stand on the feature branch today.
8. Close with one line inviting review: "what is unclear, what would you push back on?"

Every number must be real (from the notes, the branch, or a test run); if one is not measured yet, say "not
measured" instead of inventing it. Keep it under ~250 words. Plain words, ideas compressed, not sentences.

## After his review

- Fold his pushback into the mental model, then answer only what he asked.
- When he says "make the PR" / "let's do it", switch to `pr-marin` and use the reviewed text as the draft body.
