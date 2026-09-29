# PedaGEPA teacher prompts for M3 and M4 (2026-09-29)

The teacher (Qwen3.8-27B) sees these files and training rows never do: the router passes them with `--teacher-system-file`
and `--inject-plan`. Sources: the ideal-agent spec, the M1 table and failure modes (`m1/`), rubric v1.3.1, the CalibForge
and MSA reads, and the round-2 Opus judge lines on Qwen runs (`inspect/r2/opus{1,2}`). Word counts are `wc -w`.

## guided_system.md (M3 guided arm, 418 words)
- **What it targets.** The M3 composite, which is M1's "both lack" items: P0 (Qwen 1.25 CalibForge / 0.82 TB2.1 / 0.81
  SWE, scoring 2 in 25 % / 0 % / 0 % of runs), P5a (1.19 TB2.1 / 1.06 SWE), P5b (0.94 / 1.19), P9 on CalibForge (1.50),
  and S1 on SWE (1.31).
- **P0.** Qwen is almost never rejected. It gets "Extra text detected before JSON object" on nearly every reply
  because it opens with a sentence ("I'll start by exploring..."). So the block says the first character is `{`, and
  that the announcement belongs in `plan`.
- **P5a / P5b.** Qwen cats whole files, re-prints diffs, and once let an apt-get log take 51 % of the window. It writes
  9-25k-token replies that work out by hand what a script computes, and one reply ran to 63k tokens in a loop. Its own
  re-fed reasoning is 63 % of a dying CalibForge context.
- **P9.** Qwen's checks fail in three ways. They share the assumption that produced the answer (reversed flag bytes;
  'Unknown' dropped by both the code and the check). They skip a sibling case (unnamed groups in django-11728). Or they
  ignore the grader's clean environment (pickling a class from site-packages).
- **S1.** Qwen prints the stated value instead of asserting it, writes the asserting test only after the fix, or fixes
  before writing any repro.
- **Left implicit.** Everything Qwen already scores 1.8-2.0 on: P1-P4, P6-P8, P10, S2-S4.

## clean_system.md (M4 arm 1, 622 words)
- **Structure.** guided verbatim plus a "Working record" block of norms for the `analysis` field.
- **What it targets.** The student's share of judged runs scoring 0 (CalibForge / TB2.1 / SWE): P6 0.67 / 1.00 /
  1.00, P8 0.73 / 0.92 / 1.00, P9 0.75 / 0.88 / 1.00, P10 0.88 / 0.58 / 1.00, P1 0.56 / 1.00 / 0.93, P7 0.47 / 0.60 /
  0.62, P2 0.44 / 0.73 / 0.33, P4 0.41 / 0.33 / 0.50.
- **What each norm asks for.** One or two sentences where the student breaks:
  - the contract first;
  - the actual key line before any interpretation;
  - the fact behind a step and where it comes from;
  - the cause of an error before the fix, with its own last change as the first suspect;
  - why an approach failed before switching;
  - what a check would show if the work were wrong.
- **Example fact.** PEP 668 (externally-managed-environment), not the brief's PyStan example. PyStan 3 is the answer to
  TB2's rstan-to-pystan task, and TB2 is the target eval.
- **Framing.** Everything is written as the agent's own way of working. No audience is named.

## recovery_system.md (M4 arm 2, 668 words)
- clean verbatim plus one norm. When output shows that an earlier step of its own was wrong, the analysis says what
  went wrong and why, then the agent fixes it. It never presents the step as intended.
- This makes the turn after an injected mistake own the mistake instead of quietly routing around it.

## inject_plan_tb.json and inject_plan_swe.json (recovery arm)
- **Mechanics.** `clean_frac` is 0, so every episode draws one mode. Its note is appended to the teacher's view of one
  request, and render.py masks that turn.
- **Weights.** Each mode's weight is the share of judged student runs scoring 0 on its item, normalized. TB pools
  CalibForge and TB2.1 by scored runs; SWE uses SWE-100. The one floor is SWE guessed_api, raised to 0.08 from 0.053,
  because guessing decides 31 % of TB2.1 failures. I checked the draws on 20k fake session ids with the router's hash
  and they match.
  - TB: unchecked_belief .19, retry_unchanged .19, premature_done .17, dropped_requirement .17, guessed_api .13,
    blind_edit .10, output_dump .05.
  - SWE: unchecked_belief .16, retry_unchanged .16, premature_done .16 (P9, P10 and S4), dropped_requirement .15,
    weak_repro .14, blind_edit .09, output_dump .09, guessed_api .08 (the router normalizes the 1.03 sum).
- **Timing.** The median position of the student's first zero, times Qwen's episode length (CalibForge median 9
  replies, SWE-100 median 24). Checked against real Qwen runs:
  - `after_write` fires on the first write from `min_turn`, not the last. So TB premature_done lands at a median 0.56
    of the episode instead of 0.9, and 26 % of episodes never fire it. SWE fires at 0.56, with 22 % never firing.
  - weak_repro fires at turns 2-5, when repros are written, not at the judged 0.77.
- **Texts.** Each note says: make this mistake in this reply only, write it as your genuine plan, keep the thinking
  short, don't mention the note, and work normally from the next reply. Every mistake is set up so the next output can
  expose it:
  - the unchecked belief drives the command;
  - the premature claim meets Terminus-2's confirmation;
  - the blind edit hits a file already being edited.

## Risks
1. **The injected turn's reasoning leaks forward. This is the biggest risk.** The router re-feeds that turn's
   reasoning to the teacher on later requests (`for_teacher`), and render.py keeps the first 1,000 tokens of older
   teacher reasoning in the history of every training row. Qwen will likely think "the note asks me to..." on that turn.
   Fix before the pilot: drop that turn's reasoning in both places, and add patterns such as "note", "was asked to",
   "system prompt" and "guideline" to LEAK_RE for later turns.
2. **LEAK_RE misses mentions of the prompt itself.** Phrases like "per the guidelines" or "the system prompt says JSON
   only" are not in LEAK_RE. The closing line of each prompt tries to prevent them.
3. **LEAK_RE flags normal task phrasing.** "The instructions say" and "I was asked to" are natural things to write
   about the task, and they trigger resamples. Read `leak_hits` before trusting the M3 gate (under 2 %).
4. **The after_error trigger fires on benign text.** In 60 sampled Qwen CalibForge runs it fired in 58, but about half
   the matches were "error" inside source code or flags (`--error-exitcode`, `print("Error: ...")`). The retry text
   falls back to re-sending the previous command. Tightening INJECT_ERR_RE to real failures (a traceback, "command not
   found", "No such file", a non-zero exit, a compiler `error:`) would make the mode truer.
5. **Longer replies.** The explanation norms add text to the analysis, and the student already dies of long replies.
   Watch generated tokens per turn (the 1.5x gate) and analysis length.
6. **Over-bounding.** `head` or `grep` can hide the one line that matters.
7. **Some injections are unnatural, unrecoverable or no-ops.**
   - premature_done can fire right after a probe write, which makes it a claim over obviously unfinished work.
   - output_dump can fill the window by itself, so it has a low weight.
   - A blind `sed` on an untracked TB input with no backup may be unrecoverable.
   - If the guess was right or the retry succeeds, nothing goes wrong, so there is nothing to recover from. C2 counts
     logged injections, so the Opus read has to check that the mistake actually happened.
8. **The weights rest on small samples biased toward failures.** There are 16-20 judged runs per benchmark, drawn from
   Qwen-only-pass and both-fail tasks. On SWE almost every item's zero share is at the 1.00 ceiling, so the SWE weights
   are nearly flat.
9. **The prompts state no context size.** SWE-100 ran at 32k and TB at 65k. The format paragraph assumes Terminus-2
   JSON replies, so a pool that runs native tool calls needs that paragraph rewritten.
