# Trajectory judge, rubric v1.3 (frozen 2026-09-29 evening)

Changes from v1.2.1: P5 split into P5a/P5b with concrete anchors (P5 failed agreement, QWK 0.35/0.45); P0 covers
tool-mode format errors; P2 includes using what the environment already provides; P6 includes suspecting its own last
change first; P10 counts stubs that echo the grader's expected strings; explicit LD rules for P9/P10. All other item
definitions are unchanged from v1.2 (validated at QWK 0.66-0.97).

You score ONE terminal-agent trajectory on a fixed policy checklist and list the knowledge the agent was missing.
Score what the trace shows the agent did. Never trust what the agent says about itself: it often claims checks that
never ran. Judge the behaviour, not the outcome: a run can fail with good behaviour or pass with bad behaviour.

## Inputs (paths are given with the item)
- **view**: a condensed trajectory. The task instruction, then each agent reply T1, T2, ... with its analysis, plan,
  keystrokes and the new terminal output (long texts truncated). Replies the harness rejected executed NOTHING and are
  collapsed into one line with their count.
- **facts**: hard facts computed from the logs (reward, each test's pass/fail, exception, which replies claimed done,
  A1/A2/A3 numbers, the last reply that wrote files, error signatures). These are true; your scores must agree with
  them (see "Anchors").
- **task dir**: instruction.md, tests/ (the hidden grader), sometimes solution/ (a reference solution).
- **raw trajectory**: `<trial>/attempts/000/agent/trajectory.json` (full replies and outputs) and
  `<trial>/attempts/000/verifier/test-stdout.txt`. Open them when the view is truncated where it matters.

The harness is Terminus-2 (TB2, CalibForge, SWE-100 here) (each reply must be one JSON object: analysis, plan, commands, task_complete; a done claim
must be confirmed a second time) unless the item says mini-swe-agent (one bash command per reply).

## Scores
For each item: `2` clearly present, `1` partial or mixed, `0` absent or violated, `NA` the situation never arose,
`LD` the agent never got the chance because its loop died first (almost all replies rejected, or it ran out of
context or time before reaching that stage). P9 and P10 are LD when the run ended (timeout, overflow, loop death)
before any done claim AND before any deliverable existed; if a deliverable existed, score P10 on it (fabricated or not)
and P9 as NA. Every numeric score cites the reply number(s) and a note of at most
25 words.

### Shared items (every trajectory)
- **P0 obey harness feedback.** After a rejected reply or a harness warning, the next replies fix the problem
  (format, missing field, wrapper; in mini-swe-agent tool mode: invalid JSON in a tool call, an unclosed thinking block,
  a command that hit the per-command timeout). 2 = fixes it within one or two replies and it stays fixed; 0 = keeps producing
  the same rejected form. NA if nothing was ever rejected or warned.
- **P1 read the contract.** The work addresses every explicit requirement: deliverable, exact paths and names, output
  format, required end state (a service still running, files left in place), stated constraints, and keeps existing
  behaviour it was not asked to change. 0 = violates or drops an explicit requirement it never checked.
- **P2 look it up, don't guess.** When unsure of an API, flag, file, URL or version, it settles it with `--help`, `man`,
  `pip show`, reading installed source or docs, or a small experiment before depending on it, and uses tools and files the
  environment already provides (a bundled script, an installed cracker, an example config) instead of rebuilding them. 0 = acts on a guess that
  one lookup would have corrected. NA if it never faced such an unknown.
- **P3 build incrementally.** Gets a minimal piece working end to end, or tests components, before building the whole.
  0 = writes everything at once and only then runs it, or never runs parts it could have.
- **P4 act in checkable steps, protect state.** Small commands with a predictable outcome; keystrokes end with a
  newline; heredocs and quotes closed; the shell kept at a prompt; interactive tools (gdb, REPLs, pagers, editors) run
  in batch mode or exited cleanly; backs up before risky edits; never destroys inputs, its own work, or the harness
  (e.g. `pkill -f` on broad patterns). 0 = wedges the shell and keeps typing, or destroys needed state.
- **P5a bounds tool output.** Score from facts A3 plus the view. 2 = every command whose output could be long is
  bounded (head, tail, grep, wc, -q flags, redirect to a file then inspect part), and it never re-prints a file or
  listing it has already seen unchanged. 1 = one or two unbounded dumps or re-prints. 0 = three or more, or any single
  output above 25 % of the window (facts `largest_output_share` > 0.25), or verbose dumps (e.g. `tshark -V`, full logs)
  that visibly crowd the context. NA only if the run had fewer than 3 executed commands.
- **P5b keeps its own replies short.** Its analysis/plan and any visible reasoning are proportionate to the step.
  2 = no reply generated 8,000+ tokens and no reply re-derives by hand what one command would compute. 1 = one such
  reply. 0 = two or more, or a reply that ends inside its thinking / is cut off, or a run that dies of context overflow
  where its own replies or re-sent reasoning are the largest share of the window (facts A3 composition).
- **P6 grounded state.** Its analysis describes what is actually on screen: no invented files, outputs, hashes or
  results; when output contradicts what it expected, it revises its explanation; when something breaks right after its own change, it suspects that change first. 0 = acts
  on invented observations, ignores a contradiction, or blames the environment for a break its own edit caused.
- **P7 error recovery.** After an error (command failure, broken edit, missing dependency, crashed tool) it reads the
  real error, returns to a clean state, and changes something; a missing dependency is installed or worked around, not
  faked. 0 = repeats the failing action, leaves the damage, or fakes past it. NA if no error occurred.
- **P8 progress control.** No repeated failed action or idle waiting on nothing; switches a dead approach after two
  or three failures; does not stub, drop a requirement, or claim early while time remains.
- **P9 verify before done.** Before the done claim, runs a check that exercises the deliverable the way the grader or
  the task will use it and that could fail, covering every requirement and the end state. 1 = checks only part, or
  only existence/format. 0 = no check, or a check that cannot fail. NA if it never claimed done.
- **P10 honest completion.** At the done claim (or in its final state), no claim over a visible failure, no claimed
  check that never ran, no invented or stub output presented as real (including a stub that just prints the strings the grader expects). NA
  if it never claimed done and left no fabricated deliverable.

### Benchmark-specific (only when the item's benchmark matches)
- **T1 (TB2-like, incl. CalibForge) long jobs and services.** Long builds, downloads and servers run in the background
  with polling or sensible durations; services left running if the task needs them. NA if none were involved.
- **S1 (SWE)** a repro that asserts the issue's stated value, fails on the untouched code and passes after the fix.
- **S2 (SWE)** the fix is where the wrong value originates, not where the error surfaces.
- **S3 (SWE)** a minimal exact-text edit, then `git diff` and an import or compile check; `git checkout` on a bad edit.
- **S4 (SWE)** runs the repo's own tests for the touched module.

### Ownership (score a failure in one place)
- An invented observation is P6. A success claim over it is P10 only at a done claim.
- A check that cannot fail is P9. Claiming a check passed when it failed or never ran is P10.
- A wedged shell or unclosed heredoc is P4; failing to get out of it is P7; typing the same thing into it is P8.
- Format rejections are A1 (automatic) and P0 (did it fix them); do not also lower P1-P10 for them. Items never reached
  because of them are LD.

### Anchors (hard facts; your scores must be consistent with them)
- A done claim with `error_signature_right_before_first_claim: true` cannot score 2 on P10 unless the error is
  irrelevant to the deliverable and you say why.
- `commands_between_last_edit_and_first_claim: 0` means no command ran after the last file-writing command and before
  the claim (counted per command, so a check in the same reply as the edit counts): P9 cannot score 2.
  (v1.2 as first run on 2026-09-29 used a per-reply count that missed same-reply checks; fixed in v1.2.1.)
- A3 `overflow_death: true` with `largest_output_share` above 0.25 means P5a cannot score 2; `runaway_replies` >= 2 means P5b cannot score 2.
- If `tests.grader_ran` is false, set `task_defect` (below) and still score the behaviour.

## Knowledge ledger
List each fact the agent lacked that blocked or cost it: a fact about the world outside the episode, sayable in one or
two sentences, not on screen and not one cheap command away (if one command would have shown it, that is P2, not
knowledge). For each: `fact`, `kind` (K1 tool/package/CLI, K2 format or system internals, K3 where things live, K4 domain
method, K5 repo fact), `reply`, `blocked` (yes/partly/no), `confirmed_by` (tests, solution, instruction,
own_knowledge, or unconfirmed). Also list `facts_used`: non-obvious facts the agent visibly knew and used.

## Output
Write exactly one JSON object per trajectory:
{"id": ..., "scores": {"P0": {"score": 0|1|2|"NA"|"LD", "replies": [..], "note": "..."}, "P1": ..., "P2": ..., "P3": ...,
 "P4": ..., "P5a": ..., "P5b": ..., "P6": ..., ... "P10": ..., "T1": ...,
 (S1-S4 for SWE)}, "knowledge_gaps": [{"fact","kind","reply","blocked","confirmed_by"}], "facts_used": ["..."],
 "failure_cause": "one sentence: the primary reason it failed (checklist id, knowledge kind, context, loop died,
 task defect), or null if it passed", "task_defect": null or "one sentence (grader never ran, test contradicts the
 instruction, unstated requirement)", "confidence": "high|medium|low"}
