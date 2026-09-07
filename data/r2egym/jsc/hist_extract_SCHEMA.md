# history-think probe extracts (2026-09-06 probes, base arms)

Three arms of the same probe differ ONLY in what the model is shown of its own PAST turns:
  keep  = Marin contract, every prior <think> span re-fed (what the Stage-3 SFT saw)
  drop  = all prior think spans stripped from the request (current turn still thinks)
  last2 = the two most recent think spans kept, older ones stripped
Model laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888; harness terminus-2 (JSON action protocol);
64k parity budget (61,440 in / 4,096 out), no summarisation, k=8, eval wall 1800 s, verifier 1200 s.
Tasks = 441 R2E-Gym tt_v2 validation tasks. Task ids look like r2egym-v1-07916.

## files (one line = one ATTEMPT)
<probe>_turns.jsonl   numeric, per-turn. THE file for quantitative work.
<probe>_text.jsonl    truncated text per turn, same attempt order. For reading trajectories.
<probe>_index.jsonl   {n: tar member name, o: byte offset of data, s: size} -> random access back into
                      trace_archive.tar (open the tar file raw, f.seek(o), f.read(s)) for a FULL trajectory.

probes: snowball_hist_keep_base, snowball_hist_drop_base, snowball_hist_last2_base
tars:   /e/fscratch/reformo/lee27/experiments/<probe>/<probe>/trace_archive.tar

## _turns.jsonl record
{probe, task, tname, trial, sess, att,
 reward: 1.0 win / 0.0 loss / null = INFRA NULL (not a model failure; exclude from denominators),
 exc: exception_type or null. ContextLengthExceededError = ran out of window.
      BridgeOutageError / BridgeOperationTimeoutError / BridgeOperationError = the 18:30 PT bridge outage, exclude.
 n_turns, in_tok, out_tok, stop,
 turns: [ {
   i    turn index (0-based)
   th   TRUE if the completion's FIRST token is the think-start id 128002 (a thinking turn)
   th_ch  characters inside the think span
   cl   completion tokens this turn      pl  prompt tokens this turn
   nc   number of shell commands emitted
   k    per-command kind: test_repo | repro | edit | git | search | install | shell | wait | other
   dur  sum of the model's own declared command durations (its sleeps), seconds
   tc   this turn set task_complete true
   pj   the JSON action parsed (false = malformed action)
   ol   characters of terminal output that came back
   er   the observation carried an error signature (traceback / SyntaxError / not found / ...)
   pt   pytest counters found in the observation tail, e.g. [[3,"passed"],[1,"failed"]]
   rep  this turn re-sent EXACTLY the previous turn's command batch
   es   this turn WROTE to repo source PROPER (non-scratch, not under tests/)   -- edit_target.py
   et   this turn wrote to a repo TEST file. window.py's edit_src kind = es or et; this probe's question is es alone
   ws   this turn wrote to a scratch file (/tmp, or a file at the repo root where source never lives)
   pf   this turn wrote a PATCH FILE (a .patch/.diff target, or a `git diff`/`diff -u` redirected into a file)
   uw   this turn wrote somewhere we can see but cannot resolve (target in a shell/python variable, `patch -p1`, `git checkout`)
 } ] }

`es`/`ws`/`pf`/`uw` come from `data/r2egym/jsc/edit_target.py`, which lifts window.py's path predicates verbatim and adds
two things window.py cannot do: it scans EVERY line of a keystroke blob (not just the first), and it reads python heredoc /
`python -c` bodies for their write calls. Both matter for the 2026-09-07 in-place-clause A/B, whose treatment prompt
recommends exactly the python-rewrite form window.py is blind to. `edit_target.py selftest` runs the worked cases.

## _text.jsonl record
{probe, task, trial, att, reward, exc,
 turns: [{i, think (<=700 ch), an (analysis <=400), pl (plan <=400), cmds (<=8, each <=200 ch), obs (last 700 ch)}]}

## split lists (task-id per line)
/e/fscratch/reformo/lee27/experiments/tt_v2_idval.txt    150 in-distribution validation
/e/fscratch/reformo/lee27/experiments/tt_v2_oodval.txt   115 out-of-distribution (tornado + scrapy)
/e/fscratch/reformo/lee27/experiments/tt_v2_heldout.txt  176 never-solved or always-solved by the base
Repo for a task: read /e/fscratch/reformo/lee27/tasks/r2egym-tt-v2-val441/<task>/instruction.md (the repo name
appears in the issue text and paths) or environment/ for the image name.

## conventions
- a task is FULLY SAMPLED for an arm when it has >= 8 scored (reward not null) attempts. The harness retried
  through the outage, so most tasks have 8-10 scored attempts; use the FIRST 8 by attempt index if you need
  exactly 8, and say which convention you used.
- per-task pass = wins / scored (on fully-sampled tasks).
- paired deltas: compare arms only on tasks fully sampled in BOTH arms; bootstrap the per-task difference.
