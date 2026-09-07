# Takeover: Snowball re-feeding probe → trace analysis + report artifact (2026-09-06, ~21:55 PT)

You are taking over from a Fable session. Goal: turn tonight's history-think probes into a shareable report artifact like
https://claude.ai/code/artifact/d7cfe850-b666-4a24-b525-b9a411d8d0db (read it with the Artifact tool, action read; same shape:
TL;DR + outline, "I" voice, implication headlines, self-explanatory figure rows, no colons in headlines). Dispatch Opus subagents
(model: opus) for the trace analysis; you assemble and publish. Load the artifact-design skill before writing the page.

## What the probe is
Snowball 67B-A2B Stage-3 SFT (laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888) on the 441-task R2E-Gym tt_v2 validation
tree, 8 samples per task, 64k parity budget, three arms that differ only in what the model sees of its own past turns:
keep (Marin contract, every prior <think> span re-fed; what the SFT saw), drop (all prior think spans stripped from the request),
last:2 (two most recent kept). Harness switch HARBOR_TERMINUS2_HISTORY_THINK (harbor fork lukedhlee/rl-transport e6adddd8).
Decision rule Luke set: drop holds pass@8 within ~.04 of keep with fewer context deaths → RL under the dropped contract;
hard drop → last:2; both fail → the SFT side (Ben). Ben's teacher (DeepSeek-V3.2) stripped prior reasoning; Marin re-feeds it.

Per-split result already computed from the pass8 tables (see decisions.md entry 22:10 PT: drop +.014, last2 +.018 overall vs keep, ctx deaths .70 → .64, turns 24 → 33; last:2 recommended, drop alternative). Trial-level result already known (per arm: scored / successes / trial pass / median turns): keep 3521/975/.277/23,
drop 3517/1012/.288/32, last2 3520/1028/.292/33. Mechanism found on first trials: without re-fed reasoning the model
progressively stops thinking (first completion token = think-start id 128002 on 100 % of turns under keep at any depth; under
drop ~85 % at turns 6–8, ~65 % at 10–13, ~50 % by turn 15, ~17 % by turn 23; last:2 drifts the same way). The report must
separate the window effect from the thinking effect.

## Where everything is (Jupiter: `ssh -o BatchMode=yes jupiter`; times to Luke in PT = CEST − 9 h)
- E=/e/fscratch/reformo/lee27/experiments. Arms: snowball_hist_{keep,drop,last2}_base. Tables: $E/<p>/pass8_summary.json,
  $E/<p>/pass8_pass8_table.csv. Traces: $E/<p>/<p>/trace_archive.tar (tar of trace_jobs; per attempt
  eval_sessions/*/<task>__<id>/attempts/*/{result.json, agent/trajectory.json}).
- Top-up arms snowball_hist_{keep,drop,last2}_topup (same settings on the tasks the outage left under-sampled) may still be
  running when you start; tmux hist_sequencer on Jupiter runs them and writes read-outs. Its log: $E/hist_readouts/log
  (ends with ALL_DONE). Read-outs there: hist_r2egym_readout.md (strict, ≥8 scored per task), hist_r2egym_min6_readout.md,
  hist_r2egym_merged_readout.md (base + top-up per arm; use this one), *_attempts.jsonl (one record per attempt: task, reward,
  exception, turns, tokens, think_share, tc), *_curves.json (per turn: n, think share, median prompt/completion tokens, per split).
- Split lists: $E/tt_v2_{idval,oodval,heldout,train}.txt (idval 150 / oodval 115 tornado+scrapy / heldout 176 never- or always-solved).
- Scripts: /e/project1/transfernetx/lee27/code/snowball/{hist_readout.py, hist_semcheck.py, trio_topup_then_easy.sh,
  probe_history.sh}; repo copies in OpenThoughts-Agent/data/r2egym/jsc/. hist_readout.py reads trace_archive.tar directly.
- Local scratchpad (this Mac, session f4a2c148): report/{head.html (tokens+CSS), chart.js, static_sections.html (method,
  verification, outage, audits — drafted), assemble.py (read-out md → HTML tables)}, easy3_audits.md, six.json (Ben's panel).
- Ledger: ai_memory/active/snowball-r2egym/decisions.md (entries from 17:45 PT tonight onward tell the whole story, incl. the
  18:30 PT bridge outage: thread-per-request bridge hit the login node's 4,096-pid cap under 9 probes; easy probes cancelled;
  harness retries + top-up repaired the trio).
- Easy sources (TaskTrove curriculum-easy / pymethods2test-v3 / unitsyn-python-v4): probes CANCELLED by Luke; report only the
  audit verdicts (easy3_audits.md; exclusion lists $E/easy3_audit/). Do not relaunch them.

## What the Opus subagents should analyse (each gets one tar + the attempts jsonl; extract to scratch, never analyse on the
login node with heavy Python — OMP_NUM_THREADS=1, no torch/pytest/Ray there; pid cap 4,096 incl. threads)
1. Per split (idval / oodval / heldout) and per repo: paired per-task pass deltas drop−keep and last2−keep with bootstrap CIs;
   newly solved / newly lost tasks; context-death share; turns; declared-done share and P(win | done).
2. Thinking by depth: think share vs turn per arm; does a no-think turn predict a worse next action (error rate, repeated
   commands, task_complete false positives)? Completion tokens per turn by arm.
3. Where the freed window goes under drop: what the extra ~9 turns are spent on (tests, edits, repro, idle), and whether
   context deaths convert into wins or into more turns of the same failure.
4. Qualitative: read 10–20 paired trajectories where arms disagree (same task, keep 0/8 vs drop ≥3/8 and the reverse); name
   the behaviours.
5. Sanity: nulls per arm from the outage (BridgeOutageError / BridgeOperationTimeoutError) are excluded; retries mean some
   tasks have >8 scored attempts; state the fully-sampled counts.

## Deliverables
- Artifact page (new; favicon of your choice), Discord post draft, then /memo (ai_memory + memory dir). Verdict first, then the
  evidence; keep it to what changes the RL decision. Note the teacher-mismatch fact as an item for Ben.
- After tabling, move tars/tables/read-outs to /e/data1/mmlaion/lee27/experiments/<probe>/ (mmlaion is the experiments root;
  fscratch was used tonight because the launcher defaults there) and propose the launcher default fix; ask Luke before deleting.

## Guardrails
Do not launch probes or fleets; do not touch tmux hist_sequencer or bridge 9926; ssh always with BatchMode=yes; report times in
PT; no Co-Authored-By lines in commits; no hand edits on clusters (edit locally, scp the script copies to code/snowball, commit).
