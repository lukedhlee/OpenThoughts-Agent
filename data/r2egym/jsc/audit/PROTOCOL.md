# Solvability audit protocol (one Opus agent per task)

You are auditing ONE R2E-Gym task in its original environment, running in an apptainer sandbox on the JURECA cluster.
You drive it ONLY through the helper below (each call is a ~5-10 s round trip; keep commands batched and quiet):

    export SBX_JOB=<job>          # given in your task prompt
    SBX=<scratchpad>/audit/sbx.sh # given in your task prompt
    bash $SBX start  <task>                 # once, first
    bash $SBX exec   <task> '<shell cmd>'   # runs in /testbed inside the sandbox as the agent would (bash -lc)
    bash $SBX verify <task>                 # the task's OFFICIAL verifier; prints "reward: 0.0|1.0" and a pytest summary
    bash $SBX stop   <task>                 # once, last — ALWAYS, even on failure

Rules (these are the honest rules the RL policy faces — breaking them makes the audit worthless):
1. Read the issue from the local file given in your prompt (instruction.md). That text plus the repo is ALL you get.
2. No git history: never run git log/show/reflog/checkout/reset/restore/stash/revert/bisect/diff against other commits,
   never look at .git internals. `git status` and `git diff` (working tree vs HEAD) are fine.
3. No network: no pip/uv/curl/wget/apt. The repo's environment is already installed (python = /testbed/.venv/bin/python).
4. Do not read /workspace, /tests, /logs, /r2e_tests, or any file named metadata.json / expected_output* / run_tests.sh.
   The hidden tests are not yours to see. Write your own reproducer instead (a file under /tmp or /testbed/repro_*.py).
5. Edit files in place in /testbed (sed -i, python -c rewrites, or heredocs via `cat > file <<'X'`). Multi-line commands:
   base64 is already handled by the helper — just pass the command as one single-quoted string; for content with
   single quotes, write it with a heredoc using a quoted delimiter.
6. Budget: at most 40 exec calls and 25 minutes of wall clock. Call verify at most 3 times (each ~1-5 min). Stop when you
   are done or out of budget.
7. Do not touch any file on this Mac except your own report file. Do not run anything on the cluster other than the helper.

Label the task at the end with exactly one of:
- solved                 verify printed reward 1.0
- unsolvable-from-issue  you could not find a fix consistent with the issue text (say what was missing)
- env-broken             the sandbox/environment itself failed (import errors unrelated to the bug, missing deps, verifier crash)
- verifier-disagrees     your reproducer/tests pass and the fix is clearly right, but verify stays 0.0 (say what you think it wants)

Write your report as JSON to the path given in your prompt:
{"task": "...", "repo": "...", "label": "...", "execs": <int>, "verify_calls": <int>, "minutes": <float>,
 "reward": <0.0|1.0|null>, "files_edited": [...], "summary": "<3-6 sentences: what the bug was, what you changed, what
 the verifier said, anything about the environment worth knowing>"}
Then reply with just that JSON line.
