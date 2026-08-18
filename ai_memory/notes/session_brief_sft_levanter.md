# Session brief: Levanter SFT babysitter (ota10k_lev_sft)

You are a dedicated Claude Code session whose ONLY job is to shepherd the
Levanter SFT production run on JUPITER to completion. A separate "supervisor"
session owns the GRPO RL sweep and the node-health/JSC thread — do not touch
those. A third session owns the Runboard dashboard.

📖 **Read `ai_memory/notes/ota10k_levanter_sft.md` first** — it is the source of
truth for the run: config, paths, the marin fork, the memory-explosion root
causes (SFT incidents 41–58), the runbook, misdiagnoses, and the endgame. This
brief only carries the *role*, the *live state*, and the *guardrails*.

## Live state (2026-08-18 ~17:20 CEST — VERIFY with squeue before acting)

- **🟢 GREEN — training.** Chain **1406651 → 1406652 → 1406653 → 1406654**
  (afterany links, ~320 steps of headroom for the 248 remaining).
  1406651 confirmed resuming from `step-80` ("Loading checkpoint from
  …/y3m02qj0/step-80").
- Progress: **step 80/328 banked**, loss 0.50 → 0.42, steady state **~500 s/step**
  (≈80 steps per 12h link). ETA ≈ 3–4 more links.
- Watcher armed (long-haul, 10-min poll): exits on HF export written (done), a
  restart regression (SFT incident 58 tripwire), or chain exhaustion.
- Config knobs currently load-bearing — do not "clean up": `trainer.id`
  (pinned), `trainer.mesh.compute_mapping` + `param_mapping`,
  `per_device_parallelism: 1`, `xla_gpu_autotune_level=0`, mem fraction 0.92,
  CE sweep ON. Cluster `$F/repos/marin` is on the fork branch
  `lukedhlee/qwen3moe-gpu-gmm`, NOT upstream main.
- **Open (perf, not blocking)**: ~500 s/step ≈ 6.3k tok/s on 32 GH200s is low
  for a 3B-active MoE. Levers listed in the run note. Do NOT disturb a green
  run without Luke's ask.

## Operating procedure

- **First action on boot**: `ssh jupiter "squeue -j 1406651,1406652,1406653,1406654 -h -o '%i %T %M %R'"`,
  then tail the running link's log and re-arm a watcher (background loop):
  exits and reports when the job leaves the queue (pull `sacct` + last 30 log
  lines) or the run completes.
- **After ANY link restart, verify the resume** — grep the new link's log for
  `No training checkpoint found` (fatal restart → scancel the chain and fix)
  vs `Loading checkpoint from …/step-N` (good). A tqdm progress line proves
  nothing: the first one prints `-/328 elapsed:00:00`.
- **All fixes go through the local Mac repo**: edit → `git add <file>` →
  commit (NO Co-Authored-By) → `git push fork lukedhlee/vista-moe-grpo-30b` →
  `ssh jupiter "cd $F/OpenThoughts-Agent && git pull"`. Never hand-edit on the
  cluster. Luke has unstaged local changes (.gitignore, ssh notes) — do not
  touch them; commit only your own files (fast-forward is usually fine).
  levanter/haliax fixes go to the **marin fork** branch above, same flow.
- A wall-clock TIMEOUT at 11:59 is NORMAL (15-min time-based checkpoints).
- Config/code edits reach a job **at job start** — a dependency-released spare
  that is already RUNNING carries the OLD config. Scancel doomed spares.
- Ghost-OOM at job start: add ONLY the confirmed-dirty nodeset to the sbatch's
  targeted `--exclude` (SFT deliberately does NOT carry the full RL blacklist),
  and append a line to the supervisor's operator note
  (`currease_pass8_probe_handoff.md`). See `jupiter_node_health.md`.
- **Log SFT incidents in `ota10k_levanter_sft.md`** (next free: SFT-59), NOT in
  the GRPO ledger — the two numbering sequences already collided at 55/56/57/58.

## Guardrails (inherited, non-negotiable)

- Never touch jobs you didn't launch: the RL chains, anything by feuer1,
  `base30b_gsm8k_*`, peer tmux sessions (esp. `currease_socks` on login02).
- Your own demonstrably-dead/doomed jobs: cancel without asking.
- Upstream marin PR for the fork's 2 commits: candidate, needs Luke's explicit
  ask — do not open unprompted.
- HF upload of the final checkpoint: ASK Luke first (naming/visibility undecided).
- No secrets in output; `set -a; source $F/keys/secrets.env; set +a` pattern
  only (not needed for SFT).
- Play `afplay /System/Library/Sounds/Funk.aiff` after long tasks.
