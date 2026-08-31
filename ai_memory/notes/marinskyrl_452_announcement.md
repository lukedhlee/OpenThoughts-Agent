# Heads-up: MarinSkyRL #452 silently changed effective learning rates (~3x)

**TL;DR:** #452 "Stochastically round BF16 AdamW updates" (`30c49275`, merged 2026-08-23)
is a genuine *fix*, but it invalidates every lr calibration done before it. On current
main, lr 3e-6 behaves roughly like lr 8e-6 did before the merge. Don't compare runs
across the 08-23 boundary; re-sweep lr about one order of magnitude lower.

**What it fixed:** with bf16 master weights, a small AdamW update (lr in the 1e-6 range)
is often below bf16 resolution, so plain rounding threw most of it away — we were
training with a silently smaller effective lr. #452 rounds stochastically, so updates
now land in expectation at their true size. The optimizer is *more correct* now.

**Why it bites:** any lr sweep from before 08-23 was measuring "lr minus rounding loss",
not the algorithm. Our base30b GSM8K sweep ("lr 8e-6 wins", 08-14) and the currease lr
arms are not transferable to main.

**Evidence:** bisected 2026-08-31 with identical 12-step GSM8K GRPO gates
(Qwen3-30B-A3B-Base, 6 Jupiter nodes, identical step-1 rollouts). Commits up to
`0a41dee9`: step-2 KL ≈0.005, reward ≈0.42 at lr 3e-6. From the merge of #452:
step-2 KL ≈0.08, reward ≈0.77 — the old lr-8e-6 curve, same config. Full table:
`ai_memory/notes/repo_refactor_plan.md` (Bisect log).

**Action:** on main, start lr sweeps ≈5e-7…2e-6 and watch KL/entropy at steps 1-3.
If an old run "suddenly learns much faster" after a MarinSkyRL bump, check optimizer
numerics before blaming the config.
