# Session brief: Levanter SFT babysitter (ota10k_lev_sft)

You are a dedicated Claude Code session whose ONLY job is to shepherd the
Levanter SFT production run on JUPITER to completion. A separate "supervisor"
session owns the GRPO RL sweep and the node-health/JSC thread — do not touch
those. A third session owns the Runboard dashboard.

## Project context (30 seconds)

ot-agent trains LLM agents on HPC clusters (see repo CLAUDE.md). This
workstream: SFT **Qwen3-30B-A3B-Base** on **OpenThoughts-Agent-SFT-10K**
(paper recipe 2606.24855) using **raw `levanter.main.train_lm`** — per PI
choice: Levanter, NOT LLaMA-Factory; no iris, no marin-credential paths.
Cluster: JUPITER booster (JSC), GH200 nodes (4 GPUs, 96GB each), ssh alias
`jupiter` (user lee27), `F=/e/fscratch/reformo/lee27`.

## The run

- sbatch: `hpc/sbatch_sft/levanter_sft_jupiter.sbatch` (8 nodes, 11:59 wall)
- config: `sft/levanter_configs/ota10k_qwen3_30b_a3b_base.yaml`
  — 328 steps, global batch 96, seq 32768, lr 4e-5 cosine (warmup 0.1),
  beta2 0.98, max_grad_norm 1e-4, ckpt every 100 steps, HF export at 328.
- log: `$F/experiments/ota10k_lev_sft/logs/ota10k_lev_sft_<JOBID>.out`
- checkpoints: `$F/checkpoints/ota10k_sft_30ba3b_levanter/`
- tokenizer: on-disk padded copy `$F/data/qwen3_30b_a3b_base_tok_pad151936`
  (267 dummy tokens → 151936). Why: upstream levanter bug — the
  `pad_tokenizer_to_match_model` flag pads only the converter's tokenizer
  (hf_checkpoints.py:633) while train_lm.py:245 sizes Vocab from the data
  tokenizer → opt_state/model mismatch explodes on first step. The on-disk
  padded tokenizer sidesteps it. (A one-line upstream marin PR is a candidate
  but needs Luke's explicit ask — do not open it unprompted.)

## Live state (2026-08-17 ~22:50 CEST — VERIFY with squeue before acting)

- **v18 = 1400728 (head), v19 = 1400729 (afterany spare)** — carries the
  TRUE root-cause fix (incident 56): `trainer.mesh.compute_mapping` maps
  `token`/`token_repeat` onto the DP axes. Raw train_lm left them unmapped →
  the ENTIRE MoE block ran globally REPLICATED on every device (v17 HLO dump:
  bf16[8388608,2048] etc.). Mirrors marin's own harness
  (marin/experiment/train.py `_TOKEN_AXES`).
  Memory-demand trajectory (the `Can't reduce memory use below` remat line is
  the fingerprint to read in every failed log): 21.15TiB (v8, mb3) → 7.18TiB
  (v10, mb1) → 348.66GiB (v12, gmm) → v18 expected O(10-30GiB), warning
  should disappear. Marin branch @ faa0bfcb9 (use_gmm + optimize_remat) KEPT —
  still required (XLA ragged bwd is dense even locally; 48GiB probe).
  Incident 55's optimize_remat was a no-op on prod (v14 byte-identical);
  probes cleared shard_map/custom_vjp opacity — see incidents 55-56.
- Levanter-side knobs already in place and KEPT: autotune level 0 (compile
  ladder closed at incident 51), per_device_parallelism 1 (incident 52),
  mem fraction 0.92, CE sweep ON, compile cache.
- **v12 verification duties**: (a) grep the log for `ragged_dot auto
  fallback` — if present, triton kernel failed and it silently reverted to
  the XLA path → treat as failure; (b) first loss should be plausible
  (~1.5–3); nonsense loss = suspect the new kernel path numerics.
- Dead: v1 1396852, v2 1397721, v2b 1397722(?), v3 1398134, v4 1398566
  (incident 50), v5 1398567 (cancelled), v6 1399224 (incident 51), v7 1399225
  (cancelled), v8 1399676 (incident 52), v9 1399677 (cancelled), v10 1399820
  (incident 54 diagnosis), v11 1399821 (cancelled), v12 1399959 (incident
  55), v13 1399960 (cancelled), v14 1400086 (incident 56 evidence), v15
  1400087 (cancelled), v16 1400167 (diag, cache-hit no dump), v17 1400274
  (diag, produced the HLO dump; scancelled after).
- **Milestone to report: the FIRST loss line** = pipeline green. HLO changed →
  compile cache miss: expect CE sweep (~8 min) + compile (~10-20 min) first.
- Upstream marin PR for the use_gmm patch: candidate, needs Luke's explicit
  ask — do not open unprompted. Local marin clone: `/Users/lukedhlee/marin`.

## Failure history — one line each (full detail: `currease_pass8_probe_handoff.md` incidents 41–50)

1. **v1**: fused-CE autotune sweep compiled pathological candidates for 2h23m,
   then a rank died under the old default 0.75 mem fraction (71.25GiB cap).
2. **v2/v2b**: fast deaths — partly ghost-node residue from our own scancel,
   partly 258GB of core dumps blowing quota → now `ulimit -c 0` + targeted
   `--exclude` in the sbatch (deliberate, flagged deviation from Luke's
   "no SFT exclusions").
3. **v3 (incident 48 REVERSAL)**: autotune OFF is WORSE — inferred CE block
   sizes OOM train_step in 7 min at any mem limit. Verdict: sweep ON +
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` + `JAX_COMPILATION_CACHE_DIR`.
4. **v4 = 1398566 (incident 50)**: survived the CE sweep, died 1h48m in —
   **XLA's own GEMM/fusion autotuner** (default level 4: operand init +
   correctness check + redzone) RESOURCE_EXHAUSTED'd allocating 24.02GiB for
   5 of 646 instructions during train_step compile.
   Fix now in the sbatch: `XLA_FLAGS="--xla_gpu_autotune_level=1"` (still
   autotunes, skips the init/check buffer copies). Flag validated to parse in
   the levanter env (jax 0.11.0).
   **Escalation ladder if v6 dies at the same spot: level 0 next** (no
   autotune, heuristic algorithm picks — perf hit but correct).

## Misdiagnoses — do NOT repeat these

- JAX `RESOURCE_EXHAUSTED` with allocator line `Limit: 87.40GiB` (or 71.25) =
  **our own mem-fraction cap**, not ghost-node residue.
- wandb `Disk quota exceeded` staging-thread spam = non-fatal noise.
- 8-rank SIGABRT + "Shutdown barrier DEADLINE_EXCEEDED" = cascade; find the
  FIRST dead task's error, never blame the barrier.
- Real ghost-node checks/protocol: `ai_memory/notes/jupiter_node_health.md`.

## Operating procedure

- **First action on boot**: `ssh jupiter "squeue -j 1399224,1399225 -h -o '%i %T %M %R'"`,
  then tail the v6 log; arm a watcher (background loop, ~3 min period): exits
  and reports when (a) the job leaves the queue → pull `sacct` + last 30 log
  lines, or (b) the first loss line appears → report GREEN.
- **All fixes go through the local Mac repo**: edit → `git add <file>` →
  commit (NO Co-Authored-By) → `git push fork lukedhlee/vista-moe-grpo-30b` →
  `ssh jupiter "cd $F/OpenThoughts-Agent && git pull"`. Never hand-edit on the
  cluster. Luke has unstaged local changes (.gitignore, ssh notes) — do not
  touch them; if `git pull --rebase` complains, commit only your file and push
  (fast-forward usually fine).
- **Resubmit recipe** (self-contained, no proxy/DCFT needed for SFT):
  `ssh jupiter 'cd /e/fscratch/reformo/lee27/OpenThoughts-Agent && export LEV_CONFIG=$PWD/sft/levanter_configs/ota10k_qwen3_30b_a3b_base.yaml && h=$(sbatch --parsable hpc/sbatch_sft/levanter_sft_jupiter.sbatch) && sbatch --dependency=afterany:$h hpc/sbatch_sft/levanter_sft_jupiter.sbatch'`
  Always scancel a superseded/doomed link first.
- A wall-clock TIMEOUT at 11:59 is NORMAL — the spare resumes from the latest
  checkpoint (ckpt-temp preclean handles mid-save kills).
- On any ghost-OOM incident (foreign GPU memory at job start — see
  jupiter_node_health.md §1): add ONLY the confirmed-dirty nodeset to the
  sbatch's targeted `--exclude` line (SFT deliberately does NOT carry the full
  RL blacklist), and tell the supervisor session's operator note (append a
  line to `currease_pass8_probe_handoff.md`).
- Append incidents to `ai_memory/notes/currease_pass8_probe_handoff.md`
  (numbered — next free number: 51) and commit.

## Endgame

1. Run reaches step 328 → levanter writes the HF export (hf_save_steps 328).
2. Verify export loads (config + tokenizer 151936).
3. Report to Luke: loss curve summary + export path. HF upload default is
   PUBLIC to `laion/` per standing guardrail — but ASK Luke before uploading
   this one (SFT-checkpoint naming/visibility wasn't decided).
4. The follow-on (pass@8 probe of the SFT ckpt → curriculum work) belongs to
   the supervisor session — hand off, don't start it.

## Guardrails (inherited, non-negotiable)

- Never touch jobs you didn't launch: RL chains (1394805-08, 1396799-802,
  1397229-30, 1398002-03), anything by feuer1, `base30b_gsm8k_*`, peer tmux
  sessions (esp. `currease_socks` on login02).
- Your own demonstrably-dead/doomed jobs: cancel without asking.
- No secrets in output; `set -a; source $F/keys/secrets.env; set +a` pattern
  only (not needed for SFT).
- Play `afplay /System/Library/Sounds/Funk.aiff` after long tasks.
