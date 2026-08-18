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

- **🟢 GREEN: v21 = 1401031 TRAINING** (v22 = 1401032 afterany spare).
  First loss line EVER at 2026-08-18 06:06 CEST: step 4/328, loss≈0.50,
  zero remat warnings, zero ragged_dot fallbacks. Three stacked root causes,
  all fixed: (1) incident 54 — use_gmm triton MoE kernels (marin branch
  `lukedhlee/qwen3moe-gpu-gmm` @ faa0bfcb9, REQUIRED: XLA ragged bwd is dense
  even locally, 48GiB probe); (2) incident 56 — token/token_repeat axes
  unmapped in raw train_lm → MoE block globally replicated per device
  (yaml trainer.mesh.compute_mapping fix); (3) incident 57 — node-as-slice
  mesh left params/Adam fp32 only 4-way sharded, replicated across nodes
  (yaml trainer.mesh.param_mapping: embed: [replica_dcn, data]).
  Demand trajectory: 21.15TiB → 7.18TiB → 348.66GiB → 212.60GiB → fits.
- **Walltime TIMEOUT is safe**: levanter CheckpointerConfig.save_interval
  defaults to 15-min time-based checkpoints (checkpoint.py:1149) — a link
  timeout loses ≤15 min; spare resumes. Step-100 keeps are permanent.
- **Link 1 (v21 1401031) completed normally**: TIMEOUT at 11:59 after
  reaching **step 80/328**, loss 0.50 → ~0.42. Checkpoint step-80 saved 2.5
  min before the kill (time-based 15-min policy works as designed; ≤3 min of
  work lost). Steady state measured: **~500 s/step** (≈80 steps per 12h link).
- **Incident 58 (chain-breaking, FIXED)**: links restarted from HF weights
  instead of resuming — checkpoints are scoped by run id and `trainer.id`
  defaulted to a RANDOM value per link. Now pinned `trainer.id: y3m02qj0`
  (where step-80 lives). Old chain (1401032, 1406567-69) scancelled.
  **Verify every resume** by grepping the log for
  `No training checkpoint found` (fatal restart) vs `Loading/Restored
  checkpoint` — a tqdm progress line proves nothing (first one is `-/328`).
- **Live chain: 1406651 → 1406652 → 1406653 → 1406654** (4 links ≈ 320 steps
  of headroom for the 248 remaining). At step 328 levanter writes the HF
  export to `$F/checkpoints/ota10k_sft_30ba3b_levanter/hf`.
- **Open (perf, not blocking)**: ~500 s/step = ~6.3k tok/s across 32 GH200s
  for a 3B-active MoE is low. Candidate levers AFTER the run banks steps (do
  NOT disturb a green run without Luke's ask): retry autotune level 1 now
  that the HLO is sane, and check pallas-triton ragged_dot block sizes for
  GH200 (`_triton_default_block_sizes` in haliax ragged_dot.py).
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
