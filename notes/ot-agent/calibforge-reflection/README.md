# CalibForge completion reflection experiments

2026-10-13 · implementation authorized; job launch deferred to Luke and Horizon Claude Code.
Canonical repo: /Users/lukedhlee/OpenThoughts-Agent. Isolated checkout:
/Users/lukedhlee/staged-work/calibforge-reflection/OpenThoughts-Agent,
branch lukedhlee/calibforge-reflection. Evidence: agent_logs/2026-10-13_calibforge-reflection.md.

Goal: configurable vanilla SFT, true online student→student ROFT, and three completion-only reflection
variants (teacher→teacher, teacher→student, student→teacher), with resumable provenance and reflection-only loss.
All learners start at the same pinned Grug RLVR1-Step12 initialization. Existing vanilla/relay jobs are untouched.
Teacher sources are fixed existing Qwen3.8 MSA2 trials. Offline student sources must be unassisted Grug trials;
relay mixtures cannot stand in for student-only trajectories. Offline versus online is an explicit axis.

| Stage | Scope | Cost | Gate |
|---|---|---|---|
| 0 | Config, provenance, bounded prompt, Harbor import | CPU | strict arm/author/heldout checks; failed attempts retained |
| 1 | Completion generation, native rendering and masks | CPU + mock HTTP | exact token-prefix check; context zero loss; rejection census |
| 2 | Online state machine and trainer adapter | CPU tiny trainer | persistent optimizer, one accumulated update, fresh data, restart consistency |
| 3 | Config templates, read-only plans, handoff | CPU | config/CLI smoke; explicit unrun GPU gate |

Critical path: 0 → 1 → 2 → 3. Each scope is in stage<N>_scope.md.
Global invariants: additive module only; existing launchers unchanged; loss only on reflections (including nonempty
thinking and native completion suffix); no pass-only filter for ROFT; no reward weighting; deterministic split;
no stored reflection at evaluation; no dataset or checkpoint overwrite; no submission during implementation.

Borrow map (anchors drift; re-read before implementation): data/relay/sft/render_msa.py native template;
data/relay/sft/relay_rows_to_parquet.py ids/loss contract; data/tb2/jupiter/tb2_msa_rlvr1_norefeed_65k16k.yaml eval;
Marin experiments/june_tpu_67b_a2b/moe/train.py GrugTrainState and resume/data-offset logic.
Reference: https://arxiv.org/html/2609.35741 (Appendix B, D). This is a Grug/CalibForge/MSA2 adaptation, not exact
Qwen3.5/SWE-rebench reproduction. Numerical optimizer values must be read from original math, not stripped HTML.

Validation: existing files' Git diff first (empty); each behavior gate uses real tokenizer where available and tiny
CPU training plus controlled endpoint. GPU/actual HF export correctness remains a mandatory prelaunch gate; CPU
results do not establish GB200 correctness or experiment success. Track rejected/infra/unknown outcomes separately.
