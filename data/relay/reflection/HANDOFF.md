# Horizon Claude Code launch handoff

Luke requested implementation/testing first. **No new jobs or babysitters have been launched.** Your next task is
prelaunch hardware validation, then vanilla/ROFT baselines, then the three completion variants. Do not treat this
handoff as authorization to change or cancel another experiment's jobs. Confirm Luke's actual launch instruction
and resource envelope at launch time; there is no approval question pending from this implementation session.

## Read and deploy

1. Read Horizon `~/OpenThoughts-Agent/AGENTS.md`, current `.claude` TACC/Marin/Harbor ops, and
   `ai_memory/active/snowball-sft/experiments/2026-10-12_rlvr1_calibforge_pair.md`.
2. Inspect current Slurm, tmux, `relay_rlvr1/continuation.status`, and its actual logs. Earlier state: vanilla/relay
   training was blocked by final data QA; teacher MSA2 evals 56611–56613 were running. Recheck rather than using
   those snapshots as current truth. Reuse vanilla if its owner has now trained it; do not run a second owner.
3. OTA implementation branch `lukedhlee/calibforge-reflection` in Luke's fork. Marin branch
   `lukedhlee/calibforge-roft-online`, commit **2eabece16a**, based on current Horizon pair **f2b550b1e**. Deploy
   separate isolated checkouts through Git, never patch an active checkout. New backend is
   `experiments/june_tpu_67b_a2b/moe/online_reflection.py`; old trainer/launcher defaults are untouched.
4. Read README.md and every resolved config. Config templates use canonical known Horizon models, the verified
   RLVR1 native init, all 2,613 train tasks and heldout300. The requested arms are completion-only.

## Verify before a real run

- Fill operator-owned serving refresh argv, endpoints, native trainer allocation/wrapper, Python paths, environment
  and tunnel. Teacher serving must use **QWEN_SPEC=0**; original MTP teacher runs crashed and were recovered.
- Keep fixed base-student serving for offline teacher→student separate from evolving ROFT serving.
- Verify source author provenance. Teacher pool is Qwen3.8 MSA2 control/recovery; student pool must be unassisted
  RLVR1 MSA2, not relay. Do not silently use benchmark evaluation traces as CalibForge training experience.
- Run CPU/native tests in the actual selected environment, including the Marin optimizer accumulation test.
- Reserve a small explicit four-node training smoke plus owned inference resources, with bounded walltime and
  node-hours. Use one or two source attempts, one or two reflections per source, and **two optimizer updates**;
  label it a smoke, not the requested experiment. Short-reflection degenerate stopping still applies.
- Verify prompt/trajectory labels are zero, target/EOT labels one, finite losses/gradients, no overflow/NaNs,
  actual GPU utilization and memory, exact single-step increments, full native optimizer restore, and fresh data.
- At each update verify native metadata and online_state.json, exported weights/tokenizer/template/config,
  pinned qk_mult1.75 and full native-init provenance. Confirm vLLM loads the export and produces a completion;
  `/models` must identify the new root. Check pending router biases remain frozen and no teacher/student confusion.
- Failures stop. Do not weaken a gate to get a launch. Capture actual exception and retain its source/row/adapter
  artifacts. Native checkpoint plus HF export per update is large; measure smoke bytes and plan free scratch and
  checkpoint retention before full 20-update runs. Keep every source/reflection/request/result and evaluated export.

## Execute in this order

1. **Vanilla baseline**: existing matched teacher action SFT from original RLVR1 init, LR3e-4, three real passes,
   batch16×65k, frozen router bias. Repair its owner's QA gate if still blocked; never bypass it. Existing matched
   arm uses1,802 episodes/1,286 tasks; more teacher successes is a different named baseline.
2. **ROFT baseline**: configs/roft.json. Initial experiment20 updates,64 current-student attempts/update,4 current-
   student reflections/attempt, single accumulated AdamW step/update, immediate weight refresh, both outcomes,
   no replay/action loss. This is an adaptation: LR1e-5,65k sequence and MSA2, not exact paper hyperparameter parity.
3. **Completion variants**: teacher_teacher and teacher_student share the frozen teacher source list and seed;
   student_teacher uses frozen unassisted base-student sources. All explanations are post-completion, with
   evidence and available verdict as masked context. Keep offline/online differences explicit in the report.

Use `python -m data.relay.reflection.cli plan/status/run --config <path>`. Run inside allocated resources in a
persistent experiment-specific tmux. The state machine is its own durable resume source; Claude monitors it, not
an independent conflicting script. `state.json` reports phase, committed update, model and native optimizer path;
per-update census counts every rejection. Never claim all20 updates cover all2613 tasks; configure/source-report
coverage explicitly. For a full teacher offline pass set updates=ceil(actual sources/64), schedule_steps to match.

## Evaluation and reporting

MSA2 2.4.6 tool mode, no refeed/reflection context,65k context,explicit16k reply,30-minute tasks; TB2.1,SWE-r100,
TBLite,three reps each, same settings for all arms. Existing RLVR1 base averages9.3/33.3/16.7% are recorded over
86/99/100 usable tasks; preserve exclusions and use a common task intersection for paired comparisons. Report
source/reflection/supervised-token counts, updates, compute and outcome mix; action SFT versus reflection-only
are different targets and online/offline arms consume different data. Keep all traces/logs. Do not publish results
until completeness and infrastructure-error gates pass. Write status and exact resume commands into ai_memory.
