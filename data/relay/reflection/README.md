# CalibForge completion reflection training

Implementation and CPU tests are complete. No experiment jobs were launched. Start with **vanilla SFT and
online ROFT**, then compare the three completion variants. Horizon Claude Code owns eventual launch and
monitoring; read [HANDOFF.md](HANDOFF.md) first.

| Config | Source experience | Reflection author | Mode / objective |
|---|---|---|---|
| configs/vanilla.json | Qwen3.8 teacher | — | Existing vanilla action SFT |
| configs/roft.json | Current Grug student | Current Grug student | Online, reflection-only |
| configs/teacher_teacher.json | Fixed Qwen teacher | Fixed Qwen teacher | Offline, reflection-only |
| configs/teacher_student.json | Fixed Qwen teacher | Fixed base Grug student | Offline, reflection-only |
| configs/student_teacher.json | Fixed unassisted base Grug student | Fixed Qwen teacher | Offline, reflection-only |

Every learner starts from pinned RLVR1-Step12. Student and teacher revisions, endpoints, task split, generation,
optimizer, mesh, update counts and artifact paths are explicit JSON. Never reuse the existing 09-21 initialization.
The two teacher-source variants should use exactly the same teacher.jsonl and seed. Student→teacher needs fresh
unassisted base-student traces; the old relay trajectories mix authors and cannot substitute. ROFT is synchronous:
collect 64 attempts at version t, sample four explanations each using version t, make one accumulated optimizer
update, checkpoint/export, refresh serving, then collect again. This stricter zero-lag implementation preserves
ROFT's on-policy intent. Initial configs use 20 updates (1,280 source attempts), not an assertion of full-pool coverage;
tasks are shuffled from all 2,613 training tasks. Increase updates intentionally to cover more sources.

Reference: https://arxiv.org/html/2609.35741. This adapts the method to Grug/CalibForge/MSA2. The paper's rendered
HTML reports malformed optimizer powers (LR and epsilon appear as “10”); our LR 1e-5 and epsilon 1e-8 are explicit
adaptation choices, not claims of exact reproduction. Betas .9/.98, weight decay .1, scale4 then clip1, uniform
supervised-token mean, one pass/no replay and four explanations follow the described protocol. Sequence length
65,536 follows our working Grug recipe. Vanilla retains the existing LR3e-4, three-pass recipe; config sft_lr/sft_epochs
are documentation for that existing launcher, not arguments consumed by the online updater.

## Inspect without executing

Run from the OTA checkout root:

```bash
python -m data.relay.reflection.cli plan --config data/relay/reflection/configs/roft.json
python -m data.relay.reflection.cli status --config data/relay/reflection/configs/roft.json
python -m data.relay.reflection.cli validate-vanilla --config data/relay/reflection/configs/vanilla.json
```

`plan` submits nothing and identifies the missing owned-server refresh command. `validate-vanilla` checks the
already-produced ids/loss rows; it does not restart the vanilla/relay chain. Vanilla's configured source is the
existing **1,802-episode matched vanilla arm**, preserving that controlled comparison. If choosing all **3,058**
eligible teacher successes instead, use a new dataset/stage/run identity and report the changed baseline.

## Import and prepare fixed teacher experience

The importer reads `result.json`'s exact MSA `agent_result.metadata.all_messages`. It preserves raw reward and
termination separately: passing tests plus context overflow is still a passing verdict; missing feedback is unknown.
It does not discard failures. Roots must be individual Harbor job directories, not the jobs parent.

```bash
python -m data.relay.reflection.cli import-harbor \
  --roots /scratch/11584/lukedhlee/experiments/relay_rlvr1_msa_control/jobs/*_control \
  --author teacher --model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --output /scratch/11584/lukedhlee/experiments/calibforge_reflection/sources/teacher.jsonl
```

Include original and recovery job directories; compare the import census against the 5,226 completed attempts / 5,213
scored-rendered teacher census. These counts differ because failed infrastructure trials may have no experience.
Import keeps any meaningful experience with unknown verdict; it does not pretend the SFT renderer census defines
ROFT eligibility. Optional source-quality exclusions must be explicit, shared across relevant arms and counted.
The schema intentionally carries a `patch` field; CalibForge artifacts are not necessarily code patches, so the
current Harbor importer leaves it empty instead of inventing a diff. A custom source exporter may populate it.

Once resources and teacher/base-student servers are supplied, `prepare` generates and renders a complete frozen
corpus; `run` processes successive non-replayed batches directly. They are separate workflows: `run` does not
consume `prepare` output, but durable generation records are reusable under identical paths/configuration.

```bash
python -m data.relay.reflection.cli prepare --config data/relay/reflection/configs/teacher_student.json
python -m data.relay.reflection.cli run --config data/relay/reflection/configs/teacher_teacher.json
```

`run` uses config.training.updates; requesting more offline updates than ceil(sources/64) fails rather than replaying.
The final offline batch may be smaller. Both methods retain failed source attempts; incomplete/empty reflections,
overlength native rows and incompatible template boundaries are rejected with a census. No semantic critique
filter, reward weighting or action-token fallback is used. Nonempty reflection thinking and native EOT are trained;
task/trajectory/feedback and assistant header are masked. Evaluation never receives a reflection.

## Online execution contract

The controller never allocates nodes or creates public tunnels. It runs inside operator-supplied resources with
three argv adapters, each receiving request/response JSON:

- `refresh`: run an **owned** server restart command, then verify `/models` root equals the exported checkpoint.
- `collect`: pinned Harbor MSA2 tool-mode, no relay router, selected train tasks only, explicit 16k response cap.
- `train`: OTA `marin_adapter` invokes the isolated native Grug updater inside the supplied training allocation.

Set `runtime.refresh_command` before any online launch. Its placeholders are `{model}` and `{version}`; it must
restart only this experiment's server(s), wait for readiness, and preserve the configured served-model ID. Serving
resources, reverse proxy/tunnel, environment activation and Daytona credentials belong to the cluster supervisor;
credentials come from environment variables, never JSON. The controller and subprocess must run from the OTA
checkout (or have it on PYTHONPATH); the train adapter separately sets cwd to backend.marin_root. Use an allocation
where `srun` can start 16 ranks on four nodes, or replace update_command with an explicitly budgeted submission-and-
wait wrapper; running the default template on a login node is not a valid launch.

The fixed student endpoint in offline teacher→student must serve the **original base**. ROFT must use a separate
endpoint or server lifecycle so updating it cannot change the frozen-author arm. Both generation and refresh check
checkpoint roots, not merely a served-model alias.

Artifact structure: `<output_dir>/<name>/config.resolved.json`, `state.json`, and `update-NNNN/` with source rows,
per-generation records, rejected counts, rows/masks, adapter requests/responses/logs, native optimizer and HF export.
An exclusive lock prevents two owners. Re-running the same config resumes verified responses; a changed config,
source corpus, wrong policy version or stale endpoint stops. An interrupted adapter with no response is ambiguous:
reconcile its job/artifacts and write its verified response; never remove `.inflight` and blindly repeat training.

## Tests and remaining hardware gate

```bash
REFLECTION_TEST_TOKENIZER=/path/to/pinned/grug/tokenizer \
  python -m pytest data/relay/reflection/tests -q -o addopts= --confcutdir=data/relay/reflection/tests
```

Native tests require tokenizer.json, tokenizer_config.json and chat_template.jinja. Dependency versions used:
Python3.12, tokenizers0.23.3, Jinja3.1.6, pytest9.1.1. Without the env variable native tests are explicitly skipped.
The separate Marin `tests/online_reflection/test_accumulation.py` checks masked-token accumulation and Adam state
continuity on CPU. All new entry points are additive; existing SFT source files are unchanged.

Before launch, run the two-update Grug/GB200 smoke in HANDOFF.md. CPU tests establish masks and orchestration,
not full-model numerical stability, distributed memory/sharding, checkpoint I/O, HF export or serving reload parity.
