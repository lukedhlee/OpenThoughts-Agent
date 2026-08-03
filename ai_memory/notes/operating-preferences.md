# Operating preferences — how Luke works, and the cluster index

How Luke wants work done (planning style, autonomy gates, git discipline) plus the one-line index of
every cluster and its reference note. The load-bearing gates are also stated in `handoff.md`
Invariants; this file has the reasoning and the full pointer list.
Read at session start if the handoff's Invariants section leaves anything ambiguous.

Was `ai_memory/user_preference.md` until the 2026-07-29 memory refactor.

---

## Session mechanics

- **Session start:** read `ai_memory/handoff.md` first; dated deep logs under `ai_memory/logs/`;
  decisions in `ai_memory/decisions.md`; failure modes in `ai_memory/gotchas.md`.
- **Active focus:** RL on general **`Qwen/Qwen3-30B-A3B`** (branch `lukedhlee/vista-moe-grpo-30b`).
  Training cluster is now **Jupiter**; TACC Vista is concluded.
- **Primary repo:** **OpenThoughts-Agent** (`$SCRATCH/OpenThoughts-Agent`). Not `dc-agent`.

## How Luke plans

Teach-first (cluster facts, tradeoffs) → name the single bottleneck → fix that hard → ask "what's left /
what's the next action?" → only then launch. Wants the full picture and the DIY path, not black-box
fixes. Ruthless priority ("this is the most important thing").

## Autonomy gates

- On Vista/Jupiter smoke/bring-up failures, **keep fixing + relaunching in a loop** without waiting for
  the user to say "go fix it" / "status". Report only when blocked (e.g. MFA) or when a milestone lands.
- **Exception:** after the user cancels or says don't launch — **stop.** Do not auto-relaunch until they
  ask.
- Never `scancel` / relaunch without OK; diagnose+fix loops are fine.
- Ordinary non-destructive experiment submission does not need approval — state the key config, then go.

## Git / PRs (HARD)

- Never `git push`, open a PR, or merge without an **explicit** user ask in that turn. Cluster-local
  patches are OK for unblocking a job; the upstream PR is a separate, opt-in step. Luke closed a rushed
  MarinSkyRL PR #93 — next time do a **proper** PR only when asked.
- **Proper MarinSkyRL / marin-fork PR (when asked):** worktree off `main` → focused commit → local
  marin-style lint/format/type gate → push → PR with a real Summary + Test plan → iterate until CI green
  → **never self-merge** (supervisor merges). An unmerged fix rides `--skyrl-ref`, not a shared-branch
  hack. See `.claude/projects/marinskyrl/marinskyrl.md`.

## PI goal

RL general **`Qwen/Qwen3-30B-A3B`** (MoE; not Coder) → post-RL lift on **SWE-Bench Verified**,
**OT-TB Lite**, **Terminal Bench 2.0**. Sanity with a MoE smoke before full scale.

## Secrets

Mac `~/.config/otagent/secrets.env` → Vista `$SCRATCH/keys.env` (600). See
`ai_memory/notes/vista_secrets.md`. **Never commit.**

## Cluster index

Primary clusters: **Jupiter (JSC), JURECA (JSC), TACC Vista, MareNostrum**. The JURECA dense-model
agentic bring-up is parked; JURECA is now the x86_64 apptainer-bridge host.

- **TACC Vista** login: `ssh vista` (`lukedhlee`); password + TACC TOTP. Cheat sheet:
  `ai_memory/notes/tacc_vista.md`. Lab OT-Agent stack: `.claude/ops/tacc/ops.md`
  (penfever / `CCR24067`).
- **JSC JuDoor id:** **`lee27`**. Local key: `~/.ssh/id_ed25519_jsc`. Keys upload **per system**.
- **JURECA** SSH / `from=` IP workflow: `ai_memory/notes/jureca_ssh.md` +
  `ai_memory/scripts/jureca_from_clause.sh`.
- **JUDAC** SSH (data/git/HF): `ai_memory/notes/judac_ssh.md` — same key, **separate** JuDoor upload;
  `ssh judac`.
- **Jupiter** SSH: `ai_memory/notes/jupiter_ssh.md` — same key, **separate** JuDoor upload;
  `ssh jupiter` (IPv4 → login02).
- **Jupiter** cluster facts: `ai_memory/notes/jupiter_cluster.md` (4×GH200/node, `booster` 12h;
  `develbooster` reservation = 8-node smoke).
- **Jupiter** WandB: `ai_memory/notes/jupiter_wandb.md` (offline on compute → sync on login).
- **JSC** path / inode hazards: `ai_memory/notes/jsc_paths_hazards.md` (LAION iffMD).
  Jupiter = `/e/...` not `/p/...`.
- **JURECA** what-goes-where (envs/caches/logs): `ai_memory/notes/jureca_what_goes_where.md`.
- Marianna's shared env: `/p/project1/ccstdl/envs/marianna/py3.12/`.
