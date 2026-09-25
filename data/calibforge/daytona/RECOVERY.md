# CalibForge on Daytona: three shared snapshots, and how to get them back

2,457 of the 2,500 recommended CalibForge tasks run on Daytona from three snapshots. Each snapshot is the exact base
image the task images were built on (digest-pinned), plus Harbor's agent tooling. At sandbox start, the task's
`setup_files/setup.sh` downloads that task's own image layers by digest, checks each sha256, and stacks them onto the
root filesystem the way a container runtime would. The median sandbox is ready in 8 s and the p90 in 19 s. After a
purge, one command recreates only the missing snapshots from the published images (about 22 s each), never deletes
anything, and gates each base with a no-op run.

## One command on this Mac

```bash
/Users/lukedhlee/.local/share/otagent/calibforge-recovery/cf1-20260924/restore.sh
```

- `--repos ubuntu2404,bookworm` restores only those bases.
- `--require-registry` refuses the Dockerfile fallback.
- The key comes from `~/.config/otagent/daytona_eval.env`, a copy of Jupiter's `keys/daytona_eval.env` (key `...b61`).
  A key exported in the shell cannot shadow it. Override it with `--key-file`.

Exit zero means three things for every selected base. The snapshot is ACTIVE and still has the same ID at the final
check. The no-op trial raised no exception, so setup.sh applied every layer. The reward is 0 and the verifier wrote a
ctrf report with tests collected. Each run keeps its logs in the bundle's `runs/<timestamp>/`.

| base | snapshot | tasks | registry image |
|---|---|---|---|
| ubuntu2404 | `harbor__313e69c036ad__snapshot` | 1,133 | `ghcr.io/lukedhlee/calibforge/ubuntu2404@sha256:12c397e0…` |
| tb2verifier | `harbor__13bf95818376__snapshot` | 1,202 | `ghcr.io/lukedhlee/calibforge/tb2verifier@sha256:6af71399…` |
| bookworm | `harbor__239bee0dfdf6__snapshot` | 122 | `ghcr.io/lukedhlee/calibforge/bookworm@sha256:a9279f7e…` |

The full digests are in the bundle's `manifest.json`.

## What is where

- **Bundle** (`~/.local/share/otagent/calibforge-recovery/cf1-20260924/`): `task_tree.tar.gz` (2,457 tasks plus
  `pool.json` and `coverage.tsv`), the baked Dockerfiles, `manifest.json` (digests), one smoke task per base, and
  copies of every script. It is a local backup.
- **Images:** the public repo `lukedhlee/r2egym-daytona-images` holds `calibforge/`, and
  `.github/workflows/publish-calibforge.yml` builds each image on an amd64 runner. The tag is
  `cf1-<recipe sha256[:12]>`, and the workflow refuses a recipe whose checksum differs from `calibforge/images.json`.
  A changed recipe therefore gets a new tag, a new digest and a new snapshot name. It can never silently refill an
  old name. The packages are public, so Daytona pulls them anonymously.
- **Per-task layers** are not re-hosted. `setup.sh` reads them from Docker Hub's blob endpoint
  (`aweaiteam/calibforge`), by the digests recorded in the tree. Docker Hub meters manifest requests, not blob
  downloads, and nothing reads a manifest at sandbox start. `CF_MIRRORS` (blob base URLs, tried first) is the hook
  for a mirror if that ever changes (CalibForge is CC-BY-4.0, so a public mirror with attribution is allowed).

## Rebuild the tree from public sources (bundle lost)

The snapshot names depend only on `pool.py`'s recipes, so a rebuilt tree maps to the same three snapshots.

```bash
export PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src
hf download hamishivi/agent-task-calibforge task-data.tar.gz --repo-type dataset --revision f605550ac21c37cecc34e6767a753219e0ca58d3 --local-dir <dl>
# sha256 79672cdec29ec9160b663e99dd0bb17be656c74756dea59c68a4fe2d285abb80
/Users/lukedhlee/harbor/.venv/bin/python data/calibforge/daytona/build_tree.py \
  --parquet ai_memory/active/snowball-sft/research/2026-09-24_env_checks/calibforge/calibforge_tasks.parquet \
  --source <dl>/task-data.tar.gz --cache ~/.local/share/otagent/calibforge-work --out <new tree>
/Users/lukedhlee/harbor/.venv/bin/python data/calibforge/daytona/recover_pool.py prepare --tree <new tree> --bundle <new bundle>
/Users/lukedhlee/harbor/.venv/bin/python data/calibforge/daytona/recover_pool.py set-digests --bundle <new bundle> \
  --digests <digests.json from the publish-calibforge run's `digests` artifact>
```

## Running tasks

Point Harbor at the tree with `environment.kwargs.auto_snapshot: true`. Use the hook-carrying Harbor
(`lukedhlee/snowball-r2egym`, i.e. `PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src`). Without the
setup-files hook, a task runs on the bare base and silently scores 0. `task.toml` carries the image's ENV
differences (for example PATH for `/opt/venv` tasks) and `workdir = "/app"`. Verifiers need internet
(uv/pytest downloads), which Daytona sandboxes have.

## Gates run on 2026-09-24

- **Cold restore:** run `20260924-231654`. All three snapshots were registered from the digests (22 s each) and all
  three smoke gates passed. **Live pool:** run `20260924-232543` changed nothing (same IDs), passed, and took 33 s.
- **No-op gate, 50 tasks** stratified by base and category (22 ubuntu, 22 tb2verifier, 6 bookworm), at concurrency 25:
  - 50/50 started, applied their layers, and scored 0, with no exceptions.
  - 47 have a ctrf report with tests collected, and 2 have verifiers that write no ctrf (reward.txt only).
  - In the remaining task, the test module fails to import numpy. That happens in the original image too: its
    `uv pip install --system` is refused there as well, which was checked by hiding the tooling's python3.
  - Setup: sandbox create and start takes a median of 3.6 s (p90 7.6 s). setup.sh takes a median of 3.6 s (p90
    10.4 s, max 31.8 s for 395 MB of layers).
- **Fidelity, 10 tasks.** I compared each sandbox after setup.sh with the image rebuilt from its registry layers
  (151,357 paths):
  - Nothing is missing, and every image package is present at the same version.
  - Contents differ only in `/etc/hosts`, `/etc/hostname` and `/etc/resolv.conf`, and in the files the agent
    tooling rewrites (`/etc/ld.so.cache`, `/etc/shells`, `/var/lib/shells.state`, `/var/lib/dpkg/status`,
    ldconfig's aux-cache).
  - Every extra path belongs to the tooling (tmux, asciinema and, where the image had none, python3), dpkg
    bookkeeping, or Daytona's runtime.
  - Results are in `~/.local/share/otagent/calibforge-work/fid10/summary.json`.

## If recovery stops

The same rules as the R2E-Gym pool apply (`data/r2egym/daytona_artifacts/RECOVERY.md`):
- A client timeout does not stop a server build, so rerun the command.
- ERROR snapshots are preserved for inspection.
- Never delete a snapshot this workstream did not create.
- Re-count the org's custom snapshots with `snapshot_census.py` before any create. The cap is 40, and this pool uses 3.

## Verification

```bash
cd data/calibforge/daytona && PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src \
  /Users/lukedhlee/harbor/.venv/bin/python -m unittest test_snapshot_recovery -v
python gate.py sample --tree <tree> --n 50 --out s.txt && python gate.py run --tree <tree> --tasks s.txt --jobs <dir>
python fidelity_check.py --tree <tree> --tasks a,b,c --out <dir>
```
