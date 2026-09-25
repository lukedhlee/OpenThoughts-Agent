# CalibForge on Daytona: three shared snapshots, and how to get them back

2,457 of the 2,500 recommended CalibForge tasks run on Daytona from three snapshots. Each snapshot is the exact base
image the task images were built on (digest-pinned), plus Harbor's agent tooling. At sandbox start, the task's
`setup_files/setup.sh` downloads that task's own image layers by digest, checks each sha256, and stacks them onto the
root filesystem the way a container runtime would. The layers come from our public HF mirror
`laion/calibforge-daytona-layers`, and Docker Hub is only a fallback, so the pool no longer depends on AweAI's
images. The median sandbox is ready in 8–11 s and the p90 in about 20 s. After a purge, one command recreates only
the missing snapshots from our ghcr images (about 22 s each), never deletes anything, and gates each base with a
no-op run in which Docker Hub is switched off.

## One command on this Mac

```bash
/Users/lukedhlee/.local/share/otagent/calibforge-recovery/cf1-20260925/restore.sh
```

If the local bundle is gone, fetch the same bundle from HF first:
`hf download laion/calibforge-daytona-layers --repo-type dataset --include 'bundle/*' --local-dir <dir>`, then run
`bash <dir>/bundle/restore.sh`.

- `--repos ubuntu2404,bookworm` restores only those bases.
- `--require-registry` refuses the Dockerfile fallback. That fallback builds `FROM` Docker Hub bases, so it is the
  one path that still needs Docker Hub.
- The smoke gate always runs with `CF_DOCKERHUB=0` (`--mirror-only`, set in `restore.sh`), so a pass shows the
  rebuild works without Docker Hub.
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

- **Bundle** (`~/.local/share/otagent/calibforge-recovery/cf1-20260925/`, with an off-machine copy at
  `bundle/` in the HF dataset). It holds `task_tree.tar.gz` (2,457 tasks plus `pool.json` and `coverage.tsv`), the
  baked Dockerfiles, `manifest.json` (digests), one smoke task per base, and copies of every script. The older
  `cf1-20260924` bundle has the same snapshots, but its trees fetch only from Docker Hub.
- **Images:** the public repo `lukedhlee/r2egym-daytona-images` holds `calibforge/`, and
  `.github/workflows/publish-calibforge.yml` builds each image on an amd64 runner. The tag is
  `cf1-<recipe sha256[:12]>`, and the workflow refuses a recipe whose checksum differs from `calibforge/images.json`.
  A changed recipe therefore gets a new tag, a new digest and a new snapshot name. It can never silently refill an
  old name. The packages are public, so Daytona pulls them anonymously.
- **Per-task layers:** the public HF dataset
  [`laion/calibforge-daytona-layers`](https://huggingface.co/datasets/laion/calibforge-daytona-layers) holds them
  at `blobs/sha256/<first 2 hex>/<64 hex>`: 9,835 blobs, 137.1 GB, CC-BY-4.0 with credit to CalibForge (AweAI).
  - `setup.sh` tries `CF_MIRRORS` first (default: that dataset), then Docker Hub's blob endpoint unless
    `CF_DOCKERHUB=0`.
  - Every layer's sha256 is checked against its digest either way.
  - `mirror_upload.py` rebuilt the mirror by streaming 6 GB batches from Docker Hub (verify, commit, delete); it
    resumes where it stopped.
  - `registry_cache.tar.gz` in the dataset holds the manifests and configs that `build_tree.py` needs, so the tree
    can be rebuilt without Docker Hub too.

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

Point Harbor at the tree (built with the default `--mirrors`) with `environment.kwargs.auto_snapshot: true`. Use the hook-carrying Harbor
(`lukedhlee/snowball-r2egym`, i.e. `PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src`). Without the
setup-files hook, a task runs on the bare base and silently scores 0. `task.toml` carries the image's ENV
differences (for example PATH for `/opt/venv` tasks) and `workdir = "/app"`. Verifiers need internet
(uv/pytest downloads), which Daytona sandboxes have.

## Mirror gates run on 2026-09-25

- **Mirror-only no-op gate, 25 fresh tasks** (9 ubuntu, 10 tb2verifier, 6 bookworm), using a tree built with
  `--no-dockerhub`:
  - 25/25 passed, with no exceptions and reward 0.
  - 24 have a ctrf report with tests collected, and 1 verifier writes no ctrf.
  - setup.sh took a median of 6.1 s (p90 15.9 s, max 20.9 s).
- **Burst test, 200 sandboxes, mirror only** (`burst_test.py`, 85 ubuntu, 108 tb2verifier, 7 bookworm):
  - The creates took 165 s at 4 per second.
  - Then all 200 setup.sh runs started together, and 200/200 succeeded with no mirror retries.
  - setup.sh took p50 5.6 s, p90 12.9 s, max 34.4 s, and the download part took p50 2.6 s (p90 6.7 s).
  - The burst pulled 13.5 GB from HF in 34 s of wall time.
  - Every sandbox was deleted afterwards.
- **Restore on the live pool** from `cf1-20260925` in mirror-only mode: run `20260925-012828`, three passes, and the
  snapshot IDs were unchanged.

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
