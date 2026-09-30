# Recover the R2E-Gym pool after a purge

The recovery command recreates only missing snapshots, resumes unfinished builds, activates inactive images, and runs one oracle/no-op pair per repository. It never deletes a snapshot. A permanent registry image avoids repeating apt, clone, download and extraction work. All eleven images were published on 2026-09-19 and the bundle manifest now pins each one by digest, so recovery pulls; the same command still falls back to the saved Dockerfiles for any entry without an image.

## Current backup

`/Users/lukedhlee/.local/share/otagent/r2egym-recovery/v3-20260918/`

This directory contains the full 3,035-task tree in `task_tree.tar.gz`, eleven Dockerfiles with Harbor's agent tooling already appended, an exact snapshot-name manifest, eleven smoke tasks, and copies of the recovery scripts. It survives deletion of the previous session's scratch directory. It is a local backup, not a registry image or an off-machine backup.

## One command on this Mac

```bash
/Users/lukedhlee/.local/share/otagent/r2egym-recovery/v3-20260918/restore.sh
```

Add `--repos numpy,pillow` to restore only those repositories. Add `--require-registry` to refuse a Dockerfile rebuild. The default is four concurrent recovery/gate workers, with numpy first. Credentials come explicitly from `~/.config/otagent/secrets.env`; shell-exported keys cannot shadow them. Override with `--key-file`.

Each invocation makes a new timestamped directory under `runs/`, preserving logs and results. Exit zero means all selected repositories scored oracle 1 / no-op 0 and their same snapshot IDs are still ACTIVE at the final check. A failed verifier, missing result, or infrastructure error fails the command. A concurrent eviction fails the final check even if its earlier gate passed. These are representative smoke checks, not a repeat of the full 3,035-task dataset gate.

The current pool is reused rather than purged during verification. This verifies the automation and gates; it does not measure a cold registry restore.

## One command on Horizon (copy of the bundle, 2026-09-29)

The same bundle is on TACC Horizon at `~/.local/share/otagent/r2egym-recovery/v3-20260918/` (no `runs/`), with
`restore_horizon.sh`: the same `recover_pool.py restore`, using `~/snowball`'s harbor (dcf609bc) and python, the eval-org
key `~/.config/otagent/daytona_eval.env`, and `RES_OPTIONS` for Horizon's slow DNS. Run it on the login node:

```bash
~/.local/share/otagent/r2egym-recovery/v3-20260918/restore_horizon.sh --require-registry   # [--repos numpy,pillow]
```

`--require-registry` pulls the public ghcr digests and refuses a Dockerfile rebuild (Horizon has no Docker). It needs
one free snapshot slot per missing repository. On 2026-09-29 all 11 were missing and the org was at its cap.

## Publish once to make subsequent recovery faster

**Done for the v3 pool on 2026-09-19.** The images live in the public repository
[`lukedhlee/r2egym-daytona-images`](https://github.com/lukedhlee/r2egym-daytona-images), built by a GitHub Actions
matrix on amd64 runners and pushed to `ghcr.io/lukedhlee/r2egym/<repo>`. That route exists because this Mac has no
Docker engine and the local GitHub credential has no package-write scope; a workflow's built-in token has both. The
packages are public, so Daytona pulls them anonymously and the shared organization needs no registry credential.

Each image is tagged `v3-<first 12 hex of the recipe SHA-256>`, and the workflow refuses to build a Dockerfile whose
checksum no longer matches `images.json`. A changed recipe therefore gets a new tag and a new digest and can never
overwrite the image an existing snapshot name points at. This is the guard that matters: a snapshot is *named* after
Harbor's environment-directory hash but *filled* from a registry digest, and a silent drift between the two produces a
sandbox with the right name and the wrong contents, which still passes a smoke gate.

To publish a new pool, push its recipes and `images.json` to that repository and run the workflow; take the digests
from the run summary (or its `digests` artifact) into each manifest entry's `image` field. The local
`recover_pool.py publish` path below remains for a machine that does have Docker.

```bash
export PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src
/Users/lukedhlee/harbor/.venv/bin/python \
  data/r2egym/daytona_artifacts/recover_pool.py publish \
  --bundle /Users/lukedhlee/.local/share/otagent/r2egym-recovery/v3-20260918 \
  --registry ghcr.io/OWNER/r2egym
```

The publisher builds the complete baked image, pushes it, and saves its immutable digest after each repository. It resumes by skipping saved digests. Authentication stays in Docker's credential store. For a private registry, configure read access in Daytona before restoring. Use a fresh bundle when changing image recipes; do not retag a changed image as an old environment.

After publishing, the same restore command sends the digest directly to Daytona. Task Dockerfiles and Harbor environment hashes stay unchanged, so all prompt variants retain snapshot sharing. First cold restore still needs a benchmark; no speedup number has been measured yet. Documentation: https://www.daytona.io/docs/snapshots/

## If recovery stops

- **Timeout:** rerun the same command. A client timeout does not stop a server build, and the tool will not delete or resubmit that build. Default wait is 40 minutes per image.
- **Server ERROR/BUILD_FAILED:** inspect the per-repo log and snapshot. Recovery deliberately does not delete even failed snapshots. Follow `.claude/projects/daytona/daytona.md` before reclaiming the exact failed image.
- **Quota full:** use the documented cursor-paginated live-sandbox audit before reclaiming an idle image. Never delete live or queued-job dependencies. No automatic purge is hidden in recovery.
- **Credentials:** pass the correct explicit key file. A failed GET is not treated as a missing snapshot unless it is a not-found response.
- **Bad gate:** keep the evidence in `runs/`; do not launch the model probe on that repository until resolved.

## Prepare another portable bundle

```bash
python data/r2egym/daytona_artifacts/recover_pool.py prepare \
  --tree /path/to/r2egym-daytona-v3 --bundle /path/to/new-recovery-bundle
```

Use the hook-enabled Harbor Python environment. The destination must not exist. Preparation checks that every task in a repository has the expected environment hash. The manifest checksums the build recipes and task archive. On another machine, use the bundled `recover_pool.py restore --bundle <directory> --harbor <binary> --key-file <file>` with Harbor/Daytona dependencies installed and the appropriate Harbor source on `PYTHONPATH`.

## Verification

```bash
python -m unittest discover -s data/r2egym/daytona_artifacts -p test_snapshot_recovery.py -v
```

Eleven tests exercise active/inactive/building/error states, recovery after ambiguous timeouts, credential precedence, partial registry-publication failure, incorrect gate rewards, and eviction after a passing gate. Live smoke results live in the bundle's `runs/` directories.

### Registry publication on 2026-09-19

All eleven images were rebuilt from the Dockerfiles at 14:42 PT after a shared-org purge left only numpy, pyramid and
tornado; every one passed oracle 1 / no-op 0 (run `20260919-144202`). They were then published from GitHub Actions run
`35473900474`, and the bundle manifest was patched with the eleven digests after asserting, per repository, that the
published snapshot name matches the tree's environment hash and that the tag carries this recipe's checksum.

Measured: registering `harbor__18aa570afb60__snapshot`'s image (pyramid) from its digest took **58 s** end to end,
against 107.8 s to rebuild the same image from its Dockerfile. The throwaway snapshot used for that measurement was
deleted. This measures snapshot registration from the registry, not a full `restore.sh` pass with its smoke gate — the
gate has not yet been run against a digest-registered snapshot.

### Live verification on 2026-09-18

All eleven representative oracle/no-op pairs passed in run `20260918-233854`. During that run, another key purged pool snapshots. Run `20260918-234242` rebuilt pyramid from its saved baked Dockerfile in 107.8 seconds and passed oracle/no-op; this validates the Dockerfile fallback, not registry restore. The final census found only numpy, pyramid and tornado present; the other eight were missing. `snapshot_audit.json` and `live_census.json` in the bundle preserve the observations. The final-check helper rejected the old run's missing/replaced snapshots as intended.

Smoke tasks use the maintained `protocol.git.allow=never` setup fix, preventing retries against GitHub's retired git:// service. This changes no environment hash. The full task archive preserves the original tree, including its older setup scripts; apply the same maintained setup fix when deploying those tasks if needed.
