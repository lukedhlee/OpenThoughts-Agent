#!/bin/bash
# One-command recovery of the CalibForge Daytona pool (three shared snapshots). Recreates only missing snapshots
# (from the digest-pinned ghcr images in manifest.json, Dockerfile fallback otherwise), never deletes, then runs one
# no-op gate per base. Key: ~/.config/otagent/daytona_eval.env (Luke's ...b61); shell-exported keys are ignored.
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src${PYTHONPATH:+:$PYTHONPATH}
exec /Users/lukedhlee/harbor/.venv/bin/python "$BUNDLE/recover_pool.py" restore --bundle "$BUNDLE" --harbor /Users/lukedhlee/harbor/.venv/bin/harbor "$@"
