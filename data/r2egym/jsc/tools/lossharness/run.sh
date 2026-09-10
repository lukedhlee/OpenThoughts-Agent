#!/bin/bash
# Loss-equivalence harness driver: runs loss_equiv.py against two MarinSkyRL
# revisions and diffs the two JSON reports with compare.py.
#
#   ./run.sh                       -> $OLD_REPO vs $NEW_REPO
#   ./run.sh <repo-path> <label>   -> that revision vs $OLD_REPO, e.g.
#     ./run.sh ~/MarinSkyRL-wt/snowball-r2egym MERGED
#
# Parameters (environment, all optional except where noted):
#   LOSS_HARNESS_PY  python to run the harness with. REQUIRED unless `python3`
#                    on PATH already has CPU torch + omegaconf. Point it at a
#                    venv that has them, e.g. <marinskyrl-checkout>/.venv/bin/python
#                    (torch 2.11.0 / omegaconf 2.3.1 is the combination used for
#                    the 2026-09-09 migration baseline).
#   OLD_REPO         checkout of the baseline revision      (default: ./old)
#   NEW_REPO         checkout of the revision under test    (default: ./new)
#
# OLD_REPO/NEW_REPO are MarinSkyRL checkouts (root or its skyrl-train/). The
# 2026-09-09 baseline compared be413fcb (OLD) against 93d84333 (NEW).
#
# PYTHONPATH is cleared for every run so only the revision under test supplies
# skyrl_train.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${LOSS_HARNESS_PY:-python3}"
OLD_REPO="${OLD_REPO:-$HERE/old}"
NEW_REPO="${NEW_REPO:-$HERE/new}"
cd "$HERE"

run() { PYTHONPATH= "$PY" loss_equiv.py --repo "$1" --out "$2" >/dev/null 2>"${2%.json}.err"; }

if [ $# -eq 0 ]; then
  run "$OLD_REPO" old.json
  run "$NEW_REPO" new.json
  PYTHONPATH= "$PY" compare.py old.json new.json OLD NEW
else
  REPO="$1"; LABEL="${2:-OTHER}"
  [ -f old.json ] || run "$OLD_REPO" old.json
  run "$REPO" "$(echo "$LABEL" | tr 'A-Z' 'a-z').json"
  PYTHONPATH= "$PY" compare.py old.json "$(echo "$LABEL" | tr 'A-Z' 'a-z').json" OLD "$LABEL"
fi
