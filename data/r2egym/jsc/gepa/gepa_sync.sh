#!/bin/bash
# gepa_sync.sh — put the local (ground-truth) GEPA scripts on Jupiter.
#
# CLAUDE.md §Always: local clones are ground truth, clusters never diverge. For the OpenThoughts-Agent checkout that
# means commit -> push -> `git pull` on the cluster. It does NOT work for these scripts:
# /e/project1/transfernetx/lee27/code/snowball is a PLAIN DIRECTORY, not a git checkout and not a symlink into the
# OTA checkout (verified 2026-09-20: `git rev-parse` there says "not a git repository"). Every script the probe
# generator uses already lives there as a copy, so a new one has to be copied too. This script is that copy, and it
# only ever pushes -- it never pulls the cluster's version back over the repo.
#
#   bash data/r2egym/jsc/gepa/gepa_sync.sh            # dry run: show what would change
#   bash data/r2egym/jsc/gepa/gepa_sync.sh --go       # copy
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DST=/e/project1/transfernetx/lee27/code/snowball/gepa
GO=${1:-}
FILES="gepa_split.py gepa_tree.py gepa_wave.sh gepa_score.py gepa_feat.py gepa_ledger.py gepa_dump.py gepa_worst.py gepa_final.sh seed_blocks.json"
for f in $FILES; do [ -f "$HERE/$f" ] || { echo "missing $HERE/$f"; exit 1; }; done
if [ "$GO" != "--go" ]; then
  echo "DRY RUN -- would copy to jupiter:$DST"
  ( cd "$HERE" && rsync -n -avi --no-perms --no-owner --no-group $FILES "jupiter:$DST/" ) || \
    echo "(rsync dry run failed; the destination dir may not exist yet -- --go creates it)"
  echo
  echo "run: bash $HERE/gepa_sync.sh --go"
  exit 0
fi
ssh -o ConnectTimeout=25 jupiter "mkdir -p $DST"
( cd "$HERE" && rsync -avi --no-perms --no-owner --no-group $FILES "jupiter:$DST/" )
ssh -o ConnectTimeout=25 jupiter "chmod +x $DST/*.sh; ls -l $DST"
