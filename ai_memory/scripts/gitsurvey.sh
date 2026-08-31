#!/bin/bash
MAXD=$1; shift
for root in "$@"; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth "$MAXD" \( -name node_modules -o -name .venv -o -name venv -o -name site-packages -o -name hf_hub -o -name .cache -o -name envs -o -name __pycache__ -o -name experiments -o -name checkpoints -o -name apptainer_staging\* \) -prune -o \( -type d -name .git -o -type f -name .git \) -print 2>/dev/null
done | sort -u | while read g; do
  d=$(dirname "$g"); case "$d" in */.git/modules/*) continue;; esac
  cd "$d" 2>/dev/null || continue
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || continue
  echo "@@REPO $(hostname -s) $d"
  echo "WT $([ -f "$g" ] && echo true || echo false)"
  git remote -v 2>/dev/null | awk '$3=="(fetch)"{print "REMOTE " $1 " " $2}'
  echo "HEAD $(git rev-parse --abbrev-ref HEAD 2>/dev/null)|$(git rev-parse --short=10 HEAD 2>/dev/null)|$(git log -1 --format='%cI|%an|%s' 2>/dev/null)"
  echo "DIRTY $(git status --porcelain 2>/dev/null | grep -vc '^??') $(git status --porcelain 2>/dev/null | grep -c '^??')"
  git status --porcelain 2>/dev/null | head -15 | sed 's/^/DIRTYLINE /'
  echo "STASH $(git stash list 2>/dev/null | wc -l | tr -d ' ')"
  echo "FETCH $(stat -c %y .git/FETCH_HEAD 2>/dev/null || stat -f %Sm -t %Y-%m-%dT%H:%M:%S .git/FETCH_HEAD 2>/dev/null || git rev-parse --git-dir | xargs -I{} sh -c 'stat -c %y {}/FETCH_HEAD 2>/dev/null || stat -f %Sm {}/FETCH_HEAD 2>/dev/null')"
  git for-each-ref --sort=-committerdate refs/heads --format='BR %(refname:short)|%(upstream:short)|%(upstream:track,nobracket)|%(objectname:short=10)|%(committerdate:iso-strict)|%(authorname)|%(subject)' 2>/dev/null
  git worktree list --porcelain 2>/dev/null | awk '/^worktree /{w=$2} /^branch /{print "WTREE " w " " $2} /^detached/{print "WTREE " w " (detached)"}'
  echo "SUBM $(git submodule status 2>/dev/null | wc -l | tr -d ' ')"
  git submodule status 2>/dev/null | sed 's/^/SUBMLINE /'
  echo "@@END"
done
