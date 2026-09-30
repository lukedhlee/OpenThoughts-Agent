#!/bin/bash
# pair_watch.sh — poll W&B every 15 min until the Jupiter half of the SFT port pair is there, then append the verdict to
# ~/briefs/train-port.STATUS.md and exit. Runs in a login-node tmux; gives up after 5 days.
set -uo pipefail
ST=~/briefs/train-port.STATUS.md; HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 480); do
  out=$(set -a; source ~/.config/otagent/secrets.env; set +a; ~/snowball/envs/snowball/bin/python "$HERE/pair_compare.py" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    { echo; echo "## SFT pair VERDICT (auto, $(TZ=America/Los_Angeles date '+%F %H:%M PT'), pair_compare.py)"; echo '```'; echo "$out"; echo '```'; } >> "$ST"
    echo "verdict written"; exit 0
  fi
  sleep 900
done
echo "gave up after 5 days" >> "$ST"
