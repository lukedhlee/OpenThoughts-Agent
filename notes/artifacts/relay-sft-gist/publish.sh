#!/bin/bash
# publish.sh — push the figures to the figures gist, re-render pub/ with the new commit SHA, update the text gist.
set -e; export PATH=$HOME/.local/bin:$PATH; G=$(cd "$(dirname "$0")" && pwd); FID=a8ff42b325eb159091ca5d158920aa0e; TID=8669f5b6f97e4e3bae72344a8fd51f4b
W=$(mktemp -d); git clone -q https://gist.github.com/$FID.git $W; cp $G/fig*.png $W/; cd $W
git add -A; git -c user.name="Donghyun Lee" -c user.email=lukeleeai@gmail.com commit -qm "figures $(date +%F)" || true
git -c credential.helper='!gh auth git-credential' push -q; SHA=$(git rev-parse HEAD); cd $G; rm -rf $W
mkdir -p pub; sed "s#FIG_RESULTS#https://gist.githubusercontent.com/lukedhlee/$FID/raw/$SHA/fig1_results.png#; s#FIG_WHY#https://gist.githubusercontent.com/lukedhlee/$FID/raw/$SHA/fig2_why.png#" relay-sft-0921.md > pub/relay-sft-0921.md
! grep -qiE "claude|generated with|co-authored" pub/relay-sft-0921.md
gh gist edit $TID -f relay-sft-0921.md pub/relay-sft-0921.md; echo "published figures $SHA"
