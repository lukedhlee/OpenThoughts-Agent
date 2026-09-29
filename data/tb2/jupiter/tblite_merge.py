#!/usr/bin/env python3
"""Merge the job dirs of one sharded run (tblite_chain.sh with SHARD=i/n on n servers) into one directory that
summarize_tb2.py can score: symlinks to every trial dir, plus a config.json whose task list is the union.

    python3 tblite_merge.py <merged_dir> <shard_dir> [<shard_dir> ...]
"""
import json, sys
from pathlib import Path

out = Path(sys.argv[1]); shards = [Path(p) for p in sys.argv[2:]]
assert shards and not out.exists(), "give a new merged dir and at least one shard dir"
out.mkdir(parents=True); tasks = []
for s in shards:
    tasks += json.load(open(s / "config.json"))["tasks"]
    for trial in sorted(p for p in s.iterdir() if (p / "result.json").exists()):
        link = out / trial.name
        assert not link.exists(), f"duplicate trial {trial.name}"
        link.symlink_to(trial.resolve())
names = [t["path"] for t in tasks]; assert len(names) == len(set(names)), "shards overlap"
json.dump({"merged_from": [str(s) for s in shards], "tasks": tasks}, open(out / "config.json", "w"), indent=1)
print(f"{out}: {len(tasks)} planned tasks, {sum(1 for _ in out.glob('*/result.json'))} trial dirs from {len(shards)} shards")
