#!/usr/bin/env python3
"""Extract the held-out RST tasks from the source bundle into a Harbor task tree (one dir per task).

    python rst_heldout_extract.py --source <rst_source dir> --heldout rst_heldout_tasks.json --out /e/data1/mmlaion/lee27/tasks/rst_heldout

Each task package in ``data/tasks-*.tar`` sits under ``tasks/<task_id>/`` (instruction.md, task.toml, environment/Dockerfile,
tests/, solution/). The dir keeps the task_id so results join to ``rst-heldout-tasks.jsonl`` on ``task``.
"""
import argparse, collections, json, tarfile
from pathlib import Path
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, type=Path); ap.add_argument("--heldout", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
want = set(json.load(open(a.heldout)))
meta = {r["task_id"]: r for r in pq.read_table(a.source / "metadata" / "tasks.parquet", columns=["task_id", "member_prefix", "shard"]).to_pylist()
        if r["task_id"] in want}
missing = want - set(meta); assert not missing, f"{len(missing)} held-out ids not in tasks.parquet: {sorted(missing)[:3]}"
byshard = collections.defaultdict(dict)
for tid, r in meta.items(): byshard[r["shard"]][r["member_prefix"].rstrip("/")] = tid
a.out.mkdir(parents=True, exist_ok=True); got = 0
for shard, prefixes in sorted(byshard.items()):
    with tarfile.open(a.source / shard) as tf:
        for m in tf:
            top = "/".join(m.name.split("/")[:2])
            tid = prefixes.get(top)
            if tid is None or not (m.isfile() or m.isdir()): continue
            rel = m.name[len(top):].lstrip("/")
            if not rel: continue
            m.name = rel
            tf.extract(m, a.out / tid, filter="data")
    got += len(prefixes); print(f"{shard}: {len(prefixes)} tasks (total {got})", flush=True)
bad = [d.name for d in a.out.iterdir() if not ((d / "instruction.md").exists() and (d / "task.toml").exists() and (d / "environment" / "Dockerfile").exists() and (d / "tests").is_dir())]
print(f"extracted {got} of {len(want)}; incomplete task dirs: {len(bad)} {bad[:5]}")
