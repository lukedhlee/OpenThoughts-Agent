"""make_ifv2_tasks.py — extract the 477 held-out if-v2 tasks from TaskTrove into a Harbor task tree.

    python make_ifv2_tasks.py --heldout <ifv2-heldout-tasks.jsonl> --out /e/data1/mmlaion/lee27/tasks/ifv2_heldout477

Source: open-thoughts/TaskTrove laion__nemotron-gym-instruction-following-v3/tasks.parquet (path, task_binary = gzip tar).
Each task dir keeps its exact tarball name (minus .tar.gz) so results join to the held-out file on `task`.
Streams the parquet row group by row group (325 MB, 1,450 groups); run on a login node with HF access.
"""
import argparse, io, json, tarfile
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--heldout", required=True); ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
want = {json.loads(l)["task"] for l in open(a.heldout)}
p = hf_hub_download("open-thoughts/TaskTrove", "laion__nemotron-gym-instruction-following-v3/tasks.parquet", repo_type="dataset")
pf = pq.ParquetFile(p); a.out.mkdir(parents=True, exist_ok=True); got = 0
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i)
    for path, blob in zip(t["path"].to_pylist(), t["task_binary"].to_pylist()):
        if path not in want: continue
        d = a.out / path.removesuffix(".tar.gz"); d.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            members = tf.getmembers()
            top = {m.name.split("/")[0] for m in members}
            strip = 1 if len(top) == 1 and not (a.out / path.removesuffix(".tar.gz") / "task.toml").exists() and all("/" in m.name or m.isdir() for m in members) else 0
            for m in members:
                parts = m.name.split("/")[strip:]
                if not parts or not parts[-1]: continue
                m.name = "/".join(parts)
                tf.extract(m, d, filter="data")
        got += 1
print(f"extracted {got} of {len(want)} -> {a.out}")
missing = want - {d.name + ".tar.gz" for d in a.out.iterdir()}
print("missing", len(missing), sorted(missing)[:5])
