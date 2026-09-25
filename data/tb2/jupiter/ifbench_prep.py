"""ifbench_prep.py — build the IFBench prompt file for ifeval_gen.py (runs on the Mac, then scp to $IFE).

    python ifbench_prep.py <IFBench clone> ifbench_input_data.jsonl

Prompts: HF allenai/IFBench_test at a pinned revision (300 rows, the paper's single-turn OOD test set).
The parquet stores integer kwargs as floats (m: 33.0); they are cast back to int so the rows equal the scorer repo's
own data/IFBench_test.jsonl apart from trailing whitespace on two prompts (keys 13, 19). The check against that file
runs here and fails loudly on any other difference. The same rows are the scorer's --input_data (score_ifbench.py).
"""
import io, json, sys, urllib.request
import pandas as pd

REV = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"   # allenai/IFBench_test, lastModified 2025-10-17
URL = f"https://huggingface.co/datasets/allenai/IFBench_test/resolve/{REV}/data/train-00000-of-00001.parquet"

def clean(v):
    if hasattr(v, "tolist"): v = v.tolist()
    if isinstance(v, float) and v.is_integer(): return int(v)
    return v

repo, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL).read()))
rows = [{"key": str(r["key"]), "prompt": r["prompt"], "instruction_id_list": list(r["instruction_id_list"]),
         "kwargs": [{k: clean(v) for k, v in d.items()} for d in r["kwargs"]]} for _, r in df.iterrows()]
assert len(rows) == 300, len(rows)
ref = {r["key"]: r for r in map(json.loads, open(f"{repo}/data/IFBench_test.jsonl"))}
for r in rows:
    g = ref[r["key"]]
    assert r["prompt"].rstrip() == g["prompt"].rstrip(), r["key"]
    assert r["instruction_id_list"] == g["instruction_id_list"], r["key"]
    nn = lambda ks: [{k: v for k, v in d.items() if v is not None} for d in ks]
    assert nn(r["kwargs"]) == nn(g["kwargs"]), (r["key"], r["kwargs"], g["kwargs"])
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"{len(rows)} prompts, rev {REV[:8]}, match the scorer repo's copy -> {out}")
