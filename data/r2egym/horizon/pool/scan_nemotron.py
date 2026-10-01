#!/usr/bin/env python3
"""scan_nemotron.py: reduce a Nemotron SWE jsonl to one small row per trajectory (dataset tag, repo, first-user-message hash,
issue text, any 40-hex strings and docker-ish tags in the user message).

  nvidia/Nemotron-SWE-v1 data/r2e_gym.jsonl       -> nemo_v1.parquet
  nvidia/Nemotron-SFT-SWE-v2 data/agentless.jsonl -> nemo_agentless.parquet  (first user message kept whole: the agentless
                                                     prompt embeds the issue statement, matched exactly later)
  nvidia/Nemotron-SFT-SWE-v2 data/swe.jsonl       -> nemo_v2swe.parquet
"""
import argparse, hashlib, json, re, sys
import pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dest")
ap.add_argument("--keep-user", action="store_true", help="store the first user message (agentless needs it)")
a = ap.parse_args()
HEX = re.compile(r"\b[0-9a-f]{40}\b")
IMG = re.compile(r"(?:namanjain12|qingyangwu)/[\w.-]+:[0-9a-f]{7,40}")
cols = {k: [] for k in ["dataset", "repo", "user_sha", "user", "hexes", "imgs", "n_msgs"]}
n = 0
with open(a.src, "rb") as f:
    for line in f:
        d = json.loads(line)
        msgs = d.get("messages") or []
        if isinstance(msgs, str): msgs = json.loads(msgs)
        user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
        if isinstance(user, list): user = " ".join(x.get("text", "") for x in user if isinstance(x, dict))
        cols["dataset"].append(str(d.get("dataset", ""))); cols["repo"].append(str(d.get("repo", "")))
        cols["user_sha"].append(hashlib.sha1(user.encode()).hexdigest())
        cols["user"].append(user if a.keep_user else user[:6000])
        cols["hexes"].append(" ".join(sorted(set(HEX.findall(user)))))
        cols["imgs"].append(" ".join(sorted(set(IMG.findall(line.decode("utf-8", "replace"))))))
        cols["n_msgs"].append(len(msgs))
        n += 1
        if n % 20000 == 0: print(n, flush=True)
pq.write_table(pa.table(cols), a.dest)
print("rows", n, "->", a.dest)
