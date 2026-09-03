#!/usr/bin/env python3
"""Overlap of our raw R2E-Gym set vs TaskTrove r2egym-patched-full-oracle-v3, both mapped onto upstream R2E-Gym-V1.

Join key: k_eo (normalized expected_output_json), falling back to k_ps (normalized problem statement) when k_eo
is missing or ambiguous. Reports per-set counts, overlaps, and per-repo breakdown.
"""
import collections, json, os

SP = os.path.dirname(os.path.abspath(__file__))
def load(name):
    p = os.path.join(SP, name)
    return [json.loads(l) for l in open(p) if l.strip()]

up = load("r2e_upstream_keys.jsonl")
tt = load("r2e_tasktrove_keys.jsonl")
raw = load("r2e_raw_keys.jsonl")

def index(rows, key):
    d = collections.defaultdict(list)
    for r in rows:
        if r.get(key):
            d[r[key]].append(r)
    return d

up_eo, up_ps, up_img = index(up, "k_eo"), index(up, "k_ps"), index(up, "docker_image")
up_pair = collections.defaultdict(list)
for r in up:
    up_pair[(r.get("k_eo"), r.get("k_ps"))].append(r)

def to_upstream(rows, label):
    """Map each row to a unique upstream idx: docker_image (unique upstream) → (k_eo,k_ps) pair → k_eo → k_ps."""
    out, stats = {}, collections.Counter()
    for i, r in enumerate(rows):
        hit = None
        tries = [("docker_image", up_img.get(r.get("docker_image"), [])),
                 ("pair", up_pair.get((r.get("k_eo"), r.get("k_ps")), [])),
                 ("k_eo", up_eo.get(r.get("k_eo"), [])),
                 ("k_ps", up_ps.get(r.get("k_ps"), []))]
        for key, cands in tries:
            uniq = {c["idx"] for c in cands}
            if len(uniq) == 1:
                hit = next(iter(uniq)); stats[f"via_{key}"] += 1; break
            if len(uniq) > 1:
                stats[f"ambiguous_{key}"] += 1
        if hit is None:
            stats["unmatched"] += 1
        out[i] = hit
    print(f"{label}: rows={len(rows)} " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return out

tt_map = to_upstream(tt, "tasktrove")
raw_map = to_upstream(raw, "raw")

tt_idx = {v for v in tt_map.values() if v is not None}
raw_idx = {v for v in raw_map.values() if v is not None}
gate_idx = {raw_map[i] for i, r in enumerate(raw) if r.get("gate_v3") and raw_map[i] is not None}
all_idx = {r["idx"] for r in up}

print()
print(f"upstream rows            : {len(all_idx)}")
print(f"raw (4,578) mapped       : {len(raw_idx)}   gate-passed mapped: {len(gate_idx)}")
print(f"tasktrove (3,328) mapped : {len(tt_idx)}")
print(f"raw ∩ tasktrove          : {len(raw_idx & tt_idx)}")
print(f"gate ∩ tasktrove         : {len(gate_idx & tt_idx)}")
print(f"raw only                 : {len(raw_idx - tt_idx)}")
print(f"tasktrove only           : {len(tt_idx - raw_idx)}")
print(f"in neither               : {len(all_idx - raw_idx - tt_idx)}")

# per-repo breakdown
repo_of = {r["idx"]: r["repo"] for r in up}
rows = collections.defaultdict(lambda: [0, 0, 0, 0])
for i in all_idx:
    rr = rows[repo_of[i]]
    rr[0] += 1
    if i in raw_idx: rr[1] += 1
    if i in tt_idx: rr[2] += 1
    if i in raw_idx and i in tt_idx: rr[3] += 1
print("\nrepo                 upstream   raw   tasktrove   both")
for repo, (u, a, b, c) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"{repo:20s} {u:8d} {a:5d} {b:11d} {c:6d}")

# duplicates inside each set (same upstream row mapped twice)
for label, m in (("raw", raw_map), ("tasktrove", tt_map)):
    c = collections.Counter(v for v in m.values() if v is not None)
    d = sum(1 for v in c.values() if v > 1)
    if d:
        print(f"\n{label}: {d} upstream rows mapped by >1 task (duplicates inside the set)")
