#!/usr/bin/env python3
"""Kernel sequence of one decoder layer inside one engine step, from a worker trace. Usage: prof_layer.py <trace.gz> [step_idx]"""
import collections, gzip, json, sys

t = json.load(gzip.open(sys.argv[1], "rt")); ev = t["traceEvents"]
si = int(sys.argv[2]) if len(sys.argv) > 2 else 20
kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e), key=lambda e: e["ts"])
marks = [e for e in kern if e["name"].startswith("_topk_topp_kernel")]
lo = marks[si]["ts"] + marks[si]["dur"]; hi = marks[si + 1]["ts"] + marks[si + 1]["dur"]
ks = [e for e in kern if lo < e["ts"] + e["dur"] <= hi]
print(f"step kernels={len(ks)} wall={(hi - lo) / 1e3:.1f}ms busy={sum(e['dur'] for e in ks) / 1e3:.1f}ms")
idx = [i for i, e in enumerate(ks) if e["name"] == "fused_moe_kernel"]
print("fused_moe calls in step:", len(idx))
a = idx[9] + 1; b = idx[11] + 1
layer = ks[a:b]
print(f"one layer: {len(layer)} kernels, {sum(e['dur'] for e in layer):.0f} us, span {(layer[-1]['ts'] + layer[-1]['dur'] - layer[0]['ts']):.0f} us")
for e in layer:
    print(f"  {e['dur']:7.1f} us  {e['name'][:100]}")
by = collections.Counter(); n = collections.Counter()
for e in ks:
    by[e["name"]] += e["dur"]; n[e["name"]] += 1
print("--- step totals by kernel (top 25)")
for k, v in by.most_common(25):
    print(f"  {v:8.0f} us  n={n[k]:4d}  {k[:90]}")
