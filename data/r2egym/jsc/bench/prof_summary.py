#!/usr/bin/env python3
"""Summarize torch-profiler traces written by vLLM workers: CUDA kernel time by category and by kernel name.
Usage: prof_summary.py <dir with *.pt.trace.json[.gz]> [--top 40]
"""
import collections, glob, gzip, json, os, re, sys

d = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
files = sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True))
files = [f for f in files if "trace" in os.path.basename(f) or f.endswith(".json") or f.endswith(".json.gz")]
if not files:
    print("no trace files under", d); sys.exit(1)

CATS = [
    ("comm", re.compile(r"nccl|allgather|all_gather|reducescatter|reduce_scatter|allreduce|all_reduce|cross_device|custom_all_reduce|sendrecv|ncclDevKernel", re.I)),
    ("moe", re.compile(r"fused_moe|moe_align|moe_sum|silu_and_mul|grouped_topk|topk_softmax|moe_wna16|fused_experts|moe_|expert", re.I)),
    ("attention", re.compile(r"flash|fwd_kernel|attn|attention|paged|reshape_and_cache|rotary|rope|cutlass::.*fmha|flashinfer.*(decode|prefill)", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|nvjet|Cijk|cublas|sm90_|sm80_|ampere_|matmul|mm_kernel", re.I)),
    ("sampler", re.compile(r"sampl|top_k|topk|top_p|sort|radix|softmax|logprob|gumbel|multinomial|cub::", re.I)),
    ("norm/elementwise", re.compile(r"rms|norm|elementwise|vectorized|triton_(poi|red|per)|fused_add|activation|sigmoid|copy_|fill|cat|index|gather|scatter|embed|arange|where|mul|add|sub|div|cast|convert", re.I)),
]

def cat_of(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"

grand = collections.Counter(); grand_n = collections.Counter(); per_rank = {}
for f in files:
    try:
        raw = gzip.open(f, "rt") if f.endswith(".gz") else open(f)
        t = json.load(raw)
    except Exception as e:
        print("skip", f, e); continue
    ev = t.get("traceEvents", t if isinstance(t, list) else [])
    kern = [e for e in ev if e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not kern:
        continue
    by = collections.Counter(); n = collections.Counter()
    for e in kern:
        by[e["name"]] += e["dur"]; n[e["name"]] += 1
    t0 = min(e["ts"] for e in kern); t1 = max(e["ts"] + e["dur"] for e in kern)
    steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep")]
    per_rank[f] = {"by": by, "n": n, "wall_us": t1 - t0, "steps": len(steps)}
    for k, v in by.items():
        grand[k] += v; grand_n[k] += n[k]

for f, r in per_rank.items():
    tot = sum(r["by"].values())
    cats = collections.Counter()
    for k, v in r["by"].items():
        cats[cat_of(k)] += v
    print(f"\n### {os.path.basename(f)}: kernel time {tot/1e3:.1f} ms over window {r['wall_us']/1e3:.1f} ms "
          f"(GPU busy {100*tot/max(1,r['wall_us']):.0f}%), ProfilerStep events {r['steps']}")
    for c, v in cats.most_common():
        print(f"  {c:18s} {v/1e3:9.1f} ms  {100*v/tot:5.1f}%")

tot = sum(grand.values())
print(f"\n### ALL RANKS: top {top} kernels by CUDA time (total {tot/1e3:.1f} ms)")
print(f"{'cat':18s} {'ms':>9s} {'%':>6s} {'count':>7s} {'us/call':>8s}  name")
for k, v in grand.most_common(top):
    print(f"{cat_of(k):18s} {v/1e3:9.1f} {100*v/tot:6.1f} {grand_n[k]:7d} {v/grand_n[k]:8.1f}  {k[:110]}")
