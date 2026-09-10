#!/usr/bin/env python3
"""Per-step timing from a vLLM worker torch-profiler trace: step wall vs GPU-busy, and where the idle gaps are.
Steps are delimited by the sampler kernel (one per engine step). Usage: prof_steps.py <trace.json.gz> [--gaps 12]
"""
import collections, gzip, json, sys

f = sys.argv[1]
ngaps = int(sys.argv[sys.argv.index("--gaps") + 1]) if "--gaps" in sys.argv else 12
t = json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f))
ev = t.get("traceEvents", [])
kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e), key=lambda e: e["ts"])
STEP_MARK = ("_topk_topp_kernel", "gatherTopK", "sampl")
marks = [e for e in kern if e["name"].startswith("_topk_topp_kernel")]
if len(marks) < 3:
    marks = [e for e in kern if "gatherTopK" in e["name"]]
print(f"{f.split('/')[-1][:40]}: {len(kern)} kernels, {len(marks)} step markers")
if len(marks) < 3:
    sys.exit(0)
# step i = (marks[i].end, marks[i+1].end]
bounds = [m["ts"] + m["dur"] for m in marks]
steps = []
gaps_all = []
ki = 0
for i in range(len(bounds) - 1):
    lo, hi = bounds[i], bounds[i + 1]
    ks = [e for e in kern if lo < e["ts"] + e["dur"] <= hi]
    busy = sum(e["dur"] for e in ks)
    # gaps between consecutive kernels inside the step (GPU idle)
    prev_end = lo; prev_name = "STEP_START(after sampler)"
    gaps = []
    for e in sorted(ks, key=lambda e: e["ts"]):
        g = e["ts"] - prev_end
        if g > 50:  # us
            gaps.append((g, prev_name, e["name"]))
        prev_end = max(prev_end, e["ts"] + e["dur"]); prev_name = e["name"]
    steps.append({"wall": hi - lo, "busy": busy, "n": len(ks), "gaps": gaps})
    gaps_all += gaps
walls = sorted(s["wall"] for s in steps); busys = sorted(s["busy"] for s in steps)
q = lambda x, p: x[int(p * (len(x) - 1))]
print(f"steps={len(steps)} wall/step p50={q(walls,.5)/1e3:.1f} ms p90={q(walls,.9)/1e3:.1f} ms  busy/step p50={q(busys,.5)/1e3:.1f} ms  "
      f"busy fraction (sum)={100*sum(busys)/max(1,sum(walls)):.0f}%  kernels/step p50={q(sorted(s['n'] for s in steps),.5)}")
# idle attribution: group gaps by (before-kernel category)
def cat(n):
    n = n.lower()
    if "allgather" in n or "all_gather" in n: return "before AllGather"
    if "reducescatter" in n or "reduce_scatter" in n: return "before ReduceScatter"
    if "step_start" in n: return "step start (after sampler -> first kernel)"
    if "broadcast" in n or "ncclDevKernel_Reduce" in n: return "before nccl Broadcast/Reduce"
    if "fused_moe" in n: return "before fused_moe"
    if "nvjet" in n or "gemm" in n: return "before gemm"
    return "before other"
by = collections.Counter(); cnt = collections.Counter()
for g, a, b in gaps_all:
    by[cat(b)] += g; cnt[cat(b)] += 1
tot_idle = sum(s["wall"] - s["busy"] for s in steps)
print(f"total idle in steps = {tot_idle/1e3:.1f} ms of {sum(walls)/1e3:.1f} ms; gaps>50us explain {100*sum(by.values())/max(1,tot_idle):.0f}% of it")
for k, v in by.most_common():
    print(f"  {k:48s} {v/1e3:8.1f} ms  n={cnt[k]:5d}  avg={v/cnt[k]:.0f} us  ({v/max(1,len(steps))/1e3:.2f} ms/step)")
print("largest single gaps:")
for g, a, b in sorted(gaps_all, reverse=True)[:ngaps]:
    print(f"  {g/1e3:7.2f} ms  after {a[:60]!s:60s} -> {b[:60]}")
print("longest steps (wall ms | busy ms | kernels | longest kernel):")
for s in sorted(steps, key=lambda s: -s["wall"])[:6]:
    print(f"  {s['wall']/1e3:8.1f} | {s['busy']/1e3:7.1f} | {s['n']:5d}")
lk = sorted(kern, key=lambda e: -e["dur"])[:8]
print("longest kernels in trace:")
for e in lk:
    print(f"  {e['dur']/1e3:8.1f} ms  {e['name'][:90]}")
big = [e for e in kern if e["dur"] > 5000]
comm_big = [e for e in big if "nccl" in e["name"]]
print(f"kernels >5 ms: {len(big)}, of which nccl (waiting on other ranks): {len(comm_big)}, total nccl>5ms time {sum(e['dur'] for e in comm_big)/1e3:.0f} ms of trace span {(kern[-1]['ts']+kern[-1]['dur']-kern[0]['ts'])/1e3:.0f} ms")
