#!/usr/bin/env python3
"""mem_peak.py <memory_snapshots dir or pickle...>  — peak allocated CUDA memory per rank from torch _dump_snapshot pickles.
Replays device_traces (alloc/free_completed) for the running allocated total; also reports the segments' reserved
total at dump time. Prints GiB per file and the max over ranks."""
import pickle, sys, glob, os
def peak(path):
    with open(path, "rb") as f: s = pickle.load(f)
    best = 0; cur = 0; n = 0
    for tr in s.get("device_traces", []):
        cur = 0
        for e in tr:
            a = e.get("action"); sz = e.get("size", 0)
            if a == "alloc": cur += sz; best = max(best, cur); n += 1
            elif a in ("free_completed", "free_requested_and_completed"): cur -= sz
    reserved = sum(seg.get("total_size", 0) for seg in s.get("segments", []))
    allocated_now = sum(b.get("size", 0) for seg in s.get("segments", []) for b in seg.get("blocks", []) if b.get("state") == "active_allocated")
    return best / 2**30, reserved / 2**30, allocated_now / 2**30, n
files = []
for a in sys.argv[1:]: files += sorted(glob.glob(f"{a}/*.pickle")) if os.path.isdir(a) else [a]
mx = 0
for p in files:
    try: b, r, an, n = peak(p)
    except Exception as ex: print(os.path.basename(p), "ERR", ex); continue
    mx = max(mx, b); print(f"{os.path.basename(p):48s} peak_alloc={b:6.1f} GiB  reserved_at_dump={r:6.1f} GiB  alloc_at_dump={an:6.1f} GiB  (alloc events {n})")
print(f"MAX peak_alloc over {len(files)} files: {mx:.1f} GiB (GH200 usable ≈ 95.6 GiB)")
