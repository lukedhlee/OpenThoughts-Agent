#!/usr/bin/env python3
"""store_compact.py <trace_jobs dir> [--min-age-min 20] [--workers 16] [--limit N] [--dry-run]
Compact finished trial dirs inside an artifact store (run on the BatchHost via `srun --overlap`). Per trial dir older than
--min-age-min (mtime of the dir), once:
  * attempts/000/result.json  -> rewritten compact (no indent) with agent_result.rollout_details removed (token ids + logprobs,
    5 MB compact / 24 MB pretty-printed; the trainer consumed them in memory, the retention channel copied what it keeps)
  * result.json and attempts/000/lifecycle-result.json (byte-identical duplicates of the attempt result) -> deleted
  * everything else (agent/trajectory.json = the readable trace, agent + verifier logs, exception files) -> untouched
  * a `.compacted` marker is written so the dir is skipped next time.
Typical effect: 78 MB -> ~5 MB per trial. Session 20ae8169, 2026-09-05 10:05 PT."""
import argparse, json, os, sys, time
from multiprocessing import Pool

def compact(d):
    try:
        mark = os.path.join(d, ".compacted")
        if os.path.exists(mark): return ("skip", 0)
        att = os.path.join(d, "attempts", "000", "result.json")
        if not os.path.exists(att): return ("noattempt", 0)
        before = 0
        for r, _, fs in os.walk(d):
            for f in fs: before += os.path.getsize(os.path.join(r, f))
        with open(att) as fh: j = json.load(fh)
        ar = j.get("agent_result")
        if isinstance(ar, dict) and "rollout_details" in ar: ar["rollout_details"] = None
        tmp = att + ".tmp"
        with open(tmp, "w") as fh: json.dump(j, fh, separators=(",", ":"))
        os.replace(tmp, att)
        for dup in (os.path.join(d, "result.json"), os.path.join(d, "attempts", "000", "lifecycle-result.json")):
            if os.path.exists(dup): os.remove(dup)
        after = 0
        for r, _, fs in os.walk(d):
            for f in fs: after += os.path.getsize(os.path.join(r, f))
        with open(mark, "w") as fh: fh.write("%d %d\n" % (before, after))
        return ("ok", before - after)
    except Exception as e:
        return ("err:" + repr(e)[:80], 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--min-age-min", type=float, default=20)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); now = time.time(); cands = []
    for name in os.listdir(a.root):
        d = os.path.join(a.root, name)
        try:
            if not os.path.isdir(d) or os.path.exists(os.path.join(d, ".compacted")): continue
            if now - os.path.getmtime(d) < a.min_age_min * 60: continue
        except OSError:
            continue  # trial dir changed/vanished under us (concurrent writer or compactor)
        cands.append(d)
        if a.limit and len(cands) >= a.limit: break
    print("candidates", len(cands), flush=True)
    if a.dry_run: sys.exit(0)
    stats = {}; freed = 0; t0 = time.time()
    with Pool(a.workers) as p:
        for i, (st, fr) in enumerate(p.imap_unordered(compact, cands, chunksize=4), 1):
            stats[st.split(":")[0]] = stats.get(st.split(":")[0], 0) + 1; freed += fr
            if st.startswith("err") and stats["err"] <= 3: print("  ", st, flush=True)
            if i % 500 == 0: print("  %d/%d freed %.1f GB %.0fs" % (i, len(cands), freed / 1e9, time.time() - t0), flush=True)
    print("done", stats, "freed_GB", round(freed / 1e9, 1), "%.0fs" % (time.time() - t0), flush=True)
