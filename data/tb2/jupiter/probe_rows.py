#!/usr/bin/env python3
"""probe_rows.py <run dir> [<run dir> ...] — write <run>/ifv2_rows.json for a harness probe run so ifv2_pair.py can pair it.

One row per trial: [task_name, exception_type or null, reward]. Reads <trial>/result.json (harbor v0.1 layout, the
IF-477 runs) or <trial>/attempts/*/result.json (the setup-hook lineage, the RST held-out runs); an agent timeout with a
verifier reward keeps the reward (harbor counts it valid), an infrastructure exception without one gets reward null.
Prints the run's pass@1 over rows with a reward.
"""
import glob, json, os, sys

for run in sys.argv[1:]:
    rows = []
    paths = sorted(glob.glob(f"{run}/*/result.json")) + sorted(glob.glob(f"{run}/*/attempts/*/result.json"))
    for p in paths:
        if os.path.dirname(p) == run.rstrip("/"):
            continue
        r = json.load(open(p))
        if "task_name" not in r:
            continue
        exc = (r.get("exception_info") or {}).get("exception_type")
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rows.append([r["task_name"], exc, rew])
    seen = {}
    for name, exc, rew in rows:  # one row per task; a scored attempt wins over an errored one
        if name not in seen or (seen[name][2] is None and rew is not None):
            seen[name] = [name, exc, rew]
    rows = list(seen.values())
    json.dump(rows, open(f"{run}/ifv2_rows.json", "w"))
    scored = [r for r in rows if r[2] is not None]
    print(f"{os.path.basename(run.rstrip('/'))}: {len(rows)} tasks, {len(scored)} scored, pass@1 {sum(r[2] for r in scored) / max(1, len(scored)):.3f}, "
          f"errors {sum(1 for r in rows if r[2] is None)}")
