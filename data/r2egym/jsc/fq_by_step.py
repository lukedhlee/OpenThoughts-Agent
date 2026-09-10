#!/usr/bin/env python3
"""fq_by_step.py <run> [min_step] — per training step from the schema_v3 dumps: n records, f = share with stop_reason task_complete,
q = mean reward outcome among those, mean reward, length-stop share. Reads only the small fields (streams gz json)."""
import glob, gzip, json, sys, zipfile, re, collections
run = sys.argv[1]; lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
E = "/e/fscratch/reformo/lee27/experiments"
steps = sorted(glob.glob(f"{E}/{run}/{run}/exports/training_trajectories/schema_v3/archives/phase=train/step=*"))
print("step    n     f      q    reward  len_stop")
for d in steps:
    s = int(re.search(r"step=(\d+)", d).group(1))
    if s < lo: continue
    n = done = donewin = wins = ls = 0
    for zp in glob.glob(d + "/*.zip"):
        z = zipfile.ZipFile(zp)
        for name in z.namelist():
            if not name.endswith(".json.gz"): continue
            r = json.loads(gzip.decompress(z.read(name)))
            sr = r["response"]["stop_reason"]; w = float(r["reward"]["outcome"] or 0.0)
            n += 1; wins += w; ls += int(sr == "length")
            if sr == "task_complete": done += 1; donewin += w
    if n:
        print(f"{s:4d} {n:5d}  {done/n:.3f}  {donewin/done if done else 0:.3f}  {wins/n:.3f}  {ls/n:.3f}", flush=True)
