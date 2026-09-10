#!/usr/bin/env python3
"""traj_rate.py <log> [bucket_min] — completed trajectories per bucket from the RolloutCoordinator's `_process_trial_result`
lines (timestamps inside the Ray-prefixed line), plus the overall rate since the first completion and over the last hour.
The generator's true output rate, independent of the trainer's step cadence and the wave pattern."""
import re, sys, collections
from datetime import datetime
log = sys.argv[1]; bucket = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ts = []
pat = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| INFO .*_process_trial_result")
for line in open(log, errors="replace"):
    m = pat.search(line)
    if m: ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
if not ts: print("no completions"); sys.exit()
buckets = collections.Counter((t.replace(second=0, microsecond=0, minute=(t.minute // bucket) * bucket)) for t in ts)
for b, n in sorted(buckets.items()): print(f"{b:%H:%M}  {n:5d}  {n/(bucket*60):.2f} traj/s  {n*3600/(bucket*60):.0f}/h")
span = (ts[-1] - ts[0]).total_seconds(); last = [t for t in ts if (ts[-1] - t).total_seconds() <= 3600]
print(f"total {len(ts)} over {span/60:.0f} min = {len(ts)/max(span,1):.2f} traj/s ({len(ts)*3600/max(span,1):.0f}/h); last hour {len(last)} = {len(last)/3600:.2f} traj/s")
