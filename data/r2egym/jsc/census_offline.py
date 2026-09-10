#!/usr/bin/env python3
"""census_offline.py <artifact_store.img> <out.jsonl> [budget_s]
Mount an INACTIVE run's artifact-store image read-only with hpc.artifact_store.mounted (fuse2fs, shared lock; refuses
while a batch job holds the writer lock) and run tmux_census.py over its trace_jobs. Run on a compute node via
`srun --overlap` (fuse2fs is on the nodes); no compactor contends for the mount, unlike a live store. Session 5b45b53d, 2026-09-06."""
import pathlib, subprocess, sys, time
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402

image, out = pathlib.Path(sys.argv[1]), sys.argv[2]; budget = sys.argv[3] if len(sys.argv) > 3 else "2400"
t0 = time.monotonic()
with mounted(image, "ro") as root:
    tj = root / "trace_jobs"
    print(f"mounted {image} at {root}; trace_jobs exists={tj.is_dir()} ({time.monotonic()-t0:.0f}s)", file=sys.stderr, flush=True)
    subprocess.run([sys.executable, "/e/project1/transfernetx/lee27/code/snowball/tmux_census.py", str(tj), out, budget], check=False)
print(f"unmounted after {time.monotonic()-t0:.0f}s", file=sys.stderr, flush=True)
