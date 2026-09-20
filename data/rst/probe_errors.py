#!/usr/bin/env python3
"""probe_errors.py <run dir> <task tree> — what the errored trials of an RST held-out run have in common.

Maps every exception to the task's base image and sandbox size (task.toml), lists the setup-hook failures and the
setup durations, so a rerun can fix the tree (resources, timeouts, a base whose snapshot failed) instead of dropping
tasks blindly.
"""
import collections, glob, json, os, re, sys

run, tree = sys.argv[1:3]


def base_of(task):
    return open(f"{tree}/{task}/environment/Dockerfile").readline().strip()


def toml(task):
    s = open(f"{tree}/{task}/task.toml").read()
    mem = re.search(r'memory = "([^"]+)"', s); cpu = re.search(r"cpus = (\d+)", s)
    return (mem.group(1) if mem else "?", cpu.group(1) if cpu else "?")


by = collections.defaultdict(collections.Counter)
setup_secs = []
for p in glob.glob(f"{run}/*/attempts/*/result.json"):
    r = json.load(open(p)); task = r["task_name"]
    et = (r.get("exception_info") or {}).get("exception_type")
    if et:
        by["exception"][et] += 1
        by[f"{et}:base"][base_of(task)] += 1
        by[f"{et}:mem_cpu"][toml(task)] += 1
        if et == "SetupScriptError":
            by["setup_error_tasks"][task] += 1
    log = os.path.join(os.path.dirname(p), "trial.log")
    if os.path.exists(log):
        m = re.search(r"setup_files/setup.sh exited (\d+) after ([\d.]+)s", open(log, errors="replace").read())
        if m:
            setup_secs.append((float(m.group(2)), int(m.group(1)), task))
for k, v in sorted(by.items()):
    print(k, dict(v.most_common(8)))
if setup_secs:
    s = sorted(x[0] for x in setup_secs)
    print(f"setup secs: n {len(s)} median {s[len(s)//2]:.0f} p90 {s[int(len(s)*.9)]:.0f} max {s[-1]:.0f}")
    print("slowest setups:", [(f"{a:.0f}s", rc, t) for a, rc, t in sorted(setup_secs, reverse=True)[:6]])
res = collections.Counter(toml(d) for d in os.listdir(tree) if os.path.isdir(f"{tree}/{d}"))
print("tree resources (mem, cpus):", dict(res))
