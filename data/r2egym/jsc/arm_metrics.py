#!/usr/bin/env python3
"""arm_metrics.py <run> [min_step] — per-step table (reward, grad norm, entropy, truncated fraction, ctx deaths, dropped groups,
masked) across every job log of a run, via code/snowball/extract_metrics.py."""
import glob, re, subprocess, sys
run = sys.argv[1]; lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
E = "/e/fscratch/reformo/lee27/experiments"; C = "/e/project1/transfernetx/lee27/code/snowball"
want = {"reward/avg_raw_reward": "rew", "policy/policy_kl": "kl", "diag/declared_done_fraction": "f", "diag/reward_given_done": "q", "policy/grad_norm": "gnc", "policy/raw_grad_norm": "gn", "policy/policy_entropy": "ent", "diag/truncated_fraction": "trunc",
        "generate/errors/ContextLengthExceededError": "ctx", "async/rejected_count/below_minimum_group_size": "drop",
        "generate/num_masked_trajectories": "masked", "response_length/mean": "len"}
rows = {}
for log in sorted(glob.glob(f"{E}/{run}/logs/{run}_*.out"), key=lambda p: int(re.search(r"_(\d+)\.out$", p).group(1))):
    job = re.search(r"_(\d+)\.out$", log).group(1)
    out = subprocess.run(["python3", f"{C}/extract_metrics.py", log], capture_output=True, text=True).stdout
    cur = None
    for line in out.splitlines():
        m = re.match(r"=== kind=(\w+) step=(\d+)", line)
        if m: cur = (m.group(1), int(m.group(2))); continue
        if cur and cur[0] == "train":
            k, _, v = line.strip().partition(": ")
            if k in want: rows.setdefault(cur[1], {"job": job})[want[k]] = v
print("step   job     rew    gn    gnc    ent    kl    trunc   ctx drop msk   len     f     q")
for s in sorted(rows):
    if s < lo: continue
    r = rows[s]; f = lambda k: float(r.get(k, 0) or 0)
    print(f"{s:4d} {r['job']} {f('rew'):.3f} {f('gn'):.3f} {f('gnc'):.3f} {f('ent'):.3f} {f('kl'):.4f} {f('trunc'):.3f} {int(f('ctx')):5d} {int(f('drop')):4d} {int(f('masked')):3d} {f('len'):6.0f} {f('f'):.3f} {f('q'):.3f}")
