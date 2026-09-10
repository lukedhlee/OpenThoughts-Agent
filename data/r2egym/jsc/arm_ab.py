#!/usr/bin/env python3
"""arm_ab.py <runA> <runB> [min_step] [--restarts] — throughput A/B of two RL arms from their job logs.

Per-step table for each arm (step wall, rollout wait, policy train, weight sync, staleness mean, stale/total rejections,
TIS log-ratio / capped fraction, grad norm, entropy, KL, reward, pass@8), then a summary over steps >= min_step (default 2:
the first step is startup): mean step, mean wait, wait share, timed updates/h, TIS tripwires. --restarts counts restarted
attempts (attempts/001+ dirs) in each arm's trace store on its batch host via `srun --overlap`, the restart-rate metric the
WANDB records do not carry (a restarted attempt is transparent to SkyRL). Reuses code/snowball/extract_metrics.py like
arm_metrics.py. Written for the staleness-2 A/B (rl_throughput_and_acceleration.md), 2026-09-05."""
import glob, re, subprocess, sys

E = "/e/fscratch/reformo/lee27/experiments"; C = "/e/project1/transfernetx/lee27/code/snowball"
WANT = {"timing/step": "step_s", "timing/wait_for_generation_buffer": "wait_s", "timing/policy_train": "train_s",
        "timing/sync_weights": "sync_s", "timing/convert_to_training_input": "conv_s", "async/staleness_mean": "stale",
        "async/rejected_count/stale": "rej_stale", "async/rejected_count": "rej", "policy/tis/log_ratio_abs_mean": "tis_lr",
        "policy/tis/imp_ratio_capped_fraction": "tis_cap", "policy/raw_grad_norm": "gn", "policy/policy_entropy": "ent",
        "policy/policy_kl": "kl", "reward/avg_raw_reward": "rew", "reward/avg_pass_at_8": "p8",
        "generate/num_masked_trajectories": "masked", "generate/errors/ContextLengthExceededError": "ctx"}
COLS = ["step_s", "wait_s", "train_s", "sync_s", "stale", "rej_stale", "rej", "tis_lr", "tis_cap", "gn", "ent", "kl", "rew", "p8", "masked"]


def rows_of(run):
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
                if k in WANT: rows.setdefault(cur[1], {"job": job})[WANT[k]] = v
    return rows


def f(r, k):
    try: return float(r.get(k, 0) or 0)
    except ValueError: return 0.0


def table(run, rows, lo):
    print(f"\n=== {run} ===")
    print("step   job     " + " ".join(f"{c:>8s}" for c in COLS))
    for s in sorted(rows):
        if s < lo: continue
        r = rows[s]
        print(f"{s:4d} {r['job']} " + " ".join(f"{f(r, c):8.3f}" if c in ("stale", "tis_lr", "tis_cap", "gn", "ent", "kl", "rew", "p8") else f"{f(r, c):8.0f}" for c in COLS))


def summary(run, rows, lo):
    sel = [rows[s] for s in sorted(rows) if s >= lo]
    if not sel: return {"run": run, "n": 0}
    n = len(sel); mean = lambda k: sum(f(r, k) for r in sel) / n
    step, wait = mean("step_s"), mean("wait_s")
    return {"run": run, "n": n, "first": min(s for s in rows if s >= lo), "last": max(rows), "step_s": step, "wait_s": wait,
            "wait_share": wait / step if step else 0, "train_s": mean("train_s"), "sync_s": mean("sync_s"),
            "upd_per_h": 3600 / step if step else 0, "stale": mean("stale"), "rej_stale": sum(f(r, "rej_stale") for r in sel),
            "tis_lr": mean("tis_lr"), "tis_cap_max": max(f(r, "tis_cap") for r in sel), "gn_max": max(f(r, "gn") for r in sel),
            "rew": mean("rew"), "ent_last": f(sel[-1], "ent")}


def restarts(run):
    """(retried trials, total trials) from the arm's trace store on its batch host: attempts/001+ = a restart."""
    job = subprocess.run(["squeue", "-h", "-u", "lee27", "-n", run, "-o", "%i %N"], capture_output=True, text=True).stdout.split()
    if not job: return None
    jid, nodes = job[0], job[1]; host = subprocess.run(["scontrol", "show", "hostnames", nodes], capture_output=True, text=True).stdout.split()[0]
    cmd = ("D=$(ls -d /tmp/otagent-artifact-stores/%s-*/trace_jobs 2>/dev/null | head -1); [ -n \"$D\" ] || exit 3; "
           "echo $(ls -d $D/*/attempts/00[1-9] 2>/dev/null | wc -l) $(ls -d $D/*/attempts/000 2>/dev/null | wc -l)" % run)
    out = subprocess.run(["srun", f"--jobid={jid}", "--overlap", "-N1", "-n1", "-w", host, "--time=3", "--quiet", "bash", "-c", cmd],
                         capture_output=True, text=True, timeout=240).stdout.split()
    return (int(out[0]), int(out[1]), host) if len(out) == 2 else None


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; a, b = args[0], args[1]; lo = int(args[2]) if len(args) > 2 else 2
    ra, rb = rows_of(a), rows_of(b)
    table(a, ra, lo); table(b, rb, lo)
    sa, sb = summary(a, ra, lo), summary(b, rb, lo)
    print(f"\n=== summary (steps >= {lo}; timed steps only, startup and checkpointing excluded) ===")
    keys = ["n", "first", "last", "step_s", "wait_s", "wait_share", "train_s", "sync_s", "upd_per_h", "stale", "rej_stale", "tis_lr", "tis_cap_max", "gn_max", "rew", "ent_last"]
    print(f"{'metric':12s} {a[:28]:>28s} {b[:28]:>28s}")
    for k in keys:
        va, vb = sa.get(k, 0), sb.get(k, 0)
        fmt = (lambda v: f"{v:28.4f}") if isinstance(va, float) else (lambda v: f"{v:28}")
        print(f"{k:12s} {fmt(va)} {fmt(vb)}")
    if "--restarts" in sys.argv:
        for run in (a, b):
            r = restarts(run)
            print(f"restarts {run}: " + (f"{r[0]} retried attempts / {r[1]} trials = {1000 * r[0] / max(r[1], 1):.1f} per 1k (host {r[2]})" if r else "n/a (job not running or store missing)"))
