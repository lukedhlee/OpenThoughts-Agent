#!/usr/bin/env python3
"""s12_compare.py <session_distill> <session_control> [dev120.tsv]: paired per-task comparison of two bare-prompt dev120
probes (k attempts per task each; dirs <task>-pctl__id). Prints mean reward per arm, paired delta with a bootstrap 95 % CI,
wins/losses/ties, per-stratum deltas, context-death shares, and the never-solved tasks each arm unlocks against the other.
The stop rule (research/2026-09-13_p2o_distillation_test.md): PASS = delta >= +.05 with the CI excluding zero."""
import collections, csv, glob, json, os, random, statistics, sys
A, B = sys.argv[1], sys.argv[2]
DEV = sys.argv[3] if len(sys.argv) > 3 else "/e/fscratch/reformo/lee27/experiments/p2o/dev120.tsv"
dev = {r["task"]: r for r in csv.DictReader(open(DEV), delimiter="\t")}


def find_reward(o, depth=0):
    if depth > 6:
        return None
    if isinstance(o, dict):
        for k in ("reward", "rewards"):
            if k in o:
                v = o[k]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, (int, float)):
                            return float(vv)
        for k, v in o.items():
            if k in ("config", "trajectory", "messages", "steps"):
                continue
            r = find_reward(v, depth + 1)
            if r is not None:
                return r
    return None


def load(session):
    per = collections.defaultdict(list); ctx = 0; n = 0
    for f in glob.glob(session + "/*/result.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        name = d.get("task_name") or os.path.basename(os.path.dirname(f)).split("__")[0]
        task = name.rsplit("-p", 1)[0]
        e = d.get("exception_info") or {}
        et = (e.get("exception_type") if isinstance(e, dict) else "") or ""
        r = find_reward(d)
        if r is None:
            continue
        n += 1; ctx += "ContextLength" in et; per[task].append(r)
    return per, n, ctx


pa, na, ca = load(A); pb, nb, cb = load(B)
tasks = sorted(set(pa) & set(pb))
ds = [statistics.mean(pa[t]) - statistics.mean(pb[t]) for t in tasks]
rng = random.Random(0); n = len(ds); m = statistics.mean(ds)
bs = sorted(sum(rng.choice(ds) for _ in range(n)) / n for _ in range(4000)); lo, hi = bs[100], bs[3899]
w = sum(d > 0 for d in ds); l = sum(d < 0 for d in ds)
ma = statistics.mean(x for t in tasks for x in pa[t]); mb = statistics.mean(x for t in tasks for x in pb[t])
print("A=%s\nB=%s" % (A, B))
print("pairs=%d  mean reward A=%.3f B=%.3f  paired delta A-B=%+.3f 95%%CI=[%+.3f,%+.3f]  wins=%d losses=%d ties=%d" % (n, ma, mb, m, lo, hi, w, l, n - w - l))
print("trials A=%d B=%d | ctx-death share A=%.2f B=%.2f" % (na, nb, ca / max(1, na), cb / max(1, nb)))
for s in ("zero", "hard", "medium", "success"):
    ts = [t for t in tasks if dev.get(t, {}).get("stratum") == s]
    if not ts:
        continue
    dd = [statistics.mean(pa[t]) - statistics.mean(pb[t]) for t in ts]
    print("  %-8s pairs=%3d A=%.3f B=%.3f delta=%+.3f wins=%d losses=%d" % (
        s, len(ts), statistics.mean(statistics.mean(pa[t]) for t in ts), statistics.mean(statistics.mean(pb[t]) for t in ts),
        statistics.mean(dd), sum(d > 0 for d in dd), sum(d < 0 for d in dd)))
ua = [t for t in tasks if dev.get(t, {}).get("stratum") == "zero" and max(pa[t]) > 0 and max(pb[t]) == 0]
ub = [t for t in tasks if dev.get(t, {}).get("stratum") == "zero" and max(pb[t]) > 0 and max(pa[t]) == 0]
print("never-solved unlocked by A only: %d %s | by B only: %d %s" % (len(ua), ua[:8], len(ub), ub[:8]))
verdict = "PASS" if (m >= 0.05 and lo > 0) else ("FAIL" if m <= 0.02 else "INCONCLUSIVE")
print("stop rule (dev120 delta >= +.05, CI excluding 0):", verdict)
