#!/usr/bin/env python3
"""Per-token rollout-vs-trainer log-ratio distribution from SkyRL dumped_data jsonl (stdlib only)."""
import json, sys, math, bisect
files = sys.argv[1:]
first = json.loads(open(files[0]).readline())
print("record keys:", sorted(first.keys())[:40])
rk = [k for k in first if "rollout" in k.lower() and "logprob" in k.lower()]
tk = [k for k in first if k.lower() in ("old_log_probs", "action_log_probs", "log_probs", "old_logprobs", "action_logprobs", "logprobs", "trainer_logprobs")]
print("rollout key:", rk, "trainer key:", tk)
if not rk or not tk: sys.exit(0)
rk, tk = rk[0], tk[0]
xs = []; adv_pos = []; adv_neg = []; n_rec = 0
advk = "advantages" if "advantages" in first else None
maskk = "loss_mask" if "loss_mask" in first else ("response_mask" if "response_mask" in first else None)
for f in files:
    for line in open(f):
        r = json.loads(line); n_rec += 1
        a = r[tk]; b = r[rk]
        if not isinstance(a, list) or not isinstance(b, list): continue
        n = min(len(a), len(b))
        m = r.get(maskk) if maskk else None
        adv = r.get(advk) if advk else None
        for i in range(n):
            if m is not None and i < len(m) and not m[i]: continue
            if a[i] is None or b[i] is None: continue
            d = a[i] - b[i]
            xs.append(d)
            if adv is not None:
                av = adv[i] if isinstance(adv, list) and i < len(adv) else (adv if not isinstance(adv, list) else None)
                if av is not None: (adv_pos if av > 0 else adv_neg).append(d)
xs.sort(); n = len(xs)
def q(p): return xs[min(n - 1, int(p * n))]
ab = sorted(abs(x) for x in xs)
def qa(p): return ab[min(n - 1, int(p * n))]
def frac_gt(t): return 1 - bisect.bisect_right(ab, t) / n
print(f"records {n_rec}, tokens {n}")
print(f"log ratio: mean {sum(xs)/n:+.4f}, mean|x| {sum(ab)/n:.4f}, p50 {q(.5):+.4f}, p1 {q(.01):+.4f}, p99 {q(.99):+.4f}, p0.1 {q(.001):+.4f}, p99.9 {q(.999):+.4f}, min {xs[0]:+.3f}, max {xs[-1]:+.3f}")
print("|x| quantiles: " + " ".join(f"p{int(p*100)}={qa(p):.3f}" for p in (.5, .9, .95, .99, .999)) + f" p99.99={qa(.9999):.3f}")
print("fraction |log r| > eps: " + " ".join(f"{t}:{frac_gt(t):.4%}" for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.693)))
up = sum(1 for x in xs if x > 0.693) / n; lo = sum(1 for x in xs if x < -0.693) / n
print(f"one-sided: ratio>2.0 {up:.4%} (TIS cap check), ratio<0.5 {lo:.4%}")
for name, arr in (("A>0", adv_pos), ("A<0", adv_neg)):
    if arr:
        arr.sort(); m = len(arr)
        print(f"{name}: tokens {m}, mean {sum(arr)/m:+.4f}, frac x>+0.2 {sum(1 for x in arr if x > .2)/m:.4%}, frac x<-0.2 {sum(1 for x in arr if x < -.2)/m:.4%}")
