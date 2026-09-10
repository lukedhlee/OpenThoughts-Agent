import json, statistics, sys
def q(a, f):
    a = sorted(a); return a[min(len(a) - 1, int(f * len(a)))]
for name, limit in [("snowball_probe_val_eval_step0", 24576), ("snowball_probe_val_b30k_eval_step0", 30720)]:
    d = json.load(open(name + "_budget.json")); A, T = d["A"], d["T"]
    usable = limit - 2048; i = 0; rows = []
    for a in A:
        ts = T[i:i + a["turns"]]; i += a["turns"]
        if a["exc"] != "ContextLengthExceededError": continue
        n = len(ts); fixed = a["fixed"]
        full = sum(t["r"] + t["j"] + t["o"] + 10 for t in ts) / n
        no_r = sum(t["j"] + t["o"] + 10 for t in ts) / n
        cap = sum(t["r"] + t["j"] + min(t["o"], 2000) + 10 for t in ts) / n
        both = sum(t["j"] + min(t["o"], 2000) + 10 for t in ts) / n
        cap1k = sum(t["r"] + t["j"] + min(t["o"], 1000) + 10 for t in ts) / n
        both1k = sum(t["j"] + min(t["o"], 1000) + 10 for t in ts) / n
        rows.append([n, (usable - fixed) / full, (usable - fixed) / no_r, (usable - fixed) / cap, (usable - fixed) / both, (usable - fixed) / cap1k, (usable - fixed) / both1k, (65536 - 2048 - fixed) / full, (131072 - 2048 - fixed) / full])
    labs = ["observed turns at death", "model (same costs)", "no re-fed reasoning", "obs capped 2k", "no reasoning + obs 2k", "obs capped 1k", "no reasoning + obs 1k", "64k window as-is", "128k window as-is"]
    print(f"== {name} (n={len(rows)} context deaths, usable {usable}): estimated turns that fit, p50 / p90")
    for k, lab in enumerate(labs):
        col = [r[k] for r in rows]; print(f"   {lab:26s} {q(col, .5):5.1f} / {q(col, .9):5.1f}")
