"""ifv2_pair.py — paired readout of two held-out if-v2 probe runs (ifv2_rows.json written per run).

    python ifv2_pair.py <rows_A.json> <rows_B.json> [--seed 0]

Rows = [task_name, exception_type|None, reward|None]. Pairs on task_name; a timed-out agent counts as 0 (the task
was attempted), an infrastructure exception on either side drops the pair. Prints pass@1 per side, the paired
difference with a 10,000-draw bootstrap 95 % CI, and the unlock / lose counts.
"""
import json, random, sys
INFRA = {"BridgeOutageError", "TmuxBatchProtocolError", "TmuxSessionEndedError", "VerifierTimeoutError", "TmuxCommandError", "SandboxBuildFailedError"}
def load(p):
    out = {}
    for name, exc, rew in json.load(open(p)):
        if exc in INFRA: continue
        out[name] = 1.0 if (exc is None and rew == 1.0) else 0.0
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b)); seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
xa = [a[t] for t in common]; xb = [b[t] for t in common]; n = len(common)
d = sum(xb) / n - sum(xa) / n
rng = random.Random(seed); diffs = [xb[i] - xa[i] for i in range(n)]; boots = []
for _ in range(10000):
    s = [diffs[rng.randrange(n)] for _ in range(n)]; boots.append(sum(s) / n)
boots.sort(); lo, hi = boots[int(0.025 * 10000)], boots[int(0.975 * 10000)]
print(f"paired tasks {n} (A has {len(a)}, B has {len(b)} after dropping infra)")
print(f"A pass@1 {sum(xa)/n:.3f}   B pass@1 {sum(xb)/n:.3f}   B-A {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"unlocked (A=0,B=1) {sum(1 for i in range(n) if xa[i]==0 and xb[i]==1)}   lost (A=1,B=0) {sum(1 for i in range(n) if xa[i]==1 and xb[i]==0)}   both {sum(1 for i in range(n) if xa[i]==1 and xb[i]==1)}")
