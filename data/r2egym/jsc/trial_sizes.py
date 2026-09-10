import json, os, sys
root = sys.argv[1]
ds = sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))), key=lambda d: os.path.getmtime(os.path.join(root, d)))
d = ds[-1]; base = os.path.join(root, d); print("TRIAL", d, "n_trials", len(ds))
tot = 0; big = []
for r, _, fs in os.walk(base):
    for f in fs:
        p = os.path.join(r, f); s = os.path.getsize(p); tot += s
        if s > 500_000: big.append((s, p.replace(base + "/", "")))
print("total_MB", round(tot / 1e6, 1)); [print("  %7.1f MB  %s" % (s / 1e6, p)) for s, p in sorted(big, reverse=True)[:12]]
rp = [os.path.join(r, f) for r, _, fs in os.walk(base) for f in fs if f == "result.json"]
if rp:
    j = json.load(open(rp[0])); sz = lambda o: len(json.dumps(o))
    print("result.json", round(os.path.getsize(rp[0]) / 1e6, 1), "MB; top keys:", list(j)[:14])
    def walk(o, path, depth=0):
        if depth > 3: return
        if isinstance(o, dict):
            for k, v in o.items():
                s = sz(v)
                if s > 200_000: print("  %7.1f MB  %s" % (s / 1e6, path + "." + k)); walk(v, path + "." + k, depth + 1)
        elif isinstance(o, list) and o and sz(o) > 200_000: print("  list len", len(o), "at", path)
    walk(j, "result")
