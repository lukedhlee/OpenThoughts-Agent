#!/usr/bin/env python3
"""tt_daytona_allowlist.py — model-free gate verdicts for TaskTrove r2egym tasks run on Daytona through harbor jobs.

  tt_daytona_allowlist.py --map overlap/tasktrove_v3_upstream_map.tsv --oracle <jobdir>... --nop <jobdir>... \
      --out allowlist_r2egym_tt_daytona_v1.txt [--retry retry.txt] [--report report.md]

Per task and mode the LAST job dir listed that has a result wins (so `--oracle tree1_job tree2ri_job` lets the v2
repo-image re-gate override the v1 verdicts for those repos). Statuses:
  pass     oracle reward 1.0 AND pristine (nop) reward 0.0
  setup    setup_files/setup.sh failed in either mode (SetupScriptError) — the image/setup shape is wrong for the task
  infra    sandbox start timeout / Daytona conflict / rate limit / build failure — NOT a verdict; listed in --retry
  fail     verifier disagrees (oracle < 1 or pristine > 0)
  missing  no result in one of the modes — listed in --retry
"""
import argparse, collections, csv, glob, json, os
INFRA = {"EnvironmentStartTimeoutError", "DaytonaConflictError", "DaytonaRateLimitError", "SandboxBuildFailedError",
         "TimeoutError", "DaytonaError", "ClientConnectorError", "ServerDisconnectedError"}
ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True); ap.add_argument("--oracle", nargs="+", required=True); ap.add_argument("--nop", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--retry", default=None); ap.add_argument("--report", default=None)
ap.add_argument("--exclude", nargs="*", default=[])
a = ap.parse_args()
repo = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}

def load(jobdirs):
    out = {}
    for jd in jobdirs:
        for d in sorted(glob.glob(os.path.join(jd, "r2egym-*__*"))):
            task = os.path.basename(d).split("__")[0]
            res = sorted(glob.glob(os.path.join(d, "result.json")) + glob.glob(os.path.join(d, "attempts", "*", "result.json")))
            if not res: continue
            r = json.load(open(res[-1]))
            exc = (r.get("exception_info") or {}).get("exception_type")
            rew = None
            try: rew = list(r["verifier_result"]["rewards"].values())[0]
            except Exception: pass
            out[task] = (rew, exc, jd)
    return out
orc, nop = load(a.oracle), load(a.nop)
excl = {l.strip() for f in a.exclude for l in open(f) if l.strip() and not l.startswith("#")}
tasks = sorted(set(orc) | set(nop)); rows = []
for t in tasks:
    o, p = orc.get(t), nop.get(t)
    if t in excl: st = "excluded"
    elif any(x and x[1] == "SetupScriptError" for x in (o, p)): st = "setup"
    elif any(x and x[1] in INFRA for x in (o, p)): st = "infra"
    elif o is None or p is None or o[0] is None or p[0] is None: st = "missing"
    elif o[0] == 1.0 and p[0] == 0.0: st = "pass"
    else: st = "fail:o%s/p%s" % (o[0], p[0])
    rows.append((t, repo.get(t, "?"), st, (o or (None, None, "-"))[2], (p or (None, None, "-"))[2]))
allowed = [t for t, _, s, _, _ in rows if s == "pass"]
open(a.out, "w").write("\n".join(allowed) + "\n")
if a.retry: open(a.retry, "w").write("\n".join(t for t, _, s, _, _ in rows if s in ("infra", "missing")) + "\n")
tab = collections.defaultdict(collections.Counter); tot = collections.Counter()
for t, rp, s, _, _ in rows: tab[rp][s.split(":")[0]] += 1; tot[s.split(":")[0]] += 1
cols = ["pass", "fail", "setup", "infra", "missing", "excluded"]
lines = ["repo | tasks | " + " | ".join(cols) + " | pass %", "---|---|" + "---|" * len(cols) + "---"]
for rp in sorted(tab, key=lambda r: -sum(tab[r].values())):
    c = tab[rp]; n = sum(c.values()); lines.append("%s | %d | %s | %.1f" % (rp, n, " | ".join(str(c[k]) for k in cols), 100.0 * c["pass"] / n))
n = sum(tot.values()); lines.append("**all** | %d | %s | %.1f" % (n, " | ".join(str(tot[k]) for k in cols), 100.0 * tot["pass"] / max(n, 1)))
print("\n".join(lines)); print("allowlist -> %s (%d tasks)%s" % (a.out, len(allowed), "; retry list -> %s" % a.retry if a.retry else ""))
if a.report:
    with open(a.report, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        for t, rp, s, jo, jp in rows:
            if s != "pass": f.write("%s\t%s\t%s\t%s\t%s\n" % (t, rp, s, os.path.basename(jo), os.path.basename(jp)))
