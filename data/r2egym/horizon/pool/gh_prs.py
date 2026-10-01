#!/usr/bin/env python3
"""gh_prs.py: GitHub lookups for the SWE-bench Verified filter, cached to JSON (resumable; safe to re-run).

  commit -> PRs   `gh api repos/{owner}/{repo}/commits/{sha}/pulls` for every pool fix commit of the given repos
                  (the method of 2026-09-15_stage3_overlap_with_r2egym.md). Cache: {"owner/repo@sha": [pr numbers] | null}
  Verified PR -> commits  `gh api repos/{owner}/{repo}/pulls/{n}` (merge_commit_sha) + `.../pulls/{n}/commits` for every
                  Verified instance of those repos. Cache: {"owner/repo#n": {"merge": sha, "commits": [sha...]}}
Pacing: 3 workers, retries with backoff on 403/429 (secondary rate limit); a failed lookup is stored as null and retried on
the next run.
"""
import argparse, csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--map", default=os.path.expanduser("~/OpenThoughts-Agent-rl/data/r2egym/overlap/tasktrove_v3_upstream_map.tsv"))
ap.add_argument("--tasks", default="/scratch/11584/lukedhlee/rl_pool/tree/task_tree/TASKS.txt")
ap.add_argument("--swebv", default="/scratch/11584/lukedhlee/rl_pool/src/swebv.parquet")
ap.add_argument("--repos", default="sympy=sympy/sympy")
ap.add_argument("--cache", default="/scratch/11584/lukedhlee/rl_pool/src/gh_cache.json")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--task-list", default=None, help="only these task names (default: every task of --repos)")
a = ap.parse_args()
GH = os.path.expanduser("~/.local/bin/gh")
repos = dict(x.split("=") for x in a.repos.split(","))
cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}


def gh(path, paginate=False):
    cmd = [GH, "api", "-H", "Accept: application/vnd.github+json", path] + (["--paginate"] if paginate else [])
    for k in range(6):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            out = p.stdout.strip()
            if paginate and out.count("][") > 0: out = "[" + out[1:-1].replace("][", ",") + "]"
            return json.loads(out)
        err = p.stderr + p.stdout
        if "404" in err or "Not Found" in err or "No commit found" in err or "422" in err:
            return {"__missing__": err.strip()[:200]}
        time.sleep(min(120, 10 * 2 ** k))
    print("FAILED", path, err[:200], file=sys.stderr, flush=True)
    return None


tasks = set(open(a.tasks).read().split())
if a.task_list: tasks &= set(open(a.task_list).read().split())
jobs = []
for r in csv.DictReader(open(a.map), delimiter="\t"):
    if r["path"] in tasks and r["repo"] in repos:
        key = f"{repos[r['repo']]}@{r['commit']}"
        if cache.get(key) is None: jobs.append(key)
jobs = sorted(set(jobs))


def commit_prs(key):
    full, sha = key.split("@")
    res = gh(f"repos/{full}/commits/{sha}/pulls")
    if res is None: return key, None
    if isinstance(res, dict): return key, {"missing": res.get("__missing__", "")}
    return key, sorted({x["number"] for x in res if x.get("base", {}).get("repo", {}).get("full_name", full).lower() == full.lower()})


def save():
    json.dump(cache, open(a.cache + ".tmp", "w"), indent=0, sort_keys=True); os.replace(a.cache + ".tmp", a.cache)


print("commit lookups to do:", len(jobs), flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    for i, (k, v) in enumerate(ex.map(commit_prs, jobs)):
        cache[k] = v
        if i % 50 == 0: save(); print(" ", i, k, v, flush=True)
save()

sv = pq.read_table(a.swebv).to_pylist()
owners = {v.lower(): v for v in repos.values()}
pr_jobs = []
for r in sv:
    if r["repo"].lower() in owners:
        n = int(r["instance_id"].rsplit("-", 1)[1]); key = f"{owners[r['repo'].lower()]}#{n}"
        if cache.get(key) is None: pr_jobs.append(key)


def pr_commits(key):
    full, n = key.split("#")
    pr = gh(f"repos/{full}/pulls/{n}")
    cs = gh(f"repos/{full}/pulls/{n}/commits?per_page=100", paginate=True)
    if pr is None or cs is None: return key, None
    return key, {"merge": pr.get("merge_commit_sha"), "commits": [c["sha"] for c in cs] if isinstance(cs, list) else []}


print("Verified PR lookups to do:", len(pr_jobs), flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    for k, v in ex.map(pr_commits, pr_jobs): cache[k] = v
save()
print("done; null entries:", sum(v is None for v in cache.values()), flush=True)
