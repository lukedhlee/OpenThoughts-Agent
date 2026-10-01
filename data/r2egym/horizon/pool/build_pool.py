#!/usr/bin/env python3
"""build_pool.py: the clean R2E-Gym RL pool (tasks the 09-21 SFT mix never saw, no SWE-bench Verified leak) + a repo-stratified
train / held-out split. Login node, stdlib + pyarrow. Inputs come from fetch_sources.py, scan_nemotron.py and gh_prs.py.

Per task (r2egym-daytona-v3 tree, 3,035 tasks; identity from tasktrove_v3_upstream_map.tsv: upstream image
namanjain12/<repo>_final:<fix commit> and the TaskTrove base commit = the commit the task checks out):
  covered_coderforge     (repo, fix commit) is a qingyangwu/<repo>_final:<commit> image in ANY CoderForge-Preview split
  covered_nemotron_repo  repo is numpy / datalad / pandas / pillow (the audit's whole-repo rule: Nemotron-SWE-v1 rewrote the
                         statements, distinct issues = R2E-Gym-Subset counts)
  covered_nemotron_task  Nemotron-SWE-v1 R2E-Gym rows name the task's base (or fix) commit in their prompt ("compare your
                         changes with the base commit <sha>"), same repo. Per-task match the audit did not have.
  covered_agentless      Nemotron-SFT-SWE-v2 agentless: the prompt's issue block equals the task's R2E-Gym-V1 problem statement
                         after whitespace normalisation (exact), or starts with its first 300 normalised chars (prefix)
  covered_realpr         the task's GitHub PR is a real-PR instance the mix trained on: CoderForge SWE_Rebench/filtered_reward1
                         instance ids, Nemotron-SFT-SWE-v2 OpenHands and Nemotron-SWE-v1 SWE-Gym rows (base commit -> instance
                         via nebius/SWE-rebench{,-V2} + SWE-Gym), or a pool commit named in a Nemotron v2 OpenHands prompt
  swebv_pr_hit           PR number is a SWE-bench Verified instance of the same repo, or the fix/base commit is a Verified
                         base_commit, merge commit or PR commit
  clean = allowlisted (Daytona v3 allowlist, 3,020) and none of the above.
Held-out: ~10 % of clean (>= --min-heldout), largest-remainder per repo, clean val264 tasks first, then seeded random.
"""
import argparse, collections, csv, json, os, random, re
import pyarrow.parquet as pq

R = "/scratch/11584/lukedhlee/rl_pool"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=R)
ap.add_argument("--map", default=os.path.expanduser("~/OpenThoughts-Agent-rl/data/r2egym/overlap/tasktrove_v3_upstream_map.tsv"))
ap.add_argument("--split-dir", default=os.path.expanduser(
    "~/OpenThoughts-Agent/ai_memory/active/snowball-r2egym/experiments/analysis/tt_v2_split"))
ap.add_argument("--nemotron-repos", default="numpy,datalad,pandas,pillow")
ap.add_argument("--seed", type=int, default=20261001)
ap.add_argument("--heldout-frac", type=float, default=0.10)
ap.add_argument("--min-heldout", type=int, default=100)
ap.add_argument("--no-trees", action="store_true")
a = ap.parse_args()
S = f"{a.root}/src"; TREE = f"{a.root}/tree/task_tree"
WS = re.compile(r"\s+")
norm = lambda s: WS.sub(" ", (s or "").replace("\r\n", "\n")).strip()

# ---- base table
tasks = open(f"{TREE}/TASKS.txt").read().split()
allow = set(open(f"{a.root}/tasks/allowlist_r2egym_daytona_v3.txt").read().split())
umap = {r["path"]: r for r in csv.DictReader(open(a.map), delimiter="\t")}
split = {}
for s in ["train", "idval", "oodval", "heldout"]:
    for t in open(f"{a.split_dir}/tt_v2_{s}.txt").read().split(): split[t] = s
GH = {}
base = {}
for t in tasks:
    ti = json.load(open(f"{TREE}/{t}/tests/test_info.json")); u = umap[t]
    assert u["repo"] == ti["repo_name"] and u["tt_base_commit"] == ti["base_commit"], t
    GH[u["repo"]] = ti["github_repo"]
    sp = split.get(t, "rest")
    base[t] = dict(task_name=t, repo=u["repo"], commit=u["commit"], base_commit=u["tt_base_commit"], image=u["docker_image"],
                   split=sp, val264=sp in ("idval", "oodval") and t in allow, allowlisted=t in allow)
gh2repo = {v.lower(): k for k, v in GH.items()}

# ---- real-PR index: base commit -> {(github repo lower, pr)}
idx = collections.defaultdict(set)
for r in pq.read_table(f"{S}/swe_rebench_index.parquet").to_pylist():
    m = re.match(r"(.+)__(.+)-(\d+)$", r["instance_id"] or "")
    if m: idx[r["base_commit"]].add((f"{m.group(1)}/{m.group(2)}".lower(), int(m.group(3))))
seen_pr = collections.defaultdict(set)  # (gh repo lower, pr) -> sources


def inst_to_pr(s):
    m = re.search(r"sweb\.eval\.x86_64\.([\w.-]+?)_1776_([\w.-]+)-(\d+)", s) or re.match(r"([\w.-]+?)__([\w.-]+)-(\d+)", s)
    return (f"{m.group(1)}/{m.group(2)}".lower(), int(m.group(3))) if m else None


# ---- CoderForge
cf = set(); cf_rows = collections.Counter()
IMG = re.compile(r"qingyangwu[/_]([a-z0-9]+)_final[:_]([0-9a-f]{40})")
with open(f"{S}/coderforge_images.tsv") as f:
    next(f)
    for line in f:
        sp, img, _ = line.rstrip("\n").split("\t")
        m = IMG.search(img)
        if m: cf.add((m.group(1), m.group(2))); cf_rows[(m.group(1), m.group(2))] += 1
        else:
            p = inst_to_pr(img)
            if p: seen_pr[p].add(f"coderforge_{sp}")

# ---- Nemotron-SWE-v1
nv1 = pq.read_table(f"{S}/nemo_v1.parquet", columns=["dataset", "repo", "hexes"]).to_pylist()
nv1_hex = collections.defaultdict(set)
for r in nv1:
    for h in r["hexes"].split():
        if "R2E" in r["dataset"]: nv1_hex[r["repo"].lower()].add(h)
        else:
            for p in idx.get(h, ()): seen_pr[p].add("nemotron_v1_swegym")

# ---- Nemotron-SFT-SWE-v2 OpenHands (swe.jsonl)
pool_commits = {}
for t, b in base.items():
    pool_commits.setdefault(b["commit"], set()).add(t); pool_commits.setdefault(b["base_commit"], set()).add(t)
v2_direct = set()
for r in pq.read_table(f"{S}/nemo_v2swe.parquet", columns=["hexes"]).to_pylist():
    for h in r["hexes"].split():
        v2_direct |= pool_commits.get(h, set())
        for p in idx.get(h, ()): seen_pr[p].add("nemotron_v2_openhands")

# ---- agentless (Nemotron-SFT-SWE-v2 agentless.jsonl) vs R2E-Gym-V1 statements
v1 = {r["docker_image"]: norm(r["problem_statement"]) for r in pq.read_table(
    f"{S}/r2egym_v1.parquet", columns=["docker_image", "problem_statement"]).to_pylist()}
stmt_of = {t: v1.get(b["image"]) for t, b in base.items()}
missing_stmt = [t for t, s in stmt_of.items() if not s]
full = collections.defaultdict(set); pre = collections.defaultdict(set)
for t, s in stmt_of.items():
    if s: full[s].add(t); pre[s[:300]].add(t)
ISS = [re.compile(r"### GitHub Problem Description ###\n(.*?)\n###\n\n### Repository Structure", re.S),
       re.compile(r"--- BEGIN ISSUE ---\n(.*?)\n--- END ISSUE ---", re.S)]
ag_exact = collections.Counter(); ag_prefix = collections.Counter(); ag_rows = 0; ag_unparsed = 0
for u in pq.read_table(f"{S}/nemo_agentless.parquet", columns=["user"]).column("user").to_pylist():
    m = ISS[0].search(u) or ISS[1].search(u)
    if not m: ag_unparsed += 1; continue
    s = norm(m.group(1))
    hit = full.get(s)
    if hit:
        ag_rows += 1
        for t in hit: ag_exact[t] += 1
    for t in pre.get(s[:300], ()): ag_prefix[t] += 1

# ---- GitHub PRs + SWE-bench Verified
gh = json.load(open(f"{S}/gh_cache.json"))
sv = pq.read_table(f"{S}/swebv.parquet").to_pylist()
sv_repos = collections.Counter(r["repo"].lower() for r in sv)
sv_pr = {(r["repo"].lower(), int(r["instance_id"].rsplit("-", 1)[1])) for r in sv}
sv_commits = {r["base_commit"] for r in sv}
for k, v in gh.items():
    if "#" in k and v:
        sv_commits.add(v["merge"]); sv_commits |= set(v["commits"])
pool_repos_in_sv = sorted(r for r in GH.values() if r.lower() in sv_repos)

rows = []
for t in tasks:
    b = base[t]; full_gh = GH[b["repo"]]; key = f"{full_gh}@{b['commit']}"
    prs = gh.get(key)
    prs = prs if isinstance(prs, list) else []
    swebv = any((full_gh.lower(), p) in sv_pr for p in prs) or b["commit"] in sv_commits or b["base_commit"] in sv_commits
    realpr_src = sorted({s for p in prs for s in seen_pr.get((full_gh.lower(), p), ())} | ({"nemotron_v2_openhands_commit"} if t in v2_direct else set()))
    hx = nv1_hex.get(full_gh.lower(), set())
    row = dict(b, pr_lookup=key in gh, pr_number=";".join(map(str, prs)),
               covered_coderforge=(b["repo"], b["commit"]) in cf, cf_trajectories=cf_rows.get((b["repo"], b["commit"]), 0),
               covered_nemotron_repo=b["repo"] in a.nemotron_repos.split(","),
               covered_nemotron_task=b["base_commit"] in hx or b["commit"] in hx,
               covered_agentless=ag_exact[t] > 0 or ag_prefix[t] > 0, agentless_exact_rows=ag_exact[t],
               agentless_prefix_rows=ag_prefix[t], covered_realpr=bool(realpr_src), realpr_sources=";".join(realpr_src),
               swebv_pr_hit=swebv)
    row["clean"] = row["allowlisted"] and not any(row[k] for k in ["covered_coderforge", "covered_nemotron_repo",
                                                                    "covered_nemotron_task", "covered_agentless",
                                                                    "covered_realpr", "swebv_pr_hit"])
    rows.append(row)

# candidates that still need a PR lookup (non-sympy/orange3 repos) -> written out, gh_prs.py --task-list, rerun
need = [r["task_name"] for r in rows if r["clean"] and not r["pr_lookup"]]
open(f"{a.root}/need_pr_lookup.txt", "w").write("\n".join(need) + ("\n" if need else ""))
if need: print(f"WARNING {len(need)} clean candidates have no PR lookup yet -> need_pr_lookup.txt; they are NOT clean until looked up")
for r in rows:
    if r["clean"] and not r["pr_lookup"]: r["clean"] = False

# ---- split
rng = random.Random(a.seed)
clean = [r for r in rows if r["clean"]]
H = max(a.min_heldout, round(a.heldout_frac * len(clean)))
by = collections.defaultdict(list)
for r in clean: by[r["repo"]].append(r["task_name"])
quota = {g: len(v) * H / len(clean) for g, v in by.items()}
alloc = {g: int(q) for g, q in quota.items()}
for g in sorted(by, key=lambda g: (quota[g] - alloc[g], g), reverse=True)[:H - sum(alloc.values())]: alloc[g] += 1
held = []
for g in sorted(by):
    v = sorted(by[g]); valt = [t for t in v if base[t]["val264"]]; rest = [t for t in v if not base[t]["val264"]]
    rng.shuffle(valt); rng.shuffle(rest)
    held += (valt + rest)[:alloc[g]]
held = sorted(held); train = sorted({r["task_name"] for r in clean} - set(held))
for r in rows: r["rl_set"] = "heldout" if r["task_name"] in held else ("train" if r["clean"] else "")

cols = ["task_name", "repo", "commit", "base_commit", "split", "val264", "allowlisted", "covered_coderforge", "cf_trajectories",
        "covered_nemotron_repo", "covered_nemotron_task", "covered_agentless", "agentless_exact_rows", "agentless_prefix_rows",
        "covered_realpr", "realpr_sources", "swebv_pr_hit", "pr_number", "clean", "rl_set"]
with open(f"{a.root}/coverage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
open(f"{a.root}/train.txt", "w").write("\n".join(train) + "\n")
open(f"{a.root}/heldout.txt", "w").write("\n".join(held) + "\n")

keys = ["pool", "allowlisted", "covered_coderforge", "covered_nemotron_repo", "covered_nemotron_task", "covered_agentless",
        "covered_realpr", "any_sft", "swebv_pr_hit", "swebv_drop_only", "clean", "train", "heldout", "val264", "val264_clean"]
per = collections.defaultdict(lambda: dict.fromkeys(keys, 0))
for r in rows:
    sft = any(r[k] for k in ["covered_coderforge", "covered_nemotron_repo", "covered_nemotron_task", "covered_agentless", "covered_realpr"])
    vals = dict(pool=1, allowlisted=r["allowlisted"], any_sft=sft, swebv_drop_only=r["swebv_pr_hit"] and not sft and r["allowlisted"],
                train=r["rl_set"] == "train", heldout=r["rl_set"] == "heldout", val264=r["val264"], val264_clean=r["val264"] and r["clean"],
                **{k: r[k] for k in ["covered_coderforge", "covered_nemotron_repo", "covered_nemotron_task", "covered_agentless",
                                     "covered_realpr", "swebv_pr_hit", "clean"]})
    for g in (r["repo"], "TOTAL"):
        for k, v in vals.items(): per[g][k] += int(bool(v))
summary = dict(seed=a.seed, heldout_target=H, per_repo={g: per[g] for g in sorted(per)},
               agentless=dict(rows_exact_matched=ag_rows, prompts_unparsed=ag_unparsed, tasks_without_v1_statement=len(missing_stmt)),
               swebv=dict(instances=len(sv), pool_repos_in_verified=pool_repos_in_sv, verified_pr_count=len(sv_pr)),
               seen_real_prs={g: len([p for p in seen_pr if p[0] == g.lower()]) for g in GH.values()},
               need_pr_lookup=len(need))
json.dump(summary, open(f"{a.root}/summary.json", "w"), indent=1)

if not a.no_trees:
    for name, lst in (("train_tree", train), ("heldout_tree", held)):
        d = f"{a.root}/{name}"; os.makedirs(d, exist_ok=True)
        for x in os.listdir(d):
            if os.path.islink(f"{d}/{x}"): os.unlink(f"{d}/{x}")
        for t in lst: os.symlink(f"{TREE}/{t}", f"{d}/{t}")
hdr = ["repo"] + keys
print("\t".join(hdr))
for g in sorted(per, key=lambda g: (g == "TOTAL", g)): print("\t".join([g] + [str(per[g][k]) for k in keys]))
print(json.dumps({k: v for k, v in summary.items() if k != "per_repo"}))
