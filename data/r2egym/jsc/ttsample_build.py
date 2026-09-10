#!/usr/bin/env python3
"""ttsample_build.py — draw the overlap/disjoint pass@8 sample and build one mixed task tree (python3.9-safe, stdlib)."""
import csv, os, random, shutil, sys
E = "/e/fscratch/reformo/lee27/experiments/ttsample"; T = "/e/fscratch/reformo/lee27/tasks"
N_OVER, N_TTONLY, N_RAWONLY = 16, 16, 16; TT_STRATA = {"sympy": 10, "moto": 3, "matplotlib": 3}
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
raw_img = dict(l.rstrip("\n").split("\t") for l in open(E + "/raw_images.tsv") if "\t" in l)
img_raw = {}
for t, i in raw_img.items(): img_raw.setdefault(i, []).append(t)
tt = {}
for r in csv.DictReader(open(E + "/tasktrove_v3_upstream_map.tsv"), delimiter="\t"): tt[r["path"]] = (r["docker_image"], r["repo"])
allow = {l.strip() for l in open(T + "/allowlist_r2egym_tt_v1.txt") if l.strip()}
empty = {l.strip() for l in open(E + "/tasktrove_v3_empty_issue_tasks.txt") if l.strip()}
prior = {}
for f in ("/e/fscratch/reformo/lee27/experiments/snowball_pool60k_band/union_pass8_table.csv",
          "/e/fscratch/reformo/lee27/experiments/snowball_probe_val_b60k/pass8_b60k_merged_pass8_table.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("full") == "True" and r.get("pass_at_k"): prior.setdefault(r["task"], float(r["pass_at_k"]))
tt_imgs = {v[0] for v in tt.values()}
shared = [(p, img_raw[img][0], repo) for p, (img, repo) in sorted(tt.items()) if img in img_raw and p in allow and p not in empty]
ttonly = [(p, repo) for p, (img, repo) in sorted(tt.items()) if img not in img_raw and p in allow and p not in empty]
rawonly = [t for t, i in sorted(raw_img.items()) if i not in tt_imgs and os.path.isdir(T + "/r2egym-raw-v3-train/" + t)]
print("shared(allowlisted,text) %d  tt-only %d  raw-only(train) %d  priors %d" % (len(shared), len(ttonly), len(rawonly), len(prior)))
mixed = lambda t: t in prior and 0.25 <= prior[t] <= 0.75
sh_pool = [x for x in shared if mixed(x[1]) and os.path.isdir(T + "/r2egym-raw-v3-train/" + x[1])]
ro_pool = [t for t in rawonly if mixed(t)]
print("mixed-prior pools: shared %d raw-only %d" % (len(sh_pool), len(ro_pool)))
sel_sh = random.sample(sh_pool, N_OVER); sel_ro = random.sample(ro_pool, N_RAWONLY)
# TaskTrove-only text-bearing tasks are (almost) all sympy: every moto/matplotlib row is text-less. Take all non-sympy
# ones that exist, fill the rest with sympy.
from collections import Counter
print("tt-only repos:", dict(Counter(r for _, r in ttonly)))
non_sympy = [(p, r) for p, r in ttonly if r != "sympy"]
sel_tt = random.sample(non_sympy, min(len(non_sympy), 4))
sel_tt += [(p, "sympy") for p in random.sample([p for p, r in ttonly if r == "sympy"], N_TTONLY - len(sel_tt))]
src = T + "/ttsample-src"
if os.path.isdir(src): shutil.rmtree(src)
os.makedirs(src)
rows = []
for i, (p, rt, repo) in enumerate(sel_sh):
    shutil.copytree(T + "/r2egym-tt-raw/" + p, src + "/" + p); rows.append((p, "overlap_tt", "pair%02d" % i, repo, prior.get(rt, "")))
    shutil.copytree(T + "/r2egym-raw-v3-train/" + rt, src + "/" + rt); rows.append((rt, "overlap_raw", "pair%02d" % i, repo, prior.get(rt, "")))
for p, repo in sel_tt:
    shutil.copytree(T + "/r2egym-tt-raw/" + p, src + "/" + p); rows.append((p, "tt_only", "", repo, ""))
for t in sel_ro:
    shutil.copytree(T + "/r2egym-raw-v3-train/" + t, src + "/" + t); rows.append((t, "raw_only", "", raw_img[t].split("/")[1].split("_")[0], prior.get(t, "")))
with open(E + "/ttsample_manifest.tsv", "w") as f:
    f.write("task\tgroup\tpair\trepo\tprior_pass8\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
print("tree %s: %d task dirs; manifest %s/ttsample_manifest.tsv" % (src, len(os.listdir(src)), E))
for r in rows: print("\t".join(str(x) for x in r))
