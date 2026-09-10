#!/usr/bin/env python3
"""ttwave_build.py [seed] [n_overlap] [n_ttonly] — TaskTrove-side pass@8 wave at pool-probe scale (python3.9-safe, stdlib).
overlap = TaskTrove tasks shared with our raw set whose raw copy is fully sampled in BOTH the 28k (pool28k) and 60k (pool60k
union) probes; tt_only = TaskTrove tasks our raw set lacks (allowlisted, text-bearing). TaskTrove copies only (the raw side
already has its 28k/60k tables). Builds /e/fscratch/reformo/lee27/tasks/ttwave-src + ttwave_manifest.tsv."""
import csv, glob, os, random, shutil, sys
E = "/e/fscratch/reformo/lee27/experiments/ttsample"; T = "/e/fscratch/reformo/lee27/tasks"; X = "/e/fscratch/reformo/lee27/experiments"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0; N_OVER = int(sys.argv[2]) if len(sys.argv) > 2 else 512; N_TT = int(sys.argv[3]) if len(sys.argv) > 3 else 512
random.seed(seed)
raw_img = dict(l.rstrip("\n").split("\t") for l in open(E + "/raw_images.tsv") if "\t" in l)
img_raw = {}
for t, i in raw_img.items(): img_raw.setdefault(i, []).append(t)
tt = {}
for r in csv.DictReader(open(E + "/tasktrove_v3_upstream_map.tsv"), delimiter="\t"): tt[r["path"]] = (r["docker_image"], r["repo"])
allow = {l.strip() for l in open(T + "/allowlist_r2egym_tt_v1.txt") if l.strip()}
empty = {l.strip() for l in open(E + "/tasktrove_v3_empty_issue_tasks.txt") if l.strip()}
def load(files):
    d = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r.get("full") == "True" and r.get("pass_at_k") not in (None, ""): d.setdefault(r["task"], (float(r["pass_at_k"]), r.get("med_turns", "")))
    return d
p60 = load([X + "/snowball_pool60k_band/union_pass8_table.csv"])
p28 = load(sorted(glob.glob(X + "/snowball_pool28k_s*/pass8_partial28k_pass8_table.csv")))
shared = [(p, img_raw[img][0], repo) for p, (img, repo) in sorted(tt.items()) if img in img_raw and p in allow and p not in empty]
over = [(p, rt, repo) for p, rt, repo in shared if rt in p60 and rt in p28 and os.path.isdir(T + "/r2egym-tt-raw/" + p)]
ttonly = [(p, repo) for p, (img, repo) in sorted(tt.items()) if img not in img_raw and p in allow and p not in empty and os.path.isdir(T + "/r2egym-tt-raw/" + p)]
print("shared %d, of which raw fully sampled at 28k AND 60k: %d; tt-only %d; priors 60k %d 28k %d" % (len(shared), len(over), len(ttonly), len(p60), len(p28)))
sel_over = random.sample(over, min(N_OVER, len(over))); sel_tt = random.sample(ttonly, min(N_TT, len(ttonly)))
src = T + "/ttwave-src"
if os.path.isdir(src): shutil.rmtree(src)
os.makedirs(src); rows = []
for p, rt, repo in sel_over:
    shutil.copytree(T + "/r2egym-tt-raw/" + p, src + "/" + p); rows.append((p, "overlap_tt", rt, repo, p28[rt][0], p60[rt][0], p60[rt][1]))
for p, repo in sel_tt:
    shutil.copytree(T + "/r2egym-tt-raw/" + p, src + "/" + p); rows.append((p, "tt_only", "", repo, "", "", ""))
with open(E + "/ttwave_manifest.tsv", "w") as f:
    f.write("task\tgroup\traw_task\trepo\traw_pass8_28k\traw_pass8_60k\traw_med_turns_60k\n")
    for r in rows: f.write("\t".join(str(x) for x in r) + "\n")
from collections import Counter
print("tree %s: %d dirs; overlap repos %s; tt-only repos %s" % (src, len(os.listdir(src)), dict(Counter(r[3] for r in rows if r[1] == "overlap_tt")), dict(Counter(r[3] for r in rows if r[1] == "tt_only"))))
ov = [r for r in rows if r[1] == "overlap_tt"]
print("overlap raw priors: mean pass8 28k %.3f 60k %.3f; 60k buckets zero/mixed/full = %d/%d/%d" % (
    sum(r[4] for r in ov) / len(ov), sum(r[5] for r in ov) / len(ov), sum(r[5] == 0 for r in ov), sum(0 < r[5] < 1 for r in ov), sum(r[5] == 1 for r in ov)))
