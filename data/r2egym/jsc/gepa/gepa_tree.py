#!/usr/bin/env python3
"""gepa_tree.py — build one GEPA wave's task tree: every split task x every candidate block, plus the control.

Adapted by diff from p2o/p2o_trees.py (which built the six-variant wave-0 tree). Same method, same guarantees:
  <task>-pctl    instruction.md byte-identical to the source
  <task>-p<cand> instruction.md = source + DELIM + block + "\\n"   (a STRICT suffix; a later strip is exact)
and the same verification: for each task all variants are byte-identical except instruction.md, and each variant's
instruction.md starts with the control's bytes and ends with exactly DELIM + block + "\\n".

Deltas from p2o_trees.py:
  - candidates come from a wave file {"delim": ..., "blocks": {"<cand>": "<text>"}} instead of the fixed A..E set,
    and cand ids are checked against ^[A-Za-z0-9]{1,12}$ (never "ctl") so the `-p<cand>` suffix stays parseable;
  - every block is LINTED before anything is written: <= --max-tokens (400) and no task / repo / file / test name
    (the population is a general terminal-agent procedure, never knowledge about these tasks);
  - one shard, not three: all variants of a task must share a shard so every candidate runs under the same load,
    and make_tt_wave.py --shards 1 keeps the whole wave in ONE probe job (the p2o6all_s0 shape);
  - writes experiments/gepa/<wave>/{allow.txt,cands.json,tree.md} so the scorer knows the wave's candidate set.

Usage (Jupiter login node, python3.9 / stdlib):
  gepa_tree.py --wave w1 --candidates /e/fscratch/reformo/lee27/experiments/gepa/w1_cands.json --split dev_mini
"""
import argparse, collections, csv, hashlib, json, os, re, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_feat import read_list, read_split, wave_sample  # noqa: E402

E = "/e/fscratch/reformo/lee27/experiments"
T = "/e/fscratch/reformo/lee27/tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--wave", required=True, help="wave name; tree -> tasks/gepa-<wave>, run prefix -> gepa<wave>")
ap.add_argument("--candidates", required=True, help='JSON {"delim": ..., "blocks": {"<cand>": "<block text>"}}')
ap.add_argument("--split", default="feedback,dev",
                help="comma-separated. PER-WAVE minibatches: `feedback` (64 fresh from train, the only traces the "
                     "session reads) and `gate` (32 fresh from dev, the cheap gate). FIXED: `dev` (500), `oodmini` "
                     "(32 OOD-repo train tasks), `dev_mini` (the pilot's fixed 32), `test`. Or a path to a list/tsv.")
ap.add_argument("--feedback-n", type=int, default=64, help="size of this wave's fresh feedback draw")
ap.add_argument("--gate-n", type=int, default=32, help="size of this wave's fresh gate draw from dev")
ap.add_argument("--src", default=T + "/r2egym-tt-daytona", help="source task tree (Daytona shape)")
ap.add_argument("--dst", default=None, help="default tasks/gepa-<wave>")
ap.add_argument("--splits-dir", default=E + "/gepa")
ap.add_argument("--max-tokens", type=int, default=400, help="hard cap per block (estimated, see est_tokens)")
ap.add_argument("--no-ctl", action="store_true", help="omit the control arm (only for a re-probe of candidates alone)")
ap.add_argument("--verify", type=int, default=0, help="byte-verify only N tasks (0 = all; use for dev-500 waves)")
a = ap.parse_args()
dst = a.dst or (T + "/gepa-" + a.wave)
out = "%s/%s" % (a.splits_dir, a.wave)

# ---------------------------------------------------------------- candidates + lint
B = json.load(open(a.candidates))
DELIM, blocks = B["delim"], B["blocks"]
if not blocks:
    sys.exit("no candidate blocks in %s" % a.candidates)
CAND_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
# The 11 repos in the tree, plus the dataset's own name: a block naming any of these is task knowledge, not procedure.
REPOS = ["sympy", "pillow", "pandas", "pyramid", "tornado", "numpy", "scrapy", "datalad", "aiohttp", "coveragepy", "orange3"]
FORBID = [(re.compile(r"\b%s\b" % r, re.I), "names the repo %r" % r) for r in REPOS] + [
    (re.compile(r"r2egym|tasktrove|swe-?bench|terminal-?bench|\btb2\b", re.I), "names a dataset or benchmark"),
    (re.compile(r"r2egym-v1-\d+", re.I), "names a task id"),
    (re.compile(r"\btest_\w+\.py\b|\br2e_tests\b|\btest_info\.json\b|\btest\.sh\b"), "names a specific test file"),
]
WARN = [(re.compile(r"/testbed\b"), "mentions /testbed (harness-specific, allowed but not portable)"),
        (re.compile(r"(?<![\w/])(?:\.{0,2}/)?(?:[\w.-]+/){2,}[\w.-]+\.\w+"), "contains a deep concrete file path")]


def est_tokens(s):
    """Upper-ish estimate without a tokenizer on the login node: English prose runs ~1.3 tokens/word and ~4 chars/token."""
    return int(max(len(s.split()) * 1.3, len(s) / 4.0) + 0.5)


bad = []
for c in sorted(blocks):
    if not CAND_RE.match(c) or c == "ctl":
        bad.append("%s: candidate id must match [A-Za-z0-9]{1,12} and not be 'ctl'" % c)
    n = est_tokens(blocks[c])
    if n > a.max_tokens:
        bad.append("%s: ~%d tokens > %d" % (c, n, a.max_tokens))
    for rx, why in FORBID:
        m = rx.search(blocks[c])
        if m:
            bad.append("%s: %s (%r)" % (c, why, m.group(0)))
if bad:
    print("BLOCK LINT FAILED:")
    for b in bad:
        print("  " + b)
    sys.exit(2)
for c in sorted(blocks):
    for rx, why in WARN:
        m = rx.search(blocks[c])
        if m:
            print("lint warning: %s %s (%r)" % (c, why, m.group(0)))

ARMS = ([] if a.no_ctl else ["ctl"]) + sorted(blocks)

# ---------------------------------------------------------------- task list
# The test split is the loop's one held-out number and it is scored ONCE, by gepa_final.sh. Building a tree over it
# is the step that would make an accidental rollout possible, so it needs FINAL=1 in the environment -- a deliberate
# act, not a flag someone can copy off an old command line.
splits = [s.strip() for s in a.split.split(",") if s.strip()]
if any(s == "test" for s in splits) and os.environ.get("FINAL") != "1":
    sys.exit("refusing to build a tree over the test split: it is rolled out once, by gepa_final.sh.\n"
             "If this really is the final run, gepa_final.sh sets FINAL=1 for you.")


os.makedirs(out, exist_ok=True)
# PER-WAVE minibatches. A fixed 64 gets fitted to its own quirks after a few waves, so `feedback` and `gate` are
# redrawn per wave, seeded by the wave name -- reproducible if the wave is rebuilt, fresh for the next one. The list
# is written into the wave dir and everything downstream (the queue's legs, gepa_worst, gepa_dump's refusal) reads
# it from there rather than re-deriving it.
PER_WAVE = {"feedback": ("train", a.feedback_n, ""), "gate": ("dev", a.gate_n, "gate")}


def load_split(s):
    if s in PER_WAVE:
        src, n, salt = PER_WAVE[s]
        p = "%s/%s.txt" % (out, s)
        if os.path.exists(p):
            return read_list(p)
        pool = sorted(read_split(src, a.splits_dir))
        if not pool:
            sys.exit("no %s split to draw %s from -- run gepa_split.py first" % (src, s))
        v = wave_sample(a.wave, pool, n, salt, a.splits_dir)
        open(p, "w").write("\n".join(v) + "\n")
        print("wave %s drew a fresh %s batch: %d of %d %s tasks -> %s" % (a.wave, s, len(v), len(pool), src, p))
        return v
    p = s if os.path.exists(s) else "%s/split_%s.txt" % (a.splits_dir, s)
    if not os.path.exists(p):
        sys.exit("no such split list: %s" % p)
    raw = [l.rstrip("\n") for l in open(p) if l.strip() and not l.startswith("#")]
    if raw and "\t" in raw[0]:  # a tsv (dev120.tsv shape): take the `task` column
        return [r["task"] for r in csv.DictReader(open(p), delimiter="\t")]
    return raw


by_split, tasks = {}, []
for s in splits:
    v = load_split(s)
    dup = set(v) & set(tasks)
    if dup:
        sys.exit("split %s overlaps an earlier split on %d tasks, e.g. %s" % (s, len(dup), sorted(dup)[:3]))
    by_split[s] = v
    tasks += v
missing = [t for t in tasks if not os.path.isdir(os.path.join(a.src, t))]
if missing:
    sys.exit("%d split tasks are not in %s, e.g. %s" % (len(missing), a.src, missing[:3]))

# ---------------------------------------------------------------- build
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst)


def tree_hash(d, skip="instruction.md"):
    h = hashlib.md5()
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel == skip:
                continue
            h.update(rel.encode())
            h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


def suffix(arm):
    b = blocks[arm]
    return (DELIM + b + ("" if b.endswith("\n") else "\n")).encode()


verify = set(tasks if not a.verify else tasks[:: max(1, len(tasks) // a.verify)][:a.verify])
failures = 0
for t in tasks:
    src = os.path.join(a.src, t)
    ctl_bytes = open(os.path.join(src, "instruction.md"), "rb").read()
    hashes = set()
    for arm in ARMS:
        d = os.path.join(dst, "%s-p%s" % (t, arm))
        shutil.copytree(src, d)
        if arm != "ctl":
            with open(os.path.join(d, "instruction.md"), "ab") as f:
                f.write(suffix(arm))
        got = open(os.path.join(d, "instruction.md"), "rb").read()
        ok = (got == ctl_bytes) if arm == "ctl" else (got.startswith(ctl_bytes) and got[len(ctl_bytes):] == suffix(arm))
        if not ok:
            failures += 1
            print("BYTE CHECK FAILED", t, arm)
        if t in verify:
            hashes.add(tree_hash(d))
    if len(hashes) > 1:
        failures += 1
        print("NON-INSTRUCTION FILES DIFFER", t)

names = ["%s-p%s" % (t, arm) for t in tasks for arm in ARMS]
open("%s/allow.txt" % out, "w").write("\n".join(names) + "\n")
json.dump({"wave": a.wave, "split": a.split, "splits": by_split, "src": a.src, "dst": dst, "delim": DELIM,
           "blocks": blocks, "arms": ARMS, "tasks": tasks}, open("%s/cands.json" % out, "w"), indent=1)
md = ["# GEPA wave %s\n" % a.wave,
      "splits %s (%d tasks) x arms %s = %d dirs under `%s`"
      % (", ".join("`%s` %d" % (s, len(v)) for s, v in by_split.items()), len(tasks), ",".join(ARMS), len(names), dst),
      "byte-identity failures: %d; hashed %d of %d tasks\n" % (failures, len(verify), len(tasks)),
      "| cand | ~tokens | words | bytes |", "|---|---|---|---|"]
for c in sorted(blocks):
    md.append("| %s | %d | %d | %d |" % (c, est_tokens(blocks[c]), len(blocks[c].split()), len(blocks[c].encode())))
open("%s/tree.md" % out, "w").write("\n".join(md) + "\n")
print("\n".join(md[1:]))
print("allowlist -> %s/allow.txt; manifest -> %s/cands.json" % (out, out))
sys.exit(1 if failures else 0)
