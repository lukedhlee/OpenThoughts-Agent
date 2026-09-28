"""if_table.py — one IF comparison table from score_ifbench.py / score_ifeval.py per-prompt files (runs on the Mac).

    python if_table.py --row "Stage-3|no|base_ifbench.jsonl.scored.jsonl|base.jsonl.scored.jsonl" --row ... [--pair A B ...]

Each row: name | saw if-v2 | IFBench scored file | IFEval scored file. Prints a markdown table of prompt-level loose and
strict accuracy with 95 % percentile-bootstrap CIs over prompts (10,000 resamples), truncation % and mean completion
tokens over both suites, then paired differences (B − A, same prompts resampled jointly) for every --pair, or every
pair against the first row when none is given.

2026-09-28: multi-sample files (one scored row per prompt and sample). A prompt's score is the mean over its samples,
and the bootstrap resamples prompts, so the CIs stay over prompts. Sample indices that do not cover every prompt (a
job cut by its wall) are dropped and reported. Extra columns: responses ending inside an unclosed think span, responses
of 8,192+ tokens (what an 8,192 cap would have truncated), samples per prompt.
"""
import argparse, json
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--row", action="append", required=True); ap.add_argument("--pair", nargs=2, action="append")
ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

def load(path):
    """{key: per-prompt means} over the sample indices that cover every prompt in the file."""
    recs = [json.loads(l) for l in open(path)]
    keys = {r["key"] for r in recs}
    cover = defaultdict(set)
    for r in recs: cover[r.get("sample", 0)].add(r["key"])
    full = {s for s, ks in cover.items() if ks == keys}
    if len(full) < len(cover): print(f"<!-- {path}: dropped partial samples {sorted(set(cover) - full)} -->")
    g = defaultdict(list)
    for r in recs:
        if r.get("sample", 0) in full: g[r["key"]].append(r)
    mean = lambda xs: float(np.mean(xs))
    return {k: dict(loose=mean([x["loose"] for x in v]), strict=mean([x["strict"] for x in v]),
                    trunc=mean([x["finish_reason"] == "length" for x in v]),
                    over8k=mean([x["completion_tokens"] >= 8192 for x in v]),
                    unclosed=mean([x["unclosed"] for x in v]) if all("unclosed" in x for x in v) else float("nan"),
                    tokens=mean([x["completion_tokens"] for x in v]), n=len(v)) for k, v in g.items()}

rows = {}
for spec in a.row:
    name, ifv2, fb, fe = [s.strip() for s in spec.split("|")]
    rows[name] = dict(ifv2=ifv2, ifbench=load(fb), ifeval=load(fe))

def ci(v):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), (a.n_boot, len(v)))
    b = v[idx].mean(1); return v.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100

fmt = lambda t: f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"
both = lambda r, f, spec: " / ".join(spec.format(f(r[s])) for s in ("ifbench", "ifeval"))
print("| checkpoint | IFBench loose | IFBench strict | IFEval loose | IFEval strict | truncated % (IFBench / IFEval) "
      "| ≥ 8,192 tokens % | unclosed think % | mean tokens | samples / prompt | saw if-v2? |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, r in rows.items():
    cells = []
    for suite in ("ifbench", "ifeval"):
        d = r[suite]; keys = sorted(d)
        for m in ("loose", "strict"):
            cells.append(fmt(ci([d[k][m] for k in keys])))
    col = lambda f: (lambda d: 100 * np.mean([x[f] for x in d.values()]))
    tok = lambda d: np.mean([x["tokens"] for x in d.values()])
    smp = lambda d: "{}".format(min(x["n"] for x in d.values())) + ("" if len({x["n"] for x in d.values()}) == 1 else "+")
    print(f"| {name} | " + " | ".join(cells) + f" | {both(r, col('trunc'), '{:.0f}')} | {both(r, col('over8k'), '{:.0f}')} "
          f"| {both(r, col('unclosed'), '{:.0f}')} | {both(r, tok, '{:,.0f}')} | {both(r, smp, '{}')} | {r['ifv2']} |")

names = list(rows)
pairs = a.pair or [(names[0], n) for n in names[1:]]
print()
print("| B − A (paired, 95 % CI) | IFBench loose | IFBench strict | IFEval loose | IFEval strict |")
print("|---|---|---|---|---|")
for A, B in pairs:
    cells = []
    for suite in ("ifbench", "ifeval"):
        da, db = rows[A][suite], rows[B][suite]; keys = sorted(set(da) & set(db))
        for m in ("loose", "strict"):
            diff = np.array([float(db[k][m]) - float(da[k][m]) for k in keys])
            t = ci(diff); cells.append(f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]")
    print(f"| {B} − {A} | " + " | ".join(cells) + " |")
