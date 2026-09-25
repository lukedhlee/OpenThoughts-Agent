"""if_table.py — one IF comparison table from score_ifbench.py / score_ifeval.py per-prompt files (runs on the Mac).

    python if_table.py --row "Stage-3|no|base_ifbench.jsonl.scored.jsonl|base.jsonl.scored.jsonl" --row ... [--pair A B ...]

Each row: name | saw if-v2 | IFBench scored file | IFEval scored file. Prints a markdown table of prompt-level loose and
strict accuracy with 95 % percentile-bootstrap CIs over prompts (10,000 resamples), truncation % and mean completion
tokens over both suites, then paired differences (B − A, same prompts resampled jointly) for every --pair, or every
pair against the first row when none is given.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--row", action="append", required=True); ap.add_argument("--pair", nargs=2, action="append")
ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

def load(path):
    return {r["key"]: r for r in map(json.loads, open(path))}

rows = {}
for spec in a.row:
    name, ifv2, fb, fe = [s.strip() for s in spec.split("|")]
    rows[name] = dict(ifv2=ifv2, ifbench=load(fb), ifeval=load(fe))

def ci(v):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), (a.n_boot, len(v)))
    b = v[idx].mean(1); return v.mean() * 100, np.percentile(b, 2.5) * 100, np.percentile(b, 97.5) * 100

fmt = lambda t: f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"
print("| checkpoint | IFBench loose | IFBench strict | IFEval loose | IFEval strict | truncated % (IFBench / IFEval) | mean tokens (IFBench / IFEval) | saw if-v2? |")
print("|---|---|---|---|---|---|---|---|")
for name, r in rows.items():
    cells = []
    for suite in ("ifbench", "ifeval"):
        d = r[suite]; keys = sorted(d)
        for m in ("loose", "strict"):
            cells.append(fmt(ci([d[k][m] for k in keys])))
    tr = [100 * np.mean([x["finish_reason"] == "length" for x in r[s].values()]) for s in ("ifbench", "ifeval")]
    tk = [np.mean([x["completion_tokens"] for x in r[s].values()]) for s in ("ifbench", "ifeval")]
    print(f"| {name} | " + " | ".join(cells) + f" | {tr[0]:.0f} / {tr[1]:.0f} | {tk[0]:,.0f} / {tk[1]:,.0f} | {r['ifv2']} |")

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
