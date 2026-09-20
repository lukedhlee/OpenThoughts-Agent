#!/usr/bin/env python3
"""Contamination screen: Recursive-Task-Synthesis instructions vs Terminal-Bench 2.0 and SWE-bench Verified.

Same matcher as the 2026-09-19 OTA review (notes/artifacts/ota-swebench-leak-20260919): word 8-grams after
NFKC/lower/alnum normalisation, grams with 3+ numeric tokens or traceback words dropped, eval-side grams kept
only if they occur in <= 2 eval documents, train-side grams only if they occur in <= 3 unique tasks; an eval
document is flagged when it shares >= 3 such grams with one task.

    python data/rst/rst_screen.py --tasks rts_tasks.parquet --gold gold-qualified.jsonl \
        --tb2 tb2_instr --verified swebench_verified.parquet --out screen_out
"""
import argparse, collections, glob, json, re, unicodedata
from pathlib import Path
import pyarrow.parquet as pq

N = 8
NOISE = re.compile(r"\b(line|site|packages|file|traceback|most|recent|call|last|py)\b")

def toks(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).split()

def grams(s):
    t = toks(s); out = set()
    for i in range(len(t) - N + 1):
        g = " ".join(t[i:i + N])
        if sum(x.isdigit() for x in t[i:i + N]) >= 3 or NOISE.search(g): continue
        out.add(hash(g))
    return out

def screen(train: dict, ev: dict, label: str, eval_df_max=2, train_df_max=3, flag_at=3):
    tg = {k: grams(v) for k, v in train.items()}
    eg = {k: grams(v) for k, v in ev.items()}
    tdf = collections.Counter(); [tdf.update(g) for g in tg.values()]
    edf = collections.Counter(); [edf.update(g) for g in eg.values()]
    keep = {k for k, v in tdf.items() if v <= train_df_max} & {k for k, v in edf.items() if v <= eval_df_max}
    inv = collections.defaultdict(set)
    for t, g in tg.items():
        for k in g & keep: inv[k].add(t)
    rows = []
    for e, g in eg.items():
        c = collections.Counter()
        for k in g & keep:
            for t in inv[k]: c[t] += 1
        best = c.most_common(1)[0] if c else (None, 0)
        rows.append({"eval": e, "task": best[0], "shared": best[1]})
    flagged = [r for r in rows if r["shared"] >= flag_at]
    print(f"{label}: {len(ev)} eval docs vs {len(train)} tasks; >=1: {sum(r['shared']>=1 for r in rows)}, "
          f">=3: {len(flagged)}, >=5: {sum(r['shared']>=5 for r in rows)}, max: {max(r['shared'] for r in rows)}")
    return rows, flagged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--gold", required=True)
    ap.add_argument("--tb2", required=True); ap.add_argument("--verified", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    gold = {json.loads(l)["task_id"] for l in open(a.gold)}
    t = pq.read_table(a.tasks, columns=["task_id", "instruction"]).to_pylist()
    all_tasks = {r["task_id"]: r["instruction"] for r in t}
    gold_tasks = {k: v for k, v in all_tasks.items() if k in gold}
    tb2 = {Path(p).parent.name: open(p).read() for p in sorted(glob.glob(f"{a.tb2}/*/instruction.md"))}
    ver = {r["instance_id"]: r["problem_statement"] for r in pq.read_table(a.verified, columns=["instance_id", "problem_statement"]).to_pylist()}
    summary = {}
    for tlabel, tr in (("gold", gold_tasks), ("all", all_tasks)):
        for elabel, ev in (("tb2", tb2), ("verified", ver)):
            rows, flagged = screen(tr, ev, f"{tlabel} x {elabel}")
            (a.out / f"{tlabel}_{elabel}.json").write_text(json.dumps(rows, indent=1))
            summary[f"{tlabel}_{elabel}"] = {"flagged_ge3": len(flagged), "ge1": sum(r["shared"] >= 1 for r in rows),
                                             "max": max(r["shared"] for r in rows), "flagged": flagged}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    # show the strongest pairs for hand reading
    for key in ("gold_tb2", "gold_verified"):
        rows = json.loads((a.out / f"{key}.json").read_text())
        for r in sorted(rows, key=lambda r: -r["shared"])[:3]:
            if r["shared"] == 0: break
            print(f"--- {key} {r['eval']} <-> {r['task']} shared={r['shared']}")
            src = tb2 if key.endswith("tb2") else ver
            print("EVAL:", src[r["eval"]][:400].replace("\n", " "))
            print("TASK:", all_tasks[r["task"]][:400].replace("\n", " "))

if __name__ == "__main__":
    main()
