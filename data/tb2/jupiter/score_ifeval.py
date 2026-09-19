"""score_ifeval.py — score ifeval_gen.py outputs on the Mac with lm_eval's IFEval checker.

    python score_ifeval.py base.jsonl std210.jsonl std630.jsonl

Strips <think>…</think> spans (and a leading unclosed think span) before checking, reports prompt- and
instruction-level strict/loose accuracy, and how many responses were truncated, thought, or empty.
"""
import json, re, sys
from lm_eval.tasks.ifeval import utils

def strip_think(s):
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    if "<think>" in s:              # unclosed: everything after it is reasoning
        s = s.split("<think>", 1)[0]
    if "</think>" in s:             # unopened (Tezos-style): everything before it is reasoning
        s = s.split("</think>", 1)[1]
    return s.strip()

def score(path):
    rows = [json.loads(l) for l in open(path)]
    agg = {"prompt_level_strict_acc": [], "inst_level_strict_acc": [], "prompt_level_loose_acc": [], "inst_level_loose_acc": []}
    trunc = thought = empty = err = 0; toks = 0
    for r in rows:
        raw = r["response"] or ""
        if r.get("reasoning"): thought += 1
        if "<think>" in raw or "</think>" in raw: thought += 1
        resp = strip_think(raw)
        if not resp: empty += 1
        if r["finish_reason"] == "length": trunc += 1
        if r["finish_reason"] == "error": err += 1
        if r.get("usage"): toks += r["usage"].get("completion_tokens", 0)
        doc = {"key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
        out = utils.process_results(doc, [resp])
        for k in agg: 
            v = out[k]; agg[k].extend(v if isinstance(v, list) else [v])
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    return dict(n=len(rows), **m, truncated=trunc, thought=thought, empty=empty, errors=err, mean_completion_tokens=toks / max(1, len(rows)))

if __name__ == "__main__":
    res = {p: score(p) for p in sys.argv[1:]}
    print(f"{'file':28s} {'n':>4s} {'P-strict':>8s} {'I-strict':>8s} {'P-loose':>8s} {'I-loose':>8s} {'trunc':>5s} {'think':>5s} {'empty':>5s} {'err':>3s} {'tok':>6s}")
    for p, r in res.items():
        print(f"{p[-28:]:28s} {r['n']:4d} {r['prompt_level_strict_acc']*100:8.1f} {r['inst_level_strict_acc']*100:8.1f} {r['prompt_level_loose_acc']*100:8.1f} {r['inst_level_loose_acc']*100:8.1f} {r['truncated']:5d} {r['thought']:5d} {r['empty']:5d} {r['errors']:3d} {r['mean_completion_tokens']:6.0f}")
    json.dump(res, open("ifeval_scores.json", "w"), indent=1)
