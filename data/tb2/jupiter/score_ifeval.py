"""score_ifeval.py — score ifeval_gen.py outputs on the Mac with lm_eval's IFEval checker.

    python score_ifeval.py base.jsonl std210.jsonl std630.jsonl

Strips <think>…</think> spans (and a leading unclosed think span) before checking, reports prompt- and
instruction-level strict/loose accuracy, and how many responses were truncated, thought, or empty.
Writes per-prompt results next to each input as <file>.scored.jsonl (if_table.py bootstraps and pairs them).
strip_think is shared with score_ifbench.py, which is why lm_eval is imported inside score().
"""
import json, re, sys

OPEN, CLOSE = r"(?:<think>|<\|start_think\|>)", r"(?:</think>|<\|end_think\|>)"
LENIENT = "--lenient" in sys.argv          # keep all text, drop only the markers (what a user with no parser sees)

def strip_think(s):
    if LENIENT:
        return re.sub(OPEN + "|" + CLOSE, "", s).strip()
    s = re.sub(OPEN + ".*?" + CLOSE, "", s, flags=re.S)
    if re.search(OPEN, s):          # unclosed: everything from the last opener on is reasoning
        s = re.split(OPEN, s)[0]
    if re.search(CLOSE, s):         # stray closer: everything before the last one is reasoning
        s = re.split(CLOSE, s)[-1]
    s = re.sub(r"<\|[a-z_]+\|>", "", s)   # any other special token
    return s.strip()

def score(path):
    from lm_eval.tasks.ifeval import utils
    rows = [json.loads(l) for l in open(path)]
    per = []
    agg = {"prompt_level_strict_acc": [], "inst_level_strict_acc": [], "prompt_level_loose_acc": [], "inst_level_loose_acc": []}
    trunc = thought = empty = err = 0; toks = 0
    for r in rows:
        raw = r["response"] or ""
        if r.get("reasoning"): thought += 1
        if re.search(OPEN, raw): thought += 1
        resp = strip_think(raw)
        if not resp: empty += 1
        if r["finish_reason"] == "length": trunc += 1
        if r["finish_reason"] == "error": err += 1
        if r.get("usage"): toks += r["usage"].get("completion_tokens", 0)
        doc = {"key": r["key"], "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
        out = utils.process_results(doc, [resp])
        per.append({"key": r["key"], "strict": bool(out["prompt_level_strict_acc"]), "loose": bool(out["prompt_level_loose_acc"]),
                    "finish_reason": r["finish_reason"], "completion_tokens": (r.get("usage") or {}).get("completion_tokens", 0)})
        for k in agg: 
            v = out[k]; agg[k].extend(v if isinstance(v, list) else [v])
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    with open(path + (".lenient" if LENIENT else "") + ".scored.jsonl", "w") as f:
        for p in per: f.write(json.dumps(p) + "\n")
    return dict(n=len(rows), **m, truncated=trunc, thought=thought, empty=empty, errors=err, mean_completion_tokens=toks / max(1, len(rows)))

if __name__ == "__main__":
    res = {p: score(p) for p in sys.argv[1:] if not p.startswith("--")}
    print(f"{'file':28s} {'n':>4s} {'P-strict':>8s} {'I-strict':>8s} {'P-loose':>8s} {'I-loose':>8s} {'trunc':>5s} {'think':>5s} {'empty':>5s} {'err':>3s} {'tok':>6s}")
    for p, r in res.items():
        print(f"{p[-28:]:28s} {r['n']:4d} {r['prompt_level_strict_acc']*100:8.1f} {r['inst_level_strict_acc']*100:8.1f} {r['prompt_level_loose_acc']*100:8.1f} {r['inst_level_loose_acc']*100:8.1f} {r['truncated']:5d} {r['thought']:5d} {r['empty']:5d} {r['errors']:3d} {r['mean_completion_tokens']:6.0f}")
    json.dump(res, open("ifeval_scores%s.json" % ("_lenient" if LENIENT else ""), "w"), indent=1)
