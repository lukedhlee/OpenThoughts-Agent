"""score_ifbench.py — score ifeval_gen.py outputs on IFBench with AI2's official checker (runs on the Mac).

    uv run --frozen --project <IFBench clone> python score_ifbench.py --repo <IFBench clone> \
        --inp ifbench_input_data.jsonl a_ifbench.jsonl b_ifbench.jsonl ...  [--raw]

Scorer: github.com/allenai/IFBench at SCORER_COMMIT (its uv.lock pins the checker's dependencies); the clone's
evaluation_lib.test_instruction_following_{strict,loose} are called unchanged, strict first as run_eval.py does.
Think spans are stripped with score_ifeval.py's strip_think (the same rule as the IFEval numbers); --raw skips it
(used to check this wrapper against run_eval.py on the repo's sample_output.jsonl). Reports prompt-level loose
(the paper's headline) and strict, instruction-level both, truncations and mean completion tokens, and writes
per-prompt results to <file>.scored.jsonl for if_table.py.
"""
import argparse, json, os, subprocess, sys

SCORER_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"   # allenai/IFBench main, 2026-09-09

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--inp", required=True)
ap.add_argument("--raw", action="store_true"); ap.add_argument("files", nargs="+")
a = ap.parse_args()
head = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head == SCORER_COMMIT, f"IFBench clone at {head}, expected {SCORER_COMMIT}"
sys.path.insert(0, a.repo)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation_lib
from score_ifeval import strip_think

def score(path):
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(path)}
    inputs = evaluation_lib.read_prompt_list(a.inp)
    assert len(rows) == len(inputs) == 300, (path, len(rows), len(inputs))
    p2r = {}
    for inp in inputs:
        r = rows[inp.key]
        assert r["prompt"] == inp.prompt, inp.key
        p2r[inp.prompt] = (r["response"] or "") if a.raw else strip_think(r["response"] or "")
    strict = [evaluation_lib.test_instruction_following_strict(inp, p2r) for inp in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(inp, p2r) for inp in inputs]
    per = []
    for inp, s, l in zip(inputs, strict, loose):
        r = rows[inp.key]
        per.append({"key": inp.key, "strict": s.follow_all_instructions, "loose": l.follow_all_instructions,
                    "inst_strict": s.follow_instruction_list, "inst_loose": l.follow_instruction_list,
                    "finish_reason": r["finish_reason"], "completion_tokens": (r.get("usage") or {}).get("completion_tokens", 0),
                    "empty": not p2r[inp.prompt]})
    with open(path + ".scored.jsonl", "w") as f:
        for p in per: f.write(json.dumps(p) + "\n")
    n = len(per); inst = lambda k: [x for p in per for x in p[k]]
    return dict(n=n, p_loose=sum(p["loose"] for p in per) / n, p_strict=sum(p["strict"] for p in per) / n,
                i_loose=sum(inst("inst_loose")) / len(inst("inst_loose")), i_strict=sum(inst("inst_strict")) / len(inst("inst_strict")),
                truncated=sum(p["finish_reason"] == "length" for p in per), errors=sum(p["finish_reason"] == "error" for p in per),
                empty=sum(p["empty"] for p in per), mean_tokens=sum(p["completion_tokens"] for p in per) / n)

print(f"{'file':32s} {'n':>4s} {'P-loose':>8s} {'P-strict':>8s} {'I-loose':>8s} {'I-strict':>8s} {'trunc':>5s} {'empty':>5s} {'err':>3s} {'tok':>6s}")
res = {}
for p in a.files:
    r = res[p] = score(p)
    print(f"{p[-32:]:32s} {r['n']:4d} {r['p_loose']*100:8.1f} {r['p_strict']*100:8.1f} {r['i_loose']*100:8.1f} {r['i_strict']*100:8.1f} "
          f"{r['truncated']:5d} {r['empty']:5d} {r['errors']:3d} {r['mean_tokens']:6.0f}")
