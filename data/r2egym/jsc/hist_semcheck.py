#!/usr/bin/env python3
"""Semantic check of the history-think probes on the first finished attempts.
Per probe: prompt tokens at turn t (median across attempts with >= t turns), per-turn growth vs previous completion,
think-span presence in the recorded all_messages, exception types, task_complete share."""
import glob, json, statistics, sys, collections
E = "/e/fscratch/reformo/lee27/experiments"
THINK_MARKERS = ("<|start_think|>", "<think>", "<|think|>")
for name in sys.argv[1:]:
    files = sorted(glob.glob(f"{E}/{name}/{name}/trace_jobs/eval_sessions/*/*/attempts/*/result.json"))
    at = []
    for p in files[:400]:
        try: d = json.load(open(p))
        except Exception: continue
        ar = d.get("agent_result") or {}
        rd = ar.get("rollout_details") or []
        x = rd[0] if rd else {}
        pl = [len(t) for t in (x.get("prompt_token_ids") or [])]
        cl = [len(t) for t in (x.get("completion_token_ids") or [])]
        am = (ar.get("metadata") or {}).get("all_messages") or []
        asst = [m.get("content") or "" for m in am if isinstance(m, dict) and m.get("role") == "assistant"]
        think = sum(1 for c in asst if any(k in c for k in THINK_MARKERS))
        exc = (d.get("exception_info") or {}).get("exception_type")
        v = d.get("verifier_result"); r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
        at.append(dict(pl=pl, cl=cl, n_asst=len(asst), think=think, exc=exc, reward=r, in_tok=ar.get("n_input_tokens"), out_tok=ar.get("n_output_tokens")))
    print(f"\n== {name}: {len(at)} attempts (of {len(files)} result files)")
    if not at: continue
    print("  exceptions:", dict(collections.Counter(a["exc"] for a in at)))
    print("  rewards:", dict(collections.Counter(a["reward"] for a in at)))
    turns = [len(a["pl"]) for a in at if a["pl"]]
    print("  turns: median", statistics.median(turns) if turns else None, "max", max(turns) if turns else None)
    for t in (1, 2, 3, 5, 8, 10, 15, 20, 30):
        v = [a["pl"][t-1] for a in at if len(a["pl"]) >= t]
        if v: print(f"  prompt@t{t}: median {statistics.median(v):.0f} (n={len(v)})")
    # growth vs completion: prompt[i+1]-prompt[i] compared with completion[i]
    g, ratio = [], []
    for a in at:
        for i in range(min(len(a["pl"]), len(a["cl"])) - 1):
            grow = a["pl"][i+1] - a["pl"][i]; comp = a["cl"][i]
            g.append(grow - comp)
            if comp > 0: ratio.append(grow / comp)
    if g: print(f"  growth minus prev completion: median {statistics.median(g):.0f}; growth/completion ratio median {statistics.median(ratio):.2f} (n={len(g)})")
    comp = [c for a in at for c in a["cl"]]
    if comp: print(f"  completion tokens per turn: median {statistics.median(comp):.0f}, mean {statistics.mean(comp):.0f}")
    th = [a["think"] / a["n_asst"] for a in at if a["n_asst"]]
    print(f"  recorded assistant turns with a think marker: mean share {statistics.mean(th):.2f} (n={len(th)} attempts)" if th else "  no assistant messages recorded")
    print("  n_input_tokens median:", statistics.median([a["in_tok"] for a in at if a["in_tok"]]) if any(a["in_tok"] for a in at) else None)
