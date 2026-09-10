#!/usr/bin/env python3
"""parity_check.py <eval_sessions dir> <pass8 summary.json>  -> prints PASS/FAIL line for the SFT-parity contract.
Checks on up to 150 trajectories: (1) completions start with <|start_think|> (128002) >= 95%; (2) observations starting with
the 'Extra text detected before JSON' warning <= 2% of turns; (3) served prompts: assistant<|eot_id|> is followed directly by
<|start_header_id|>user (128009,128006,882) and NEVER by a stray newline (128009,198,128006,882); the template's own
user<|eot_id|>\\n<|start_header_id|>assistant (128009,198,128006,78191) is expected; (4) trial pass >= 5% over >= 40 scored."""
import json, glob, sys, re
S, sp = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(f"{S}/*/*/attempts/*/agent/trajectory.json"))[:150]
turns = ft = extra = good = bad = tmpl = 0
for p in paths:
    try: d = json.load(open(p))
    except Exception: continue
    for s in d.get("steps", []):
        if s.get("source") != "agent": continue
        m = s.get("metrics") or {}; c = m.get("completion_token_ids") or []; pids = m.get("prompt_token_ids") or []
        if not c: continue
        turns += 1; ft += (c[0] == 128002)
        obs = ((s.get("observation") or {}).get("results") or []); oc = (obs[0].get("content") or "") if obs else ""
        extra += ("Extra text detected" in oc.split("\n\n")[0]) if oc.startswith("Previous response had warnings") else 0
        for i, t in enumerate(pids[:-3]):
            if t != 128009: continue
            a, b, cc = pids[i+1], pids[i+2], pids[i+3]
            if a == 128006 and b == 882: good += 1
            elif a == 198 and b == 128006 and cc == 882: bad += 1
            elif a == 198 and b == 128006 and cc == 78191: tmpl += 1
s = json.load(open(sp)); tp = s["successes"] / max(1, s["scored"])
f1 = ft / max(1, turns); ex = extra / max(1, turns)
ok = turns > 0 and f1 >= 0.95 and ex <= 0.02 and bad == 0 and good > 0 and s["scored"] >= 40 and tp >= 0.05
print(("PASS" if ok else "FAIL") + f" trajs={len(paths)} turns={turns} first128002={f1:.3f} extra_text_warn={ex:.4f} asst_eot_clean={good} asst_eot_stray_nl={bad} user_eot_tmpl_nl={tmpl} scored={s['scored']} successes={s['successes']} trial_pass={tp:.3f}")
