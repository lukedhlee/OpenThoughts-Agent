#!/usr/bin/env python3
"""ledger.py <eval_sessions dir> [--min-turns N] [--max-trajs K] [--task SUBSTR] [--reward 0|1]
Classifies every attempt result (exception, reward, trajectory present) and prints a per-turn ledger
(think chars, task_complete, commands, observation head) for up to K trajectories."""
import json, glob, sys, os, collections, argparse
ap = argparse.ArgumentParser(); ap.add_argument("S"); ap.add_argument("--min-turns", type=int, default=1)
ap.add_argument("--max-trajs", type=int, default=1); ap.add_argument("--task", default=None); ap.add_argument("--reward", default=None)
ap.add_argument("--no-ledger", action="store_true"); a = ap.parse_args()
S = a.S; rs = sorted(glob.glob(f"{S}/*/*/attempts/*/result.json")); c = collections.Counter(); rew = {}
for p in rs:
    d = json.load(open(p)); e = (d.get("exception_info") or {}).get("exception_type")
    v = d.get("verifier_result"); r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
    has = os.path.exists(os.path.join(os.path.dirname(p), "agent", "trajectory.json")); c[(e, r, has)] += 1
    rew[os.path.dirname(p)] = r
    if e: print("EXC", p.split("/")[-4], e, str((d.get("exception_info") or {}).get("exception_message", ""))[:160])
print("results:", len(rs), dict(c))
if a.no_ledger: sys.exit()
shown = 0
for p in sorted(glob.glob(f"{S}/*/*/attempts/*/agent/trajectory.json")):
    if a.task and a.task not in p: continue
    d = json.load(open(p)); ag = [s for s in d["steps"] if s.get("source") == "agent"]
    if len(ag) < a.min_turns: continue
    r = rew.get(os.path.dirname(os.path.dirname(p)))
    if a.reward is not None and str(r) != a.reward and not (a.reward == "1" and r == 1.0) and not (a.reward == "0" and r == 0.0): continue
    print(f"\n=== TASK {p.split('/')[-5]} turns {len(ag)} reward {r}")
    for i, s in enumerate(ag):
        m = s.get("message", "") or ""; think = m.split("<|end_think|>")[0] if "<|end_think|>" in m else ""
        body = m.split("<|end_think|>", 1)[1] if "<|end_think|>" in m else m
        j = body.find("{"); cmds = ["<parse-fail>"]; tc = None
        try:
            js = json.loads(body[j:body.rfind("}") + 1]); cmds = [x.get("keystrokes", "").strip().replace("\n", "⏎")[:100] for x in js.get("commands") or []]; tc = js.get("task_complete")
        except Exception: pass
        obs = ((s.get("observation") or {}).get("results") or []); oc = (obs[0].get("content") or "") if obs else ""
        print(f"{i+1:2d} think={len(think):5d} tc={tc} {cmds} | obs={oc[:70]!r}")
    shown += 1
    if shown >= a.max_trajs: break
