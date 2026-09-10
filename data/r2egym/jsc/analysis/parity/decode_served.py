import json, glob, sys, os
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M)
S = "/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
paths = sorted(glob.glob(f"{S}/*/attempts/000/agent/trajectory.json"))
print("n trajectories:", len(paths))
picked = []
for p in paths:
    d = json.load(open(p))
    ag = [s for s in d["steps"] if s.get("source") == "agent"]
    if len(ag) >= 4 and (ag[2].get("metrics") or {}).get("prompt_token_ids"):
        picked.append((p, d, ag))
    if len(picked) == 2: break
HDR = "<|start_header_id|>"
for p, d, ag in picked:
    task = p.split("/")[-5]
    print("\n" + "=" * 100 + f"\nTASK {task}  n_agent_steps={len(ag)}  agent={d['agent']}")
    s = ag[2]; m = s["metrics"]; pids = m["prompt_token_ids"]; cids = m["completion_token_ids"]
    txt = tok.decode(pids, skip_special_tokens=False)
    print(f"turn3 prompt: n_ids={len(pids)} first ids={pids[:6]} -> {tok.decode(pids[:6], skip_special_tokens=False)!r}")
    print(f"  count 128002={pids.count(128002)} 128003={pids.count(128003)} 128009={pids.count(128009)} 128000={pids.count(128000)} 128001={pids.count(128001)}")
    print(f"  'Reasoning:' in prompt: {'Reasoning:' in txt}  '<|start_header_id|>system' in prompt: {(HDR+'system') in txt}")
    print(f"  literal '<think>' in prompt: {'<think>' in txt}  '</think>': {'</think>' in txt}")
    print(f"  HEAD[:200]: {txt[:200]!r}")
    # every header position
    idx = [i for i in range(len(txt)) if txt.startswith(HDR, i)]
    print(f"  n headers={len(idx)} roles={[txt[i+len(HDR):i+len(HDR)+12].split('<')[0] for i in idx]}")
    for j, i in enumerate(idx):
        role = txt[i+len(HDR):i+len(HDR)+12].split('<')[0]
        seg_end = idx[j+1] if j+1 < len(idx) else len(txt)
        seg = txt[i:seg_end]
        if role == "assistant" and j < len(idx)-1:
            print(f"  --- assistant turn @{i} (len {len(seg)}): HEAD {seg[:260]!r}")
            print(f"      TAIL {seg[-160:]!r}")
        elif role == "user" and j > 0:
            print(f"  --- user turn @{i} (len {len(seg)}): HEAD {seg[:220]!r}")
            print(f"      TAIL {seg[-80:]!r}")
    print(f"  GEN PROMPT tail[-60:]: {txt[-60:]!r}  last ids={pids[-6:]}")
    print(f"  turn3 completion: n={len(cids)} first ids={cids[:4]} -> {tok.decode(cids[:4], skip_special_tokens=False)!r} last ids={cids[-4:]} -> {tok.decode(cids[-4:], skip_special_tokens=False)!r}")
    ctxt = tok.decode(cids, skip_special_tokens=False)
    k = ctxt.find("<|end_think|>")
    print(f"  completion around <|end_think|>: {ctxt[max(0,k-60):k+60]!r}")
    print(f"  stored step message (raw_content) head: {s['message'][:200]!r}")
    print(f"  stored step message tail: {s['message'][-80:]!r}  reasoning_content field: {str(s.get('reasoning_content'))[:60]!r}")
    # step0/1 messages vs how they appear in prompt
    for t in (0, 1):
        msg = ag[t]["message"]
        print(f"  step{t} message starts with: {msg[:60]!r} ; appears verbatim in turn3 prompt: {msg in txt} ; msg.strip() in prompt: {msg.strip() in txt}")
        obs = (ag[t].get("observation") or {}).get("results") or []
        if obs:
            oc = obs[0].get("content") or ""
            print(f"  step{t} observation head: {oc[:120]!r} ; in prompt: {oc.strip() in txt}")
    # per-turn stats across all steps
    firsts = [ (s2["metrics"]["completion_token_ids"] or [None])[0] for s2 in ag if s2.get("metrics") and s2["metrics"].get("completion_token_ids")]
    print(f"  first completion id per turn: {firsts}")
    warn = [i for i, s2 in enumerate(ag) if any('WARNINGS' in ((r.get('content') or '')) for r in ((s2.get('observation') or {}).get('results') or []))]
    print(f"  turns whose observation carries WARNINGS: {warn}")
# aggregate over 200 trajectories: first-token and warnings frequency
import collections
ft = collections.Counter(); warnc = 0; turns = 0; extra_before = 0; think_in_content = 0
for p in paths[:300]:
    d = json.load(open(p)); ag = [s for s in d["steps"] if s.get("source") == "agent"]
    for s2 in ag:
        m = s2.get("metrics") or {}; c = m.get("completion_token_ids") or []
        if c: ft[c[0]] += 1; turns += 1
        obs = ((s2.get("observation") or {}).get("results") or [])
        if obs and "WARNINGS" in (obs[0].get("content") or ""): warnc += 1
        if obs and "Extra text detected before JSON" in (obs[0].get("content") or ""): extra_before += 1
        if "<|start_think|>" in (s2.get("message") or "") or "<think>" in (s2.get("message") or ""): think_in_content += 1
print(f"\nAGG over {min(300,len(paths))} trajs: turns={turns} first-token dist={ft.most_common(4)} warnings-obs={warnc} ({warnc/max(1,turns):.1%}) extra-text-before={extra_before} ({extra_before/max(1,turns):.1%}) think-markers-in-stored-content={think_in_content}")
