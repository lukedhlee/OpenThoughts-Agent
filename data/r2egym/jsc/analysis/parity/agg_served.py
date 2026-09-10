import json, glob, collections, re
from transformers import AutoTokenizer
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
tok = AutoTokenizer.from_pretrained(M)
S = "/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
paths = sorted(glob.glob(f"{S}/*/attempts/000/agent/trajectory.json"))[:400]
ft = collections.Counter(); warn_kinds = collections.Counter(); turns = 0; obs_warn = 0; obs_nto = 0; obs_sure = 0; traj_sure = 0
endthink = 0; endthink_nn_brace = 0; text_after_json = 0; lastid = collections.Counter(); tc_true = 0; shown = False
newline_after_eot = collections.Counter()
for p in paths:
    d = json.load(open(p)); ag = [s for s in d["steps"] if s.get("source") == "agent"]
    tsure = False
    for s in ag:
        m = s.get("metrics") or {}; c = m.get("completion_token_ids") or []; pids = m.get("prompt_token_ids") or []
        if not c: continue
        turns += 1; ft[c[0]] += 1; lastid[c[-1]] += 1
        ct = tok.decode(c, skip_special_tokens=False)
        endthink += "<|end_think|>" in ct; endthink_nn_brace += "<|end_think|>\n\n{" in ct
        body = ct.split("<|end_think|>", 1)[1] if "<|end_think|>" in ct else ct
        tc_true += bool(re.search(r'"task_complete":\s*true', body))
        j = body.rfind("}"); text_after_json += bool(body[j+1:].replace("<|eot_id|>", "").strip()) if j >= 0 else 0
        obs = ((s.get("observation") or {}).get("results") or [])
        oc = (obs[0].get("content") or "") if obs else ""
        if oc.startswith("Previous response had warnings"):
            obs_warn += 1
            for w in re.findall(r"- ([^\n]+)", oc.split("\n\n")[0]): warn_kinds[w] += 1
        obs_nto += "New Terminal Output:" in oc
        if "Are you sure you want to mark the task as complete" in oc:
            obs_sure += 1; tsure = True
            if not shown:
                shown = True; print("SAMPLE completion-confirmation observation head:", repr(oc[:220]))
        if pids:
            for k, t in enumerate(pids[:-1]):
                if t == 128009: newline_after_eot[pids[k+1]] += 1
    traj_sure += tsure
print(f"trajs={len(paths)} turns={turns}")
print(f"first completion id: {ft.most_common(3)}  -> 128002 frac {ft[128002]/turns:.4f}")
print(f"last completion id: {lastid.most_common(3)}")
print(f"completions with <|end_think|>: {endthink/turns:.4f}; with '<|end_think|>\\n\\n{{': {endthink_nn_brace/turns:.4f}; trailing text after final '}}': {text_after_json/turns:.4f}; task_complete true: {tc_true/turns:.4f}")
print(f"observations starting 'Previous response had warnings': {obs_warn/turns:.4f}; containing 'New Terminal Output:': {obs_nto/turns:.4f}; completion-confirmation obs: {obs_sure} (trajs with it: {traj_sure}/{len(paths)})")
print("warning kinds:", warn_kinds.most_common(6))
print("token following <|eot_id|> in served prompts:", newline_after_eot.most_common(4))
