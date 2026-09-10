#!/usr/bin/env python3
"""rollout_profile.py <dumped_data dir or jsonl...> [--csv out.csv]
Per-training-step behaviour profile from MarinSkyRL dumped train rollouts (global_step_N_train_rollouts.jsonl).
Same command classifier and metric names as experiments/analysis/scripts/build_index.py (60k baseline), so the
curves are comparable: reward, turns, tokens/turn, reasoning share, never-edit, first-edit turn, test-after-edit,
task_complete, C-c, exact repeats, parse-fail turns, warning observations."""
import json, re, sys, os, glob, collections, statistics, csv
CAT = [("test", r"^(python3? -m pytest|pytest|python3? -m unittest|tox|nosetests|make test|python3? -c .*(assert|import))"),
       ("edit", r"(sed -i|cat > |cat >> |tee |apply_patch|patch -p|python3? - ?<<|cat <<|printf .* > |echo .* > |>> ?[A-Za-z0-9_./-]+\.py|mv |cp )"),
       ("run_python", r"^python3? (-c|[A-Za-z0-9_./-]+\.py)"),
       ("explore", r"^(ls|cat |head|tail|grep|rg|find|sed -n|wc|tree|pwd|cd |which|git (log|status|diff|show|grep)|less|more|awk|nl |stat )"),
       ("install", r"^(pip|uv |apt|conda|npm)"), ("git", r"^git ")]
LEAK = re.compile(r"/workspace|metadata\.json|expected_output|/logs/verifier|reward\.txt")
GIT_HIST = re.compile(r"\bgit\s+(log|reflog|show)\b")
GIT_RESTORE = re.compile(r"\bgit\s+(checkout|reset|restore|stash|revert)\b")
FAIL_OBS = re.compile(r"\b(FAILED|failed|Error|Traceback|error:)\b")
def classify(k):
    k = k.strip()
    if not k: return "empty"
    for c, pat in CAT:
        if re.search(pat, k): return c
    return "other"
ROLE = re.compile(r"(?:^|(?<=[}\n]))(?:<\|start_header_id\|>)?(assistant|user)(?:<\|end_header_id\|>)?\n\n?")
def turns_of(text):
    """split a decoded multi-turn response into [(role, content)]"""
    out = []; pos = 0; cur = None
    for m in ROLE.finditer(text):
        if cur is not None: out.append((cur, text[pos:m.start()]))
        cur = m.group(1); pos = m.end()
    if cur is not None: out.append((cur, text[pos:]))
    return out
def q(a, f):
    a = sorted(a); return a[min(len(a) - 1, int(f * len(a)))] if a else float("nan")
def profile_record(r):
    segs = turns_of(r["response"]); ag = [c for role, c in segs if role == "assistant"]; us = [c for role, c in segs if role == "user"]
    cmds_all = []; think_chars = total_chars = 0; tc = False; pf = 0; edits = []; tests = []; ctrlc = 0; nturn = len(ag)
    leak = git_hist = git_restore = 0; tc_over_fail = False
    for i, m in enumerate(ag):
        total_chars += len(m)
        if "<|end_think|>" in m: think = m.split("<|end_think|>")[0]
        else:
            j = m.find("{"); think = m[:j] if j > 0 else ""
        think_chars += len(think)
        body = m.split("<|end_think|>", 1)[1] if "<|end_think|>" in m else m
        j = body.find("{"); k = body.rfind("}")
        cmds = []
        try:
            js = json.loads(body[j:k + 1]); cmds = [str(c.get("keystrokes") or "") for c in js.get("commands") or [] if isinstance(c, dict)]
            if js.get("task_complete") is True:
                tc = True
                if i < len(us) and FAIL_OBS.search(us[i][-3000:] if i < len(us) else "") and not any(classify(c) == "edit" for c in cmds): tc_over_fail = True
        except Exception: pf += 1
        for c in cmds:
            if LEAK.search(c): leak += 1
            if GIT_HIST.search(c): git_hist += 1
            if GIT_RESTORE.search(c): git_restore += 1
        cats = [classify(c) for c in cmds]; cmds_all.append(tuple(c.strip() for c in cmds))
        if "edit" in cats: edits.append(i + 1)
        if "test" in cats: tests.append(i + 1)
        ctrlc += sum(1 for c in cmds if c.strip() in ("C-c", "C-d", "C-z"))
    rep_prev = sum(1 for i in range(1, len(cmds_all)) if cmds_all[i] and cmds_all[i] == cmds_all[i - 1])
    seen = collections.Counter(cmds_all); rep_any = sum(v - 1 for k, v in seen.items() if k and v > 1)
    warn = sum(1 for u in us if u.lstrip().startswith("Previous response had warnings"))
    return dict(reward=r.get("reward"), turns=nturn, resp_len=r.get("response_len"), tok_per_turn=(r.get("response_len") or 0) / max(1, nturn),
                reasoning_share=think_chars / max(1, total_chars), never_edit=not edits, first_edit=edits[0] if edits else None,
                test_after_edit=bool(edits and any(t > edits[0] for t in tests)), task_complete=tc, ctrlc=ctrlc, rep_prev=rep_prev, rep_any=rep_any,
                parse_fail=pf, warn_obs=warn, stop=r.get("stop_reason"),
                leak=leak, git_hist=git_hist, git_restore=git_restore, tc_over_fail=tc_over_fail)
def summarize(step, recs):
    P = [profile_record(r) for r in recs]; n = len(P); sc = [p for p in P if p["reward"] is not None]
    succ = [p for p in sc if p["reward"] >= 1.0]; ed = [p for p in P if not p["never_edit"]]
    row = dict(step=step, n=n, reward_mean=round(statistics.mean(p["reward"] for p in sc), 4) if sc else None, n_success=len(succ),
               turns_p50=q([p["turns"] for p in P], .5), turns_p90=q([p["turns"] for p in P], .9),
               tok_per_turn_p50=round(q([p["tok_per_turn"] for p in P], .5)), resp_len_p50=q([p["resp_len"] or 0 for p in P], .5),
               reasoning_share=round(statistics.mean(p["reasoning_share"] for p in P), 3),
               never_edit=round(sum(p["never_edit"] for p in P) / n, 3), first_edit_p50=q([p["first_edit"] for p in P if p["first_edit"]], .5),
               test_after_edit_of_editors=round(sum(p["test_after_edit"] for p in ed) / max(1, len(ed)), 3),
               task_complete=round(sum(p["task_complete"] for p in P) / n, 3), ctrlc_attempts=round(sum(p["ctrlc"] > 0 for p in P) / n, 3),
               exact_repeat_attempts=round(sum((p["rep_prev"] + p["rep_any"]) > 0 for p in P) / n, 3),
               parse_fail_turns=round(sum(p["parse_fail"] for p in P) / max(1, sum(p["turns"] for p in P)), 4),
               warn_obs_share=round(sum(p["warn_obs"] for p in P) / max(1, sum(p["turns"] for p in P)), 4),
               stop=dict(collections.Counter(p["stop"] for p in P)),
               leak_touch_attempts=round(sum(p["leak"] > 0 for p in P) / n, 4), git_history_attempts=round(sum(p["git_hist"] > 0 for p in P) / n, 4),
               git_restore_attempts=round(sum(p["git_restore"] > 0 for p in P) / n, 4), tc_over_fail_attempts=round(sum(p["tc_over_fail"] for p in P) / n, 4),
               succ_task_complete=round(sum(p["task_complete"] for p in succ) / max(1, len(succ)), 3) if succ else None,
               succ_turns_p50=q([p["turns"] for p in succ], .5) if succ else None)
    return row
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    files = []
    for a in args: files += sorted(glob.glob(f"{a}/global_step_*_train_rollouts.jsonl")) if os.path.isdir(a) else [a]
    files.sort(key=lambda f: int(re.search(r"global_step_(\d+)", f).group(1)) if re.search(r"global_step_(\d+)", f) else 0)
    rows = []
    for f in files:
        recs = [json.loads(l) for l in open(f)]; step = int(re.search(r"global_step_(\d+)", f).group(1)) if re.search(r"global_step_(\d+)", f) else -1
        rows.append(summarize(step, recs))
    keys = list(rows[0].keys()) if rows else []
    for r in rows: print(json.dumps(r))
    if out and rows:
        with open(out, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}) for r in rows]
        print("wrote", out)
