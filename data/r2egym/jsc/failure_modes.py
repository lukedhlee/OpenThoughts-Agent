#!/usr/bin/env python3
"""failure_modes.py <dumped_data dir>... — per-trial failure-pattern frequency in FAILED (reward 0) vs SUCCESSFUL (reward>=1) trials,
with lift = P(pattern | fail) / P(pattern | success). Patterns: the rollout_profile counters (parse-fail turns, never-edit, exact repeats,
C-c, leak, git history/restore, task_complete over a failing observation, warning observations, length stops) plus edit-block markers
(<<<<<<< SEARCH / >>>>>>> REPLACE / ======= conflict-style blocks), unterminated heredocs (cat <<EOF without a closing line in the same
command), python heredocs, and shape flags (>=60 turns, >=3k tok/turn, reasoning share < .2). Session 04b0b663, 2026-09-04 — for the
token-level reward-shaping idea (Luke, 00:58 PT)."""
import sys, os, glob, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_profile import profile_record, turns_of
SEARCH_BLOCK = re.compile(r"<<<<<<<\s*SEARCH|>>>>>>>\s*REPLACE|^=======\s*$", re.M)
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
def cmds_of(resp):
    out = []
    for role, m in turns_of(resp):
        if role != "assistant": continue
        body = m.split("<|end_think|>", 1)[1] if "<|end_think|>" in m else m
        j = body.find("{"); k = body.rfind("}")
        try:
            js = json.loads(body[j:k + 1]); out += [str(c.get("keystrokes") or "") for c in js.get("commands") or [] if isinstance(c, dict)]
        except Exception: pass
    return out
def feats(r):
    p = profile_record(r); cmds = cmds_of(r["response"]); joined = "\n".join(cmds)
    unterminated = 0
    for c in cmds:
        for m in HEREDOC.finditer(c):
            if not re.search(r"^" + re.escape(m.group(1)) + r"\s*$", c, re.M): unterminated += 1
    return dict(parse_fail=p["parse_fail"] > 0, never_edit=p["never_edit"], repeat=(p["rep_prev"] + p["rep_any"]) > 0, ctrlc=p["ctrlc"] > 0,
                leak=p["leak"] > 0, git_hist=p["git_hist"] > 0, git_restore=p["git_restore"] > 0, tc_over_fail=p["tc_over_fail"], warn_obs=p["warn_obs"] > 0,
                stop_length=(p["stop"] == "length"), no_task_complete=not p["task_complete"],
                search_block=bool(SEARCH_BLOCK.search(joined)), unterminated_heredoc=unterminated > 0, python_heredoc=("python" in joined and "<<" in joined),
                many_turns=p["turns"] >= 60, long_turns=p["tok_per_turn"] >= 3000, low_reasoning=p["reasoning_share"] < 0.2)
rows = []
for d in sys.argv[1:]:
    arm = d.rstrip("/").split("/")[-3]
    files = sorted(glob.glob(os.path.join(d, "global_step_*_train_rollouts.jsonl")), key=lambda f: int(re.search(r"step_(\d+)_", f).group(1)))
    for f in files:
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("reward") is None: continue
            fe = feats(r); fe["_fail"] = r["reward"] < 1.0; fe["_arm"] = arm; rows.append(fe)
fail = [x for x in rows if x["_fail"]]; succ = [x for x in rows if not x["_fail"]]
print(f"trials={len(rows)} fail={len(fail)} success={len(succ)} arms={sorted(set(x['_arm'] for x in rows))}")
print(f"{'pattern':22s} {'P(fail)':>8s} {'P(succ)':>8s} {'lift':>6s} {'n_fail':>7s} {'n_succ':>7s}")
keys = [k for k in rows[0] if not k.startswith("_")]
out = []
for k in keys:
    nf = sum(x[k] for x in fail); ns = sum(x[k] for x in succ)
    pf = nf / max(1, len(fail)); ps = ns / max(1, len(succ))
    out.append((k, pf, ps, (pf / ps) if ps > 0 else float("inf"), nf, ns))
for k, pf, ps, lift, nf, ns in sorted(out, key=lambda t: -t[3]):
    print(f"{k:22s} {pf:8.3f} {ps:8.3f} {lift:6.2f} {nf:7d} {ns:7d}")
