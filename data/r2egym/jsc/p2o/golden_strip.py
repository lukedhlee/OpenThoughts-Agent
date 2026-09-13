#!/usr/bin/env python3
"""golden_strip.py: does the trainer's text strip reproduce the control prompt on REAL prompted trajectories?

For N task pairs of a P2O probe tree (<task>-pA vs <task>-pctl), takes the first user message of each harbor trajectory
(ATIF steps[0], source user), applies the same edit the trainer applies (remove [start_marker, end_marker) once) and
checks (1) the stripped text equals the control's first user message byte for byte, (2) the real tokenizer renders the
stripped text to the same ids as the control text, (3) the served prompt ids of turn 1 (steps[first agent].metrics
.prompt_token_ids) equal apply_chat_template(user0, add_generation_prompt=True) for BOTH arms, i.e. the runner's
re-tokenized initial prompt matches what the engine served (the pre-existing assumption the strip relies on).
Runs on a Jupiter login node with the snowball-v2 venv (OMP_NUM_THREADS=1); read-only.
"""
import argparse, glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
S = "/e/fscratch/reformo/lee27/experiments/p2oAc_s0/p2oAc_s0/trace_jobs/eval_sessions/p2oAc_s0_eval_step0"
ap = argparse.ArgumentParser()
ap.add_argument("--session", default=S); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--model", default="/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888")
ap.add_argument("--start", default="\n\n---\nWorking guidance:\n"); ap.add_argument("--end", default="\n\nCurrent terminal state:")
a = ap.parse_args()


def strip(content, start_marker, end_marker):
    n = content.count(start_marker)
    if n != 1:
        return None, "absent" if n == 0 else "start_repeated"
    s = content.index(start_marker); e = content.find(end_marker, s + len(start_marker))
    if e < 0:
        return None, "end_missing"
    return content[:s] + content[e:], "stripped"


def first_user_and_p0(tj):
    d = json.load(open(tj)); steps = d["steps"]
    user0 = next(s for s in steps if s.get("source") == "user")
    agent0 = next((s for s in steps if s.get("source") == "agent" and s.get("metrics", {}).get("prompt_token_ids")), None)
    return user0["message"], (agent0["metrics"]["prompt_token_ids"] if agent0 else None)


def ids(x):
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
tasks = sorted({os.path.basename(p).split("-pA__")[0] for p in glob.glob(a.session + "/*-pA__*")})[: a.n]
ok_text = ok_ids = ok_served = pairs = 0; first_bad = None
for t in tasks:
    pa = sorted(glob.glob("%s/%s-pA__*/attempts/*/agent/trajectory.json" % (a.session, t)))
    pc = sorted(glob.glob("%s/%s-pctl__*/attempts/*/agent/trajectory.json" % (a.session, t)))
    if not pa or not pc:
        continue
    pairs += 1
    ua, p0a = first_user_and_p0(pa[0]); uc, p0c = first_user_and_p0(pc[0])
    stripped, status = strip(ua, a.start, a.end)
    if stripped == uc:
        ok_text += 1
    elif first_bad is None:
        i = next((k for k in range(min(len(stripped or ""), len(uc))) if (stripped or "")[k] != uc[k]), min(len(stripped or ""), len(uc)))
        first_bad = (t, status, repr((stripped or "")[max(0, i - 60): i + 60]), repr(uc[max(0, i - 60): i + 60]))
    r_strip = ids(tok.apply_chat_template([{"role": "user", "content": stripped or ""}], add_generation_prompt=False, tokenize=True))
    r_ctl = ids(tok.apply_chat_template([{"role": "user", "content": uc}], add_generation_prompt=False, tokenize=True))
    ok_ids += int(r_strip == r_ctl)
    # served turn-1 prompt vs re-tokenized user0 + generation prompt, both arms
    served_ok = True
    for u, p0 in ((ua, p0a), (uc, p0c)):
        if p0 is None:
            served_ok = False; continue
        r_gp = ids(tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=True))
        served_ok = served_ok and (list(p0) == r_gp)
        if list(p0) != r_gp and first_bad is None:
            k = next((k for k in range(min(len(p0), len(r_gp))) if p0[k] != r_gp[k]), min(len(p0), len(r_gp)))
            first_bad = (t, "served!=retok", "served len %d retok len %d first diff at %d: %r vs %r" % (len(p0), len(r_gp), k, tok.decode(p0[max(0, k - 5): k + 5]), tok.decode(r_gp[max(0, k - 5): k + 5])), "")
    ok_served += int(served_ok)
print("pairs %d | stripped text == control text: %d | tokenized equal: %d | served p0 == retok(user0)+gen prompt (both arms): %d" % (pairs, ok_text, ok_ids, ok_served))
print("first mismatch:", first_bad)
sys.exit(0 if (ok_text == pairs and ok_ids == pairs) else 1)
