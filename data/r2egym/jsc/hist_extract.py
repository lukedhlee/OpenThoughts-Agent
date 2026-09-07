#!/usr/bin/env python3
"""hist_extract.py — ONE streaming pass over a history-think probe's trace_archive.tar.

Writes three files into --out (few inodes on purpose; reformo is over its inode soft quota):
  <probe>_turns.jsonl  compact per-attempt record, per-turn numeric + command KINDS + write TARGETS (no free text) -> all quantitative work
  <probe>_text.jsonl   per-attempt truncated text (think head, analysis, plan, keystrokes, observation tails) -> qualitative reading
  <probe>_index.jsonl  {name, offset, size} for every result.json / trajectory.json member -> random access back into the tar

Python 3.9 / stdlib, single process, OMP_NUM_THREADS=1. Safe on the Jupiter login node (I/O + json.loads only).
"""
import argparse, json, os, re, sys, tarfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edit_target import attempt_flags  # per-turn write targets (repo source / scratch / patch file)

THINK_ID = 128002
E = "/e/fscratch/reformo/lee27/experiments"

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tar", default=None)
ap.add_argument("--tasks", default=None, help="file of task ids; emit only these attempts (default: all)")
a = ap.parse_args()
KEEP = {l.strip() for l in open(a.tasks) if l.strip()} if a.tasks else None

TAR = a.tar or f"{E}/{a.probe}/{a.probe}/trace_archive.tar"
os.makedirs(a.out, exist_ok=True)

# ---------- command classification ----------
RE = [
    ("test_repo",  re.compile(r"\b(pytest|python -m pytest|py\.test|tox|nosetests|run_tests\.sh|unittest)\b")),
    ("repro",      re.compile(r"(reproduce|repro|bug_?repro|test_issue|check_issue|poc)[\w./-]*\.py|python3? +/?(testbed/)?repro")),
    ("edit",       re.compile(r"(sed -i|>\s*/testbed/|cat\s*>\s*|tee\s+/testbed|patch -p|apply_patch|python3? +- +<<|<<\s*['\"]?EOF)")),
    ("git",        re.compile(r"\bgit +(diff|status|log|checkout|stash|apply|add|restore|show)\b")),
    ("search",     re.compile(r"\b(grep|rg|find|ls|cat|head|tail|awk|wc|nl|which|tree)\b|sed -n")),
    ("install",    re.compile(r"\b(pip install|apt-get|conda install|python setup\.py)\b")),
    ("shell",      re.compile(r"\b(cd|pwd|echo|export|mkdir|cp|mv|rm|chmod|source)\b")),
]
RE_WAIT = re.compile(r"^\s*$|^C-c\s*$|^\s*\\n\s*$")
ERR = re.compile(r"Traceback \(most recent|SyntaxError|IndentationError|command not found|No such file or directory|ModuleNotFoundError|ImportError|Permission denied|error:", re.I)
PYT = re.compile(r"(\d+) (passed|failed|error|errors|xfailed|xpassed|skipped)")


def kind_of(ks):
    if not ks or RE_WAIT.match(ks):
        return "wait"
    for k, r in RE:
        if r.search(ks):
            return k
    return "other"


def clip(s, n):
    if not s:
        return ""
    s = s.replace("\n", "\\n")
    return s[:n]


def parse_assistant(content):
    """-> (think_chars, has_think, obj_or_None, raw_json_head)"""
    think = ""
    body = content or ""
    if "<|start_think|>" in body:
        seg = body.split("<|start_think|>", 1)[1]
        if "<|end_think|>" in seg:
            think, body = seg.split("<|end_think|>", 1)
        else:
            think, body = seg, ""
    i = body.find("{")
    obj = None
    if i >= 0:
        for j in (body.rfind("}"),):
            if j > i:
                try:
                    obj = json.loads(body[i:j + 1])
                except Exception:
                    obj = None
    return len(think), bool(think.strip()), obj, body[:0]


def handle(name, data, fh_turns, fh_text):
    try:
        d = json.loads(data)
    except Exception:
        return
    parts = name.split("/")
    # trace_jobs/eval_sessions/<session>/<task>__<id>/attempts/<nnn>/result.json
    trial = parts[3] if len(parts) > 3 else ""
    task = trial.split("__")[0] or (d.get("task_name") or "")
    if KEEP is not None and task not in KEEP:
        return
    tname = d.get("task_name")
    sess = parts[2] if len(parts) > 2 else ""
    att = parts[5] if len(parts) > 5 else ""
    v = d.get("verifier_result")
    reward = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
    exc = (d.get("exception_info") or {}).get("exception_type")
    ar = d.get("agent_result") or {}
    rd = (ar.get("rollout_details") or [{}])[0]
    plens = [len(t) for t in (rd.get("prompt_token_ids") or [])]
    ctoks = rd.get("completion_token_ids") or []
    clens = [len(t) for t in ctoks]
    thinktok = [bool(t) and t[0] == THINK_ID for t in ctoks]
    md = ar.get("metadata") or {}
    am = md.get("all_messages") or []

    turns, texts = [], []
    ai = 0  # assistant index
    prev_cmds = None
    cwd = "/testbed"          # the pane's working directory, threaded across turns
    for k, m in enumerate(am):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        c = m.get("content") or ""
        th_chars, has_think, obj, _ = parse_assistant(c)
        cmds = []
        durs = []
        if isinstance(obj, dict):
            for cc in (obj.get("commands") or []):
                if isinstance(cc, dict):
                    cmds.append(cc.get("keystrokes") or "")
                    try: durs.append(float(cc.get("duration") or 0))
                    except Exception: durs.append(0.0)
        tc = bool(isinstance(obj, dict) and obj.get("task_complete") is True)
        kinds = [kind_of(x) for x in cmds]
        wf, cwd = attempt_flags(cmds, cwd)  # full keystrokes, before any clipping; cwd survives the turn (tmux pane)
        # observation = the next user message
        obs = ""
        for m2 in am[k + 1:]:
            if isinstance(m2, dict) and m2.get("role") == "user":
                obs = m2.get("content") or ""
                break
            if isinstance(m2, dict) and m2.get("role") == "assistant":
                break
        pyt = PYT.findall(obs[-4000:]) if obs else []
        sig = "|".join(x.strip() for x in cmds)
        turns.append(dict(
            i=ai,
            th=bool(thinktok[ai]) if ai < len(thinktok) else has_think,
            th_ch=th_chars,
            cl=clens[ai] if ai < len(clens) else None,
            pl=plens[ai] if ai < len(plens) else None,
            nc=len(cmds),
            k=kinds,
            dur=round(sum(durs), 1),
            tc=tc,
            pj=obj is not None,
            ol=len(obs),
            er=bool(ERR.search(obs[-4000:])) if obs else False,
            pt=[[int(n), w] for n, w in pyt][:6],
            rep=bool(prev_cmds is not None and sig == prev_cmds and sig),
            es=wf["edit_src"], et=wf["edit_test"], ws=wf["write_scratch"], pf=wf["patch_file"], uw=wf["unknown_write"],
        ))
        texts.append(dict(
            i=ai,
            think=clip((c.split("<|start_think|>", 1)[1].split("<|end_think|>", 1)[0] if "<|start_think|>" in c else ""), 700),
            an=clip((obj or {}).get("analysis") if isinstance(obj, dict) else "", 400),
            pl=clip((obj or {}).get("plan") if isinstance(obj, dict) else "", 400),
            cmds=[clip(x, 200) for x in cmds[:8]],
            obs=clip(obs[-700:], 700),
        ))
        prev_cmds = sig
        ai += 1

    rec = dict(probe=a.probe, task=task, tname=tname, trial=trial, sess=sess, att=att, reward=reward, exc=exc,
               n_turns=len(turns), in_tok=ar.get("n_input_tokens"), out_tok=ar.get("n_output_tokens"),
               stop=md.get("stop_reason"), turns=turns)
    fh_turns.write(json.dumps(rec, separators=(",", ":")) + "\n")
    fh_text.write(json.dumps(dict(probe=a.probe, task=task, trial=trial, att=att, reward=reward, exc=exc,
                                  turns=texts), separators=(",", ":")) + "\n")


t0 = time.time()
n = 0
p_turns = f"{a.out}/{a.probe}_turns.jsonl"
p_text = f"{a.out}/{a.probe}_text.jsonl"
p_idx = f"{a.out}/{a.probe}_index.jsonl"
with open(p_turns, "w") as ft, open(p_text, "w") as fx, open(p_idx, "w") as fi, tarfile.open(TAR, "r|") as tf:
    for m in tf:
        if not m.isfile():
            continue
        if (m.name.endswith("/result.json") or m.name.endswith("/trajectory.json")) and "/eval_sessions/" in m.name:
            fi.write(json.dumps(dict(n=m.name, o=m.offset_data, s=m.size), separators=(",", ":")) + "\n")
        if m.name.endswith("/result.json") and "/attempts/" in m.name and "/eval_sessions/" in m.name:
            handle(m.name, tf.extractfile(m).read(), ft, fx)
            n += 1
            if n % 500 == 0:
                sys.stderr.write(f"{a.probe}: {n} attempts ({time.time()-t0:.0f}s)\n")
                sys.stderr.flush(); ft.flush(); fx.flush(); fi.flush()
sys.stderr.write(f"{a.probe}: DONE {n} attempts in {time.time()-t0:.0f}s\n")
