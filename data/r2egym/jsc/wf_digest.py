#!/usr/bin/env python3
"""wf_digest.py --sample <ttwf_sample.tsv> --out <dir> [--tasks t1,t2,...] [--max-att 3]
Paired trace digests for the workflow-prompt re-probe: for each task, the instruction (workflow version), then the
header-prompt arm (tthd) and the workflow-prompt arm (ttwf) attempts, each as a per-turn ledger (flags, task_complete,
commands, think snippet, observation head) with the verifier reward and the test-stdout tail. Up to --max-att attempts
per arm, chosen to cover both outcomes (first win, first loss, then the rest in order). Adapted from tt_ana/digest_tt.py.
Writes <out>/<stratum>__<task>.txt and <out>/INDEX.md. Python 3.9 (Jupiter login node).
"""
import argparse, glob, json, os, re, sys, textwrap
sys.path.insert(0, "/e/fscratch/reformo/lee27/experiments/tt_ana/scripts")
from gap_agg import classify, edit_targets, obs_content, RESTORE, TMPPAT, SRCEXT  # noqa: E402
E = "/e/fscratch/reformo/lee27/experiments"; T = "/e/fscratch/reformo/lee27/tasks"
CW, OW = 150, 110
ARMS = [("hd", "tthd", T + "/r2egym-tt-hd300"), ("wf", "ttwf", T + "/r2egym-tt-wf300")]


def load(p):
    try: return json.load(open(p))
    except Exception: return None


def digest(att, tag, out):
    tj = load(os.path.join(att, "agent", "trajectory.json"))
    if tj is None: out.append("  !! unreadable %s" % att); return
    r = load(os.path.join(att, "result.json")) or {}
    exc = (r.get("exception_info") or {}).get("exception_type")
    ag = [s for s in tj["steps"] if s.get("source") == "agent"]
    out.append("\n== %s attempt=%s turns=%d exc=%s" % (tag, os.path.basename(att), len(ag), exc))
    for i, s in enumerate(ag):
        turn = i + 1
        m = s.get("message", ""); m = m if isinstance(m, str) else json.dumps(m)
        obs = obs_content(s.get("observation", "")); obs = obs if isinstance(obs, str) else json.dumps(obs)
        rej = obs.startswith("Previous response had parsing errors")
        think = ""
        mt = re.search(r"<\|start_think\|>(.*?)(<\|end_think\|>|\{)", m, re.S)
        if mt: think = mt.group(1).strip().replace("\n", " ")[:170]
        try:
            a, b = m.index("{"), m.rindex("}"); j = json.loads(m[a:b + 1])
        except Exception:
            out.append("  T%-3d[PARSEFAIL] think: %s" % (turn, think)); out.append(("        obs(%d): %s" % (len(obs), obs[:OW])).replace("\n", "|")); continue
        cmds = [c.get("keystrokes", "") for c in (j.get("commands") or []) if isinstance(c, dict)]
        tc = j.get("task_complete") is True
        flags = []
        for k in cmds:
            c = classify(k)
            if c == "edit":
                tg = edit_targets(k)
                if any((not TMPPAT.search(x)) and (re.search(SRCEXT + r"$", x) or x == "__PATCH__") for x in tg): flags.append("SRCEDIT")
                elif any(TMPPAT.search(x) for x in tg): flags.append("tmpedit")
                else: flags.append("edit?")
            elif c == "test": flags.append("TEST")
            elif c == "install": flags.append("INSTALL")
            if RESTORE.search(k): flags.append("RESTORE")
            if "reproduce_issue.py" in k: flags.append("REPRO")
            if "run_tests.sh" in k: flags.append("GRADER")
        cs = " ; ".join(k.strip().replace("\n", "\\n") for k in cmds)[:CW]
        fl = ",".join(sorted(set(flags))) or "-"
        out.append("  T%-3d[%-20s]%s%s %s" % (turn, fl, "TC" if tc else "  ", "REJ" if rej else "   ", cs))
        if think: out.append("        think: %s" % think)
        out.append(("        obs(%d): %s" % (len(obs), obs[:OW])).replace("\n", "|"))
    vr = r.get("verifier_result") or {}
    out.append("  VERIFIER reward=%s" % ((vr.get("rewards") or {}).get("reward")))
    so = os.path.join(att, "verifier", "test-stdout.txt")
    if os.path.exists(so):
        t = open(so, errors="replace").read(); out.append("  TEST-STDOUT tail: " + t[-900:].replace("\n", "|"))


def pick(atts, k):
    tagged = []
    for att in atts:
        r = load(os.path.join(att, "result.json")) or {}
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        tagged.append((att, "NULL" if rew is None else ("WIN" if rew >= 1 else "LOSS")))
    counts = {"WIN": sum(1 for _, t in tagged if t == "WIN"), "LOSS": sum(1 for _, t in tagged if t == "LOSS"), "NULL": sum(1 for _, t in tagged if t == "NULL")}
    chosen = []
    for want in ("WIN", "LOSS"):
        for att, t in tagged:
            if t == want and (att, t) not in chosen: chosen.append((att, t)); break
    for att, t in tagged:
        if len(chosen) >= k: break
        if (att, t) not in chosen and t != "NULL": chosen.append((att, t))
    return counts, chosen


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", default=E + "/ttwf_sample.tsv"); ap.add_argument("--out", default=E + "/ttwf_ana/digests")
    ap.add_argument("--tasks", default=None); ap.add_argument("--max-att", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = {}
    with open(a.sample) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("\t"))); rows[r["task"]] = r
    tasks = a.tasks.split(",") if a.tasks else sorted(rows)
    idx = ["# task stratum repo tt60k_succ | hd W/L/N | wf W/L/N | file"]
    for task in tasks:
        r = rows[task]; out = ["###### %s stratum=%s repo=%s base-probe(frozen harness, header prompt)=%s/8" % (task, r["stratum"], r["repo"], r["tt60k_succ"])]
        ip = os.path.join(ARMS[1][2], task, "instruction.md")
        if os.path.exists(ip): out.append("--- instruction.md (workflow arm; the header arm has only the first 4 lines + the issue)"); out.append(textwrap.indent(open(ip, errors="replace").read()[:4500], "  "))
        summ = []
        for short, prefix, tree in ARMS:
            atts = sorted(glob.glob("%s/%s_s[0-9]/%s_s[0-9]/trace_jobs/eval_sessions/*/%s__*/attempts/*" % (E, prefix, prefix, task)), key=lambda p: int(os.path.basename(p)))
            counts, chosen = pick(atts, a.max_att)
            out.append("\n\n######## ARM %s (%s): %dW/%dL/%dN over %d attempts; showing %d" % (short, "header prompt" if short == "hd" else "workflow prompt", counts["WIN"], counts["LOSS"], counts["NULL"], len(atts), len(chosen)))
            for att, tag in chosen: digest(att, "%s-%s" % (short, tag), out)
            summ.append("%dW/%dL/%dN" % (counts["WIN"], counts["LOSS"], counts["NULL"]))
        fn = "%s/%s__%s.txt" % (a.out, r["stratum"], task); open(fn, "w").write("\n".join(out) + "\n")
        idx.append("%s %s %s %s | %s | %s | %s" % (task, r["stratum"], r["repo"], r["tt60k_succ"], summ[0], summ[1], os.path.basename(fn))); print(idx[-1], flush=True)
    open(a.out + "/INDEX.md", "w").write("\n".join(idx) + "\n")


if __name__ == "__main__":
    main()
