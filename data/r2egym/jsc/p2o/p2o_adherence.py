#!/usr/bin/env python3
"""p2o_adherence.py <trace roots...> --out <tsv> [--limit N]
Per-trial adherence detectors for the five P2O guidance blocks, read from the trajectory command stream
(trace_jobs/eval_sessions/<session>/<task>__<hash>/attempts/<k>/agent/trajectory.json), plus think-share by turn.
Task dir names are <task>-p<arm> (wave 0) or plain <task> (the 09-06 control, arm = ctl). Python 3.9 / stdlib.

Detectors (per trial, last attempt with a trajectory):
  A  a_replace_script: a python -c / heredoc that reads a file, .replace(...) once, writes it back
     a_check_after_edit: py_compile or git diff run in a turn after a source edit
     a_sed_patch: sed -i / patch applied to a repo source file (violation)
     adhere_A = a_replace_script and a_check_after_edit and not a_sed_patch
  B  first_edit_turn (turn index of the first source edit), b_repeat (an identical explore command run twice),
     adhere_B = first_edit_turn <= 10 and not b_repeat
  C  c_repro_tmp (wrote a /tmp/*.py containing assert), c_run_before (ran it before the first source edit),
     c_run_after (ran it after the last source edit), c_test_edit (edited a file under tests/ or test_*.py)
     adhere_C = c_repro_tmp and c_run_before and c_run_after and not c_test_edit
  D  d_grep_first (a grep/rg for an identifier taken from the issue text before any file was opened),
     d_hypothesis (first agent step's plan/analysis names a cause: 'root cause', 'hypothes', 'likely', 'because')
     adhere_D = d_grep_first and d_hypothesis
  E  adhere_E = adhere_A and adhere_B and adhere_C and adhere_D
Also: turns, reward, ctx_death, declared_done, think_share (fraction of agent turns whose message opens a think span),
think_by_turn (per-turn 0/1 list, for the per-turn profile).
"""
import argparse, collections, csv, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+", help="trace_jobs dirs (or their parents); globbed for eval_sessions/*/*__*/attempts/*")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

SRC_EXT = (".py", ".pyx", ".c", ".h", ".rs", ".js", ".ts", ".go", ".java")
EXPLORE = re.compile(r"^\s*(grep|rg|find|sed\s+-n|cat|head|tail|ls|awk)\b")
SED_I = re.compile(r"\bsed\s+(-[a-zA-Z]*i|--in-place)")
PATCH = re.compile(r"(^|\s|\|)\s*(patch|git\s+apply)\b")
PYCOMPILE = re.compile(r"py_compile|ast\.parse|python[0-9.]*\s+-m\s+compileall")
GITDIFF = re.compile(r"\bgit\s+diff\b")
REPLACE_SCRIPT = re.compile(r"\.replace\(", re.S)
OPEN_WRITE = re.compile(r"open\([^)]*['\"]w['\"]", re.S)
HEREDOC_TARGET = re.compile(r"(?:cat|tee)\s*(?:>|>>|<<\s*['\"]?\w+['\"]?\s*>)\s*([^\s;&|]+)")
REDIR_TARGET = re.compile(r">\s*([^\s;&|]+\.py)\b")
PYWRITE_TARGET = re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]['\"]")
RUN_PY = re.compile(r"python[0-9.]*\s+(/tmp/[^\s;&|]+\.py)")
TESTS_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$")
HYP = re.compile(r"root cause|hypothes|likely|because|caused by|the bug is|the issue is", re.I)
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]{3,})`|\b([A-Z][a-z]+[A-Z][A-Za-z]+)\b|\b([a-z]+_[a-z_]+)\b")


def parse_response(msg):
    """terminus-2 JSON response -> (analysis, plan, [keystrokes]); tolerant of think spans and extra text."""
    m = msg
    if "<|end_think|>" in m:
        m = m.split("<|end_think|>", 1)[1]
    i, j = m.find("{"), m.rfind("}")
    if i < 0 or j < 0:
        return "", "", [], False
    try:
        d = json.loads(m[i:j + 1])
    except Exception:
        return "", "", [], False
    cmds = [c.get("keystrokes", "") for c in d.get("commands", []) if isinstance(c, dict)]
    return d.get("analysis", "") or "", d.get("plan", "") or "", cmds, bool(d.get("task_complete"))


def edited_files(cmd):
    """files written by this command (heredoc / redirect / python open-write / sed -i / patch targets)"""
    out = set()
    for rx in (HEREDOC_TARGET, REDIR_TARGET, PYWRITE_TARGET):
        out |= {m.group(1) for m in rx.finditer(cmd)}
    if SED_I.search(cmd):
        out |= set(re.findall(r"(\S+\.(?:py|pyx|c|h|rs|js|ts|go|java))\b", cmd))
    return out


def is_src(path):
    return path.endswith(SRC_EXT) and not path.startswith("/tmp") and "reproduce" not in path


def features(tj, issue_text):
    d = json.load(open(tj))
    steps = [s for s in d["steps"] if s.get("source") == "agent"]
    f = collections.OrderedDict()  # type: collections.OrderedDict[str, object]
    f["turns"] = len(steps)
    think = [1 if (s.get("message") or "").lstrip().startswith("<|start_think|>") else 0 for s in steps]
    f["think_share"] = round(sum(think) / len(think), 3) if think else None
    f["think_by_turn"] = "".join(map(str, think))
    idents = set()
    for m in IDENT.finditer(issue_text or ""):
        idents.add(next(g for g in m.groups() if g))
    first_edit = last_edit = None; opened_file = False
    a_replace = a_check = a_sedpatch = False
    seen_explore = set(); b_repeat = False
    repro_files = set(); c_before = c_after = False; c_test_edit = False
    d_grep_first = False; d_hyp = False; declared = False
    for t, s in enumerate(steps, 1):
        analysis, plan, cmds, tc = parse_response(s.get("message") or "")
        declared = declared or tc
        if t == 1 and HYP.search(analysis + " " + plan):
            d_hyp = True
        for cmd in cmds:
            c = cmd.strip()
            if not c:
                continue
            edits = edited_files(c)
            src_edits = {p for p in edits if is_src(p)}
            if src_edits:
                if first_edit is None:
                    first_edit = t
                last_edit = t
                if REPLACE_SCRIPT.search(c) and OPEN_WRITE.search(c):
                    a_replace = True
                if SED_I.search(c) or PATCH.search(c):
                    a_sedpatch = True
                if any(TESTS_PATH.search(p) for p in src_edits):
                    c_test_edit = True
            if first_edit is not None and t > first_edit and (PYCOMPILE.search(c) or GITDIFF.search(c)):
                a_check = True
            tmp_py = {p for p in edits if p.startswith("/tmp") and p.endswith(".py")}
            if tmp_py and "assert" in c:
                repro_files |= tmp_py
            for m in RUN_PY.finditer(c):
                if m.group(1) in repro_files:
                    if first_edit is None:
                        c_before = True
                    if last_edit is not None and t > last_edit:
                        c_after = True
            if EXPLORE.match(c):
                key = re.sub(r"\s+", " ", c)[:200]
                if key in seen_explore:
                    b_repeat = True
                seen_explore.add(key)
                if not opened_file and re.match(r"^\s*(grep|rg)\b", c) and any(i in c for i in idents):
                    d_grep_first = True
                if re.match(r"^\s*(cat|sed\s+-n|head|tail)\b", c):
                    opened_file = True
        # c_after must be re-evaluated at the end: a repro run after the last edit only counts if no later edit happened
    # recompute c_after strictly: last repro run turn > last edit turn
    f["reward"] = None
    try:
        f["reward"] = float(open(os.path.join(os.path.dirname(os.path.dirname(tj)), "verifier", "reward.txt")).read().strip())
    except Exception:
        pass
    exc = os.path.join(os.path.dirname(os.path.dirname(tj)), "exception.txt")
    f["ctx_death"] = int(os.path.exists(exc) and "ContextLengthExceeded" in open(exc).read())
    f["declared_done"] = int(declared)
    f["first_edit_turn"] = first_edit
    f["a_replace_script"] = int(a_replace); f["a_check_after_edit"] = int(a_check); f["a_sed_patch"] = int(a_sedpatch)
    f["adhere_A"] = int(a_replace and a_check and not a_sedpatch)
    f["b_repeat"] = int(b_repeat)
    f["adhere_B"] = int(first_edit is not None and first_edit <= 10 and not b_repeat)
    f["c_repro_tmp"] = int(bool(repro_files)); f["c_run_before"] = int(c_before); f["c_run_after"] = int(c_after); f["c_test_edit"] = int(c_test_edit)
    f["adhere_C"] = int(bool(repro_files) and c_before and c_after and not c_test_edit)
    f["d_grep_first"] = int(d_grep_first); f["d_hypothesis"] = int(d_hyp)
    f["adhere_D"] = int(d_grep_first and d_hyp)
    f["adhere_E"] = int(f["adhere_A"] and f["adhere_B"] and f["adhere_C"] and f["adhere_D"])
    return f


recs = []
for root in a.roots:
    for td in sorted(glob.glob(root + "/**/eval_sessions/*/*__*", recursive=True)):
        name = os.path.basename(td).split("__", 1)[0]
        task, arm = (name.rsplit("-p", 1) if re.search(r"-p(ctl|[A-E])$", name) else (name, "ctl"))
        atts = sorted(glob.glob(td + "/attempts/*/agent/trajectory.json"))
        if not atts:
            continue
        tj = atts[-1]
        try:
            issue = open(os.path.join(td, "..", "..", "..", "..", "instruction.md")).read()
        except Exception:
            issue = ""
        try:
            f = features(tj, issue)
        except Exception as e:
            print("skip", td, e, file=sys.stderr); continue
        rec = collections.OrderedDict()  # type: collections.OrderedDict[str, object]
        rec["task"] = task; rec["arm"] = arm; rec["trial"] = os.path.basename(td); rec.update(f); recs.append(rec)
        if a.limit and len(recs) >= a.limit:
            break
    if a.limit and len(recs) >= a.limit:
        break
if not recs:
    sys.exit("no trajectories found")
with open(a.out, "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(recs)
by = collections.defaultdict(list)
for r in recs:
    by[r["arm"]].append(r)
print("arm\tn\tpass\tctx\tdone\tthink\tadhA\tadhB\tadhC\tadhD\tadhE\tfirst_edit_med")
for arm in sorted(by):
    rs = by[arm]; n = len(rs)
    def m(k):
        return sum(float(r[k] or 0) for r in rs) / n  # type: ignore[arg-type]
    fe = sorted(int(r["first_edit_turn"]) for r in rs if r["first_edit_turn"])  # type: ignore[arg-type]
    print("%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s" % (
        arm, n, m("reward"), m("ctx_death"), m("declared_done"), m("think_share"),
        m("adhere_A"), m("adhere_B"), m("adhere_C"), m("adhere_D"), m("adhere_E"), fe[len(fe) // 2] if fe else "-"))
print("wrote", a.out, len(recs))
