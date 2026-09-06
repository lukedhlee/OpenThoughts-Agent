#!/usr/bin/env python3
"""wf_feat.py <trace_jobs dir or exp name>... --out <jsonl> [--arm NAME]
Per-trial behaviour features for the workflow-prompt re-probe, read straight from the (uncompacted) trace tree:
  trace_jobs/eval_sessions/<session>/<task>__<hash>/attempts/<k>/{agent/trajectory.json, verifier/reward.txt, exception.txt}
Reuses feat() from agent_scratch/behaviour/heldout/heldout_feat.py (turns, tokens, edits, pytest, grader, task_complete
timing, ...) and adds the workflow-specific ones:
  repro_written        wrote /testbed/reproduce_issue.py (or any scratch repro script) -- step 2 of the workflow prompt
  repro_runs           how many turns ran a repro/scratch script; repro_before_edit / repro_after_edit relative to src edits
  pytest_after_edit    ran the repo's tests after the last source edit (step 4)
  shadow_files         top-level /testbed/<stdlib-or-package name>.py written by the agent (the verifier hardening case)
  git_clean / git_restore   destructive git commands in /testbed
One record per trial = the last attempt that has a trajectory. Python 3.9 / stdlib (Jupiter login node, OMP_NUM_THREADS=1).
"""
import argparse, glob, json, os, re, sys, time
sys.path.insert(0, "/e/fscratch/reformo/lee27/experiments/agent_scratch/behaviour/heldout")
from heldout_feat import feat, parse_action, edit_targets, norm_path, PYTEST, REDIR, TEE  # type: ignore  # noqa: E402

STDLIB = set("""abc argparse array ast asyncio base64 binascii bisect builtins bz2 calendar cmd code codecs collections
concurrent configparser contextlib copy copyreg csv ctypes curses dataclasses datetime decimal difflib doctest email enum
errno fnmatch fractions functools gc getopt gettext glob gzip hashlib heapq hmac html http importlib inspect io ipaddress
itertools json keyword locale logging lzma math mmap multiprocessing numbers operator os parser pathlib pdb pickle pkgutil
platform pprint profile queue random re reprlib runpy sched secrets select selectors shlex shutil signal site socket ssl
stat statistics string struct subprocess sys sysconfig tarfile tempfile textwrap threading time timeit token tokenize
trace traceback types typing unicodedata unittest urllib uuid warnings weakref xml zipfile zlib
pytest numpy pandas six attr attrs yaml requests setuptools pkg_resources scipy matplotlib PIL sympy tornado aiohttp
scrapy pyramid datalad coverage Orange""".split())
REPRO = re.compile(r"reproduce_issue\.py")
REPRO_RUN = re.compile(r"python\d?(\.\d+)?\s+(-u\s+)?(/testbed/|\./)?reproduce_issue\.py")
SCRATCH_RUN = re.compile(r"python\d?(\.\d+)?\s+(-u\s+)?(/testbed/|\./)?(repro|reproduce|test_issue|check|verify|debug|edge_case)[A-Za-z0-9_]*\.py")
GIT_CLEAN = re.compile(r"\bgit\s+clean\b")
GIT_RESTORE = re.compile(r"\bgit\s+(checkout\s+(--\s+)?(\.|/testbed|[A-Za-z0-9_./-]+\.py)|restore\b|stash\b|reset\s+--hard)")
TOPLEVEL_PY = re.compile(r"(?:^|[\s>|'\"=])(?:/testbed/|\./)?([A-Za-z_][A-Za-z0-9_]*)\.py(?=[\s'\"|;&]|$)")


def write_targets(c):
    """paths the command writes to (heredoc / redirect / tee / cp / mv targets)"""
    out = []
    for m in REDIR.finditer(c):
        p = m.group(1)
        if not p.startswith("&") and not p.startswith("/dev/"): out.append(norm_path(p))
    for m in TEE.finditer(c): out.append(norm_path(m.group(1)))
    m = re.match(r"^\s*(cp|mv)\s+(?:-\S+\s+)*\S+\s+(\S+)", c)
    if m: out.append(norm_path(m.group(2)))
    return out


def extra(d):
    ag = [s for s in (d.get("steps") or []) if s.get("source") == "agent"]
    src_edit_turns, repro_runs, scratch_runs, pytest_turns = [], [], [], []
    repro_written = False; shadow = set(); git_clean = 0; git_restore = 0
    script_targets = {}
    for i, s in enumerate(ag):
        m = s.get("message") or ""
        body = m.split("<|end_think|>", 1)[1] if "<|end_think|>" in m else m
        d2, _ = parse_action(body)
        if d2 is None: continue
        cmds = [str(c.get("keystrokes") or "") for c in (d2.get("commands") or []) if isinstance(c, dict)]
        for c in cmds:
            if PYTEST.search(c): pytest_turns.append(i + 1)
            if REPRO_RUN.search(c): repro_runs.append(i + 1)
            elif SCRATCH_RUN.search(c): scratch_runs.append(i + 1)
            if GIT_CLEAN.search(c): git_clean += 1
            if GIT_RESTORE.search(c): git_restore += 1
            wt = write_targets(c)
            if any(REPRO.search(p) for p in wt) or (REPRO.search(c) and re.search(r"(cat\s*>|>\s*|tee\s|<<)", c)): repro_written = True
            for p in wt:
                rel = p[len("/testbed/"):] if p.startswith("/testbed/") else p
                if "/" not in rel and rel.endswith(".py") and rel[:-3] in STDLIB: shadow.add(rel)
            tg, scripts = edit_targets(c, script_targets)
            for sp, targets in scripts: script_targets[sp] = targets
            if any(k == "src" for k, p in tg): src_edit_turns.append(i + 1)
    fe = src_edit_turns[0] if src_edit_turns else None; le = src_edit_turns[-1] if src_edit_turns else None
    runs = sorted(set(repro_runs + scratch_runs))
    return dict(repro_written=repro_written, repro_runs=len(repro_runs), scratch_runs=len(scratch_runs),
                repro_before_edit=bool(runs) and (fe is None or runs[0] < fe),
                repro_after_edit=bool(runs) and le is not None and runs[-1] > le,
                pytest_after_edit=bool(pytest_turns) and le is not None and pytest_turns[-1] > le,
                pytest_before_edit=bool(pytest_turns) and (fe is None or pytest_turns[0] < fe),
                shadow_files=sorted(shadow), git_clean=git_clean, git_restore=git_restore,
                first_src_edit_x=fe, last_src_edit=le)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("trees", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--arm", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); E = "/e/fscratch/reformo/lee27/experiments"
    t0 = time.time(); n = 0; errs = 0; out = open(a.out, "w")
    for tree in a.trees:
        tj = tree if os.path.isdir(tree) and tree.endswith("trace_jobs") else "%s/%s/%s/trace_jobs" % (E, tree, tree)
        shard = os.path.basename(tree.rstrip("/")) if not tree.endswith("trace_jobs") else tree.split("/")[-3]
        trials = sorted(glob.glob(tj + "/eval_sessions/*/*__*"))
        for td in trials:
            task, h = os.path.basename(td).split("__", 1)
            atts = sorted(glob.glob(td + "/attempts/*"), key=lambda p: int(os.path.basename(p)))
            have = [p for p in atts if os.path.exists(p + "/agent/trajectory.json")]
            if not have: continue
            p = have[-1]; rec = dict(task=task, trial=os.path.basename(td), shard=shard, arm=a.arm or shard.split("_")[0], att=int(os.path.basename(p)), n_att=len(atts))
            try:
                rw = open(p + "/verifier/reward.txt").read().strip(); rec["reward"] = float(rw)
            except Exception: rec["reward"] = None
            rec["exc"] = None
            if os.path.exists(p + "/exception.txt"):
                lines = [l for l in open(p + "/exception.txt", errors="replace").read().splitlines() if l.strip()]
                rec["exc"] = (lines[-1] if lines else "").split(":")[0].split(".")[-1][:60]
            rec["ctx"] = rec["exc"] == "ContextLengthExceededError"
            try:
                d = json.load(open(p + "/agent/trajectory.json")); rec.update(feat(d)); rec.update(extra(d))
            except Exception as e:
                rec["err"] = str(e)[:200]; errs += 1
            out.write(json.dumps(rec) + "\n"); n += 1
            if n % 200 == 0: sys.stderr.write("%d trials %.0fs\n" % (n, time.time() - t0)); sys.stderr.flush()
            if a.limit and n >= a.limit: break
    out.close(); sys.stderr.write("done: %d trials, %d errs, %.0fs -> %s\n" % (n, errs, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
