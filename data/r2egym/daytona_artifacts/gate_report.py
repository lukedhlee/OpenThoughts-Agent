#!/usr/bin/env python3
"""Final readout of the Daytona gate (gate_run.py job dirs): per-repo pass counts vs the JSC gate, sandbox-start timeouts and
setup times by concurrency, the allowlist (oracle 1 and nop 0), and the failures split into verifier failures (scored, wrong
reward) and infra failures (never scored).

  python3 gate_report.py --jobs <jobs_dir> --map tasktrove_v3_upstream_map.tsv --jsc-fails jsc_gate_fails.tsv \
      --tasks allow_r2egym_all3035.txt --out <dir> [--archive <resume_archive>]

Each task counts once per phase: its latest scored trial, else its latest unscored one. `--archive` is where the unscored trial
dirs were moved (as <archive>/<job>/<trial>/) before `harbor jobs resume` re-ran them; their attempts still count toward the
start-timeout rate of the job's original concurrency, and the re-run trials are counted under a separate "resume" level.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics as st
from datetime import datetime

SETUP_RE = re.compile(r"setup_files/setup\.sh exited (-?\d+) after ([\d.]+)s")


def signature(tdir):
    """A short failure kind from the last attempt's verifier output: the grader's assertion plus the first telling error in
    the R2E-Gym test run (digits folded so like failures group)."""
    atts = sorted(glob.glob(os.path.join(tdir, "attempts", "*")))
    try:
        log = open(os.path.join(atts[-1], "verifier", "test-stdout.txt"), errors="replace").read() if atts else ""
    except OSError:
        return "no verifier output"
    hint = ""
    for pat in (r"unrecognized arguments: (\S+)", r"(ModuleNotFoundError: No module named \S+)", r"(ImportError: [^\n]{0,80})",
                r"(AttributeError: [^\n]{0,80})", r"(SyntaxError: [^\n]{0,60})", r"(Timeout[^\n]{0,60})"):
        m = re.search(pat, log)
        if m:
            hint = ("unrecognized pytest arg " + m.group(1)) if "unrecognized" in pat else m.group(1)
            break
    ms = re.findall(r"E\s+AssertionError: ([^\n]*)", log)   # the grader (test_state.py) runs last
    head = re.split(r"[.(:]", re.sub(r"\d+", "N", ms[-1]))[0][:60].strip() if ms else ("no grader assertion" if log else "no verifier output")
    return f"{head}{' | ' + hint if hint else ''}"


def ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def dur(span):
    span = span or {}
    if span.get("started_at") and span.get("finished_at"):
        return (ts(span["finished_at"]) - ts(span["started_at"])).total_seconds()
    return None


def reward_of(r):
    return ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def exc_of(r):
    e = r.get("exception_info") or {}
    return e.get("exception_type") if isinstance(e, dict) else None


def load_json(f):
    try:
        return json.load(open(f))
    except Exception:  # noqa: BLE001
        return None


def pct(xs, p):
    return round(sorted(xs)[max(0, int(p * len(xs) + 0.5) - 1)], 1) if xs else None


def scan_trial(tdir):
    """(final result row, [attempt rows]) for one trial dir."""
    r = load_json(os.path.join(tdir, "result.json"))
    if r is None:
        return None, []
    row = {"task": r["task_name"], "reward": reward_of(r), "exc": exc_of(r), "finished": r.get("finished_at") or "",
           "dir": tdir, "stdout": ((r.get("verifier_result") or {}).get("stdout") or "")[-600:]}
    atts = []
    for ad in sorted(glob.glob(os.path.join(tdir, "attempts", "*"))):
        a = load_json(os.path.join(ad, "result.json"))
        if a is None:
            continue
        setup = None
        try:
            m = SETUP_RE.search(open(os.path.join(ad, "trial.log"), errors="replace").read())
            if m:
                setup = float(m.group(2))
        except OSError:
            pass
        atts.append({"first": not atts, "exc": exc_of(a), "start_s": dur(a.get("environment_setup")) if a.get("agent_setup") else None,
                     "setup_s": setup})
    if row["reward"] is None and not row["stdout"] and atts:
        # harbor's top-level result mirrors the last attempt; keep the verifier tail from there when present
        last = load_json(os.path.join(sorted(glob.glob(os.path.join(tdir, "attempts", "*")))[-1], "result.json")) or {}
        row["stdout"] = ((last.get("verifier_result") or {}).get("stdout") or "")[-600:]
    return row, atts


def better(new, old):
    """Latest scored trial wins; an unscored trial only replaces another unscored one that finished earlier."""
    if old is None:
        return True
    if (new["reward"] is None) != (old["reward"] is None):
        return new["reward"] is not None
    return new["finished"] > old["finished"]


def level_stats(atts, trials):
    n = len(atts)
    to = sum(1 for a in atts if a["exc"] == "EnvironmentStartTimeoutError")
    start = [a["start_s"] for a in atts if a["start_s"] is not None]
    setup = [a["setup_s"] for a in atts if a["setup_s"] is not None]
    first_to = sum(1 for a in atts if a["first"] and a["exc"] == "EnvironmentStartTimeoutError")
    return {"trials": trials, "attempts": n, "start_timeouts": to, "start_timeout_rate": round(to / n, 3) if n else None,
            "first_attempt_timeout_rate": round(first_to / trials, 3) if trials else None,
            "other_exc": dict(collections.Counter(a["exc"] for a in atts if a["exc"] and a["exc"] != "EnvironmentStartTimeoutError")),
            "sandbox_start_s": {"p50": pct(start, .5), "p90": pct(start, .9), "max": round(max(start), 1)} if start else None,
            "setup_sh_s": {"p50": pct(setup, .5), "p90": pct(setup, .9), "max": round(max(setup), 1)} if setup else None}


def load_phase(jobs, archive, phase):
    best, levels = {}, collections.defaultdict(lambda: {"atts": [], "trials": 0})
    for jd in sorted(glob.glob(os.path.join(jobs, f"{phase}_shard*"))):
        if not os.path.isdir(jd):
            continue
        job = os.path.basename(jd)
        conc = int(job.split("_c")[-1].split("_")[0])
        arch = os.path.join(archive, job) if archive else None
        arch_dirs = sorted(glob.glob(os.path.join(arch, "r2egym*"))) if arch and os.path.isdir(arch) else []
        resumed_tasks = {os.path.basename(d).split("__")[0] for d in arch_dirs}
        for tdir, origin in [(d, "job") for d in sorted(glob.glob(os.path.join(jd, "r2egym*")))] + [(d, "archive") for d in arch_dirs]:
            row, atts = scan_trial(tdir)
            if row is None:
                continue
            key = f"resume_{phase}" if origin == "job" and row["task"] in resumed_tasks else f"{phase}_c{conc}"
            levels[key]["atts"] += atts
            levels[key]["trials"] += 1
            row["job"] = job
            if better(row, best.get(row["task"])):
                best[row["task"]] = row
    return best, {k: level_stats(v["atts"], v["trials"]) for k, v in levels.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--jsc-fails", default=None)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", default=None, help="dir holding unscored trial dirs moved out before a resume")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    repo_of = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}
    tasks = [l.strip() for l in open(a.tasks) if l.strip()]
    jsc_fail = {}
    if a.jsc_fails:
        for line in open(a.jsc_fails):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                jsc_fail[p[0]] = p[2]
    oracle, olv = load_phase(a.jobs, a.archive, "oracle")
    nop, nlv = load_phase(a.jobs, a.archive, "nop")

    def show(x):
        return "missing" if x is None else (x["reward"] if x["reward"] is not None else f"ERR:{x['exc']}")

    ok, vfail, infra, missing = [], {}, {}, []
    for t in tasks:
        o, n = oracle.get(t), nop.get(t)
        if o is None or n is None:
            missing.append(t)
        elif o["reward"] == 1.0 and n["reward"] == 0.0:
            ok.append(t)
        elif (o["reward"] is not None and o["reward"] != 1.0) or (n["reward"] is not None and n["reward"] != 0.0):
            vfail[t] = f"oracle={show(o)} nop={show(n)}"   # a scored wrong reward is a real failure even if the other phase errored
        else:
            infra[t] = f"oracle={show(o)} nop={show(n)}"
    okset = set(ok)
    sigs = {}
    by_repo = collections.defaultdict(lambda: {"tasks": 0, "pass": 0, "verifier_fail": 0, "infra": 0, "missing": 0,
                                               "jsc_pass": 0, "fail_kinds": collections.Counter()})
    for t in tasks:
        r = by_repo[repo_of.get(t, "?")]
        r["tasks"] += 1
        r["pass"] += t in okset
        r["verifier_fail"] += t in vfail
        r["infra"] += t in infra
        r["missing"] += t in missing
        r["jsc_pass"] += t not in jsc_fail
        if t in vfail:
            o = oracle[t]
            bad = o if o["reward"] is not None and o["reward"] != 1.0 else nop[t]
            sigs[t] = ("oracle: " if bad is o else "nop: ") + signature(bad["dir"])
            r["fail_kinds"][sigs[t]] += 1
        elif t in infra:
            r["fail_kinds"]["infra"] += 1
    for r in by_repo.values():
        r["fail_kinds"] = dict(r["fail_kinds"])
    scored = [t for t in tasks if t not in missing and t not in infra]
    agree = sum(1 for t in scored if (t in okset) == (t not in jsc_fail))
    with open(os.path.join(a.out, "allowlist_r2egym_daytona_v3.txt"), "w") as f:
        f.write("\n".join(sorted(ok)) + "\n")
    with open(os.path.join(a.out, "gate_fail.tsv"), "w") as f:
        for kind, d in (("verifier", vfail), ("infra", infra)):
            for t, why in sorted(d.items()):
                o, n = oracle[t], nop[t]
                src = o if o["reward"] != 1.0 else n
                tail = (src["stdout"] or "").replace("\n", " ⏎ ")[-300:]
                f.write(f"{t}\t{repo_of.get(t, '?')}\t{kind}\t{why}\tjsc={jsc_fail.get(t, 'pass')}\t{src['job']}\t{sigs.get(t, '')}\t{tail}\n")
    summary = {
        "tasks": len(tasks), "pass": len(ok), "verifier_fail": len(vfail), "infra_fail": len(infra), "missing": len(missing),
        "agree_with_jsc_on_scored": agree, "scored": len(scored),
        "pass_on_daytona_not_jsc": sorted(t for t in ok if t in jsc_fail),
        "fail_on_daytona_pass_on_jsc": sorted(t for t in vfail if t not in jsc_fail),
        "infra_tasks": infra, "missing_tasks": missing,
        "by_repo": dict(sorted(by_repo.items())), "levels": {**dict(sorted(olv.items())), **dict(sorted(nlv.items()))},
    }
    json.dump(summary, open(os.path.join(a.out, "gate_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_repo", "levels", "missing_tasks")}, indent=1))
    print(f"{'repo':12s} {'tasks':>5s} {'pass':>5s} {'vfail':>5s} {'infra':>5s} {'miss':>4s} {'jsc':>5s}  kinds")
    for k, r in summary["by_repo"].items():
        print(f"{k:12s} {r['tasks']:5d} {r['pass']:5d} {r['verifier_fail']:5d} {r['infra']:5d} {r['missing']:4d} {r['jsc_pass']:5d}  {r['fail_kinds']}")
    for k, v in summary["levels"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
