#!/usr/bin/env python3
"""required_tests.json per task from an oracle gate pass whose verifier saved the whole short summary.

For every task, the latest oracle trial that scored 1: read attempts/<n>/verifier/test_summary.txt (written by the test.sh
that tt_daytona_tree_v3.py emits), keep the staged test ids (from r2e_tests/ on) whose name is in test_info's expected map,
with the status the oracle run reported. The grader (test_state_required.py) then requires each of them to run with that status.

  derive_required_tests.py --jobs <jobs dir> --job-glob 'oracle_*_req1' --tree <task tree> --out required_tests.json.gz

Writes <out> ({task: {id: status}}) and <out>.tsv (task, expected, staged ids seen, required, status mismatches). Tasks with
no scoring oracle trial or no staged id in the summary are left out (they grade as before) and counted on stdout.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

ANSI = re.compile("\x1b" + r"\[\d+m")
STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL")


def norm(key: str) -> str:
    k = ANSI.sub("", key).split(" - ")[0].strip()
    return ".".join(k.split("::")[1:]) if "::" in k else k


def staged(summary: str) -> dict[str, str]:
    out = {}
    for line in summary.split("short test summary info", 1)[-1].splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        for st in STATUSES:
            if st in line:
                k = ANSI.sub("", line.split(st, 1)[1] if line.startswith(st) else line).split(" - ")[0].strip()
                i = k.rfind("r2e_tests/")
                if i >= 0 and "::" in k:
                    out[k[i:]] = st
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--job-glob", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best: dict[str, tuple[str, Path]] = {}
    for job in sorted(Path(a.jobs).glob(a.job_glob)):
        if not job.is_dir():
            continue
        for trial in job.iterdir():
            rj = trial / "result.json"
            if not rj.is_file():
                continue
            r = json.load(open(rj))
            rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if rew != 1:
                continue
            atts = sorted((trial / "attempts").glob("*/verifier/test_summary.txt"))
            if not atts:
                continue
            fin = r.get("finished_at") or ""
            if r["task_name"] not in best or fin > best[r["task_name"]][0]:
                best[r["task_name"]] = (fin, atts[-1])
    req: dict[str, dict[str, str]] = {}
    rows, empty = [], []
    for task, (_, summ) in sorted(best.items()):
        info = json.load(open(Path(a.tree) / task / "tests" / "test_info.json"))
        exp = info.get("expected_output_json", "{}")
        exp = {norm(k): v for k, v in (json.loads(exp) if isinstance(exp, str) else exp).items()}
        seen = staged(summ.read_text(errors="replace"))
        # status = what the oracle run reported: equal to the expected status except where two staged files define a test
        # of the same name (6 tasks), which the name-keyed expected map cannot tell apart
        r = {k: seen[k] for k in seen if norm(k) in exp}
        mism = sum(1 for k in r if seen[k] != exp[norm(k)])
        rows.append(f"{task}\t{len(exp)}\t{len(seen)}\t{len(r)}\t{mism}")
        if r:
            req[task] = r
        else:
            empty.append(task)
    with gzip.open(a.out, "wt") as f:
        json.dump(req, f, sort_keys=True)
    Path(a.out + ".tsv").write_text("task\texpected\tstaged_seen\trequired\tstatus_mismatch\n" + "\n".join(rows) + "\n")
    n = sum(len(v) for v in req.values())
    print(f"oracle-passing tasks with a summary: {len(best)}; with required tests: {len(req)} ({n} ids); "
          f"none staged: {len(empty)} {empty[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
