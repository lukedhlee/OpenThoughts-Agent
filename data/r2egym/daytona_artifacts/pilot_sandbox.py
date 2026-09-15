#!/usr/bin/env python3
"""Time a Daytona sandbox built from the artifact snapshot: start latency, per-commit setup, and the test gate.

Setup per task (what a harbor `setup_files/setup.sh` would do at rollout):
  git checkout -f <base_commit>; git clean -fdxq; untar <task>.venv.tar.zst + <task>.delta.tar.zst (mtimes preserved,
  so setuptools never recompiles); remove files install.sh had deleted; import check.
Gate per task: pytest on /r2e_tests (from <task>.tests.tar.zst) pristine (expect the expected_output tests NOT all
PASSED) and after copying the oracle's patched files over /testbed (expect all PASSED).

Usage (harbor venv, Mac):
  /Users/lukedhlee/harbor/.venv/bin/python pilot_sandbox.py --snapshot harbor__pilot-pandas-artifacts__snapshot \
      --summary <summary.tsv> --gate-dir <dir with <task>/{tests/test_info.json,solution/patched_files}> \
      --gate-tasks r2egym-v1-03043,r2egym-v1-03407,r2egym-v1-04265 --out report.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path


def load_secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in open(os.path.expanduser("~/.config/otagent/secrets.env")):
        line = line.strip()
        if line.startswith((f"{name}=", f"export {name}=")):
            return line.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'").split()[0]
    raise SystemExit(f"{name} not found")


SETUP = r"""
set -e
cd /testbed
T={task}; BASE={base}
t0=$(date +%s%N)
rm -rf .venv /r2e_tests/* 2>/dev/null || true
git -c safe.directory='*' checkout -q -f "$BASE"
git -c safe.directory='*' clean -fdxq
t1=$(date +%s%N)
tar --zstd -xpf /opt/artifacts/$T.venv.tar.zst
t2=$(date +%s%N)
tar --zstd -xpf /opt/artifacts/$T.delta.tar.zst
if [ -f /opt/artifacts/manifests/$T.deleted.list ]; then while read -r p; do [ -n "$p" ] && rm -f -- "$p"; done < /opt/artifacts/manifests/$T.deleted.list; fi
(cd / && tar --zstd -xpf /opt/artifacts/$T.tests.tar.zst)
t3=$(date +%s%N)
.venv/bin/python -c "import pandas, sys; print('pandas', pandas.__version__, 'python', sys.version.split()[0])"
t4=$(date +%s%N)
echo "TIMING checkout_ms=$(( (t1-t0)/1000000 )) venv_ms=$(( (t2-t1)/1000000 )) delta_ms=$(( (t3-t2)/1000000 )) import_ms=$(( (t4-t3)/1000000 )) total_ms=$(( (t4-t0)/1000000 ))"
echo "PYTHON $(readlink -f .venv/bin/python)"
echo "SO $(find . -name '*.so' -newer /opt/artifacts/$T.delta.tar.zst | wc -l) newer-than-tarball (0 = mtimes preserved)"
"""

PYTEST = r"""
cd /testbed
rm -rf /logs/test_output.log; mkdir -p /logs
timeout 600 .venv/bin/python -m pytest /r2e_tests -rA -p no:cacheprovider -q -o addopts= > /logs/test_output.log 2>&1; grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL) ' /logs/test_output.log; tail -1 /logs/test_output.log
"""


def parse_statuses(log: str) -> dict:
    out = {}
    for line in log.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL)\s+(.*)$", line.strip())
        if not m:
            continue
        status, tid = m.group(1), m.group(2).split(" - ")[0].strip()
        if "::" in tid:
            tid = ".".join(tid.split("::")[1:])
        out[tid] = status
    return out


def gate_verdict(statuses: dict, expected: dict) -> tuple[bool, str]:
    missing = [k for k in expected if k not in statuses]
    bad = [k for k, v in expected.items() if statuses.get(k) != v]
    return (not bad and not missing), f"expected={len(expected)} matched={len(expected) - len(bad)} missing={len(missing)} bad={bad[:3]}"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--summary", required=True, help="harvest summary.tsv (task, image, head, ...)")
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--gate-tasks", required=True)
    ap.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    ap.add_argument("--out", default="pilot_report.json")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams, DaytonaConfig, FileUpload

    rows = [l.rstrip("\n").split("\t") for l in open(a.summary) if l.strip() and not l.startswith("task\t")]
    tasks = [(r[0], r[2]) for r in rows]  # (task, head = base commit checked out in the image)
    gate_tasks = a.gate_tasks.split(",")
    report = {"snapshot": a.snapshot, "setup": {}, "gate": {}}
    key = load_secret(a.api_key_env)
    async with AsyncDaytona(DaytonaConfig(api_key=key)) as d:
        t0 = time.time()
        sb = await d.create(
            CreateSandboxFromSnapshotParams(snapshot=a.snapshot, labels={"harbor.instance": "pilot-pandas-artifacts"}, auto_stop_interval=60),
            timeout=600,
        )
        report["start_seconds"] = round(time.time() - t0, 1)
        print(f"sandbox {sb.id} started in {report['start_seconds']}s state={sb.state}")
        try:
            r = await sb.process.exec("cat /etc/os-release | head -2; ls /opt/artifacts | head -40; du -sh /opt/artifacts; ls /root/.local/share/uv/python; df -h / | tail -1", timeout=120)
            print(r.result)
            for task, base in tasks:
                r = await sb.process.exec(SETUP.format(task=task, base=base), timeout=900)
                tail = "\n".join(r.result.strip().splitlines()[-4:])
                print(f"[{task}] exit={r.exit_code}\n{tail}")
                m = re.search(r"TIMING (.*)", r.result)
                report["setup"][task] = {"exit": r.exit_code, "timing": m.group(1) if m else None, "tail": tail}
                if task in gate_tasks and r.exit_code == 0:
                    gd = Path(a.gate_dir) / task
                    expected = json.loads(json.load(open(gd / "tests/test_info.json"))["expected_output_json"])
                    r1 = await sb.process.exec(PYTEST, timeout=900)
                    s1 = parse_statuses(r1.result)
                    ok1, why1 = gate_verdict(s1, expected)
                    # oracle: overwrite patched files
                    files = []
                    for p in (gd / "solution/patched_files").rglob("*"):
                        if p.is_file():
                            files.append(FileUpload(source=p.read_bytes(), destination="/testbed/" + str(p.relative_to(gd / "solution/patched_files"))))
                    await sb.fs.upload_files(files)
                    r2 = await sb.process.exec(PYTEST, timeout=900)
                    s2 = parse_statuses(r2.result)
                    ok2, why2 = gate_verdict(s2, expected)
                    report["gate"][task] = {"pristine_all_pass": ok1, "pristine": why1, "oracle_all_pass": ok2, "oracle": why2,
                                           "pristine_tail": r1.result.strip().splitlines()[-3:], "oracle_tail": r2.result.strip().splitlines()[-3:]}
                    print(f"  GATE {task}: pristine all-pass={ok1} ({why1}) | oracle all-pass={ok2} ({why2})")
        finally:
            if not a.keep:
                await sb.delete()
                print("sandbox deleted")
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps(report["setup"], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
