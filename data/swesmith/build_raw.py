#!/usr/bin/env python3
"""Build a faithful ("raw") SWE-smith task tree from the flattened TaskTrove set.

WHY. `laion/swesmith-oracle-filtered-v2` is the pool Ben's TaskTrove panel measured, but it was
flattened for Daytona: the per-repo prebuilt image `jyangballin/swesmith.x86_64.<repo>.<sha8>`
(repo at /testbed, conda env `testbed` installed, pytest ready) was swapped for a generic
`python:3.x-bookworm` base plus an instruction preamble that tells the AGENT to git-clone the
swesmith fork, check out the task branch and `pip install -e .` with auto-discovered extras. On an
air-gapped cluster that preamble cannot run, and even where it can it spends agent turns and
context on setup that original SWE-smith never asks for: there the environment is prepared and the
agent only sees an issue. Same story as R2E-Gym (`data/r2egym/PATCHING.md` vs `data/r2egym/build_raw.py`).

WHAT. For every selected v2 task this emits the original shape, in the harbor layout the
apptainer bridge already handles for SweSmith-style tasks (worker.py "base image SIF" path):

  environment/Dockerfile   FROM <prebuilt image> + RUN git fetch && git checkout <instance_id>
                           (the worker resolves swesmith_base_<sha256(image)[:12]>.sif and replays
                           the RUN lines after instance start; git goes to the offline mirror through
                           the bridge's /etc/gitconfig insteadOf rewrite)
  instruction.md           short "already at /testbed, no network" header + the problem statement
  tests/config.json        instance_id, image_name, repo, FAIL_TO_PASS, PASS_TO_PASS (no patch)
  tests/test.sh            offline verifier: env-ready check, restore the hidden F2P/P2P test files
                           from the "Bug Patch" commit, run pytest on exactly those files inside the
                           image's conda env, grade with tests/grade.py -> /logs/verifier/reward.txt
  tests/grade.py           F2P all PASSED and P2P all PASSED, from `pytest --verbose` output
  solution/solve.sh        oracle: reverse-apply the bug patch (v2's `patch` field is the bug diff)
  task.toml                same timeouts/metadata shape as the raw r2egym tasks

Branch structure of every swesmith fork (verified on oauthlib__oauthlib.1fd52536, 2026-09-03):
  main = "Initial commit" (fixed code, all tests)  ->  "Bug Patch"  ->  "Remove F2P Tests" (= branch tip)
so the agent starts at the tip (bug present, hidden tests deleted) and the verifier restores the
test files from the Bug Patch commit before running them. The verifier never checks anything out
wholesale, so an agent that commits its fix is graded on its fix.

Known, deliberate leak (same class as r2egym's "fix commit in history"): the fork's git history is
intact, so `git diff main` or `git show HEAD~1` reveals the bug patch / hidden tests. Count it per
step (rollout_profile hacking counters), do not pretend it is absent.

USAGE (local Mac, reads the HF parquet):
  python data/swesmith/build_raw.py --out /path/swesmith-raw-v1 [--repos oauthlib,sqlfluff]
        [--per-repo 2] [--ids id1,id2] [--limit N] [--list-repos]
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

V2_REPO = "laion/swesmith-oracle-filtered-v2"

HEADER = """You are working in an existing Python repository checked out at `/testbed`.
The repository is ALREADY present at the correct commit and its development environment is
ALREADY installed (the conda env `testbed` is active in your shell; if `python` is ever not the
project's interpreter, run `source /opt/miniconda3/bin/activate testbed`). Do NOT clone anything and
do not expect network access. Fix the issue described below by editing the code in `/testbed`.

"""

TASK_TOML = """
version = "1.0"

[agent]
timeout_sec = 900.0

[metadata]
author_name = "SWE-smith"
author_email = "swesmith@example.com"
difficulty = "hard"
category = "software-engineering"
tags = [
    "swesmith",
    "code-repair",
    "bug-fixing",
]

[verifier]
restart_environment = false
timeout_sec = 720.0
"""

DOCKERFILE = """FROM {image}

# Prebuilt SWE-smith image: repo at /testbed, conda env `testbed` installed, pytest ready.
# The apptainer bridge resolves this to swesmith_base_<sha256(image)[:12]>.sif and replays the
# RUN lines below after instance start (worker.py base-image path). git reaches the offline
# mirror through the bridge's /etc/gitconfig insteadOf rewrite (BRIDGE_GIT_MIRRORS).
WORKDIR /testbed
RUN git fetch && git checkout {instance_id}
RUN mkdir -p /tmp/fakehome && printf 'source /opt/miniconda3/bin/activate testbed 2>/dev/null\\ncd /testbed 2>/dev/null\\n' >> /tmp/fakehome/.bashrc
{extra_run}"""

# Offline time bomb for astropy-based repos (sunpy): the image's leap-second table expires ~1 year after
# the build, astropy then tries to download IERS data, and the repo's pytest config turns the warning
# into a collection error. With internet (Daytona, Docker) it silently refreshes. Offline equivalent:
# ship a fresh leap-seconds.list in the task's environment dir (-> /workspace) and point astropy at it
# from both config locations HOME can resolve to. Verified on sunpy 07054: pristine 0 / oracle 1.
ASTROPY_REPOS = {"sunpy"}
ASTROPY_RUN = (
    "RUN mkdir -p /tmp/fakehome/.astropy/config /tmp/fakehome/.config/astropy && "
    "printf '[utils.iers.iers]\\nauto_download = False\\niers_degraded_accuracy = ignore\\n"
    "system_leap_second_file = /workspace/leap-seconds.list\\n' "
    "| tee /tmp/fakehome/.astropy/config/astropy.cfg > /tmp/fakehome/.config/astropy/astropy.cfg\n"
)
LEAP_SECONDS_URL = "https://data.iana.org/time-zones/data/leap-seconds.list"

TEST_SH = r"""#!/bin/bash
# SWE-smith raw verifier (offline). Reward 1 iff every FAIL_TO_PASS and PASS_TO_PASS test passes.
# No wholesale checkout: the agent's working tree is graded as-is; only the hidden test files are
# restored from the fork's "Bug Patch" commit (the branch tip deleted them).
set -u
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt
cd /testbed || { echo "ENV_NOT_READY: no /testbed"; exit 3; }

PY=/opt/miniconda3/bin/python3
[ -x "$PY" ] || PY=python3
ID=$($PY -c 'import json;print(json.load(open("/tests/config.json"))["instance_id"])')
BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
if [ "$BR" != "$ID" ] && ! git rev-parse --verify -q "refs/heads/$ID" >/dev/null; then
    echo "ENV_NOT_READY: HEAD=$BR, branch $ID absent (Dockerfile RUN checkout did not happen)"
    exit 3
fi

TEST_FILES=$($PY - <<'PYEOF'
import ast, json
cfg = json.load(open('/tests/config.json'))
def lst(v):
    if isinstance(v, list): return v
    try: return json.loads(v)
    except Exception: return ast.literal_eval(v)
files = sorted({t.split('::')[0] for t in lst(cfg['FAIL_TO_PASS']) + lst(cfg['PASS_TO_PASS'])})
print(' '.join(files))
PYEOF
)

# Restore the hidden test files from the "Bug Patch" commit (bug present, tests intact).
BP=$(git log --format=%H --grep='^Bug Patch$' -n1 "refs/heads/$ID" 2>/dev/null)
[ -n "$BP" ] || BP=$(git log --format=%H --grep='^Bug Patch$' -n1 --all 2>/dev/null)
if [ -n "$BP" ]; then
    git checkout "$BP" -- $TEST_FILES 2>/dev/null || true
else
    echo "WARN: no 'Bug Patch' commit found; restoring test files from main"
    git checkout main -- $TEST_FILES 2>/dev/null || git checkout origin/main -- $TEST_FILES 2>/dev/null || true
fi

# Run exactly the graded test files inside the image's env (SWE-smith python profile command).
# The log goes under /logs/verifier so harbor keeps it with the trial (only /logs/verifier|agent|artifacts
# are host-bound on the apptainer bridge; a plain /logs/test_output.log would vanish with the overlay).
source /opt/miniconda3/bin/activate testbed 2>/dev/null || { echo "ENV_NOT_READY: conda env testbed missing"; exit 3; }
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
TEST_LOG=/logs/verifier/test_output.log
timeout -k 10 1500 python -m pytest --disable-warnings --color=no --tb=short --verbose -p no:cacheprovider \
    $TEST_FILES > $TEST_LOG 2>&1
echo "pytest exit: $?"
tail -n 5 $TEST_LOG

$PY /tests/grade.py $TEST_LOG /tests/config.json
"""

GRADE_PY = r"""#!/usr/bin/env python3
# Grade `pytest --verbose` output against FAIL_TO_PASS / PASS_TO_PASS (SWE-smith semantics:
# resolved iff every listed test PASSED). Writes /logs/verifier/reward.txt.
import ast, json, re, sys
from pathlib import Path

log_path, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))

def lst(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)

STATUS = re.compile(r"^(?P<id>\S.*?) (?P<st>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
status = {}
try:
    for line in Path(log_path).read_text(errors="replace").splitlines():
        line = ANSI.sub("", line).strip()
        m = STATUS.match(line)
        if m:
            status[m.group("id")] = m.group("st")
except FileNotFoundError:
    pass

f2p, p2p = lst(cfg["FAIL_TO_PASS"]), lst(cfg["PASS_TO_PASS"])
def ok(t):
    return status.get(t) == "PASSED"
f2p_ok = sum(ok(t) for t in f2p)
p2p_ok = sum(ok(t) for t in p2p)
resolved = bool(f2p) and f2p_ok == len(f2p) and p2p_ok == len(p2p)
print(f"grade: F2P {f2p_ok}/{len(f2p)}  P2P {p2p_ok}/{len(p2p)}  parsed={len(status)}  resolved={resolved}")
missing = [t for t in f2p + p2p if t not in status][:5]
if missing:
    print("grade: not in log (first 5):", missing)
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text("1" if resolved else "0")
"""

SOLVE_SH = """#!/bin/bash
# Oracle: reverse-apply the bug patch (v2's `patch` field is the bug-introducing diff).
set -e
cd /testbed
cat > /tmp/bug_patch.diff << '__BUG_PATCH__'
{patch}
__BUG_PATCH__
git apply --verbose --reject --reverse /tmp/bug_patch.diff
"""


def load_v2(path: str | None):
    import pyarrow.parquet as pq

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(V2_REPO, "tasks.parquet", repo_type="dataset")
    t = pq.read_table(path)
    for p, blob in zip(t.column("path").to_pylist(), t.column("task_binary").to_pylist()):
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        cfg = None
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith("config.json"):
                fh = tf.extractfile(m)
                if fh is not None:
                    cfg = json.loads(fh.read().decode())
        if cfg is None:
            print(f"WARN: {p} has no config.json, skipped", file=sys.stderr)
            continue
        yield p, cfg


def repo_short(cfg) -> str:
    # swesmith/<owner>__<name>.<sha8>  ->  <name>
    return cfg["repo"].split("/", 1)[1].split("__", 1)[1].rsplit(".", 1)[0]


def as_list(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


def load_leap_seconds(path: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    import urllib.request

    data = urllib.request.urlopen(LEAP_SECONDS_URL, timeout=30).read()
    if b"#@" not in data:
        raise SystemExit("leap-seconds.list download looks wrong (no #@ expiry line)")
    return data


def write_task(out: Path, name: str, cfg: dict, leap_seconds: bytes | None) -> None:
    d = out / name
    (d / "environment").mkdir(parents=True, exist_ok=True)
    (d / "tests").mkdir(exist_ok=True)
    (d / "solution").mkdir(exist_ok=True)
    (d / "task.toml").write_text(TASK_TOML.lstrip("\n"))
    ps = re.sub(r"^\[ISSUE\]\s*", "", cfg["problem_statement"].strip())
    ps = re.sub(r"\s*\[/ISSUE\]\s*$", "", ps).strip()
    (d / "instruction.md").write_text(HEADER + ps + "\n")
    extra_run = ""
    if repo_short(cfg) in ASTROPY_REPOS:
        if leap_seconds is None:
            raise SystemExit(f"{name}: astropy repo needs a leap-seconds.list (use --leap-seconds or allow download)")
        (d / "environment" / "leap-seconds.list").write_bytes(leap_seconds)
        extra_run = ASTROPY_RUN
    (d / "environment" / "Dockerfile").write_text(
        DOCKERFILE.format(image=cfg["image_name"], instance_id=cfg["instance_id"], extra_run=extra_run)
    )
    slim = {
        "instance_id": cfg["instance_id"],
        "image_name": cfg["image_name"],
        "repo": cfg["repo"],
        "FAIL_TO_PASS": as_list(cfg["FAIL_TO_PASS"]),
        "PASS_TO_PASS": as_list(cfg["PASS_TO_PASS"]),
        "source": "swesmith-oracle-filtered-v2",
    }
    (d / "tests" / "config.json").write_text(json.dumps(slim, indent=1))
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "tests" / "grade.py").write_text(GRADE_PY)
    (d / "solution" / "solve.sh").write_text(SOLVE_SH.format(patch=cfg["patch"].rstrip("\n")))
    for f in ("tests/test.sh", "solution/solve.sh"):
        os.chmod(d / f, 0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output task root (created)")
    ap.add_argument("--parquet", help="local tasks.parquet (default: download from HF)")
    ap.add_argument("--repos", help="comma-separated short repo names (e.g. oauthlib,sqlfluff)")
    ap.add_argument("--ids", help="comma-separated instance_ids")
    ap.add_argument("--per-repo", type=int, default=0, help="take the first N tasks of each selected repo")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-repos", action="store_true")
    ap.add_argument("--leap-seconds", help="local IERS leap-seconds.list (default: download once from IANA)")
    a = ap.parse_args()

    repos = set(a.repos.split(",")) if a.repos else None
    ids = set(a.ids.split(",")) if a.ids else None
    taken = collections.Counter()
    images = {}
    n = 0
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    leap: bytes | None = None
    for p, cfg in load_v2(a.parquet):
        r = repo_short(cfg)
        images.setdefault(cfg["image_name"], [r, 0])
        images[cfg["image_name"]][1] += 1
        if a.list_repos:
            continue
        if repos and r not in repos:
            continue
        if ids and cfg["instance_id"] not in ids:
            continue
        if a.per_repo and taken[r] >= a.per_repo:
            continue
        taken[r] += 1
        n += 1
        if out:
            if r in ASTROPY_REPOS and leap is None:
                leap = load_leap_seconds(a.leap_seconds)
            write_task(out, p, cfg, leap)
        if a.limit and n >= a.limit:
            break
    if a.list_repos:
        for img, (r, c) in sorted(images.items(), key=lambda kv: -kv[1][1]):
            print(f"{c:6d}  {r:32s}  {img}  swesmith_base_{hashlib.sha256(img.encode()).hexdigest()[:12]}.sif")
        return
    print(f"tasks written: {n}  per repo: {dict(taken)}  out: {out}")
    if out:
        need = sorted({(cfg_img, hashlib.sha256(cfg_img.encode()).hexdigest()[:12]) for cfg_img in
                       {json.load(open(out / t / 'tests' / 'config.json'))['image_name'] for t in os.listdir(out)}})
        for img, h in need:
            print(f"needs SIF swesmith_base_{h}.sif  <- docker://{img}")


if __name__ == "__main__":
    main()
