#!/usr/bin/env python3
"""Artifact-shape task tree for TaskTrove r2egym on Daytona: ONE snapshot per repo, every task set up in ~2 s, no build.

Each task becomes harness-prepared:
  environment/Dockerfile   the repo's artifact image (daytona_artifacts/repo_image.py, bundle layout of
                           laion/r2egym-build-artifacts): byte-identical across the repo's allowlisted tasks (asserted), so
                           harbor's environment-dir hash -> one `harbor__<hash>__snapshot` per repo. Nothing task-specific
                           lives in environment/.
  setup_files/setup.sh     run by harbor before the agent (setup-files hook): git checkout <base>, materialise the harvested
                           venv from the image's blob store (outside the repo, symlinked as $W/.venv) and this commit's
                           build delta, remove what install.sh removed, symlink python/pytest into /usr/local/bin, import
                           check. ~1-3 s, no network.
  instruction.md           issue-only (TaskTrove's `## Environment Setup` preamble stripped).
  tests/                   TaskTrove v5.1's verifier verbatim when the task is in v5.1 (test.sh with the git gate + trusted-
                           test restore, install_trusted_test_paths.sh, trusted_test_paths.txt, test_state.py, test_info.json,
                           r2e_tests/); for tasks v5.1 dropped (pandas, numpy, most of orange3) the v3 tests plus the v5.1
                           scripts templated on the task's base commit.
  solution/, task.toml     as TaskTrove ships them.

  tt_daytona_tree_v3.py --allow <list> --out <dir> [--artifacts laion/r2egym-build-artifacts] [--tasks a,b,c]

Writes TASKS.txt and SNAPSHOTS.md (per repo: tasks, Dockerfile hash, predicted snapshot name, size estimate).

Gate it with daytona_artifacts/gate_{oracle,nop}.yaml. Run harbor with the hook-carrying checkout on PYTHONPATH
(`export PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src`): the harbor venv's `harbor.pth` still points at the
old ~/harbor checkout, which has no setup-files hook, so without it every task runs unprepared and scores 0.
Proven 2026-09-14 on the 8 pilot pandas tasks: oracle 8 × 1.0, nop 8 × 0.0, setup 3 s (research/2026-09-14_daytona_tree_v3.md).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import io
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "daytona_artifacts"))
from repo_image import ArtifactIndex, dockerfile_for_repo, estimate_image_mb, setup_sh  # noqa: E402

HUB = os.path.expanduser("~/.cache/huggingface/hub")
FENCE = re.compile(r"^## Environment Setup[^\n]*\n\n```bash\n(.*?)\n```\n\n---\n\n", re.S)
SHA40 = re.compile(r"[0-9a-f]{40}")
V51_FILES = ("tests/test.sh", "tests/install_trusted_test_paths.sh", "tests/trusted_test_paths.txt")
# tasks whose base commit's pytest ini adds a test path to every pytest call (pillow's `addopts = -vx Tests`)
ADDOPTS_OVERRIDE = HERE / "daytona_artifacts" / "pytest_addopts_override.txt"
FIRST_PYTEST = 'python -m pytest $PYTEST_ARGS "$TESTS_DIR/"'
# The verifier prints only the last 100 lines of the test run; keep the whole short summary for the gate readout and for
# deriving required_tests.json. The grader then also requires every staged test the oracle reported (test_state_required.py).
END_TEST_OUTPUT = 'echo "=== END TEST OUTPUT ==="\n'
SAVE_SUMMARY = ('# the whole short test summary (the stdout above keeps 100 lines): gate readout + required_tests.json derivation\n'
                'sed -n "/short test summary info/,\\$p" "$TEST_OUTPUT" > /logs/verifier/test_summary.txt 2>/dev/null || true\n')
REQUIRED_GRADER = HERE / "daytona_artifacts" / "test_state_required.py"
# v5.1's test.sh passes --no-header, which pytest < 6 rejects ("unrecognized arguments"), so no test runs and every trial
# scores 0; R2E-Gym's own run_tests.sh never passed it. Tasks whose harvested venv has pytest < 6 (25 coveragepy tasks on
# py3.7, pytest 4.6) pass it only when the pytest in the venv supports it. The grader reads only the short test summary.
PYTEST_ARGS_V51 = 'PYTEST_ARGS="-v --tb=short --no-header -rA -p no:cacheprovider -W ignore::DeprecationWarning"\n'
PYTEST_ARGS_OLD_PYTEST = (
    "NO_HEADER=--no-header\n"
    "python -c 'import sys, pytest; sys.exit(int(pytest.__version__.split(\".\")[0]) < 6)' 2>/dev/null || NO_HEADER=\n"
    'PYTEST_ARGS="-v --tb=short $NO_HEADER -rA -p no:cacheprovider -W ignore::DeprecationWarning"\n'
)


def find_parquet(pattern: str) -> str | None:
    c = sorted(glob.glob(pattern))
    return c[-1] if c else None


def load_blobs(parquet: str, want: set[str]) -> dict[str, bytes]:
    out = {}
    pf = pq.ParquetFile(parquet)
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
        for p, b in zip(d["path"], d["task_binary"]):
            if p in want:
                out[p] = b
    return out


def collect_staged_tests_only(test_sh: str) -> str:
    """The verifier's first pytest call with the repo's ini addopts cleared (its retry already clears them)."""
    assert test_sh.count(FIRST_PYTEST) == 1, "test.sh: first pytest call not found"
    return test_sh.replace(FIRST_PYTEST, 'python -m pytest $PYTEST_ARGS --override-ini=\'addopts=\' "$TESTS_DIR/"')


def env_dir_hash(env_dir: Path) -> str:
    """harbor's snapshot hash when importable (the same function names the snapshot); sha256 of the Dockerfile otherwise."""
    try:
        from harbor.utils.container_cache import environment_dir_hash_truncated
        return environment_dir_hash_truncated(env_dir, truncate=12)
    except Exception:  # noqa: BLE001
        return hashlib.sha256((env_dir / "Dockerfile").read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", default=None, help="task list file (one TaskTrove path per line)")
    ap.add_argument("--tasks", default=None, help="comma list of tasks (instead of / in addition to --allow)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifacts", default="laion/r2egym-build-artifacts", help="HF dataset with the harvested artifacts")
    ap.add_argument("--tasktrove-v51", default=None, help="TaskTrove v5.1 r2egym parquet (default: HF cache)")
    ap.add_argument("--tasktrove-v3", default=None, help="laion/r2egym-patched-full-oracle-v3 parquet (default: HF cache)")
    ap.add_argument("--map", default=str(HERE / "overlap" / "tasktrove_v3_upstream_map.tsv"))
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--clean", action="store_true", help="remove <out> first")
    ap.add_argument("--required-tests", default=None,
                    help="json(.gz) {task: {staged test id: status}} from derive_required_tests.py; tasks absent from it grade as before")
    a = ap.parse_args()

    v51 = a.tasktrove_v51 or find_parquet(f"{HUB}/datasets--open-thoughts--TaskTrove/snapshots/*/laion__r2egym-patched-full-oracle-v3/tasks.parquet")
    v3 = a.tasktrove_v3 or find_parquet(f"{HUB}/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet")
    assert v3, "v3 parquet not found (laion/r2egym-patched-full-oracle-v3)"
    repo_of = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}
    excl = {l.strip() for l in open(a.exclude)} if a.exclude else set()
    addopts_override = {l.strip() for l in open(ADDOPTS_OVERRIDE) if l.strip() and not l.startswith("#")}
    required_tests: dict[str, dict[str, str]] = {}
    if a.required_tests:
        import gzip, json
        required_tests = json.load((gzip.open if a.required_tests.endswith(".gz") else open)(a.required_tests, "rt"))
    grader_tail = REQUIRED_GRADER.read_text()
    want: list[str] = []
    if a.allow:
        want += [l.strip() for l in open(a.allow) if l.strip() and not l.startswith("#")]
    if a.tasks:
        want += [t for t in a.tasks.split(",") if t]
    want = sorted({t for t in want if t not in excl})
    assert want, "no tasks"

    index = ArtifactIndex(a.artifacts, repo_of=repo_of)
    index.ensure_tool_uploaded()   # the Dockerfile pins tools/harvest_materialize.py by sha256
    missing = [t for t in want if t not in index.tasks]
    assert not missing, f"{len(missing)} tasks have no artifacts in {a.artifacts}: {missing[:5]}"

    blobs_v51 = load_blobs(v51, set(want)) if v51 else {}
    blobs_v3 = load_blobs(v3, set(want) - set(blobs_v51))
    absent = set(want) - set(blobs_v51) - set(blobs_v3)
    assert not absent, f"not in TaskTrove v5.1 or v3: {sorted(absent)[:5]}"

    # v5.1 verifier templates from any v5.1 task of the majority (cd /testbed) variant: only the base commit varies
    templates: dict[str, str] = {}
    if v51:
        pf = pq.ParquetFile(v51)
        d = pf.read_row_group(0, columns=["path", "task_binary"]).to_pydict()
        for p, b in zip(d["path"], d["task_binary"]):
            tf = tarfile.open(fileobj=io.BytesIO(b))
            files = {m.name: tf.extractfile(m).read().decode() for m in tf.getmembers() if m.isfile() and m.name in V51_FILES}
            if len(files) == 3 and "cd /testbed" in files["tests/test.sh"]:
                templates = {k: SHA40.sub("{BASE}", v) for k, v in files.items()}
                assert templates["tests/test.sh"].count("{BASE}") == 2, "unexpected test.sh template"
                break
    assert templates, "no v5.1 verifier template found (need open-thoughts/TaskTrove in the HF cache)"

    out = Path(a.out)
    if a.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    by_repo: dict[str, list] = collections.defaultdict(list)
    source = collections.Counter()
    old_pytest: list[str] = []
    for task in want:
        art = index.tasks[task]
        repo = repo_of.get(task) or art.repo
        assert repo == art.repo, f"{task}: map says {repo}, artifacts say {art.repo}"
        blob, src = (blobs_v51[task], "v5.1") if task in blobs_v51 else (blobs_v3[task], "v3")
        source[src] += 1
        dst = out / task
        if dst.exists():
            shutil.rmtree(dst)
        tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz").extractall(dst, filter="data")
        # instruction: issue only; cross-check the preamble's checkout commit against the artifact's base commit
        ins_p = dst / "instruction.md"
        ins = ins_p.read_text()
        m = FENCE.match(ins)
        assert m, f"{task}: no environment preamble"
        cm = re.search(r"git checkout ([0-9a-f]{40})", m.group(1))
        assert cm and cm.group(1) == art.base, f"{task}: preamble commit {cm and cm.group(1)} != artifact base {art.base}"
        ins_p.write_text(ins[m.end():])
        # setup hook
        (dst / "setup_files").mkdir(exist_ok=True)
        sh = dst / "setup_files" / "setup.sh"
        sh.write_text(setup_sh(art))
        sh.chmod(0o755)
        # verifier: v5.1 scripts, templated on this task's base commit when the task is not in v5.1
        for name, tmpl in templates.items():
            p = dst / name
            if src == "v3" or not p.exists():
                p.write_text(tmpl.replace("{BASE}", art.base))
        if art.pytest and int(art.pytest.split(".")[0]) < 6:
            ts = dst / "tests" / "test.sh"
            body = ts.read_text()
            assert body.count(PYTEST_ARGS_V51) == 1, f"{task}: test.sh has no v5.1 PYTEST_ARGS line to guard"
            ts.write_text(body.replace(PYTEST_ARGS_V51, PYTEST_ARGS_OLD_PYTEST))
            old_pytest.append(task)
        # R2E-Gym grades the staged tests only. On pillow before 7da17ad41 the ini addopts `-vx Tests` made the verifier's
        # pytest run the repo's whole Tests/ suite first and stop at its first failure, Tests/test_features.py (`_tkinter`
        # is built into the uv python, no __file__, the same in R2E-Gym's image): the staged tests never ran, and a no-op
        # scored 1 whenever the expected tests shared names with repo tests that had passed before the stop.
        if task in addopts_override:
            (dst / "tests" / "test.sh").write_text(collect_staged_tests_only((dst / "tests" / "test.sh").read_text()))
        ts = dst / "tests" / "test.sh"
        body = ts.read_text()
        assert body.count(END_TEST_OUTPUT) == 1, f"{task}: test.sh has no END TEST OUTPUT marker"
        ts.write_text(body.replace(END_TEST_OUTPUT, END_TEST_OUTPUT + SAVE_SUMMARY))
        st = dst / "tests" / "test_state.py"
        assert "def test_r2egym_tests_resolved" in st.read_text(), f"{task}: unexpected test_state.py"
        st.write_text(st.read_text() + grader_tail)
        if task in required_tests:
            (dst / "tests" / "required_tests.json").write_text(json.dumps(required_tests[task], indent=0, sort_keys=True) + "\n")
        (dst / "tests" / "test.sh").chmod(0o755)
        (dst / "tests" / "install_trusted_test_paths.sh").chmod(0o755)
        # environment: the per-repo artifact image, written after the loop (needs the repo's whole task set)
        for p in (dst / "environment").iterdir():
            if p.name != "Dockerfile":
                raise SystemExit(f"{task}: unexpected file in environment/: {p.name} (would change the snapshot hash)")
        by_repo[repo].append(art)

    rows = ["repo | tasks | Dockerfile sha256[:12] | predicted snapshot | est. image MB (pythons/blobs raw/tasks bundle/base)", "---|---|---|---|---"]
    for repo, arts in sorted(by_repo.items(), key=lambda kv: -len(kv[1])):
        df = dockerfile_for_repo(repo, arts, index)
        hashes = set()
        for art in arts:
            (out / art.task / "environment" / "Dockerfile").write_text(df)
            hashes.add(env_dir_hash(out / art.task / "environment"))
        assert len(hashes) == 1, f"{repo}: {len(hashes)} distinct environment hashes"
        h = hashes.pop()
        est = estimate_image_mb(repo, arts, index)
        rows.append(f"{repo} | {len(arts)} | {hashlib.sha256(df.encode()).hexdigest()[:12]} | harbor__{h}__snapshot | "
                    f"{est['total_mb']} ({est['python_mb']}/{est['blobs_raw_mb']}/{est['tasks_bundle_mb']}/{est['base_mb']})")
        (out / f"Dockerfile.{repo}").write_text(df)
    rows.append(f"**all** | {len(want)} | {len(by_repo)} images | | tasks from TaskTrove {dict(source)}")
    (out / "TASKS.txt").write_text("\n".join(want) + "\n")
    (out / "TASKS_OLD_PYTEST.txt").write_text("".join(t + "\n" for t in old_pytest))
    (out / "SNAPSHOTS.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
