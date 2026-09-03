#!/usr/bin/env python3
"""Build TaskTrove's r2egym selection in the ORIGINAL R2E-Gym environment shape.

REFERENCE DESIGN (R2E-Gym). One prebuilt x86_64 image per task, `namanjain12/<repo>_final:<fix commit>`: the repo
sits at /testbed at the buggy parent commit with every dependency installed; the agent gets only the issue text and
edits in place; the verifier runs the hidden tests inside that same environment and compares the status map with
expected_output_json. No clone, no pip, no network.

WHAT WE HOLD. `laion/r2egym-patched-full-oracle-v3` (TaskTrove; the pool Ben measured) keeps R2E-Gym's TASK SELECTION
but not its environment: `data/r2egym/PATCHING.md` swapped the per-task image for a generic python image plus an
instruction preamble that makes the AGENT clone + install, and a verifier that fetches from the network — a constant
reward on an air-gapped cluster (0 % learnable band). This builder keeps the selection (task dir = TaskTrove `path`,
so Ben's tracker lines up) and restores the environment: the Dockerfile is byte-identical to the raw R2E-Gym rebuild
(`build_raw.py`), so `sha256(Dockerfile)[:12]` resolves the SAME SIFs harbor's apptainer worker already has
(glob `*-<hash>.sif`: 1,746 in our cache, 1,582 prebuilt by Marianna under the `build_r2egym-v1-*` names). The
verifier is the raw r2egym `tests/test.sh` (offline-patched by the worker at run time). Nothing is pulled or built.

Per task `<out>/<tasktrove path>/`:
  environment/Dockerfile             raw template with `FROM <upstream image>`        (hash == SIF name)
  environment/workspace/metadata.json raw keys (instance_id, docker_image, base_commit = fix commit, problem_statement
                                      as upstream incl. the [ISSUE] wrapper, repo_name, expected_output_json, source)
                                      + tasktrove_path / tasktrove_dataset / tasktrove_base_commit for cross-reference
  tests/test.sh, task.toml            raw templates (tt_raw_template/, verified byte-identical to the raw set)
  instruction.md                      build_raw.py's header + the issue body with the [ISSUE] wrapper stripped

Mapping = overlap/tasktrove_v3_upstream_map.tsv (overlap/tasktrove_map.py; test files + expected output verified).
Usage: build_tt_raw.py --out <dir> [--limit N] [--sif-list <file of cached SIF names>] --apply
"""
import argparse, glob, hashlib, json, os, re, sys

SP = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/huggingface/hub")
HEADER = """You are working in an existing Python repository checked out at `/testbed`.
The repository is ALREADY present at the correct commit -- do NOT clone anything, and do
not expect network access. Fix the issue described below by editing the code in
`/testbed`.

"""
TT_DATASET = "laion/r2egym-patched-full-oracle-v3"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--map", default=os.path.join(SP, "overlap", "tasktrove_v3_upstream_map.tsv"))
ap.add_argument("--upstream-glob", default=f"{HUB}/datasets--R2E-Gym--R2E-Gym-V1/snapshots/*/data/*.parquet")
ap.add_argument("--sif-list", default=None, help="file listing SIF names in the cache; every task must hit `*-<hash>.sif`")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

tpl = {n: open(os.path.join(SP, "tt_raw_template", n), "rb").read() for n in ("Dockerfile.body", "test.sh", "task.toml")}
# self-test: the template must reproduce a known SIF hash (raw r2egym-0000 -> build_r2egym-0000-bafb1d2ccf58.sif)
_probe = b"FROM namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d\n" + tpl["Dockerfile.body"]
assert hashlib.sha256(_probe).hexdigest()[:12] == "bafb1d2ccf58", "Dockerfile template drifted from the raw set"

rows = []
with open(a.map) as f:
    hdr = f.readline().rstrip("\n").split("\t")
    for line in f:
        r = dict(zip(hdr, line.rstrip("\n").split("\t")))
        assert r["verified"] == "yes", r
        rows.append(r)
if a.limit:
    rows = rows[: a.limit]
want = {r["docker_image"] for r in rows}

import pyarrow.parquet as pq  # noqa: E402
up = {}
cols = ["repo_name", "docker_image", "commit_hash", "problem_statement", "expected_output_json"]
for p in sorted(glob.glob(a.upstream_glob)):
    for r in pq.read_table(p, columns=cols).to_pylist():
        if r["docker_image"] in want:
            up[r["docker_image"]] = r
missing = want - set(up)
assert not missing, f"{len(missing)} mapped images not in upstream: {sorted(missing)[:3]}"

sifs = None
if a.sif_list:
    sifs = {}
    for line in open(a.sif_list):
        m = re.match(r"(?:.*/)?(\S+)-([0-9a-f]{12})\.sif$", line.strip())
        if m:
            sifs.setdefault(m.group(2), m.group(1))

made, nosif = 0, []
for r in rows:
    u = up[r["docker_image"]]
    df = b"FROM " + u["docker_image"].encode() + b"\n" + tpl["Dockerfile.body"]
    h = hashlib.sha256(df).hexdigest()[:12]
    if sifs is not None and h not in sifs:
        nosif.append((r["path"], u["docker_image"], h))
    ps = (u["problem_statement"] or "").strip()
    body = re.sub(r"^\[ISSUE\]\s*", "", ps)
    body = re.sub(r"\s*\[/ISSUE\]\s*$", "", body).strip()
    meta = {
        "instance_id": u["docker_image"], "docker_image": u["docker_image"], "base_commit": u["commit_hash"],
        "problem_statement": u["problem_statement"], "repo_name": u["repo_name"],
        "expected_output_json": u["expected_output_json"], "source": "r2egym",
        "tasktrove_path": r["path"], "tasktrove_dataset": TT_DATASET, "tasktrove_base_commit": r["tt_base_commit"],
    }
    made += 1
    if not a.apply:
        continue
    d = os.path.join(a.out, r["path"])
    os.makedirs(os.path.join(d, "environment", "workspace"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    open(os.path.join(d, "environment", "Dockerfile"), "wb").write(df)
    with open(os.path.join(d, "environment", "workspace", "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(d, "tests", "test.sh"), "wb").write(tpl["test.sh"])
    open(os.path.join(d, "task.toml"), "wb").write(tpl["task.toml"])
    open(os.path.join(d, "instruction.md"), "w").write(HEADER + body + "\n")

print(f"mapped tasks   : {len(rows)}")
print(f"built          : {made}")
print(f"dest           : {a.out}")
if sifs is not None:
    print(f"SIF hits       : {made - len(nosif)} / {made}" + (f"   MISSING: {nosif[:5]}" if nosif else ""))
print("DRY RUN -- pass --apply to write" if not a.apply else "APPLIED")
if nosif:
    sys.exit(1)
