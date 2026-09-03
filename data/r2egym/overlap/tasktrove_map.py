#!/usr/bin/env python3
"""Map every TaskTrove `laion/r2egym-patched-full-oracle-v3` task to its upstream R2E-Gym-V1 row (= the original
prebuilt image), and verify the mapping independently of any docker_image field.

WHY NOT the docker_image field. TaskTrove v3 rewrote `test_info.json:{docker_image,base_commit}` to the buggy PARENT
commit (its flattened agent clones the repo and checks that commit out). The parent of one fix commit is often the
previous task's fix commit in the same repo, so 391 of those rewritten tags exist upstream and point at the WRONG task.
A join that trusts docker_image first (r2e_overlap.py, 2026-09-03 16:00) therefore double-mapped 255 rows.

SOURCE OF THE MAPPING. Marianna's `r2e_baked_all.txt` (3,328 lines: `<tasktrove path> <sha256(Dockerfile)[:12]>
<upstream image>`), built from the pre-v3 TaskTrove whose docker_image was still the real image; committed verbatim as
`marianna_r2e_baked_all.txt`. This script does not trust it either: for each TaskTrove task it extracts the hidden
test files (`tests/r2e_tests/*`) and expected_output_json from the task tarball and requires the mapped upstream row
to carry byte-identical `execution_result_content.test_file_codes` AND the same expected output. 2026-09-03: 3,328/3,328
pass both checks; 2,103 tasks are unique by test files alone, the other 1,225 are disambiguated by the mapping.

OUTPUT `tasktrove_v3_upstream_map.tsv` (one line per TaskTrove task):
  path  docker_image  repo  commit  tt_base_commit  n_test_candidates  verified

Inputs default to the local HF hub cache; override with --upstream-glob / --tasktrove.
"""
import argparse, collections, glob, hashlib, io, json, os, sys, tarfile
import pyarrow.parquet as pq

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from normkeys import k_eo  # noqa: E402

HUB = os.path.expanduser("~/.cache/huggingface/hub")
ap = argparse.ArgumentParser()
ap.add_argument("--upstream-glob", default=f"{HUB}/datasets--R2E-Gym--R2E-Gym-V1/snapshots/*/data/*.parquet")
ap.add_argument("--tasktrove", default=None, help="tasks.parquet of laion/r2egym-patched-full-oracle-v3")
ap.add_argument("--baked", default=os.path.join(SP, "marianna_r2e_baked_all.txt"))
ap.add_argument("--out", default=os.path.join(SP, "tasktrove_v3_upstream_map.tsv"))
a = ap.parse_args()
if a.tasktrove is None:
    c = glob.glob(f"{HUB}/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet")
    assert len(c) == 1, c
    a.tasktrove = c[0]


def test_hash(codes):
    return hashlib.sha1("\x00".join(sorted(codes)).encode()).hexdigest()


up = {}
for p in sorted(glob.glob(a.upstream_glob)):
    cols = ["repo_name", "docker_image", "commit_hash", "expected_output_json", "execution_result_content"]
    for r in pq.read_table(p, columns=cols).to_pylist():
        erc = json.loads(r["execution_result_content"])
        up[r["docker_image"]] = {"repo": r["repo_name"], "commit": r["commit_hash"], "k_eo": k_eo(r["expected_output_json"]),
                                 "th": test_hash(erc.get("test_file_codes") or [])}
assert len(up) == 8101, len(up)
by_th = collections.defaultdict(list)
for img, r in up.items():
    by_th[r["th"]].append(img)

baked = {}
for line in open(a.baked):
    if line.strip():
        path, _h, img = line.split()
        baked[path] = img
assert len(baked) == 3328, len(baked)

pf = pq.ParquetFile(a.tasktrove)
rows, stats = [], collections.Counter()
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
    for path, blob in zip(d["path"], d["task_binary"]):
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        codes = [tf.extractfile(m).read().decode("utf-8", "replace") for m in tf.getmembers() if m.isfile() and "/r2e_tests/" in m.name]
        ti = json.loads(tf.extractfile(next(m for m in tf.getmembers() if m.name.endswith("test_info.json"))).read())
        cands = by_th.get(test_hash(codes), [])
        img = baked.get(path)
        u = up.get(img)
        ok = bool(u) and img in cands and u["k_eo"] == k_eo(ti.get("expected_output_json"))
        stats["verified" if ok else "FAILED"] += 1
        rows.append((path, img, u["repo"] if u else "", u["commit"] if u else "", ti.get("base_commit", ""), len(cands), "yes" if ok else "NO"))
rows.sort()
imgs = collections.Counter(r[1] for r in rows)
assert max(imgs.values()) == 1, "an upstream image is mapped by more than one TaskTrove path"
with open(a.out, "w") as f:
    f.write("path\tdocker_image\trepo\tcommit\ttt_base_commit\tn_test_candidates\tverified\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(dict(stats), "->", a.out)
print("per repo:", dict(collections.Counter(r[2] for r in rows).most_common()))
if stats["FAILED"]:
    sys.exit(1)
