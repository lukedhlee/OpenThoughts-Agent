import io, json, re, tarfile, hashlib
import pyarrow.parquet as pq
from normkeys import k_eo, k_ps, norm_ws, strip_issue_markers

p = open("tt_path.txt").read().strip()
pf = pq.ParquetFile(p)

FENCE_END = re.compile(r"^```\s*$", re.M)

def split_preamble(text):
    """Preamble = '## Environment Setup' heading + its ```bash fence, terminated by a '---' rule.
    Returns text after the first '\n---\n' that follows the closing fence; falls back to whole text."""
    if text.startswith("## Environment Setup"):
        m = re.search(r"\n---\n", text)
        if m:
            return text[m.end():], "hr"
    return text, "none"

def issue_inner(text):
    a = text.find("<issue_description>")
    b = text.find("</issue_description>")
    if a != -1 and b != -1 and b > a:
        return text[a + len("<issue_description>"):b], True
    return text, False

def read_member(tf, suffix):
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith(suffix):
            return tf.extractfile(m).read()
    return None

n = 0; eos = set(); pss = set(); froms = set()
stats = {"boundary_hr": 0, "boundary_none": 0, "issue_tag": 0, "no_issue_tag": 0, "no_test_info": 0, "no_dockerfile": 0}
with open("r2e_tasktrove_keys.jsonl", "w") as out:
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
        for path, blob in zip(d["path"], d["task_binary"]):
            tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
            ins = (read_member(tf, "instruction.md") or b"").decode("utf-8", "replace")
            post, how = split_preamble(ins)
            stats["boundary_" + how] += 1
            inner, tagged = issue_inner(post)
            stats["issue_tag" if tagged else "no_issue_tag"] += 1
            ti_raw = read_member(tf, "test_info.json")
            ti = json.loads(ti_raw) if ti_raw else {}
            if not ti_raw: stats["no_test_info"] += 1
            df = read_member(tf, "environment/Dockerfile")
            frm = None
            if df is None:
                stats["no_dockerfile"] += 1
            else:
                for line in df.decode("utf-8", "replace").splitlines():
                    if line.strip().upper().startswith("FROM "):
                        frm = line.strip()[5:].strip(); break
            rec = {
                "src": "tasktrove", "path": path,
                "repo": ti.get("github_repo"), "base_commit": ti.get("base_commit"),
                "docker_image": ti.get("docker_image"),
                "from": frm,
                "k_eo": k_eo(ti.get("expected_output_json")),
                "k_ps": k_ps(inner),
                "k_post": hashlib.sha1(norm_ws(post).encode()).hexdigest(),
            }
            eos.add(rec["k_eo"]); pss.add(rec["k_ps"]); froms.add(frm)
            out.write(json.dumps(rec) + "\n"); n += 1
print(json.dumps({"rows": n, "distinct_k_eo": len(eos), "distinct_k_ps": len(pss), "distinct_from": len(froms), **stats}, indent=1))
print("FROM values:", sorted(x for x in froms if x))
