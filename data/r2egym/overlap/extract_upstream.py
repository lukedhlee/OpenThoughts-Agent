import json, sys
import pyarrow.parquet as pq
from normkeys import k_eo, k_ps
paths = [l.strip() for l in open("upstream_paths.txt") if l.strip()]
cols = ["repo_name", "docker_image", "commit_hash", "problem_statement", "expected_output_json"]
n = 0
imgs, eos, pss = set(), set(), set()
with open("r2e_upstream_keys.jsonl", "w") as out:
    for p in paths:
        t = pq.read_table(p, columns=cols)
        d = t.to_pydict()
        for i in range(t.num_rows):
            rec = {
                "src": "upstream", "idx": n,
                "repo": d["repo_name"][i],
                "docker_image": d["docker_image"][i],
                "commit": d["commit_hash"][i],
                "k_eo": k_eo(d["expected_output_json"][i]),
                "k_ps": k_ps(d["problem_statement"][i]),
            }
            imgs.add(rec["docker_image"]); eos.add(rec["k_eo"]); pss.add(rec["k_ps"])
            out.write(json.dumps(rec) + "\n"); n += 1
        print(p, t.num_rows, flush=True)
print(json.dumps({"rows": n, "distinct_docker_image": len(imgs), "distinct_k_eo": len(eos), "distinct_k_ps": len(pss)}))
