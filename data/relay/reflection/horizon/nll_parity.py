"""Train/serve parity: vLLM's teacher-forced NLL on an update's supervised tokens vs the Marin backend's loss on them.

    python -m data.relay.reflection.horizon.nll_parity --update-dir <run>/update-0001 [--n 64] --out parity.json

The served model must be the update's model_before (the weights the backend computed its loss with). vLLM returns
prompt logprobs for the exact token ids; the token-weighted mean of -logprob over loss==1 positions is the quantity the
backend reports as `loss` (global_token_mean, temperature 1). A gap well beyond bf16 noise means the trainer and the
server run different functions of the same weights.
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKETS = (0, 2048, 4096, 8192, 16384)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-dir", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--url", default="http://127.0.0.1:29010/v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.update_dir)
    rows = [json.loads(line) for line in (d / "rows.jsonl").read_text().splitlines() if line.strip()][: a.n]
    trained = json.loads((d / "train.response.json").read_text())

    def one(row):
        payload = {"model": "cf-student", "prompt": row["ids"], "max_tokens": 1, "prompt_logprobs": 0,
                   "temperature": 1.0}
        req = urllib.request.Request(a.url + "/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as f:
            plp = json.load(f)["choices"][0]["prompt_logprobs"]
        total, count = 0.0, 0
        buckets = {}
        for i, (token, weight) in enumerate(zip(row["ids"], row["loss"])):
            if not weight:
                continue
            entry = plp[i]
            lp = entry.get(str(token)) if entry else None
            lp = lp["logprob"] if isinstance(lp, dict) else lp
            total += -lp
            count += 1
            lo = max(b for b in BUCKETS if b <= i)
            acc = buckets.setdefault(f"pos_{lo}", [0.0, 0])
            acc[0] += -lp
            acc[1] += 1
        per_row.append({"sid": row["sid"], "tokens": count, "nll": total / count, **buckets})
        return total, count

    per_row = []
    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(one, rows))
    nll = sum(t for t, _ in parts) / sum(c for _, c in parts)
    # The backend's loss covers all rows; also recompute it over the same subset is impossible from the response, so
    # report the subset's vLLM NLL next to the backend's all-row loss plus per-row spread.
    spread = sorted(t / c for t, c in parts)
    report = {"rows": len(rows), "tokens": sum(c for _, c in parts), "vllm_nll": round(nll, 4),
              "backend_loss_all_rows": round(trained["loss"], 4), "backend_rows": trained.get("supervised_tokens"),
              "vllm_row_nll_p10_p50_p90": [round(spread[int(q * (len(spread) - 1))], 4) for q in (0.1, 0.5, 0.9)],
              "per_row": per_row}
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_row"}))


if __name__ == "__main__":
    main()
