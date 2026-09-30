#!/usr/bin/env python3
"""The Horizon <-> Jupiter SFT pair set: Kimi SWE-smith traces in the Grug Datakit 09-21 row format, ~30 steps.

Input: the kimi_swesmith_v1 train parquet (kimi_sft_convert.py; assistant turns are
``<|start_think|>\\n<think>\\n<|end_think|>\\n\\n<json>``). Output: one parquet in the format the datakit stages read
(marin grug_datakit_chat.SnowballDatakitChatFormat): ``conversations`` = list of {role, content, reasoning} with the
think text moved to ``reasoning`` (-> reasoning_content, rendered by the 09-21 training template's think span), and
a row-level ``enable_thinking`` = True (the /think header, the bespoke "think" variant). User turns are unchanged.

Rows are taken in a seeded random order until their 09-21-template token total reaches --target-steps x
--seq-len x --batch (the launcher's token-derived epoch = ceil(tokens / (seq_len x batch)) steps), so one epoch is
exactly --target-steps steps. Rows over --seq-len tokens are skipped. Writes train-00000-of-00001.parquet,
parquet.list (--list-root), census.json and SHA256SUMS.

    python kimi_0921_convert.py --src kimi_swesmith_v1/train-00000-of-00001.parquet --tok <09-21 dir> \\
        --out kimi0921_pair_v1 --list-root <dir the parquet will live in>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TURN_RE = re.compile(r"^<\|start_think\|>\n(.*?)\n<\|end_think\|>\n\n(.*)$", re.S)
TEMPLATE_SHA = "6f55d2ce5974bbcc5d1636c564cddddc945465edecc56e3a391486268af4d930"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tok", required=True, help="09-21 export dir (tokenizer + training_chat_template.jinja)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-root", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--target-steps", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    tpl_path = Path(a.tok) / "training_chat_template.jinja"
    tpl = tpl_path.read_text()
    assert hashlib.sha256(tpl_path.read_bytes()).hexdigest() == TEMPLATE_SHA, "not the 09-21 training template"
    tok = AutoTokenizer.from_pretrained(a.tok)
    rows = pq.read_table(a.src).to_pylist()
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    step_tokens = a.seq_len * a.batch
    budget = a.target_steps * step_tokens - step_tokens // 2   # ceil(total / step_tokens) == target_steps
    out, total, skipped_long = [], 0, 0
    for i in order:
        r = rows[i]
        conv = []
        for m in r["conversations"]:
            if m["role"] == "assistant":
                mt = TURN_RE.match(m["content"])
                assert mt, f"assistant turn not in serve format: {r['instance_id']}"
                conv.append({"role": "assistant", "content": mt.group(2), "reasoning": mt.group(1)})
            else:
                conv.append({"role": m["role"], "content": m["content"], "reasoning": ""})
        hf = [{"role": m["role"], "content": m["content"], **({"reasoning_content": m["reasoning"]} if m["reasoning"].strip() else {})}
              for m in conv]
        text = tok.apply_chat_template(hf, chat_template=tpl, tokenize=False, enable_thinking=True)
        n = len(tok(text, add_special_tokens=False)["input_ids"])   # the template emits bos itself
        if n > a.seq_len:
            skipped_long += 1
            continue
        if total + n > budget:
            break
        total += n
        out.append({"task": r["task"], "instance_id": r["instance_id"], "repo": r["repo"], "result": r["result"],
                    "tokens_0921": n, "enable_thinking": True, "conversations": conv})
    steps = -(-total // step_tokens)
    assert steps == a.target_steps, (steps, total)
    schema = pa.schema([("task", pa.string()), ("instance_id", pa.string()), ("repo", pa.string()), ("result", pa.string()),
                        ("tokens_0921", pa.int64()), ("enable_thinking", pa.bool_()),
                        ("conversations", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning", pa.string())])))])
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out, schema=schema), od / "train-00000-of-00001.parquet", compression="zstd")
    sha = hashlib.sha256((od / "train-00000-of-00001.parquet").read_bytes()).hexdigest()
    (od / "parquet.list").write_text(f"{a.list_root.rstrip('/')}/train-00000-of-00001.parquet\n")
    (od / "SHA256SUMS").write_text(f"{sha}  train-00000-of-00001.parquet\n")
    census = {"src": a.src, "seed": a.seed, "rows": len(out), "tokens_0921": total, "token_derived_steps": steps,
              "seq_len": a.seq_len, "batch": a.batch, "skipped_over_seq_len": skipped_long,
              "content_sha256": hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest(), "parquet_sha256": sha}
    (od / "census.json").write_text(json.dumps(census, indent=1))
    print(json.dumps(census, indent=1))


if __name__ == "__main__":
    main()
