#!/usr/bin/env python3
"""Build the Snowball SFT parquet from the Kimi-K2.5 SWE-smith Terminus-2 traces.

Input: the seven public shards of
``open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k`` (9,356 rows,
column ``task`` = ``swesmith-NNNNN``), the ``task`` -> SWE-smith instance-id map and the train / held-out
instance lists chosen in ``ai_memory/active/snowball-sft/data/kimi_swesmith_set.md`` (Stage-3 overlap,
pygments, verifier errors removed; three repos held out whole; repo cap).

Output: ``train-00000-of-00001.parquet`` and ``heldout-00000-of-00001.parquet`` in the column layout the
``r2egym`` stage already reads (``conversations`` = list of {role, content}), plus ``task_ids.txt``,
``parquet.list`` (the Jupiter path the chain's prep step expects), ``census.json`` and ``SHA256SUMS``.

Format rules (train == serve, see memory ``sft-think-tags-must-be-special-tokens``):
  * every assistant turn is rewritten to ``<|start_think|>\\n<think text>\\n<|end_think|>\\n\\n<json>``,
    the form Snowball's own Delphi template renders and the model emits at serve time; the text tags
    ``<think>``/``</think>`` the release carries tokenize as plain text and are never left in place;
  * a turn with no think block gets the template's empty block ``<|start_think|>\\n\\n<|end_think|>\\n\\n``;
  * user turns are untouched;
  * rows over the 32,768-token packing length (Marin tokenizer, chat template applied) are dropped.

    python kimi_sft_convert.py --shards 'kimi_src/*.parquet' --tokenizer-dir marin_tok \\
        --idmap taskindex_to_instance.json --train-ids kimi_train_ids.txt --heldout-ids kimi_heldout_ids.txt \\
        --out kimi_swesmith_v1 --jupiter-root /e/data1/mmlaion/lee27/snowball-sft/data/kimi_swesmith_v1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

START, END = "<|start_think|>", "<|end_think|>"
THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)
PACK_LEN = 32_768
OUT_COLUMNS = ["task", "result", "conversations", "model", "agent", "trial_name", "episode", "instance_id", "repo"]


def rewrite_assistant(content: str, stats: Counter) -> str | None:
    """Return the assistant turn in Snowball's serve format, or None if it is malformed."""
    if START in content or END in content:
        stats["already_special"] += 1
        return content
    m = THINK_RE.match(content)
    if m:
        think = m.group(1).strip("\n").strip()
        rest = content[m.end():].strip()
        stats["think_rewritten"] += 1
    else:
        if "<think>" in content or "</think>" in content:
            stats["malformed_think"] += 1  # a tag somewhere other than a leading block
            return None
        think, rest = "", content.strip()
        stats["empty_think_inserted"] += 1
    if not rest:
        stats["empty_body"] += 1
        return None
    return f"{START}\n{think}\n{END}\n\n{rest}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="glob of the source parquet shards")
    ap.add_argument("--tokenizer-dir", required=True, help="dir with tokenizer.json + tokenizer_config.json + chat_template.jinja")
    ap.add_argument("--idmap", required=True)
    ap.add_argument("--train-ids", required=True)
    ap.add_argument("--heldout-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jupiter-root", required=True, help="where the out dir will live on Jupiter (for parquet.list)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert tok.convert_tokens_to_ids(START) == 128002 and tok.convert_tokens_to_ids(END) == 128003, "wrong tokenizer"
    idmap = json.load(open(args.idmap))
    train_ids = set(Path(args.train_ids).read_text().split())
    heldout_ids = set(Path(args.heldout_ids).read_text().split())
    assert not (train_ids & heldout_ids), "train / held-out overlap"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    rows = {"train": [], "heldout": []}
    lengths = {"train": [], "heldout": []}
    think_tok = []
    files = sorted(glob.glob(args.shards))
    assert files, f"no shards match {args.shards}"
    for f in files:
        t = pq.read_table(f)
        for r in t.to_pylist():
            stats["rows_in"] += 1
            inst = idmap.get(r["task"])
            if inst is None:
                stats["no_instance_id"] += 1
                continue
            split = "train" if inst in train_ids else "heldout" if inst in heldout_ids else None
            if split is None:
                stats["not_selected"] += 1
                continue
            conv, bad = [], False
            for m in r["conversations"]:
                if m["role"] == "assistant":
                    new = rewrite_assistant(m["content"] or "", stats)
                    if new is None:
                        bad = True
                        break
                    if split == "train":
                        tm = re.search(re.escape(START) + r"\n(.*?)\n" + re.escape(END), new, re.S)
                        think_tok.append(len(tok.encode(tm.group(1), add_special_tokens=False)) if tm else 0)
                    conv.append({"role": "assistant", "content": new})
                else:
                    conv.append({"role": m["role"], "content": m["content"]})
            if bad:
                stats[f"dropped_malformed_{split}"] += 1
                continue
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]  # the template already emits bos
            n = len(ids)
            if n > PACK_LEN:
                stats[f"dropped_over_{PACK_LEN}_{split}"] += 1
                continue
            assert ids.count(128002) == sum(1 for m in conv if m["role"] == "assistant"), "special think token count != assistant turns"
            lengths[split].append(n)
            rows[split].append(
                {
                    "task": r["task"],
                    "result": r["result"],
                    "conversations": conv,
                    "model": r["model"],
                    "agent": r["agent"],
                    "trial_name": r["trial_name"],
                    "episode": r["episode"],
                    "instance_id": inst,
                    "repo": inst.split(".")[0],
                }
            )

    schema = pa.schema(
        [
            ("task", pa.string()),
            ("result", pa.string()),
            ("conversations", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            ("model", pa.string()),
            ("agent", pa.string()),
            ("trial_name", pa.string()),
            ("episode", pa.string()),
            ("instance_id", pa.string()),
            ("repo", pa.string()),
        ]
    )
    written = {}
    for split, name in [("train", "train-00000-of-00001.parquet"), ("heldout", "heldout-00000-of-00001.parquet")]:
        rs = sorted(rows[split], key=lambda x: x["instance_id"])
        table = pa.Table.from_pylist(rs, schema=schema)
        pq.write_table(table, out / name, compression="zstd")
        written[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "task_ids.txt").write_text("\n".join(sorted(x["instance_id"] for x in rows["train"])) + "\n")
    (out / "heldout_ids.txt").write_text("\n".join(sorted(x["instance_id"] for x in rows["heldout"])) + "\n")
    (out / "parquet.list").write_text(f"{args.jupiter_root.rstrip('/')}/train-00000-of-00001.parquet\n")
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in written.items()))

    def q(v, p):
        v = sorted(v)
        return v[int(p * len(v))] if v else 0

    tr = lengths["train"]
    census = {
        "source_shards": [Path(f).name for f in files],
        "counts": dict(stats),
        "train_rows": len(rows["train"]),
        "heldout_rows": len(rows["heldout"]),
        "train_pass_rows": sum(1 for x in rows["train"] if str(x["result"]) in ("1.0", "1")),
        "train_repos": len({x["repo"] for x in rows["train"]}),
        "train_tokens_total": sum(tr),
        "train_tokens_median": st.median(tr) if tr else 0,
        "train_tokens_p90": q(tr, 0.9),
        "train_tokens_max": max(tr) if tr else 0,
        "heldout_tokens_total": sum(lengths["heldout"]),
        "think_tokens_median": st.median(think_tok) if think_tok else 0,
        "think_tokens_mean": round(st.mean(think_tok), 1) if think_tok else 0,
        "think_tokens_p90": q(think_tok, 0.9),
        "pack_len": PACK_LEN,
        "format": f"{START}\\n<think>\\n{END}\\n\\n<json> per assistant turn; empty block when the source turn had none",
        "sha256": written,
    }
    (out / "census.json").write_text(json.dumps(census, indent=1))
    print(json.dumps(census, indent=1))


if __name__ == "__main__":
    main()
