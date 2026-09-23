#!/usr/bin/env python3
"""Render the SFT variants from bespoke_to_terminus.py output (run it with --strip-watchdog).

Variants (x {all, noglm}):
  fold  : /nothink. Reasoning is dropped, except that short, code-free reasoning (<= --fold-max tokens, 09-21
          tokenizer) is prepended to the reply's "analysis". enable_thinking = False.
  think : /think. Reasoning is kept for the think span (reasoning -> reasoning_content at tokenization).
          enable_thinking = True.
Both: stuck streaks are trimmed. Horizon's watchdog fires after several turns whose terminal output did not change;
the converter strips the nudge text, and here the repeated (assistant, unchanged observation) pairs just before it are
cut down to the first one, so the recovery reply follows a single unchanged observation.
Rows whose rendered length exceeds --max-tokens are dropped (the 65,536 packing length).

PRIVATE DATA: outputs stay local (scratchpad) and on Jupiter in a mode-700 dir; never upload to HF.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

CODE_RE = re.compile(r"```|^\s{4}\S|\bdef \w+\(|\bimport \w|<<'?EOF|<<'?PY", re.M)


def norm_obs(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("Current terminal state:", "")).strip()


def trim_stuck(conv, had_watchdog, stats):
    """conv alternates user/assistant after the first user turn. had_watchdog[i] marks user turns that carried a nudge."""
    out = list(conv)
    i = len(out) - 1
    while i > 0:
        if out[i]["role"] == "user" and had_watchdog.get(id(out[i])):
            s = norm_obs(out[i]["content"])
            j = i  # walk back over (assistant, user) pairs whose observation equals this one
            while j - 2 > 0 and out[j - 1]["role"] == "assistant" and out[j - 2]["role"] == "user" \
                    and norm_obs(out[j - 2]["content"]) == s:
                j -= 2
            # pairs out[j..i-1] are (assistant, unchanged obs) repeats before the nudge; keep the first pair (j-1, j)
            drop = out[j + 1:i + 1] if j < i else []
            if drop:
                stats["stuck_pairs_trimmed"] += len(drop) // 2
                out = out[:j + 1] + out[i + 1:]
                i = j
        i -= 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="bespoke_to_terminus.py --strip-watchdog output dir (has all/)")
    ap.add_argument("--raw", required=True, help="unstripped converter output dir, to know which turns had a nudge")
    ap.add_argument("--tok", required=True, help="09-21 tokenizer dir (tokenizer.json + training_chat_template.jinja)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-max", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=65_536)
    ap.add_argument("--jupiter-root", default="/e/data1/mmlaion/lee27/snowball-sft/data/bespoke_v1")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tok)
    tpl = (Path(args.tok) / "training_chat_template.jinja").read_text()
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False))
    out = Path(args.out)
    census = {}
    for split in ("train", "heldout"):
        src = pq.read_table(Path(args.src) / "all" / f"{split}-00000-of-00001.parquet").to_pylist()
        raw = {r["rollout_id"]: r for r in pq.read_table(Path(args.raw) / "all" / f"{split}-00000-of-00001.parquet").to_pylist()}
        for mode in ("fold", "think"):
            for sub in ("all", "noglm"):
                stats, rows, lens = Counter(), [], []
                for r in src:
                    if sub == "noglm" and r["teacher"] == "glm52":
                        continue
                    conv = [dict(m) for m in r["conversations"]]
                    rawc = raw[r["rollout_id"]]["conversations"]
                    hw = {id(conv[k]): ("[horizon watchdog]" in rawc[k]["content"]) for k in range(len(conv)) if conv[k]["role"] == "user"}
                    conv = trim_stuck(conv, hw, stats)
                    msgs = []
                    for m in conv:
                        if m["role"] != "assistant":
                            msgs.append({"role": "user", "content": m["content"], "reasoning": ""})
                            continue
                        reason = m["reasoning"] or ""
                        if mode == "think":
                            msgs.append({"role": "assistant", "content": m["content"], "reasoning": reason})
                            stats["think_turns_with_reasoning"] += bool(reason)
                            continue
                        reply = json.loads(m["content"])
                        if reason and ntok(reason) <= args.fold_max and not CODE_RE.search(reason):
                            reply["analysis"] = reason.strip() + "\n\n" + reply["analysis"]
                            stats["fold_turns_folded"] += 1
                        elif reason:
                            stats["fold_turns_dropped_reasoning"] += 1
                        msgs.append({"role": "assistant", "content": json.dumps(reply, indent=2, ensure_ascii=False), "reasoning": ""})
                    hf = [{"role": m["role"], "content": m["content"], **({"reasoning_content": m["reasoning"]} if m["reasoning"] else {})} for m in msgs]
                    text = tok.apply_chat_template(hf, chat_template=tpl, tokenize=False, enable_thinking=(mode == "think"))
                    n = len(tok(text, add_special_tokens=False)["input_ids"])
                    if n > args.max_tokens:
                        stats["dropped_over_max"] += 1
                        continue
                    stats["assistant_turns"] += sum(m["role"] == "assistant" for m in msgs)
                    lens.append(n)
                    rows.append({"task": r["task"], "task_name": r["task_name"], "teacher": r["teacher"], "rollout_id": r["rollout_id"],
                                 "enable_thinking": mode == "think", "conversations": msgs})
                v = f"{mode}_{sub}"
                d = out / v
                d.mkdir(parents=True, exist_ok=True)
                schema = pa.schema([("task", pa.string()), ("task_name", pa.string()), ("teacher", pa.string()), ("rollout_id", pa.string()),
                                    ("enable_thinking", pa.bool_()),
                                    ("conversations", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("reasoning", pa.string())])))])
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), d / f"{split}-00000-of-00001.parquet", compression="zstd")
                if split == "train":
                    (d / "parquet.list").write_text(f"{args.jupiter_root}/{v}/train-00000-of-00001.parquet\n")
                lens.sort()
                census.setdefault(v, {})[split] = {"rows": len(rows), "tokens": sum(lens),
                                                   "median_tokens": lens[len(lens) // 2] if lens else 0,
                                                   "by_teacher": dict(Counter(x["teacher"] for x in rows)), "stats": dict(stats)}
    (out / "census.json").write_text(json.dumps(census, indent=1))
    for v, c in census.items():
        t = c["train"]
        print(f'{v:12} train rows {t["rows"]:4} tokens {t["tokens"]/1e6:5.1f}M median {t["median_tokens"]:6} | heldout {c["heldout"]["rows"]} | {t["stats"]}')


if __name__ == "__main__":
    main()
