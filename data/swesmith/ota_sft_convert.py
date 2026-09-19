#!/usr/bin/env python3
"""Build a Snowball-format SFT parquet from the pinned OpenThoughts-Agent-SFT-100K shards.

Same output contract as ``kimi_sft_convert.py`` (assistant turns on the
``<|start_think|>``/``<|end_think|>`` special tokens, 32,768-token packing length, one parquet per
split) with three deltas the OTA release forces:

  * selection is by ROW ID (``<shard>:<offset>``) against the 2026-09-18 audit manifest, because OTA
    rows carry no canonical identity outside the SWE-smith slice;
  * rows are screened here rather than upstream. ``--screen reference`` (the default since 2026-09-19,
    Luke: "put them back") keeps every row the release trained on and drops only rows flagged for
    evaluation leakage; ``--screen strict`` is the 2026-09-18 screen (structurally incomplete traces,
    Stage-3 overlap and the pygments environment dropped too, 37 % of the release);
  * ``summarization-*`` rows are KEPT. They are Harbor's own proactive-compaction path — the handoff
    prompt is byte-identical to ``terminus_2.py`` — so their trailing prose turns are admitted while
    every earlier turn must still parse as a Terminus-2 action.

Splits are drawn per slice by task hash, so a task's rollouts never straddle train and held-out.

    python data/swesmith/ota_sft_convert.py --tokenizer-dir <dir with tokenizer.json> \\
        --out sft/ota --jupiter-root /p/data1/mmlaion/lee27/sft/ota
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import statistics as st
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kimi_sft_convert import END, PACK_LEN, START, rewrite_assistant

# Tezos turns carry the teacher's CLOSING </think> only: the opening tag was in the prefilled prompt,
# so 38 % of that slice's rows would be dropped as malformed. Restore the tag, never invent a block.
def normalize_think(content: str, stats: collections.Counter) -> str:
    if "<think>" in content or content.count("</think>") != 1:
        return content
    stats["reopened_think"] += 1
    return "<think>\n" + content.lstrip("\n")

SLICES = ("swesmith", "superuser", "issue", "tezos")
STRUCT = {"summarization_fragment", "recorded_error_or_failure", "invalid_action_turn", "no_final_completion"}
DECON = {"stage3_same_task", "stage3_same_issue", "eval_same_issue_or_prompt", "known_broken_pygments_environment"}
SUMM_OK = {"summarization_fragment", "invalid_action_turn", "no_final_completion"}
SWE = {"swesmith", "issue"}
MAX_PROSE_TURNS = 3


EVAL_LEAK = {"eval_same_issue_or_prompt"}


def verdict(r: dict, screen: str = "reference") -> tuple[str | None, str | None]:
    """Return (kind, drop_reason). kind is 'action' or 'compaction' for rows we keep.

    reference: the release's own training set (every trace, timed-out and mid-work ones included, as
    OpenThoughts-Agent trained it) minus rows flagged for evaluation leakage. The screen that gates
    the model is the evaluation, not trace completeness.
    strict: the 2026-09-18 screen, kept for reproducing ota_sft_100k_v1.
    """
    reasons = set(r["reasons"])
    summarization = (r["trace_source"] or "").startswith("summarization")
    if screen == "reference":
        if reasons & EVAL_LEAK:
            return None, "decon"
        if r["source"] in SWE and "eval_patch_line_overlap_review" in r["flags"]:
            return None, "decon_review_flag"
        return ("compaction" if summarization else "action"), None
    if r["source"] in SWE and "missing_original_task_prompt" in reasons:
        return None, "no_task_identity"
    if reasons & DECON:
        return None, "decon"
    if r["source"] in SWE and "eval_patch_line_overlap_review" in r["flags"]:
        return None, "decon_review_flag"
    prose_turns = r["assistant_turns"] - r["valid_action_turns"]
    if summarization:
        if r["result"] not in (None, "1", "1.0"):
            return None, "compaction_run_errored"
        if not (reasons & STRUCT) <= SUMM_OK:
            return None, "compaction_other_defect"
        if prose_turns > MAX_PROSE_TURNS:
            return None, "compaction_too_many_prose_turns"
        return "compaction", None
    if reasons & STRUCT:
        return None, "structural"
    return "action", None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("scratchpad/ota_dedup"), help="dir with the pinned ota-*.parquet shards")
    ap.add_argument("--manifest", type=Path, default=Path("notes/artifacts/ota-sft-audit-20260918/manifest.jsonl"))
    ap.add_argument("--tokenizer-dir", required=True, help="dir with tokenizer.json + tokenizer_config.json + chat_template.jinja")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jupiter-root", required=True, help="where the out dir will live on Jupiter (for parquet.list)")
    ap.add_argument("--slices", default=",".join(SLICES), help="comma list of slices to emit")
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--no-compaction", action="store_true", help="drop the summarization/compaction rows")
    ap.add_argument("--screen", choices=("reference", "strict"), default="reference", help="see verdict()")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert tok.convert_tokens_to_ids(START) == 128002 and tok.convert_tokens_to_ids(END) == 128003, "wrong tokenizer"
    wanted = [s for s in args.slices.split(",") if s]
    assert set(wanted) <= set(SLICES), f"unknown slice in {wanted}"

    selected: dict[str, dict] = {}
    stats: collections.Counter = collections.Counter()
    for line in open(args.manifest):
        r = json.loads(line)
        kind, why = verdict(r, args.screen)
        if kind is None:
            stats[f"drop:{why}"] += 1
            continue
        if kind == "compaction" and args.no_compaction:
            stats["drop:compaction_disabled"] += 1
            continue
        if r["source"] not in wanted:
            stats["drop:slice_not_requested"] += 1
            continue
        selected[r["row_id"]] = {
            "slice": r["source"], "kind": kind, "base_task": r["base_task"], "instance_id": r["instance_id"],
            "audit_reasons": ",".join(sorted(r["reasons"])), "final_complete": bool(r["final_complete"]),
        }

    def split_of(base_task: str) -> str:
        h = int(hashlib.sha256(base_task.encode()).hexdigest()[:8], 16)
        return "heldout" if (h % 10_000) < args.heldout_frac * 10_000 else "train"

    rows: dict[tuple[str, str], list] = collections.defaultdict(list)
    lengths: dict[tuple[str, str], list] = collections.defaultdict(list)
    think_tokens: dict[str, list] = collections.defaultdict(list)
    for shard in sorted(glob.glob(str(args.cache / "ota-*.parquet"))):
        name = Path(shard).name
        table = pq.read_table(shard)
        for offset, r in enumerate(table.to_pylist()):
            pick = selected.get(f"{name}:{offset}")
            if pick is None:
                continue
            stats["rows_in"] += 1
            conv, bad = [], False
            for m in r["conversations"]:
                if m["role"] != "assistant":
                    conv.append({"role": m["role"], "content": m["content"]})
                    continue
                new = rewrite_assistant(normalize_think(m["content"] or "", stats), stats)
                if new is None:
                    bad = True
                    break
                conv.append({"role": "assistant", "content": new})
            if bad:
                stats[f"dropped_malformed_{pick['kind']}"] += 1
                continue
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > PACK_LEN:
                stats[f"dropped_over_{PACK_LEN}"] += 1
                continue
            assert ids.count(128002) == sum(1 for m in conv if m["role"] == "assistant"), "think token count != assistant turns"
            split = split_of(pick["base_task"])
            key = (pick["slice"], split)
            lengths[key].append(len(ids))
            think_tokens[pick["slice"]].append(sum(1 for i in ids if i == 128002))
            rows[key].append(
                {
                    "task": r["task"],
                    "base_task": pick["base_task"],
                    "row_id": f"{name}:{offset}",
                    "slice": pick["slice"],
                    "kind": pick["kind"],
                    "trace_source": r["trace_source"],
                    "result": r["result"],
                    "conversations": conv,
                    "model": r["model"],
                    "agent": r["agent"],
                    "trial_name": r["trial_name"],
                    "episode": r["episode"],
                    "instance_id": pick["instance_id"],
                    "audit_reasons": pick["audit_reasons"],
                    "final_complete": pick["final_complete"],
                }
            )

    schema = pa.schema(
        [
            ("task", pa.string()),
            ("base_task", pa.string()),
            ("row_id", pa.string()),
            ("slice", pa.string()),
            ("kind", pa.string()),
            ("trace_source", pa.string()),
            ("result", pa.string()),
            ("conversations", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
            ("model", pa.string()),
            ("agent", pa.string()),
            ("trial_name", pa.string()),
            ("episode", pa.string()),
            ("instance_id", pa.string()),
            ("audit_reasons", pa.string()),
            ("final_complete", pa.bool_()),
        ]
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for (slice_name, split), rs in sorted(rows.items()):
        rs = sorted(rs, key=lambda x: x["row_id"])
        name = f"{slice_name}-{split}-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(rs, schema=schema), out / name, compression="zstd")
        written[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    root = args.jupiter_root.rstrip("/")
    (out / "parquet.list").write_text("".join(f"{root}/{n}\n" for n in sorted(written) if "-train-" in n))
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(written.items())))

    def q(v, p):
        v = sorted(v)
        return v[int(p * len(v))] if v else 0

    census = {
        "source": "open-thoughts/OpenThoughts-Agent-SFT-100K @ 45fb28fcc38d352133cb28a1c8a43a2f14fea97b",
        "manifest": str(args.manifest),
        "slices": wanted,
        "heldout_frac": args.heldout_frac,
        "screen": args.screen,
        "compaction_rows_kept": not args.no_compaction,
        "counts": dict(stats),
        "per_slice": {
            s: {
                "train_rows": len(rows[(s, "train")]),
                "heldout_rows": len(rows[(s, "heldout")]),
                "train_tokens": sum(lengths[(s, "train")]),
                "heldout_tokens": sum(lengths[(s, "heldout")]),
                "train_tokens_median": st.median(lengths[(s, "train")]) if lengths[(s, "train")] else 0,
                "train_tokens_p90": q(lengths[(s, "train")], 0.9),
                "compaction_rows": sum(1 for x in rows[(s, "train")] + rows[(s, "heldout")] if x["kind"] == "compaction"),
                "tasks_train": len({x["base_task"] for x in rows[(s, "train")]}),
                "tasks_heldout": len({x["base_task"] for x in rows[(s, "heldout")]}),
            }
            for s in wanted
        },
        "train_tokens_total": sum(sum(v) for k, v in lengths.items() if k[1] == "train"),
        "pack_len": PACK_LEN,
        "format": f"{START}\\n<think>\\n{END}\\n\\n<body> per assistant turn; empty block when the source turn had none",
        "sha256": written,
    }
    (out / "census.json").write_text(json.dumps(census, indent=1))
    print(json.dumps(census, indent=1))


if __name__ == "__main__":
    main()
