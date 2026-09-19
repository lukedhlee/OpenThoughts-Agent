#!/usr/bin/env python3
"""Build the instruction-following slice for the OTA + IF mix from the if-v2 Terminus-2 traces.

Source: open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces @ 50b7f77 (25,187 traces, teacher
Qwen3.5-122B-A10B-FP8, Nemotron-RL instruction-following tasks: write a constrained answer to /app/answer.txt).
Selection = the clean subset: verifier result 1.0, no harness parse-error / warning feedback turn, every
assistant turn rewritable into Snowball's serve format (kimi_sft_convert.rewrite_assistant), <= 32,768 tokens.
Failing rows are NOT training data (a silently wrong answer with no in-context feedback); they are the RL pool.
A held-out set of HELDOUT tasks (by task hash) is written for the harness probe and never trained on.
Output schema = ota_sft_convert.py's, slice "ifv2", so ota_if_merge.py can append it to ota-all-train.

    python data/swesmith/ifv2_sft_convert.py --tokenizer-dir <S3 export> --out sft/ifv2
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, statistics as st
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from kimi_sft_convert import PACK_LEN, rewrite_assistant

REPO, REV = "open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces", "50b7f77"
FEEDBACK = ("Previous response had parsing errors", "parsing error", "WARNINGS:", "Extra text detected")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--heldout", type=int, default=500, help="tasks held out for the harness probe")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    assert tok.convert_tokens_to_ids("<|start_think|>") == 128002
    d = snapshot_download(REPO, repo_type="dataset", revision=REV, allow_patterns=["data/*.parquet"])
    stats = collections.Counter(); rows = {"train": [], "heldout": []}; lengths = []; heldout_tasks = []
    def split_of(task):
        h = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 25_187
        return "heldout" if h < a.heldout else "train"
    for f in sorted(glob.glob(d + "/data/*.parquet")):
        for r in pq.read_table(f).to_pylist():
            stats["rows_in"] += 1
            split = split_of(r["task"])
            if split == "heldout":
                heldout_tasks.append({"task": r["task"], "result": r["result"], "verifier_output": r["verifier_output"]})
                stats["heldout_tasks"] += 1
                continue
            if r["result"] != "1.0":
                stats["drop:not_passed"] += 1; continue
            if any(m["role"] == "user" and any(k in (m["content"] or "") for k in FEEDBACK) for m in r["conversations"]):
                stats["drop:harness_feedback"] += 1; continue
            conv, bad = [], False
            for m in r["conversations"]:
                if m["role"] != "assistant":
                    conv.append({"role": m["role"], "content": m["content"]}); continue
                new = rewrite_assistant(m["content"] or "", stats)
                if new is None: bad = True; break
                conv.append({"role": "assistant", "content": new})
            if bad: stats["drop:malformed"] += 1; continue
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) > PACK_LEN: stats["drop:over_len"] += 1; continue
            assert ids.count(128002) == sum(1 for m in conv if m["role"] == "assistant")
            lengths.append(len(ids))
            rows["train"].append({"task": r["task"], "base_task": r["task"], "row_id": f"{Path(f).name}:{r['trial_name']}",
                "slice": "ifv2", "kind": "action", "trace_source": "nemotron-gym-if-v2", "result": r["result"],
                "conversations": conv, "model": r["model"], "agent": r["agent"], "trial_name": r["trial_name"],
                "episode": r["episode"], "instance_id": r["task"], "audit_reasons": "", "final_complete": True})
    schema = pa.schema([("task", pa.string()), ("base_task", pa.string()), ("row_id", pa.string()), ("slice", pa.string()),
        ("kind", pa.string()), ("trace_source", pa.string()), ("result", pa.string()),
        ("conversations", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))), ("model", pa.string()),
        ("agent", pa.string()), ("trial_name", pa.string()), ("episode", pa.string()), ("instance_id", pa.string()),
        ("audit_reasons", pa.string()), ("final_complete", pa.bool_())])
    a.out.mkdir(parents=True, exist_ok=True)
    name = "ifv2-train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(sorted(rows["train"], key=lambda x: x["row_id"]), schema=schema), a.out / name,
                   compression="zstd", row_group_size=1000)
    (a.out / "ifv2-heldout-tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in heldout_tasks))
    census = {"source": f"{REPO} @ {REV}", "counts": dict(stats), "train_rows": len(rows["train"]), "train_tokens": sum(lengths),
              "tokens_median": st.median(lengths), "tokens_p90": sorted(lengths)[int(.9 * len(lengths))], "tokens_max": max(lengths),
              "heldout_tasks": len(heldout_tasks), "heldout_pass_rate_teacher": sum(t["result"] == "1.0" for t in heldout_tasks) / max(1, len(heldout_tasks)),
              "sha256": {name: hashlib.sha256((a.out / name).read_bytes()).hexdigest()}}
    (a.out / "census.json").write_text(json.dumps(census, indent=1)); print(json.dumps(census, indent=1))

if __name__ == "__main__":
    main()
