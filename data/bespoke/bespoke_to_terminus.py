#!/usr/bin/env python3
"""Convert the private Bespoke Horizon trajectory export into Terminus-2 SFT rows (template-agnostic).

Input: the unpacked export (``manifest.json`` + ``trajectories/<model>/<rollout>.json``, each a Horizon message
array). The prompt is already the harbor terminus-2 prompt; assistant replies are tool calls
(``bash_command`` {keystrokes, duration}, ``mark_task_complete``) with "Analysis: ...\\nPlan: ..." text and the
reasoning in ``content_json['reasoning']``.

Output (one parquet per variant, ``--variants all,noglm``): column ``conversations`` = list of
{role, content, reasoning}; assistant ``content`` is the terminus-2 JSON reply
{"analysis", "plan", "commands": [{"keystrokes", "duration"}], "task_complete"}, ``reasoning`` the raw thinking
("" when the export has none). Observation turns become ``user`` turns, as terminus-2 sends them. Rendering the
thinking (special think tokens, /think vs /nothink header) is the tokenization step's job, not this script's.

Held-out: a fixed set of whole tasks (``--heldout-frac`` of the 162, hashed by task_id) for held-out NLL; the same
tasks are held out in every variant.

PRIVATE DATA: never upload the input or the output anywhere public (HF, public gists). Outputs stay under the
session scratchpad and a mode-700 directory on Jupiter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

AP_RE = re.compile(r"^\s*Analysis:\s*(.*?)\n\s*Plan:\s*(.*)\s*$", re.S)
WATCHDOG_RE = re.compile(r"\n?\[horizon watchdog\].*?(?=\n\S|\Z)", re.S)
# Horizon appends a terminal-usage guidance block to every task prompt; harbor terminus-2 never sends it
GUIDANCE_RE = re.compile(r"\n*## Terminal usage \(important\).*?<!-- horizon:terminal-usage-guidance -->", re.S)
TEACHERS = {"z-ai/glm-5.2": "glm52", "Qwen/Qwen3.8-Max": "qwen38", "Qwen/Qwen3.6-35B-A3B": "qwen36"}


def parse_cj(cj):
    if cj in (None, "None", ""):
        return {}
    if isinstance(cj, dict):
        return cj
    try:
        return ast.literal_eval(cj)
    except Exception:
        return json.loads(cj)


def convert(msgs, stats: Counter, strip_watchdog: bool):
    conv = []
    for x in msgs:
        role, text = x["role"], x.get("content") or ""
        if role == "assistant":
            d = parse_cj(x.get("content_json"))
            m = AP_RE.match(text)
            if m:
                analysis, plan = m.group(1).strip(), m.group(2).strip()
            else:
                analysis, plan = text.strip(), ""
                stats["no_analysis_plan_split"] += 1
            cmds, done = [], False
            for tc in d.get("tool_calls") or []:
                if tc["function_name"] == "bash_command":
                    a = tc.get("arguments") or {}
                    cmds.append({"keystrokes": a.get("keystrokes", ""), "duration": a.get("duration", 1.0)})
                elif tc["function_name"] == "mark_task_complete":
                    done = True
                else:
                    stats[f"unknown_tool:{tc['function_name']}"] += 1
            reply = {"analysis": analysis, "plan": plan, "commands": cmds, "task_complete": done}
            reasoning = (d.get("reasoning") or "").strip()
            stats["assistant_turns"] += 1
            stats["assistant_with_reasoning"] += bool(reasoning)
            conv.append({"role": "assistant", "content": json.dumps(reply, indent=2, ensure_ascii=False),
                         "reasoning": reasoning})
        else:
            if strip_watchdog and "<!-- horizon:terminal-usage-guidance -->" in text:
                text, n = GUIDANCE_RE.subn("", text)
                stats["guidance_blocks_stripped"] += n
            if "[horizon watchdog]" in text:
                stats["watchdog_turns"] += 1
                if strip_watchdog:
                    text = WATCHDOG_RE.sub("", text)
            conv.append({"role": "user", "content": text, "reasoning": ""})
    # the reply must end the conversation; drop a trailing observation
    while conv and conv[-1]["role"] != "assistant":
        conv.pop()
        stats["trailing_obs_dropped"] += 1
    return conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unpacked export dir (manifest.json + trajectories/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="all,noglm")
    ap.add_argument("--heldout-frac", type=float, default=0.05)
    ap.add_argument("--strip-watchdog", action="store_true", help="remove Horizon-only text: [horizon watchdog] nudges and the terminal-usage guidance block")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(src / "manifest.json"))["trajectories"]
    tasks = sorted({r["task_id"] for r in man})
    heldout = {t for t in tasks if int(hashlib.sha256(t.encode()).hexdigest(), 16) % 10_000 < args.heldout_frac * 10_000}
    stats = Counter()
    rows = []
    for r in man:
        msgs = json.load(open(src / r["path"]))
        conv = convert(msgs, stats, args.strip_watchdog)
        rows.append({"task": r["task_id"], "task_name": r["task_name"], "teacher": TEACHERS[r["model"]],
                     "rollout_id": r["rollout_id"], "split": "heldout" if r["task_id"] in heldout else "train",
                     "conversations": conv})
    schema = pa.schema([("task", pa.string()), ("task_name", pa.string()), ("teacher", pa.string()),
                        ("rollout_id", pa.string()), ("split", pa.string()),
                        ("conversations", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()),
                                                               ("reasoning", pa.string())])))])
    census = {"stats": dict(stats), "heldout_tasks": len(heldout), "tasks": len(tasks), "variants": {}}
    for v in args.variants.split(","):
        keep = [r for r in rows if v == "all" or (v == "noglm" and r["teacher"] != "glm52")]
        for split in ("train", "heldout"):
            sel = sorted([r for r in keep if r["split"] == split], key=lambda r: r["rollout_id"])
            p = out / v / f"{split}-00000-of-00001.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(sel, schema=schema), p, compression="zstd")
            census["variants"].setdefault(v, {})[split] = {"rows": len(sel), "tasks": len({r["task"] for r in sel}),
                                                           "by_teacher": dict(Counter(r["teacher"] for r in sel))}
    (out / "census.json").write_text(json.dumps(census, indent=1))
    print(json.dumps(census, indent=1))


if __name__ == "__main__":
    main()
