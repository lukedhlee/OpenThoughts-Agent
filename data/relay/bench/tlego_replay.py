#!/usr/bin/env python3
"""Build a bench_client2.py replay file from Terminal-Lego's public Terminus-2 trajectories.

Each row is one trajectory: the real alternating user/assistant history, plus per-turn prompt and completion token
counts under the Qwen3.8-27B tokenizer, so qwen_bench.sbatch replays Terminus-2-shaped multi-turn load. Rows are
sampled evenly across the teacher files (ShareGPT `conversations`, human/gpt) and are not filtered by length, so the
turn-count and context-length mix matches the public runs.

Usage: tlego_replay.py <out.jsonl> <traj.json> [<traj.json> ...] [--n 1200] [--tokenizer <dir or hf id>]
"""
import argparse
import json
import random

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("traj", nargs="+")
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = Tokenizer.from_file(hf_hub_download(a.tokenizer, "tokenizer.json"))
ROLE = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
PER_MESSAGE_OVERHEAD = 5  # chat-template markers per message

random.seed(a.seed)
per_file = a.n // len(a.traj)
rows = []
for path in a.traj:
    data = json.load(open(path))
    random.shuffle(data)
    kept = 0
    for i, r in enumerate(data):
        conv = r.get("conversations") or r.get("messages") or []
        msgs = [{"role": ROLE.get(m.get("from") or m.get("role")), "content": m.get("value") or m.get("content") or ""} for m in conv]
        if not msgs or any(m["role"] != ("user" if j % 2 == 0 else "assistant") for j, m in enumerate(msgs)):
            continue
        n_turns = len(msgs) // 2
        if n_turns < 1:
            continue
        lens = [len(tok.encode(m["content"]).ids) + PER_MESSAGE_OVERHEAD for m in msgs[: 2 * n_turns]]
        prompt_len = [sum(lens[: 2 * t + 1]) for t in range(n_turns)]
        completion_len = [lens[2 * t + 1] for t in range(n_turns)]
        rows.append({"task": (r.get("metadata") or {}).get("task_id") or f"{path.rsplit('/', 1)[-1]}:{i}",
                     "trial": f"{path.rsplit('/', 1)[-1]}:{i}", "n_turns": n_turns,
                     "completion_len": completion_len, "prompt_len": prompt_len, "api_ms": [0] * n_turns,
                     "messages": msgs[: 2 * n_turns]})
        kept += 1
        if kept >= per_file:
            break

random.shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
turns = sorted(r["n_turns"] for r in rows)
ctx = sorted(r["prompt_len"][-1] + r["completion_len"][-1] for r in rows)
q = lambda x, p: x[int(p * (len(x) - 1))]
print(f"wrote {len(rows)} trajectories to {a.out}; turns p50={q(turns, .5)} p90={q(turns, .9)}; "
      f"final context p50={q(ctx, .5)} p90={q(ctx, .9)} tokens")
