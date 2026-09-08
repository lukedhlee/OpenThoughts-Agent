"""Does the dropped separator token actually change what the model does?

Harbor's continuation drops the newline a Qwen-style template puts after the
assistant's end-of-turn token. "The prompt is malformed" is a weaker claim than
"the model behaves differently", so this measures the second directly: the same
conversation, served both ways, same seeds, and how far the outputs diverge.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import httpx


def tokenize_chat(client, base_url, model, messages, add_generation_prompt) -> List[int]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    r = client.post(f"{root}/tokenize",
                    json={"model": model, "messages": messages,
                          "add_generation_prompt": add_generation_prompt}, timeout=60.0)
    r.raise_for_status()
    return r.json()["tokens"]


def complete(client, base_url, model, prompt_ids, max_tokens, seed, temperature):
    r = client.post(f"{base_url}/completions",
                    json={"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
                          "temperature": temperature, "seed": seed, "logprobs": 0,
                          "return_token_ids": True}, timeout=300.0)
    r.raise_for_status()
    payload = r.json()
    choice = payload["choices"][0]
    return choice.get("text") or "", choice.get("token_ids") or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--separator-id", type=int, default=198, help="the token harbor drops (Qwen: newline)")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = httpx.Client()
    tasks = [
        "List three shell commands to find large files.",
        "Write a Python one-liner that reverses a dict.",
        "Explain what `set -euo pipefail` does.",
        "Give a regex for an ISO date.",
    ]
    observation = "$ ls -la\ntotal 8\n-rw-r--r-- 1 root root 0 config.json\n"

    identical = 0
    first_token_same = 0
    rows: List[Dict[str, Any]] = []
    for i in range(args.trials):
        task = tasks[i % len(tasks)]
        convo = [
            {"role": "system", "content": "You are a terminal agent."},
            {"role": "user", "content": task},
            {"role": "assistant", "content": "Sure. Here is what I would run."},
            {"role": "user", "content": observation},
        ]
        canonical = tokenize_chat(client, args.base_url, args.model, convo, True)
        # Harbor's version: the separator after the assistant turn-end is missing.
        # Find the assistant end-of-turn and drop the separator that follows it.
        broken = list(canonical)
        for j in range(len(broken) - 1, 0, -1):
            if broken[j] == args.separator_id:
                # the separator immediately preceding the next <|im_start|> block
                broken.pop(j)
                break

        seed = 1000 + i
        text_a, ids_a = complete(client, args.base_url, args.model, canonical, args.max_tokens, seed, args.temperature)
        text_b, ids_b = complete(client, args.base_url, args.model, broken, args.max_tokens, seed, args.temperature)
        same = ids_a == ids_b
        identical += same
        ft = bool(ids_a) and bool(ids_b) and ids_a[0] == ids_b[0]
        first_token_same += ft
        if not same and len(rows) < 5:
            k = next((x for x, (p, q) in enumerate(zip(ids_a, ids_b)) if p != q), min(len(ids_a), len(ids_b)))
            rows.append({"trial": i, "diverges_at_token": k,
                         "canonical": text_a[:160], "harbor": text_b[:160]})

    result = {
        "model": args.model,
        "trials": args.trials,
        "identical_completions": identical,
        "identical_fraction": identical / max(args.trials, 1),
        "same_first_token": first_token_same,
        "same_first_token_fraction": first_token_same / max(args.trials, 1),
        "examples": rows,
    }
    print(json.dumps(result, indent=2)[:2600])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
