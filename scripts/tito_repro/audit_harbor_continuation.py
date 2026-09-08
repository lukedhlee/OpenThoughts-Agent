"""Audit Harbor's token-preserving continuation against the chat endpoint.

marin-community/harbor#111 stops re-tokenizing history: it renders only the new
user turn against a fixed dummy base and appends it to the exact served prompt +
completion IDs. Whether that reproduces the template depends on how the previous
turn's end-of-turn marker is accounted for, and that logic is gated on vLLM
reporting a non-null integer ``stop_reason``.

This checks the assembled prompt against ground truth: what /v1/chat/completions
would itself have tokenized for the same conversation. Harbor's own guard cannot
catch a mismatch here -- it only verifies that vLLM ran on the IDs it was sent,
which is true even when those IDs are malformed.

Read-only against a running vLLM.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import httpx

class UnstablePrefix(Exception):
    """Harbor's own precondition failed: base is not a prefix of base+user."""

    def __init__(self, base_ids, continued_ids, offset):
        super().__init__("The vLLM chat template does not have a stable continuation prefix")
        self.base_ids = base_ids
        self.continued_ids = continued_ids
        self.offset = offset


# Verbatim from harbor/src/harbor/llms/lite_llm.py.
_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE = [
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "ok", "reasoning_content": "thinking"},
]


def tokenize_chat(client: httpx.Client, base_url: str, model: str, messages: List[Dict[str, Any]],
                  add_generation_prompt: bool, template_options: Optional[Dict[str, Any]] = None) -> List[int]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    payload = {"model": model, "messages": messages,
               "add_generation_prompt": add_generation_prompt, **(template_options or {})}
    r = client.post(f"{root}/tokenize", json=payload, timeout=60.0)
    r.raise_for_status()
    return r.json()["tokens"]


def build_continuation_prompt_token_ids(
    client, base_url, model, prompt: str,
    previous_prompt_token_ids: List[int],
    previous_completion_token_ids: List[int],
    previous_stop_reason,
    template_options: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Port of harbor's build_continuation_prompt_token_ids, logic unchanged."""
    base_ids = tokenize_chat(client, base_url, model, _TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE, False, template_options)
    continued_ids = tokenize_chat(
        client, base_url, model,
        [*_TOKEN_IN_TOKEN_OUT_TEMPLATE_BASE, {"role": "user", "content": prompt}],
        True, template_options)
    if continued_ids[: len(base_ids)] != base_ids:
        # Harbor raises here. Report what actually diverged instead, so the cause
        # is visible rather than just the symptom.
        i = next((k for k, (a, b) in enumerate(zip(base_ids, continued_ids)) if a != b),
                 min(len(base_ids), len(continued_ids)))
        raise UnstablePrefix(base_ids, continued_ids, i)

    cut = len(base_ids)
    if (
        isinstance(previous_stop_reason, int)
        and previous_completion_token_ids
        and previous_completion_token_ids[-1] == previous_stop_reason
        and base_ids
        and base_ids[-1] != previous_stop_reason
        and previous_stop_reason in base_ids
    ):
        cut = len(base_ids) - 1 - base_ids[::-1].index(previous_stop_reason)

    continuation_ids = continued_ids[cut:]
    if (
        continuation_ids
        and previous_completion_token_ids
        and continuation_ids[0] == previous_completion_token_ids[-1]
        and continuation_ids[0] == previous_stop_reason
    ):
        continuation_ids = continuation_ids[1:]
    return [*previous_prompt_token_ids, *previous_completion_token_ids, *continuation_ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = httpx.Client()
    system = {"role": "system", "content": "You are a terminal agent."}
    user0 = {"role": "user", "content": "Print a Python dict literal and nothing else."}
    observation = "$ python a.py\nKeyError: '{/'\n"

    # Turn 0 exactly as Harbor issues it today.
    body = {"model": args.model, "messages": [system, user0],
            "max_tokens": args.max_tokens, "temperature": 0.7, "seed": 3,
            "return_token_ids": True}
    r = client.post(f"{args.base_url}/chat/completions", json=body, timeout=300.0)
    r.raise_for_status()
    payload = r.json()
    choice = payload["choices"][0]
    prompt_ids = payload.get("prompt_token_ids") or choice.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    content = choice["message"].get("content") or ""

    report: Dict[str, Any] = {
        "model": args.model,
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "stop_reason_type": type(choice.get("stop_reason")).__name__,
        "completion_tail_ids": completion_ids[-4:] if completion_ids else None,
        "n_prompt_ids": len(prompt_ids or []),
        "n_completion_ids": len(completion_ids or []),
    }

    # What Harbor will send for turn 1.
    try:
        harbor_prompt = build_continuation_prompt_token_ids(
        client, args.base_url, args.model, observation, prompt_ids, completion_ids,
            choice.get("stop_reason"))
    except UnstablePrefix as exc:
        report["harbor_precondition"] = "FAILED — build_continuation_prompt_token_ids raises"
        report["unstable_prefix_offset"] = exc.offset
        report["base_window"] = exc.base_ids[max(0, exc.offset - 4): exc.offset + 6]
        report["continued_window"] = exc.continued_ids[max(0, exc.offset - 4): exc.offset + 6]
        report["base_len"] = len(exc.base_ids)
        report["continued_len"] = len(exc.continued_ids)
        print(json.dumps(report, indent=2))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(report, fh, indent=2)
        return 0

    # Ground truth: what the chat endpoint tokenizes for the same conversation.
    canonical = tokenize_chat(
        client, args.base_url, args.model,
        [system, user0, {"role": "assistant", "content": content}, {"role": "user", "content": observation}],
        True)

    report["harbor_len"] = len(harbor_prompt)
    report["canonical_len"] = len(canonical)
    report["matches_canonical"] = harbor_prompt == canonical
    if harbor_prompt != canonical:
        i = next((k for k, (a, b) in enumerate(zip(harbor_prompt, canonical)) if a != b),
                 min(len(harbor_prompt), len(canonical)))
        report["first_divergence"] = i
        report["harbor_window"] = harbor_prompt[max(0, i - 4): i + 6]
        report["canonical_window"] = canonical[max(0, i - 4): i + 6]
        # Also report the join region, which is where an end-of-turn marker goes missing.
        join = len(prompt_ids) + len(completion_ids)
        report["join_offset"] = join
        report["harbor_at_join"] = harbor_prompt[join - 3: join + 5]
        report["canonical_at_join"] = canonical[join - 3: join + 5] if len(canonical) > join else None

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
