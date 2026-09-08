"""Reproduce MarinSkyRL#520 against a live vLLM, and test both proposed repairs.

The bug: Harbor sends the whole conversation to vLLM as TEXT every turn. vLLM
re-tokenizes that text, so the prompt IDs it reports for turn t need not extend
the IDs it reported for turn t-1 plus the IDs the model actually sampled. When
they do not, SkyRL declines full-TITO assembly and rebuilds the sequence from a
fresh tokenization, and ``tis/exact_match_fraction`` still reads 1.0.

Two agent loops run against the same server, the same tasks and the same seeds:

  text   -- what Harbor does today: /v1/chat/completions with the message list.
  tokens -- repair A (serving side): /v1/completions with the accumulated
            integer prompt, so the IDs the model sampled are never re-tokenized.
            New observations are encoded against a fixed dummy base, as in the
            ApexAgents TITO recipe referenced from the issue.

Repair B (training side: replay each turn against its own served prefix) needs
no serving change, so it is evaluated offline from the recorded ``text`` streams
-- see ``tito_checks.replay_cost``.

Everything here is read-only against the server; nothing is written outside
--out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tito_checks import (  # noqa: E402
    PREFIX,
    check_tito,
    classify_prefix_failure,
    completion_survived,
    replay_cost,
)

# Tasks chosen to make the model emit code, paths and punctuation clusters --
# where BPE boundaries are least stable, and what a terminal agent emits anyway.
TASKS = [
    "Write a Python function that parses a URL query string into a dict. Show the code.",
    "Write a bash one-liner that finds every *.py file containing 'TODO(' and prints path:line.",
    "Write a Python regex that matches C-style block comments, and explain each group.",
    "Write a JSON config for a web server with nested objects and escaped strings.",
    "Write a Python dataclass for a tokenizer config and a __repr__ that prints it compactly.",
    "Write a shell command using awk and sed to reformat CSV columns 2 and 5.",
    "Write a Python snippet that builds a nested f-string with dict access inside it.",
    "Write a Makefile rule with pattern substitution and a multi-line recipe.",
]

# Synthetic terminal output: the agent's next-turn observation. Kept fixed so the
# two loops differ only in HOW history is transported, never in what is fed back.
OBSERVATIONS = [
    "$ python /tmp/a.py\nTraceback (most recent call last):\n  File \"/tmp/a.py\", line 3, in <module>\n    main()\nKeyError: '{/'\n",
    "$ ls -la /srv/app/{bin,lib}\n-rw-r--r-- 1 root root 0 Jan  1 00:00 /srv/app/bin/run.sh\n",
    "$ ./run.sh --flag=(/{a,b})\nbash: syntax error near unexpected token `('\n",
    "$ pytest -q\nE   assert '(/{' == '(/ {'\n1 failed, 0 passed\n",
    "$ grep -rn 'x=\\\"y\\\"' .\n./cfg.json:12:  \"x\":\"y\",\n",
]


def parse_ids(payload: Dict[str, Any]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Pull (prompt_token_ids, completion_token_ids) out of a raw vLLM response.

    Same field precedence Harbor's extractor uses, so the probe records exactly
    what Harbor would have recorded.
    """
    choice = (payload.get("choices") or [{}])[0]
    prompt_ids = payload.get("prompt_token_ids")
    if prompt_ids is None:
        prompt_ids = choice.get("prompt_token_ids")
    completion_ids = choice.get("token_ids")
    if completion_ids is None:
        psf = choice.get("provider_specific_fields") or {}
        completion_ids = psf.get("token_ids")
    return prompt_ids, completion_ids


def choice_text(payload: Dict[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    if "message" in choice:
        return choice["message"].get("content") or ""
    return choice.get("text") or ""


def _ids(x) -> List[int]:
    """Normalize whatever apply_chat_template/tokenizer returned into List[int].

    transformers 5.x can hand back a BatchEncoding or a nested list where 4.x
    returned a flat list; the probe must not care which.
    """
    if x is None:
        return []
    if isinstance(x, dict) or hasattr(x, "input_ids"):
        x = x["input_ids"] if isinstance(x, dict) else x.input_ids
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [int(v) for v in x]


class ObservationEncoder:
    """Encode a new user observation WITHOUT re-tokenizing anything before it.

    Renders the observation against a fixed dummy conversation and strips the
    dummy's prefix, so the returned IDs depend only on the observation text.
    That is the property repair A needs: prior sampled IDs are appended
    verbatim, never round-tripped through text.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        dummy = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        # Base: dummy conversation with one assistant turn already generated.
        self._base_prefix = _ids(tokenizer.apply_chat_template(
            dummy, add_generation_prompt=True, tokenize=True, **CHAT_KWARGS
        ))
        self._assistant_text = "A"
        with_assistant = _ids(tokenizer.apply_chat_template(
            dummy + [{"role": "assistant", "content": self._assistant_text}],
            add_generation_prompt=False,
            tokenize=True,
            **CHAT_KWARGS,
        ))
        # Tokens the template adds AFTER an assistant message's content, i.e. the
        # end-of-turn marker the sampled completion may or may not already carry.
        assistant_content_ids = _ids(tokenizer(self._assistant_text, add_special_tokens=False))
        head = len(self._base_prefix) + len(assistant_content_ids)
        self.assistant_suffix = list(with_assistant[head:])
        self._with_assistant = list(with_assistant)

    def encode(self, obs_text: str, completion_ids: List[int]) -> List[int]:
        """IDs to append after ``completion_ids``: end-of-turn + observation + generation prompt."""
        full = _ids(self.tok.apply_chat_template(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": self._assistant_text},
                {"role": "user", "content": obs_text},
            ],
            add_generation_prompt=True,
            tokenize=True,
            **CHAT_KWARGS,
        ))
        tail = list(full[len(self._with_assistant) :])
        suffix = list(self.assistant_suffix)
        # The sampled completion usually already ends with the stop token, which
        # is also the first token of the template's assistant suffix. Appending
        # it again would corrupt the template -- the stop-token overlap the
        # ApexAgents implementation calls out.
        overlap = 0
        for k in range(min(len(suffix), len(completion_ids)), 0, -1):
            if completion_ids[-k:] == suffix[:k]:
                overlap = k
                break
        return suffix[overlap:] + tail


CHAT_KWARGS: Dict[str, Any] = {}


async def scrape_metrics(client: httpx.AsyncClient, url: str) -> Dict[str, float]:
    """Read the prefix-cache counters so the two transports can be compared.

    A token-transported prompt is an exact extension of the previous one, so the
    KV prefix cache should hit all the way; a re-cut invalidates it from the
    divergence onward. If that is real it shows up here.
    """
    keys = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total")
    try:
        r = await client.get(url, timeout=30.0)
        r.raise_for_status()
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in keys:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    return out


async def post(client: httpx.AsyncClient, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = await client.post(url, json=body, timeout=600.0)
    r.raise_for_status()
    return r.json()


async def run_text_trajectory(
    client: httpx.AsyncClient, args, task: str, seed: int
) -> Dict[str, Any]:
    """Harbor's path today: resend the message list as text every turn."""
    messages = [
        {"role": "system", "content": "You are a terminal agent. Answer with code when asked."},
        {"role": "user", "content": task},
    ]
    prompts, completions, texts = [], [], []
    for t in range(args.turns):
        body = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": seed + t,
            "return_token_ids": True,
        }
        if CHAT_KWARGS:
            body["chat_template_kwargs"] = dict(CHAT_KWARGS)
        payload = await post(client, f"{args.base_url}/chat/completions", body)
        p_ids, c_ids = parse_ids(payload)
        content = choice_text(payload)
        prompts.append(p_ids)
        completions.append(c_ids)
        texts.append(content)
        messages.append({"role": "assistant", "content": content})
        if t + 1 < args.turns:
            messages.append({"role": "user", "content": OBSERVATIONS[t % len(OBSERVATIONS)]})
    return {"mode": "text", "task": task, "seed": seed, "prompt_token_ids": prompts,
            "completion_token_ids": completions, "texts": texts}


async def run_tokens_trajectory(
    client: httpx.AsyncClient, args, task: str, seed: int, encoder: ObservationEncoder
) -> Dict[str, Any]:
    """Repair A: carry the conversation as token IDs; never re-tokenize history."""
    tokens: List[int] = _ids(
        encoder.tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a terminal agent. Answer with code when asked."},
                {"role": "user", "content": task},
            ],
            add_generation_prompt=True,
            tokenize=True,
            **CHAT_KWARGS,
        )
    )
    prompts, completions, texts, echo_mismatch = [], [], [], []
    for t in range(args.turns):
        body = {
            "model": args.model,
            "prompt": tokens,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": seed + t,
            "return_token_ids": True,
        }
        payload = await post(client, f"{args.base_url}/completions", body)
        p_ids, c_ids = parse_ids(payload)
        # Self-check: the server must run on exactly the IDs we sent.
        echo_mismatch.append(None if p_ids is None else (list(p_ids) != list(tokens)))
        prompts.append(p_ids)
        completions.append(c_ids)
        texts.append(choice_text(payload))
        if c_ids is None:
            break
        tokens = tokens + list(c_ids)
        if t + 1 < args.turns:
            tokens = tokens + encoder.encode(OBSERVATIONS[t % len(OBSERVATIONS)], list(c_ids))
    return {"mode": "tokens", "task": task, "seed": seed, "prompt_token_ids": prompts,
            "completion_token_ids": completions, "texts": texts,
            "echo_mismatch": echo_mismatch}


def _cache_rate(counters: Dict[str, float]) -> str:
    for q, h in (("vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_hits_total"),
                 ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")):
        if counters.get(q):
            return f"{counters.get(h, 0.0) / counters[q]:.1%}"
    return "n/a"


def summarize(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    reasons: Dict[str, int] = {}
    failure_kinds: Dict[str, int] = {}
    declined, total_turns, declining_turns, recut_turns = 0, 0, 0, 0
    replay: List[float] = []
    examples: List[Dict[str, Any]] = []
    for traj in trajectories:
        p, c = traj["prompt_token_ids"], traj["completion_token_ids"]
        if any(x is None for x in p) or any(x is None for x in c):
            reasons["missing_streams"] = reasons.get("missing_streams", 0) + 1
            declined += 1
            continue
        total_turns += max(len(p) - 1, 0)
        result = check_tito(p, c)
        traj["check"] = result
        if result["reason"] is not None:
            declined += 1
            reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
            if result["reason"] == PREFIX and len(examples) < 8:
                examples.append({k: v for k, v in result.items() if k != "expected_ids" or True})
        # Per-turn rates. Two different questions: does the STRICT invariant hold
        # (what the trainer asks), and did the model's own sampled tokens survive
        # (what #520 is about). A chat template can break the first without
        # touching the second.
        for t in range(1, len(p)):
            prev = list(p[t - 1]) + list(c[t - 1])
            if list(p[t])[: len(prev)] != prev:
                declining_turns += 1
                kind = classify_prefix_failure(p[t - 1], c[t - 1], p[t])
                failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
            if not completion_survived(p[t - 1], c[t - 1], p[t]):
                recut_turns += 1
        cost = replay_cost(p, c)
        if cost:
            replay.append(cost["replay_multiplier"])
    # Mode B builds its own prompts, so a zero decline rate is true by
    # construction; what has to be checked is that the SERVER ran on the ids we
    # sent, and that the conversation stayed well formed.
    echo_mismatches = sum(
        1 for t in trajectories for v in (t.get("echo_mismatch") or []) if v
    )
    echo_checked = sum(1 for t in trajectories for v in (t.get("echo_mismatch") or []) if v is not None)
    completion_lengths = [len(c) for t in trajectories for c in t["completion_token_ids"] if c]
    empty_completions = sum(
        1 for t in trajectories for c in t["completion_token_ids"] if not c
    )
    empty_texts = sum(1 for t in trajectories for x in t["texts"] if not x.strip())

    n = len(trajectories)
    # Turn-0 fingerprints: in "text" mode vLLM applies the chat template
    # server-side, in "tokens" mode the client applies it. If these disagree the
    # two loops are not conditioned identically and the comparison is unfair, so
    # record enough to check it offline.
    fingerprints = [
        {
            "task": t["task"][:40],
            "len": len(t["prompt_token_ids"][0]) if t["prompt_token_ids"] and t["prompt_token_ids"][0] else None,
            "head": list(t["prompt_token_ids"][0][:24]) if t["prompt_token_ids"] and t["prompt_token_ids"][0] else None,
            "tail": list(t["prompt_token_ids"][0][-8:]) if t["prompt_token_ids"] and t["prompt_token_ids"][0] else None,
        }
        for t in trajectories[:8]
    ]
    return {
        "turn0_fingerprints": fingerprints,
        "server_echo_checked": echo_checked,
        "server_echo_mismatches": echo_mismatches,
        "mean_completion_tokens": (sum(completion_lengths) / len(completion_lengths)) if completion_lengths else None,
        "empty_completions": empty_completions,
        "empty_texts": empty_texts,
        "n_trajectories": n,
        "n_declined": declined,
        "decline_fraction": declined / max(n, 1),
        "n_turn_boundaries": total_turns,
        "n_declining_turn_boundaries": declining_turns,
        "turn_decline_fraction": declining_turns / max(total_turns, 1),
        "n_recut_turn_boundaries": recut_turns,
        "recut_fraction": recut_turns / max(total_turns, 1),
        "failure_kinds": failure_kinds,
        "reasons": reasons,
        "prefix_examples": examples,
        "replay_multiplier_mean": sum(replay) / len(replay) if replay else None,
    }


async def main_async(args) -> int:
    global CHAT_KWARGS
    if args.no_thinking:
        CHAT_KWARGS = {"enable_thinking": False}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    encoder = ObservationEncoder(tokenizer)

    out: Dict[str, Any] = {
        "config": vars(args),
        "assistant_suffix_ids": encoder.assistant_suffix,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        for mode in args.modes:
            jobs = []
            for i in range(args.trajectories):
                task = TASKS[i % len(TASKS)]
                seed = args.seed + 1000 * i
                if mode == "text":
                    jobs.append(run_text_trajectory(client, args, task, seed))
                else:
                    jobs.append(run_tokens_trajectory(client, args, task, seed, encoder))
            sem = asyncio.Semaphore(args.concurrency)

            async def guarded(coro):
                async with sem:
                    try:
                        return await coro
                    except Exception as exc:  # keep one bad trajectory from killing the run
                        return {"mode": "error", "error": repr(exc), "prompt_token_ids": [],
                                "completion_token_ids": [], "texts": []}

            metrics_url = args.base_url.rsplit("/v1", 1)[0] + "/metrics"
            before = await scrape_metrics(client, metrics_url)
            t0 = time.time()
            results = await asyncio.gather(*(guarded(j) for j in jobs))
            after = await scrape_metrics(client, metrics_url)
            errors = [r for r in results if r.get("mode") == "error"]
            good = [r for r in results if r.get("mode") != "error"]
            out[mode] = {
                "summary": summarize(good),
                "n_errors": len(errors),
                "errors": [e["error"] for e in errors[:5]],
                "wall_seconds": round(time.time() - t0, 1),
                "prefix_cache": {
                    k: after.get(k, 0.0) - before.get(k, 0.0)
                    for k in set(before) | set(after)
                },
                "trajectories": good if args.dump_trajectories else None,
            }
            s = out[mode]["summary"]
            print(
                f"[{mode}] trajectories={s['n_trajectories']} declined={s['n_declined']} "
                f"({s['decline_fraction']:.1%})  turn_boundaries={s['n_turn_boundaries']} "
                f"declining={s['n_declining_turn_boundaries']} ({s['turn_decline_fraction']:.1%})  "
                f"recut={s['n_recut_turn_boundaries']} ({s['recut_fraction']:.1%})  "
                f"kinds={s['failure_kinds']}  errors={len(errors)}  "
                f"echo_bad={s['server_echo_mismatches']}/{s['server_echo_checked']}  "
                f"mean_completion={s['mean_completion_tokens'] and round(s['mean_completion_tokens'])}  "
                f"empty={s['empty_completions']}  "
                f"cache={_cache_rate(out[mode]['prefix_cache'])}",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--modes", nargs="+", default=["text", "tokens"], choices=["text", "tokens"])
    ap.add_argument("--trajectories", type=int, default=32)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--dump-trajectories", action="store_true")
    ap.add_argument("--out", default="tito_probe.json")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
