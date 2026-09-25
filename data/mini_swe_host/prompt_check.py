#!/usr/bin/env python3
"""Does the served chat template re-feed a prior turn's reasoning? (Stage 2 gate, Qwen3.8.)

Takes the first recorded episode that has at least two model replies, sends its first four messages (system, task,
reply 1, observation 1) to vLLM's /tokenize with the generation prompt, detokenizes the rendered prompt, and reports
whether reply 1's reasoning appears inside a <think> block of the rendered prompt.

    python prompt_check.py <api_base .../v1> <served name> <jobs_dir>/<job_name>
"""

import glob
import json
import sys
import urllib.request


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=120))


def main() -> None:
    api_base, served, job_dir = sys.argv[1:4]
    for path in sorted(glob.glob(f"{job_dir}/*/attempts/*/agent/mini-swe-agent.trajectory.json")):
        messages = json.load(open(path))["messages"]
        if len(messages) >= 4 and messages[2]["role"] == "assistant" and messages[3]["role"] == "user":
            break
    else:
        sys.exit("no episode with a parsed first reply")
    request_messages = []
    for m in messages[:4]:
        view = {"role": m["role"], "content": m["content"]}
        if m.get("reasoning_content"):
            view["reasoning_content"] = m["reasoning_content"]
        request_messages.append(view)
    root = api_base.rstrip("/").removesuffix("/v1")
    reply = post(f"{root}/tokenize", {"model": served, "messages": request_messages, "add_generation_prompt": True})
    rendered = post(f"{root}/detokenize", {"model": served, "tokens": reply["tokens"]})["prompt"]
    reasoning = request_messages[2].get("reasoning_content") or ""
    snippet = reasoning.strip()[:80]
    at = rendered.find(snippet) if snippet else -1
    print("TRAJECTORY", path)
    print("REPLY1_HAS_REASONING_CONTENT", bool(reasoning), "len", len(reasoning))
    print("REPLY1_CONTENT_HAS_THINK_MARKER", "</think>" in request_messages[2]["content"])
    print("PROMPT_TOKENS", reply.get("count"))
    print("REASONING_PRESENT_IN_PROMPT", at >= 0)
    print("REASONING_IN_THINK_BLOCK", at >= 0 and rendered[:at].rstrip().endswith("<think>"))
    print("RENDERED_AROUND_REPLY1", json.dumps(rendered[max(at, 0) - 120: max(at, 0) + 200] if at >= 0 else rendered[-800:]))

if __name__ == "__main__":
    main()
