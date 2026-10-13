"""OpenAI-compatible completion generation with durable, non-replayed requests."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from .core import ARMS, digest, reflection_prompt, write_json


def chat_completion(endpoint, payload, timeout, api_key_env=None):
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        headers["Authorization"] = "Bearer " + os.environ[api_key_env]
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def normalize_message(message):
    content = (message.get("content") or "").strip()
    thinking = (message.get("reasoning_content") or "").strip()
    if not thinking:
        for begin, end in [
            ("<|start_think|>", "<|end_think|>"),
            ("<think>", "</think>"),
        ]:
            if content.startswith(begin) and end in content:
                thinking, content = content[len(begin) :].split(end, 1)
                thinking, content = thinking.strip(), content.strip()
                break
    content = re.sub(r"(?:<\|eot_id\|>|<\|im_end\|>)\s*$", "", content).strip()
    return {"role": "assistant", "content": content, "reasoning_content": thinking}


def generate_reflections(
    rows, config, out_dir, version=0, client=chat_completion, expected_model=None
):
    author = ARMS[config["arm"]][1]
    if author is None:
        raise ValueError("vanilla does not generate reflections")
    endpoint = config["runtime"]["endpoints"][author]
    if client is chat_completion:
        expected_model = expected_model or config[author].get("weights_path")
        if expected_model:
            headers = {}
            if endpoint.get("api_key_env"):
                headers["Authorization"] = (
                    "Bearer " + os.environ[endpoint["api_key_env"]]
                )
            req = urllib.request.Request(
                endpoint["url"].rstrip("/") + "/models", headers=headers
            )
            with urllib.request.urlopen(
                req, timeout=config["runtime"]["timeout_sec"]
            ) as f:
                models = json.load(f)["data"]
            matches = [m for m in models if m["id"] == endpoint["served_model"]]
            if (
                len(matches) != 1
                or Path(matches[0].get("root", "")).resolve()
                != Path(expected_model).resolve()
            ):
                raise ValueError(
                    "reflection server weights do not match configured author/checkpoint"
                )
    settings = config["reflection"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    rejected = {}
    for row in rows:
        prompt = reflection_prompt(row, settings)
        for sample in range(settings["per_trajectory"]):
            sid = digest(
                {
                    "source": row["id"],
                    "sample": sample,
                    "arm": config["arm"],
                    "version": version,
                }
            )
            payload = {
                "model": endpoint["served_model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "max_tokens": settings["max_tokens"],
            }
            fingerprint = digest(
                {
                    "payload": payload,
                    "endpoint": endpoint["url"],
                    "source": row,
                    "config": config,
                    "version": version,
                    "expected_model": expected_model,
                }
            )
            p = out / (sid + ".json")
            if p.exists():
                record = json.loads(p.read_text())
                if record["fingerprint"] != fingerprint:
                    raise ValueError(
                        "generation settings/source changed at existing artifact"
                    )
            else:
                record = {
                    "sid": sid,
                    "fingerprint": fingerprint,
                    "source_id": row["id"],
                    "sample": sample,
                    "task": row["task"],
                    "trajectory_author": row["author"],
                    "reflection_author": author,
                    "reflection_model": config.get(author, {}).get("model"),
                    "reflection_checkpoint": expected_model,
                    "policy_version": version,
                    "prompt": prompt,
                }
                try:
                    response = client(
                        endpoint["url"],
                        payload,
                        config["runtime"]["timeout_sec"],
                        endpoint.get("api_key_env"),
                    )
                    choice = response["choices"][0]
                    message = normalize_message(choice["message"])
                    final = message["content"]
                    thinking = message["reasoning_content"]
                    # Preserve actual thinking. Empty thinking prefixes are removed at rendering, not fabricated.
                    record.update(
                        message=message,
                        finish_reason=choice.get("finish_reason"),
                        usage=response.get("usage"),
                        raw_response=response,
                    )
                    if choice.get("finish_reason") not in ("stop",):
                        record["rejected"] = "non_stop_completion"
                    elif not (final.strip() or thinking.strip()):
                        record["rejected"] = "empty"
                except (OSError, ValueError, KeyError, IndexError) as error:
                    record["rejected"] = "generation_error"
                    record["error_type"] = type(error).__name__
                write_json(p, record)
            if record.get("rejected"):
                why = record["rejected"]
                rejected[why] = rejected.get(why, 0) + 1
            else:
                results.append(record)
    return results, rejected
