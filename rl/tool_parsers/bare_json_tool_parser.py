# SPDX-License-Identifier: Apache-2.0
"""vLLM tool parser for models that emit tool calls as bare JSON.

Some of our SFT'd checkpoints (e.g. the terminus-lineage ``g1_diverse_*`` models)
emit tool calls as a naked JSON object in the message body::

    <think>
    ...reasoning...
    </think>

    {"name": "bash", "arguments": {"command": "ls -la /testbed"}}

No ``<tool_call>`` delimiter, no XML. Every stock vLLM parser misses this by a
hair:

* ``qwen3_coder`` / ``qwen3xml`` want an XML grammar -- they never match, and
  never log, so the call silently degrades to plain assistant text.
* ``hermes`` wants exactly this JSON but wrapped in ``<tool_call>``.
* ``llama3_json`` has the right extraction logic but hard-requires a
  ``<|python_tag|>`` token that the Qwen3 vocab does not contain, so it raises
  at construction time.
* ``xlam`` handles the ``</think>`` prefix and the same key names, but rejects a
  single object because it requires a top-level JSON *array*.

This parser is the small intersection those four miss: bare JSON, single object
or array, ``arguments`` or ``parameters``, with any surrounding prose ignored.

Register with::

    --tool-parser-plugin rl/tool_parsers/bare_json_tool_parser.py
    --tool-call-parser bare_json
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser, ToolParserManager

logger = init_logger(__name__)

# A trailing unclosed <think> means the model is still reasoning; a closed block
# is stripped so the JSON that follows it is what we scan.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_START = re.compile(r"[{\[]")


def _strip_thinking(text: str) -> str:
    """Drop closed <think> blocks. An unterminated one swallows the rest."""
    text = _THINK_BLOCK.sub("", text)
    head, sep, _ = text.partition("<think>")
    return head if sep else text


def _as_tool_call_dicts(obj: object) -> list[dict] | None:
    """Normalize a decoded JSON value to a list of tool-call dicts, or None.

    Accepts a single object or an array of them. Every element must carry a
    ``name`` plus one of ``arguments`` / ``parameters``; anything else means this
    JSON was incidental prose (a code block, an example) and not a tool call.
    """
    candidates = obj if isinstance(obj, list) else [obj]
    if not candidates:
        return None
    out: list[dict] = []
    for item in candidates:
        if not isinstance(item, dict):
            return None
        if "name" not in item or not isinstance(item.get("name"), str):
            return None
        if "arguments" in item:
            args = item["arguments"]
        elif "parameters" in item:
            args = item["parameters"]
        else:
            return None
        out.append({"name": item["name"], "arguments": args})
    return out


def _extract(text: str) -> tuple[list[dict], str]:
    """Return (tool_calls, leftover_content) from a complete model output."""
    scanned = _strip_thinking(text)
    decoder = json.JSONDecoder()
    end = -1
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []

    for match in _JSON_START.finditer(scanned):
        start = match.start()
        if start <= end:  # inside a JSON value we already consumed
            continue
        try:
            obj, offset = decoder.raw_decode(scanned[start:])
        except json.JSONDecodeError:
            continue
        normalized = _as_tool_call_dicts(obj)
        if normalized is None:
            continue
        end = start + offset
        calls.extend(normalized)
        spans.append((start, end))

    leftover = scanned
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    if calls:
        return calls, leftover.strip()

    # Strict scan found nothing. If the text plainly ATTEMPTS a tool call, try
    # the conservative repair pass (see _extract_repaired) before giving up.
    # Measured on job 1251403 (thinking-ON baseline): 10/11 stalled trials
    # carried a tool call whose JSON had exactly one small defect, and 0/11 had
    # a valid call the strict pass missed.
    if _NAME_KEY.search(scanned):
        repaired = _extract_repaired(scanned)
        if repaired is not None:
            return repaired
    return calls, leftover.strip()


# Repair pass. Fires ONLY when the strict scan yields nothing, so healthy
# outputs are byte-identically handled. Defect classes it repairs are the three
# measured on real g1_diverse_tezos_100k_8b trajectories (job 1251403):
#   1. literal control chars inside strings  -> strict=False decode
#   2. missing closing quote on the LAST string, its braces textually present
#      (`"command": "ls -la /testbed}}`)     -> insert '"' before the trailing
#      brace run. In streaming this cannot fire early: until the model has
#      emitted every closing brace the candidate stays unbalanced and fails.
#   3. stray backslash-escapes outside strings (`"pattern": \"x\"`) -> unescape.
# NOT repaired (ambiguous, measured 1 case each): unquoted YAML-ish `{bash: x}`
# and an arguments dict that is not valid JSON structure. Those stay content.
_NAME_KEY = re.compile(r'"name"\s*:\s*"')
_TRAILING_JUNK = re.compile(r"(?:\s|</tool_call>|</think>)+$")
# Leftmost closer whose suffix is nothing but closers/whitespace — covers both
# compact (`}}`) and pretty-printed (`\n  }\n}`) endings.
_TRAILING_CLOSERS = re.compile(r"[}\]][\s}\]]*$")


def _repair_candidates(fragment: str):
    yield fragment
    closers = _TRAILING_CLOSERS.search(fragment)
    if closers:
        yield fragment[: closers.start()] + '"' + fragment[closers.start() :]
    unescaped = fragment.replace('\\"', '"')
    if unescaped != fragment:
        yield unescaped
        closers = _TRAILING_CLOSERS.search(unescaped)
        if closers:
            yield unescaped[: closers.start()] + '"' + unescaped[closers.start() :]


def _extract_repaired(scanned: str) -> tuple[list[dict], str] | None:
    """Best-effort recovery of ONE defective tool call; None if unrepairable."""
    lenient = json.JSONDecoder(strict=False)  # tolerates raw \n and \t in strings
    trimmed = _TRAILING_JUNK.sub("", scanned)
    for match in _JSON_START.finditer(trimmed):
        start = match.start()
        fragment = trimmed[start:]
        if not _NAME_KEY.search(fragment):
            continue
        for candidate in _repair_candidates(fragment):
            try:
                obj, _ = lenient.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            normalized = _as_tool_call_dicts(obj)
            if normalized is None:
                continue
            logger.warning(
                "bare_json: repaired a malformed tool call (defect fixed by "
                "lenient decode / close-quote insertion / unescape)"
            )
            return normalized, trimmed[:start].strip()
    return None


def _to_arguments_json(args: object) -> str:
    return json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args


@ToolParserManager.register_module("bare_json")
class BareJsonToolParser(ToolParser):
    """Parse tool calls emitted as undelimited JSON in the message body."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Streaming: emit each call once, whole, as soon as its JSON closes.
        self._emitted = 0
        self._streamed_content = 0

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        try:
            calls, leftover = _extract(model_output)
        except Exception:
            logger.exception("bare_json tool parser failed; returning raw content")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        if not calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=c["name"], arguments=_to_arguments_json(c["arguments"])
                    ),
                )
                for c in calls
            ],
            content=leftover or None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Emit each tool call in one piece the moment its JSON parses.

        Incremental argument streaming buys nothing for an agent rollout -- the
        agent cannot act on half a command -- so a call is withheld until it is
        complete and then sent whole. That avoids the partial-JSON state machine
        that makes the other parsers' streaming paths fragile.
        """
        try:
            calls, _ = _extract(current_text)
        except Exception:
            logger.exception("bare_json streaming parse failed")
            return None

        if not calls:
            # No tool call yet: stream prose through untouched.
            return DeltaMessage(content=delta_text) if delta_text else None

        if len(calls) <= self._emitted:
            return None

        new = calls[self._emitted :]
        index0 = self._emitted
        self._emitted = len(calls)
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index0 + i,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=c["name"], arguments=_to_arguments_json(c["arguments"])
                    ),
                )
                for i, c in enumerate(new)
            ]
        )
