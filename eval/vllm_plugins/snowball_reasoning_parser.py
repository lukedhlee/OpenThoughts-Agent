"""vLLM reasoning parser for Snowball's think markers.

Snowball reasons between two special tokens, ``<|start_think|>`` and
``<|end_think|>``, then emits its answer. vLLM ships parsers for the common
marker pairs (``<think>`` for Qwen3, ``<seed:think>`` for Seed, ...) but not
for these, so a stock server hands the reasoning to the client inline in
``content`` (when special tokens are kept) or as unmarked prose ahead of the
answer (when they are stripped). Either way a stock agent harness that parses
``content`` for a structured action sees "extra text before the JSON".

This parser makes Snowball behave like every other thinking model on the same
server: the reasoning goes to ``reasoning_content`` and ``content`` holds only
what follows ``<|end_think|>``. Nothing about sampling changes; token ids are
untouched.

Usage (a plugin file, no vLLM rebuild):

    vllm serve <model> --reasoning-parser snowball \\
        --reasoning-parser-plugin eval/vllm_plugins/snowball_reasoning_parser.py

In the eval registry this is ``reasoning_parser: snowball`` plus the plugin
path in ``extra_args`` (see ``model_config/_patterns.yaml``).
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"


class SnowballReasoningParser(BaseThinkingReasoningParser):
    """``<|start_think|>`` ... ``<|end_think|>`` delimited reasoning.

    The base class implements extraction, streaming and the reasoning-end
    check from the two token ids; only the marker strings are Snowball's.
    Construction fails loudly if the served tokenizer lacks either marker, so
    the parser cannot be applied to the wrong model silently.
    """

    @property
    def start_token(self) -> str:
        return START_THINK

    @property
    def end_token(self) -> str:
        return END_THINK

    # The base class assumes a turn that never closes the span is reasoning
    # from its first character (models whose template puts the start marker
    # in the prompt). Snowball generates its own start marker, so a turn
    # without one is an answer with no reasoning, not reasoning with no
    # answer. Only the no-start-marker case is changed; a span that opens and
    # is cut off by the length limit is still all reasoning, as in the base.

    def extract_reasoning(
        self, model_output: str, request
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self.start_token_id not in input_ids:
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.start_token_id not in input_ids:
            return input_ids
        return super().extract_content_ids(input_ids)


# Explicit registration: the decorator form registers lazily by module path,
# and a plugin file's module name is whatever the loader chose, so register
# the class object directly.
ReasoningParserManager.register_module(
    name="snowball", module=SnowballReasoningParser, force=True
)
