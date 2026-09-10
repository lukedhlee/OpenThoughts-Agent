"""Checks the Snowball reasoning parser against a real Snowball tokenizer.

Needs vLLM and a checkpoint directory; both live on the clusters, so the test
skips on a machine without them. Run on a login node (no GPU needed):

    SNOWBALL_TOKENIZER=/path/to/checkpoint \\
    env -u PYTHONPATH OMP_NUM_THREADS=1 python -m pytest -q eval/vllm_plugins/
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")

PLUGIN = Path(__file__).with_name("snowball_reasoning_parser.py")
TOKENIZER = os.environ.get("SNOWBALL_TOKENIZER")

pytestmark = pytest.mark.skipif(
    not TOKENIZER, reason="set SNOWBALL_TOKENIZER to a Snowball checkpoint dir"
)


class _Request:  # the parser never reads the request
    pass


@pytest.fixture(scope="module")
def parser():
    from vllm.reasoning import ReasoningParserManager

    ReasoningParserManager.import_reasoning_parser(str(PLUGIN))
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    cls = ReasoningParserManager.get_reasoning_parser("snowball")
    return cls(tok), tok


ACTION = '{"analysis": "start", "commands": [{"keystrokes": "ls\\n", "duration": 1.0}]}'


def test_markers_are_the_special_tokens(parser):
    p, _ = parser
    assert (p.start_token_id, p.end_token_id) == (128002, 128003)


def test_span_goes_to_reasoning_and_action_to_content(parser):
    p, tok = parser
    text = f"<|start_think|>I should list the files first.<|end_think|>\n\n{ACTION}"
    reasoning, content = p.extract_reasoning(text, _Request())
    assert reasoning == "I should list the files first."
    assert content.lstrip() == ACTION
    ids = tok.encode(text, add_special_tokens=False)
    assert p.is_reasoning_end(ids)
    assert tok.decode(p.extract_content_ids(ids)).lstrip() == ACTION


def test_turn_without_a_span_is_all_content(parser):
    p, tok = parser
    assert p.extract_reasoning(ACTION, _Request()) == (None, ACTION)
    ids = tok.encode(ACTION, add_special_tokens=False)
    assert p.is_reasoning_end(ids)
    assert p.extract_content_ids(ids) == ids
    delta = p.extract_reasoning_streaming("", ACTION, ACTION, [], ids, ids)
    assert delta.content == ACTION and delta.reasoning is None


def test_span_cut_by_the_length_limit_stays_reasoning(parser):
    p, _ = parser
    cut = "<|start_think|>still thinking when the limit hit"
    assert p.extract_reasoning(cut, _Request()) == ("still thinking when the limit hit", None)
