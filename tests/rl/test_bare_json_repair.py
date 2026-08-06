"""Repair pass of the bare_json tool parser.

Fixtures in tests/data/bare_json_malformed_1251403.json are the VERBATIM text
parts from job 1251403 (g1_diverse_tezos_100k_8b, thinking ON) in which the
model attempted a tool call whose JSON the strict parser rejects. They are the
measured population the repair pass was designed against — if this test fails
after a parser change, the change regressed a real, observed trajectory.

vLLM is not importable on the laptop, so its modules are stubbed before the
parser module is loaded; only the pure extraction functions are under test.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARSER = REPO / "rl" / "tool_parsers" / "bare_json_tool_parser.py"
FIXTURE = REPO / "tests" / "data" / "bare_json_malformed_1251403.json"


def _stub(name, **attrs):
    mod = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_parser_module():
    class _Base:
        def __init__(self, tokenizer, tools=None):
            pass

    class _Mgr:
        @staticmethod
        def register_module(_name):
            return lambda cls: cls

    class _Logger:
        def warning(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    def _any(*a, **k):
        return object()

    _stub("vllm")
    _stub("vllm.entrypoints")
    _stub("vllm.entrypoints.chat_utils", make_tool_call_id=lambda: "id")
    _stub("vllm.entrypoints.openai")
    _stub("vllm.entrypoints.openai.chat_completion")
    _stub("vllm.entrypoints.openai.chat_completion.protocol", ChatCompletionRequest=object)
    _stub(
        "vllm.entrypoints.openai.engine",
    )
    _stub(
        "vllm.entrypoints.openai.engine.protocol",
        DeltaFunctionCall=_any,
        DeltaMessage=_any,
        DeltaToolCall=_any,
        ExtractedToolCallInformation=_any,
        FunctionCall=_any,
        ToolCall=_any,
    )
    _stub("vllm.logger", init_logger=lambda name: _Logger())
    _stub("vllm.tokenizers", TokenizerLike=object)
    _stub(
        "vllm.tool_parsers",
    )
    _stub(
        "vllm.tool_parsers.abstract_tool_parser",
        Tool=object,
        ToolParser=_Base,
        ToolParserManager=_Mgr,
    )

    spec = importlib.util.spec_from_file_location("bare_json_tool_parser", PARSER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_parser_module()
FIXTURES = {f["trial"]: f["text"] for f in json.loads(FIXTURE.read_text())}

# Trials whose defect is in a repairable class (measured expectation; see the
# parser's repair-pass comment). The two absentees are the ambiguous classes:
# r2egym-0592__eWCXM6a (arguments not valid JSON structure) and
# r2egym-1742__GdcB4rY (both its objects damaged beyond the safe repairs).
EXPECT_REPAIRED = {
    "r2egym-0000__RWoGnTg",
    "r2egym-0000__vkeRiuc",
    "r2egym-0002__5yM9D9p",
    "r2egym-0591__UyEmUwS",
    "r2egym-1742__Ueq5tFk",
    "r2egym-2040__PwwLGK6",
    "r2egym-2514__dCmGsKH",
    "r2egym-2515__rZwmYgs",
}


def test_fixture_population_is_complete():
    assert set(FIXTURES) >= EXPECT_REPAIRED
    assert len(FIXTURES) == 10


def test_repairs_measured_malformed_calls():
    repaired = {t for t, txt in FIXTURES.items() if M._extract(txt)[0]}
    assert repaired >= EXPECT_REPAIRED, f"lost repairs: {EXPECT_REPAIRED - repaired}"


def test_repaired_calls_are_wellformed():
    for trial in EXPECT_REPAIRED:
        calls, _ = M._extract(FIXTURES[trial])
        assert calls, trial
        for c in calls:
            assert isinstance(c["name"], str) and c["name"], trial
            args = c["arguments"]
            assert isinstance(args, (dict, str)), trial


def test_healthy_outputs_unchanged():
    # Strict pass must handle these identically to the pre-repair parser.
    ok = '{"name": "bash", "arguments": {"command": "ls -la /testbed"}}'
    calls, leftover = M._extract(ok)
    assert calls == [{"name": "bash", "arguments": {"command": "ls -la /testbed"}}]
    assert leftover == ""

    with_think = f"<think>plan</think>\n\n{ok}"
    calls, _ = M._extract(with_think)
    assert len(calls) == 1

    arr = '[{"name": "read", "arguments": {"filePath": "/a"}}, {"name": "bash", "parameters": {"command": "ls"}}]'
    calls, _ = M._extract(arr)
    assert [c["name"] for c in calls] == ["read", "bash"]

    prose = "I will inspect the repo layout first."
    calls, leftover = M._extract(prose)
    assert calls == [] and leftover == prose

    nontool_json = 'Example config: {"nameserver": "8.8.8.8", "options": {"timeout": 1}}'
    calls, _ = M._extract(nontool_json)
    assert calls == []


def test_repair_does_not_fire_early_on_streaming_prefixes():
    full = '{"name": "bash", "arguments": {"command": "find /testbed -name *.py | head -20}}'
    # Every strict prefix that has not yet emitted the final brace must yield
    # no calls — otherwise streaming would emit a truncated command.
    for cut in range(1, len(full)):
        calls, _ = M._extract(full[:cut])
        assert calls == [], f"fired at prefix len {cut}: {full[:cut]!r}"
    calls, _ = M._extract(full)
    assert calls and calls[0]["arguments"]["command"] == "find /testbed -name *.py | head -20"
