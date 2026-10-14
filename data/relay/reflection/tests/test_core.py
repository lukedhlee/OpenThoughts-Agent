import json

import pytest

from data.relay.reflection.core import bounded, reflection_prompt, validate_sources
from data.relay.reflection.harbor import import_trials


def source():
    return {
        "id": "a",
        "task": "train",
        "author": "student",
        "model": "grug",
        "revision": "pin",
        "policy_version": 0,
        "task_description": "fix me",
        "messages": [
            {"role": "assistant", "content": "bad action"},
            {"role": "tool", "content": "error"},
        ],
        "feedback": {"verdict": "fail", "text": "test failed"},
        "termination": "timeout",
    }


def test_bounded_keeps_head_tail():
    assert bounded("abcdefghij", 6) == "abcd\n[TRUNCATED]\nij"


def test_failed_and_unknown_reflections():
    r = source()
    prompt = reflection_prompt(r, {})
    assert "fail" in prompt and "test failed" in prompt and "timeout" in prompt
    r["feedback"] = {"verdict": "unknown"}
    assert "unknown" in reflection_prompt(r, {})
    assert "RAW VERDICT" not in reflection_prompt(r, {"include_feedback": False})


def test_source_author_split_version(tmp_path):
    p = tmp_path / "heldout"
    p.write_text("test\n")
    c = {
        "arm": "roft",
        "data": {"heldout_tasks": str(p)},
        "student": {"model": "grug", "revision": "pin"},
    }
    validate_sources([source()], c, 0)
    for field, value in [
        ("author", "teacher"),
        ("task", "test"),
        ("policy_version", -1),
        ("model", "other"),
    ]:
        r = source()
        r[field] = value
        with pytest.raises(ValueError):
            validate_sources([r], c, 0)
    with pytest.raises(ValueError):
        validate_sources([source(), source()], c, 0)


def test_import_preserves_verdict_separate_from_error(tmp_path):
    p = tmp_path / "trial"
    p.mkdir()
    r = {
        "id": "a",
        "task_name": "train",
        "agent_result": {"metadata": {"all_messages": source()["messages"]}},
        "verifier_result": {"rewards": {"reward": 1}},
        "exception_info": {"exception_type": "ContextLengthExceededError"},
    }
    (p / "result.json").write_text(json.dumps(r))
    rows, bad = import_trials(
        [tmp_path], author="teacher", model="qwen", revision="pin"
    )
    assert not bad and rows[0]["feedback"]["verdict"] == "pass"
    assert rows[0]["termination"] == "ContextLengthExceededError"
    r["verifier_result"] = None
    (p / "result.json").write_text(json.dumps(r))
    rows, _ = import_trials([tmp_path], author="teacher", model="qwen", revision="pin")
    assert rows[0]["feedback"]["verdict"] == "unknown"


def test_import_recovers_all_malformed_episode_from_atif(tmp_path):
    from data.relay.reflection.harbor import import_trials

    trial = tmp_path / "job" / "trial-1"
    (trial / "attempts/000/agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "id": "t1", "task_name": "train0", "trial_relpath": "trial-1/attempts/000",
        "verifier_result": {"rewards": {"reward": 0.0}},
        "agent_result": {"metadata": {"all_messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
            {"role": "user", "content": "Tool call error: never closed"}]}},
    }))
    (trial / "attempts/000/agent/trajectory.json").write_text(json.dumps({"steps": [
        {"source": "system", "message": "sys"}, {"source": "user", "message": "task"},
        {"source": "agent", "message": "<|start_think|>x<|end_think|><tool_call>{",
         "observation": {"results": [{"content": "Tool call error: never closed"}]}}]}))
    rows, bad = import_trials([tmp_path / "job"], author="student", model="m", revision="r")
    assert bad == {"recovered_from_atif": 1} and rows[0]["experience_source"] == "atif_trajectory"
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant", "user"]
    assert rows[0]["feedback"]["verdict"] == "fail"
