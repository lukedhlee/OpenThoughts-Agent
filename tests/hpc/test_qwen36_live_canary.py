import json
from pathlib import Path

import pytest

from hpc.qwen36_live_canary import (
    MODEL_REVISION,
    OPENCODE_VERSION,
    TOOL_MARKER,
    CanaryFailure,
    canary_instruction,
    classify_failure,
    normalize_endpoint,
    opencode_error_text,
    select_task,
    validate_canary_artifacts,
    validate_materialized_trial_contract,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "hpc/skyrl_standard/jupiter/run_qwen36_live_canary.sh"


def valid_result(reward: float = 0.0) -> dict:
    return {
        "exception_info": None,
        "agent_info": {
            "name": "opencode",
            "version": OPENCODE_VERSION,
            "model_info": {"name": MODEL_REVISION, "provider": "vllm"},
        },
        "verifier_result": {"rewards": {"reward": reward}},
    }


def valid_trajectory() -> dict:
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "bash",
                        "arguments": {"command": f"printf {TOOL_MARKER}"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call-1", "content": TOOL_MARKER}]
                },
            }
        ]
    }


def valid_trial_config() -> dict:
    return {
        "agent": {
            "name": "opencode",
            "model_name": f"vllm/{MODEL_REVISION}",
            "kwargs": {
                "api_base": "http://jrlogin05i:18000/v1",
                "version": OPENCODE_VERSION,
                "preinstalled": True,
                "model_info": {
                    "max_input_tokens": 32_768,
                    "max_output_tokens": 4_096,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
                "opencode_config": {"compaction": {"auto": True}},
            },
        },
        "environment": {
            "type": "apptainer",
            "kwargs": {"bridge_url": "http://10.128.1.2:9920"},
        },
    }


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("prompt contains at least 28673 input tokens", "CONTEXT"),
        ("BridgeOperationTimeoutError", "BRIDGE"),
        ("Network is unreachable", "TRANSPORT"),
        ("VerifierTimeoutError", "VERIFIER_INFRA"),
        ("some other OpenCode crash", "AGENT_RUNTIME"),
    ],
)
def test_failure_classes_are_explicit(text: str, kind: str) -> None:
    assert classify_failure(text) == kind


def test_endpoint_must_be_explicit_v1_base() -> None:
    assert normalize_endpoint("http://jrlogin05i:18000/v1/") == (
        "http://jrlogin05i:18000/v1"
    )
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[ENDPOINT\]"):
        normalize_endpoint("http://jrlogin05i:18000")


def test_canary_requires_exact_identity_tool_observation_and_numeric_reward() -> None:
    assert validate_canary_artifacts(valid_result(0.0), valid_trajectory()) == 0.0
    assert validate_canary_artifacts(valid_result(1.0), valid_trajectory()) == 1.0


def test_zero_reward_is_a_valid_canary_transport_result() -> None:
    assert validate_canary_artifacts(valid_result(0.0), valid_trajectory()) == 0.0


def test_materialized_trial_keeps_exact_endpoint_agent_context_and_bridge() -> None:
    validate_materialized_trial_contract(
        valid_trial_config(),
        endpoint="http://jrlogin05i:18000/v1",
        bridge_url="http://10.128.1.2:9920",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "latest"),
        ("preinstalled", False),
        ("api_base", "http://wrong/v1"),
        ("compaction", False),
        ("max_input_tokens", 28_672),
    ],
)
def test_materialized_trial_contract_rejects_runtime_drift(field: str, value) -> None:
    config = valid_trial_config()
    kwargs = config["agent"]["kwargs"]
    if field == "compaction":
        kwargs["opencode_config"]["compaction"]["auto"] = value
    elif field == "max_input_tokens":
        kwargs["model_info"][field] = value
    else:
        kwargs[field] = value
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[CONTRACT\]"):
        validate_materialized_trial_contract(
            config,
            endpoint="http://jrlogin05i:18000/v1",
            bridge_url="http://10.128.1.2:9920",
        )


def test_missing_tool_observation_fails_loudly() -> None:
    trajectory = valid_trajectory()
    trajectory["steps"][0]["observation"]["results"] = []
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[TOOL_ACTION\]"):
        validate_canary_artifacts(valid_result(), trajectory)


def test_wrong_model_or_opencode_version_fails_identity() -> None:
    result = valid_result()
    result["agent_info"]["version"] = "latest"
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[IDENTITY\]"):
        validate_canary_artifacts(result, valid_trajectory())

    result = valid_result()
    result["agent_info"]["model_info"]["name"] = "Qwen3.6-35B-A3B"
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[IDENTITY\]"):
        validate_canary_artifacts(result, valid_trajectory())


@pytest.mark.parametrize("reward", [None, True, "0", float("nan"), float("inf")])
def test_missing_or_nonfinite_reward_is_verifier_infrastructure_failure(reward) -> None:
    result = valid_result()
    result["verifier_result"]["rewards"]["reward"] = reward
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[VERIFIER_INFRA\]"):
        validate_canary_artifacts(result, valid_trajectory())


def test_trial_exception_uses_specific_failure_class() -> None:
    result = valid_result()
    result["exception_info"] = {
        "exception_type": "NonZeroAgentExitCodeError",
        "exception_message": "Network is unreachable",
    }
    with pytest.raises(CanaryFailure, match=r"CANARY_FAIL\[TRANSPORT\]"):
        validate_canary_artifacts(result, None)


def test_only_opencode_error_events_are_scanned() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "reasoning", "part": {"text": "connection refused"}}),
            json.dumps({"type": "error", "error": {"name": "APIConnectionError"}}),
        ]
    )
    assert opencode_error_text(raw) == '{"name": "APIConnectionError"}'


def test_instruction_demands_one_observable_real_tool_action() -> None:
    prompt = canary_instruction()
    assert "bash tool" in prompt
    assert "tee /tmp/qwen36_canary_tool_ok" in prompt
    assert TOOL_MARKER in prompt


def test_task_selection_is_top_level_and_deterministic(tmp_path: Path) -> None:
    for name in ("task-b", "task-a"):
        task = tmp_path / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "instruction.md").write_text("task")
        (task / "task.toml").write_text('version = "1.0"')
        (task / "tests" / "test.sh").write_text("#!/bin/sh")
    assert select_task(tmp_path).name == "task-a"


def test_launcher_uses_no_scheduler_and_keeps_exact_runtime_defaults() -> None:
    launcher = LAUNCHER.read_text()
    assert "sbatch" not in launcher
    assert "srun" not in launcher
    assert "http://jrlogin05i:18000/v1" in launcher
    assert "http://10.128.1.2:9920" in launcher
    assert "REQUIRE_CLEAN_BRIDGE:=1" in launcher
    assert "qwen36_live_canary.py" in launcher
