import json
from pathlib import Path

import pytest

from hpc.qwen36_grpo_preflight import (
    harbor_opencode_limit,
    validate_bridge_status,
    validate_context_contract,
)


ROOT = Path(__file__).resolve().parents[2]
RL_CONFIG = ROOT / "hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml"
LAUNCHER = ROOT / "hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo.sh"


def bridge_status(**overrides: object) -> dict:
    status = {
        "envs": {
            "pending": 0,
            "starting": 0,
            "ready": 0,
            "stopping": 0,
            "stopped": 31,
        },
        "queue_size": 0,
        "active_jobs": 0,
        "workers_alive": True,
        "stats": {"envs_created": 31, "envs_stopped": 31},
    }
    status.update(overrides)
    return status


def test_bridge_requires_live_workers_not_only_http_json() -> None:
    with pytest.raises(ValueError, match="workers_alive must be true"):
        validate_bridge_status(bridge_status(workers_alive=False), require_clean=True)


@pytest.mark.parametrize(
    ("field", "status"),
    [
        ("envs.ready=2", bridge_status(envs={"pending": 0, "starting": 0, "ready": 2, "stopping": 0})),
        ("queue_size=1", bridge_status(queue_size=1)),
        ("active_jobs=3", bridge_status(active_jobs=3)),
    ],
)
def test_dedicated_smoke_rejects_stale_bridge_work(field: str, status: dict) -> None:
    with pytest.raises(ValueError, match=field):
        validate_bridge_status(status, require_clean=True)


def test_clean_bridge_check_can_be_disabled_for_intentional_sharing() -> None:
    validate_bridge_status(
        bridge_status(
            envs={"pending": 1, "starting": 2, "ready": 3, "stopping": 4},
            queue_size=5,
            active_jobs=6,
        ),
        require_clean=False,
    )


def test_exact_deployed_harbor_limit_semantics() -> None:
    assert harbor_opencode_limit(
        {"max_input_tokens": 32_768, "max_output_tokens": 4_096}
    ) == {"context": 27_648, "output": 4_096}


def test_exact_qwen36_materialized_context_contract() -> None:
    validate_context_contract(RL_CONFIG)


def test_launcher_runs_fast_contract_check_before_model_import() -> None:
    launcher = LAUNCHER.read_text()
    fast_check = "-m hpc.qwen36_grpo_preflight"
    assert ': "${REQUIRE_CLEAN_BRIDGE:=1}"' in launcher
    assert '"$DCFT/hpc/qwen36_grpo_preflight.py"' in launcher
    assert fast_check in launcher
    assert launcher.index(fast_check) < launcher.index("import vllm._C")


def test_cli_reports_invalid_bridge_json(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(bridge_status(workers_alive=False)))
    # Keep CLI error behavior covered without launching a subprocess.
    with pytest.raises(ValueError, match="HTTP 200 alone is insufficient"):
        validate_bridge_status(json.loads(status_path.read_text()), require_clean=True)
