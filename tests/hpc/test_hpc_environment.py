import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from hpc.hpc import set_environment
from hpc.rl_launch_utils import _build_rl_container_env, get_rl_env_activation


ROOT = Path(__file__).resolve().parents[2]


def test_set_environment_preserves_explicit_values(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / "cluster.env"
    dotenv.write_text(
        "export SCRATCH=/cluster/default\n"
        "export DCFT=$SCRATCH/OpenThoughts-Agent\n"
        "export WANDB_MODE=offline\n"
    )
    monkeypatch.setenv("SCRATCH", "/project/operator")
    monkeypatch.setenv("DCFT", "/project/operator/execution-checkout")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    cluster = SimpleNamespace(
        dotenv_path=str(dotenv),
        name="jupiter",
    )
    set_environment(cluster)

    assert os.environ["SCRATCH"] == "/project/operator"
    assert os.environ["DCFT"] == "/project/operator/execution-checkout"
    assert os.environ["WANDB_MODE"] == "offline"


def test_jupiter_batch_dotenv_treats_checkout_paths_as_defaults() -> None:
    dotenv = (ROOT / "hpc/dotenv/jupiter.env").read_text()

    assert 'export SCRATCH="${SCRATCH:-/e/scratch/jureap59/$USER}"' in dotenv
    assert 'export DCFT="${DCFT:-$SCRATCH/OpenThoughts-Agent}"' in dotenv


def test_rl_sbatch_skips_generic_cluster_conda_activation() -> None:
    source = (ROOT / "hpc/rl_launch_utils.py").read_text()

    assert (
        '"conda_activate": "# Generic cluster Conda activation skipped for RL"'
        in source
    )


def test_qwen36_rl_uses_only_operator_owned_ray_paths_and_no_proxy() -> None:
    config = (
        ROOT
        / "hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml"
    ).read_text()
    hpc_source = (ROOT / "hpc/hpc.py").read_text()

    assert "OT_AGENT_RAY_LOG_DIR: /tmp/ray_logs" in config
    assert "/e/scratch/reformo/lee27/ray_spill" in config
    assert 'proxychains_binary=""' in hpc_source


def test_rl_container_env_preserves_json_and_runtime_expansion() -> None:
    json_value = '{"type":"filesystem","params":{"directory_path":"/scratch/spill"}}'
    block = _build_rl_container_env(
        {
            "extra_env": {
                "RAY_object_spilling_config": json_value,
                "LIBRARY_PATH": "${CUDA_HOME}/lib64",
            }
        },
        {},
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            block
            + "\nprintf '%s\\n%s\\n' \"$RAY_object_spilling_config\" \"$LIBRARY_PATH\"",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_HOME": "/opt/cuda"},
    )
    assert result.stdout.splitlines() == [json_value, "/opt/cuda/lib64"]


def test_rl_env_selector_is_emitted_only_before_activation() -> None:
    template = (ROOT / "hpc/sbatch_rl/universal_rl.sbatch").read_text()
    early = _build_rl_container_env(
        {"extra_env": {"RL_ENV_DIR": "/runtime/rl", "NCCL_DEBUG": "WARN"}},
        {},
    )
    late = _build_rl_container_env(
        {"extra_env": {"RL_ENV_DIR": "/runtime/rl", "NCCL_DEBUG": "WARN"}},
        {},
        exclude_extra_env_keys=("RL_ENV_DIR", "DCFT_RL_ENV"),
    )

    assert template.index("{rl_container_env}") < template.index("{rl_env_activation}")
    assert 'export RL_ENV_DIR="/runtime/rl"' in early
    assert "RL_ENV_DIR" not in late
    assert 'export NCCL_DEBUG="WARN"' in late


def test_rl_venv_activation_fails_fast_without_python_and_ray() -> None:
    activation = get_rl_env_activation({})

    assert '[[ ! -x "$RL_ENV_DIR/bin/python" || ! -x "$RL_ENV_DIR/bin/ray" ]]' in activation
    assert "RL environment is not usable" in activation
    assert "Warning: RL environment not found" not in activation
