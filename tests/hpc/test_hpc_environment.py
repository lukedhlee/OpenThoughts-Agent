import os
from pathlib import Path
from types import SimpleNamespace

from hpc.hpc import set_environment


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
