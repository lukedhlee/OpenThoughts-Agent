import os
from types import SimpleNamespace

from hpc.hpc import set_environment


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
