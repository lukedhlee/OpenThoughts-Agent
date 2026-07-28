import os
import subprocess
from pathlib import Path

import pytest

from hpc.datagen_launch_utils import TracegenJobConfig, TracegenJobRunner


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _config(**kwargs) -> TracegenJobConfig:
    return TracegenJobConfig(
        job_name="test",
        harbor_config="config.yaml",
        trace_script="",
        experiments_dir="/tmp/test",
        **kwargs,
    )


def test_harbor_runtime_forwards_bridge_and_verified_checkout(
    tmp_path: Path, monkeypatch
):
    checkout = tmp_path / "harbor"
    checkout.mkdir()
    (checkout / "src" / "harbor").mkdir(parents=True)
    _git("init", cwd=checkout)
    _git("config", "user.email", "test@example.com", cwd=checkout)
    _git("config", "user.name", "Test", cwd=checkout)
    (checkout / "README").write_text("test")
    _git("add", ".", cwd=checkout)
    _git("commit", "-m", "initial", cwd=checkout)
    commit = _git("rev-parse", "HEAD", cwd=checkout)

    monkeypatch.setenv("HARBOR_HOME", str(checkout))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    runner = TracegenJobRunner(
        _config(
            harbor_ref=commit,
            apptainer_bridge_url="http://jrlogin04i:9920",
        )
    )

    assert runner.config.harbor_ref == commit
    assert os.environ["APPTAINER_BRIDGE_URL"] == "http://jrlogin04i:9920"
    assert os.environ["HARBOR_REF"] == commit
    assert os.environ["PYTHONPATH"] == str(checkout / "src")


def test_harbor_runtime_rejects_checkout_mismatch(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "harbor"
    checkout.mkdir()
    _git("init", cwd=checkout)
    _git("config", "user.email", "test@example.com", cwd=checkout)
    _git("config", "user.name", "Test", cwd=checkout)
    (checkout / "README").write_text("first")
    _git("add", ".", cwd=checkout)
    _git("commit", "-m", "first", cwd=checkout)
    old_commit = _git("rev-parse", "HEAD", cwd=checkout)
    (checkout / "README").write_text("second")
    _git("commit", "-am", "second", cwd=checkout)

    monkeypatch.setenv("HARBOR_HOME", str(checkout))
    with pytest.raises(RuntimeError, match="Harbor checkout mismatch"):
        TracegenJobRunner(_config(harbor_ref=old_commit))
