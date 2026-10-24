"""Checks that experiment provisioning cannot fall through to snapshot/declarative builds."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from data.relay.horizon.mixed_eval import runtime
from data.relay.horizon.mixed_eval.readout import passk, summarize


def test_direct_snapshot_does_not_access_client():
    called = []
    fake = SimpleNamespace(
        _compose_mode=False,
        _snapshot_template_name="harbor__exact__snapshot",
        _snapshot_sandbox_params=lambda name: called.append(name) or name,
    )
    result = asyncio.run(
        runtime.ExistingSnapshotDaytona._resolve_start_sandbox_params(
            fake, object(), None, force_build=False
        )
    )
    assert result == "harbor__exact__snapshot" and called == [result]
    with pytest.raises(ValueError):
        asyncio.run(
            runtime.ExistingSnapshotDaytona._resolve_start_sandbox_params(
                fake, object(), None, force_build=True
            )
        )


def test_dind_routing_and_rejected_unreviewed_compose(tmp_path):
    for task in runtime.PYTHON_DIND_TASKS | runtime.JAVA_DIND_TASKS:
        environment = tmp_path / task / "environment"
        environment.mkdir(parents=True)
        result = runtime.provisioning(environment, Path("/pinned/harbor"))
        assert (
            result["auto_snapshot"] is False and result["dind_managed_daemon"] is True
        )
        assert result["dind_snapshot"] == (
            "daytona-large"
            if task in runtime.JAVA_DIND_TASKS
            else "harbor__313e69c036ad__snapshot"
        )
        assert (
            result["dind_bootstrap_script"]
            == "/pinned/harbor/scripts/bootstrap_daytona_docker.sh"
        )
    environment = tmp_path / "unknown-compose/environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError):
        runtime.provisioning(environment, Path("/pinned/harbor"))
    with pytest.raises(ValueError):
        runtime.provisioning(
            tmp_path / "git-repo-forensics/environment", Path("/pinned/harbor")
        )


def test_factory_default_empty_compose_is_accepted(tmp_path):
    environment = tmp_path / "prediction-model-evaluation/environment"
    environment.mkdir(parents=True)
    with patch.object(
        runtime.DaytonaEnvironment, "__init__", return_value=None
    ) as parent:
        runtime.ExistingSnapshotDaytona(
            environment_dir=environment,
            extra_docker_compose=[],
            harbor_root="/pinned/harbor",
        )
    assert parent.call_args.kwargs["dind_managed_daemon"] is True
    assert len(parent.call_args.kwargs["extra_docker_compose"]) == 1


def test_cli_verifier_only_applies_to_cli():
    fake = object.__new__(runtime.MixedVerifier)
    fake.task = SimpleNamespace(name="ordinary-task")
    with patch.object(runtime.Verifier, "verify", new=AsyncMock(return_value="plain")):
        assert asyncio.run(fake.verify()) == "plain"
    fake.task = SimpleNamespace(name="pytorch-model-cli")
    with patch.object(
        runtime.CliCacheVerifier, "verify", new=AsyncMock(return_value="cli")
    ):
        assert asyncio.run(fake.verify()) == "cli"


def test_passk_without_replacement():
    assert passk(8, 0, 4) == 0
    assert passk(8, 8, 4) == 1
    assert passk(8, 1, 4) == 0.5
    assert passk(8, 2, 1) == 0.25
    with pytest.raises(ValueError):
        passk(8, 1, 9)


def test_unknown_excludes_entire_paired_panel_and_partial_is_not_fullsolve(tmp_path):
    import json

    groups = []
    for tag in ["cm50", "cm100"]:
        for rep in range(1, 9):
            run = f"{tag}_r{rep}"
            groups.append(
                dict(
                    tag=tag,
                    rep=rep,
                    drivers=[
                        dict(
                            run=run,
                            shard="tb21",
                            config=dict(
                                tasks=[
                                    dict(path="/tasks/pass"),
                                    dict(path="/tasks/missing"),
                                ]
                            ),
                        )
                    ],
                )
            )
            for name in ["pass", "missing"]:
                if tag == "cm100" and rep == 8 and name == "missing":
                    continue
                directory = tmp_path / "jobs" / run / f"{name}__id"
                directory.mkdir(parents=True)
                reward = 1 if tag == "cm50" else 0.8825
                (directory / "result.json").write_text(
                    json.dumps(
                        dict(
                            task_name=name,
                            verifier_result=dict(rewards=dict(reward=reward)),
                            exception_info=None,
                        )
                    )
                )
    (tmp_path / "plan.json").write_text(json.dumps(dict(groups=groups)))
    result = summarize(tmp_path)
    assert all(
        row["paired_tasks"] == 1 and row["mix50"] == 1 and row["concat"] == 0
        for row in result["comparisons"]
        if row["family"] == "tb21"
    )
    assert len(result["unknowns"]) == 1
    assert any(slot["raw_reward"] == 0.8825 for slot in result["slots"])
