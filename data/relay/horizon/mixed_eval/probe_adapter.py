"""Live provisioning-only gate for the adapter; creates/deletes only its three own sandboxes."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from harbor.environments.factory import EnvironmentFactory
from harbor.environments.daytona.snapshots import DaytonaSnapshotService
from harbor.models.task.task import Task
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


async def forbidden(*args, **kwargs):
    raise AssertionError(
        "Automatic snapshot resolution/build is forbidden in this gate"
    )


async def probe(path: Path, harbor: Path, output: Path, kind: str) -> dict:
    task = Task(path)
    trial = TrialPaths(output / kind)
    trial.mkdir()
    config = EnvironmentConfig.model_validate(
        dict(
            import_path="data.relay.horizon.mixed_eval.runtime:ExistingSnapshotDaytona",
            kwargs=dict(harbor_root=str(harbor)),
        )
    )
    environment = EnvironmentFactory.create_environment_from_config(
        config,
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="cfmix-gate-" + uuid.uuid4().hex[:12],
        trial_paths=trial,
        task_env_config=task.config.environment,
    )
    start = time.monotonic()
    sandbox_id = None
    cleanup = False
    try:
        await asyncio.wait_for(
            environment.start(force_build=False),
            timeout=task.config.environment.build_timeout_sec,
        )
        sandbox_id = environment._sandbox.id
        result = await environment.exec("printf CFMIX_ADAPTER_READY", timeout_sec=30)
        assert result.return_code == 0 and "CFMIX_ADAPTER_READY" in result.stdout, (
            result
        )
    finally:
        await environment.stop(delete=True)
        cleanup = environment._sandbox is None
    return dict(
        kind=kind,
        task=task.name,
        passed=True,
        seconds=time.monotonic() - start,
        sandbox_id=sandbox_id,
        cleanup=cleanup,
        snapshot_operations="none",
    )


async def main_async(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (
        patch.object(DaytonaSnapshotService, "ensure_auto", forbidden),
        patch.object(DaytonaSnapshotService, "resolve_template", forbidden),
    ):
        for kind, path in [
            ("direct", args.direct),
            ("python", args.python),
            ("java", args.java),
        ]:
            results.append(await probe(path, args.harbor, args.output, kind))
            (args.output / "adapter_live_gate.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["direct", "python", "java", "harbor", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
