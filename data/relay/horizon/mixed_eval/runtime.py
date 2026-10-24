"""Read-only snapshot provisioning for the pinned corrected Snowball eval stack."""

from pathlib import Path

from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.utils.container_cache import environment_dir_hash_truncated
from harbor.verifier.verifier import Verifier

from data.relay.horizon.mixed_eval.cli_cache_verifier import CliCacheVerifier

from data.relay.horizon.mixed_eval.plan import PYTHON_DIND_TASKS, JAVA_DIND_TASKS


def provisioning(environment_dir: Path, harbor_root: Path) -> dict:
    name = environment_dir.parent.name
    if name == "git-repo-forensics":
        raise ValueError("Git fixture-invalid task is excluded from this experiment")
    if name in PYTHON_DIND_TASKS | JAVA_DIND_TASKS:
        java = name in JAVA_DIND_TASKS
        return dict(
            auto_snapshot=False,
            extra_docker_compose=[
                Path(__file__).with_name(
                    "compose_java.yaml" if java else "compose.yaml"
                )
            ],
            dind_snapshot="daytona-large" if java else "harbor__313e69c036ad__snapshot",
            dind_shell="sudo -n -E sh -c" if java else "sh -c",
            dind_managed_daemon=True,
            dind_bootstrap_script=str(
                harbor_root / "scripts/bootstrap_daytona_docker.sh"
            ),
        )
    if (environment_dir / "docker-compose.yaml").exists():
        raise ValueError(f"Unreviewed compose environment: {name}")
    return dict(
        auto_snapshot=False,
        snapshot_template_name=f"harbor__{environment_dir_hash_truncated(environment_dir)}__snapshot",
        assume_global_snapshot=True,
    )


class ExistingSnapshotDaytona(DaytonaEnvironment):
    """Provision only from existing snapshots; declarative/auto builds are unreachable."""

    def __init__(self, *args, harbor_root: str, **kwargs):
        if args:
            raise ValueError("Pinned EnvironmentFactory must pass named arguments")
        environment_dir = Path(kwargs["environment_dir"])
        selected = provisioning(environment_dir, Path(harbor_root))
        if kwargs.pop("extra_docker_compose", None):
            raise ValueError("Unreviewed external compose override")
        if any(key in kwargs for key in selected):
            raise ValueError(
                "Provisioning overrides must be selected by the experiment adapter"
            )
        super().__init__(**kwargs, **selected)

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        if force_build or self._compose_mode or not self._snapshot_template_name:
            raise ValueError("Direct provisioning requires an existing named snapshot")
        return self._snapshot_sandbox_params(self._snapshot_template_name)


class MixedVerifier(CliCacheVerifier):
    """Use the deployed CLI cache wrapper only for its validated task."""

    async def verify(self):
        if self.task.name == "pytorch-model-cli":
            return await super().verify()
        return await Verifier.verify(self)
