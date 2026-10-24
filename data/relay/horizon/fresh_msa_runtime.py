"""Existing-snapshot-only CalibForge provisioning for fresh MSA2 rollouts."""
from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.utils.container_cache import environment_dir_hash_truncated


class ExistingCalibforgeSnapshot(DaytonaEnvironment):
    """Fail on missing snapshots rather than creating snapshots or building images."""

    def __init__(self, *args, **kwargs):
        if kwargs.pop("auto_snapshot", False):
            raise ValueError("Automatic snapshot creation is forbidden")
        super().__init__(*args, auto_snapshot=False, **kwargs)

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        if force_build or self._compose_mode:
            raise ValueError("This run requires an existing CalibForge snapshot")
        name = f"harbor__{environment_dir_hash_truncated(self.environment_dir)}__snapshot"
        return self._snapshot_sandbox_params(name)
