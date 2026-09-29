"""Build the harbor auto-snapshot for given TBLite tasks exactly as harbor 761fb516 would on first use
(DaytonaSnapshotService._create_with_retry: Image.from_dockerfile + bake_agent_tooling, name harbor__<env hash>__snapshot,
resources from task.toml). One task at a time; never deletes anything except an ERROR-state snapshot of the same name
(harbor's own behavior).   usage: build_snapshots.py <task> [<task> ...]"""
import asyncio, logging, sys
from pathlib import Path
from daytona import AsyncDaytona, Resources
from harbor.environments.daytona.snapshots import DaytonaSnapshotService
from harbor.models.task.config import TaskConfig

T = Path("/e/fscratch/reformo/lee27/tasks/openthoughts_tblite_2_0")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
for n in ("httpx", "httpcore", "urllib3", "aiohttp"): logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("build")

async def main(tasks):
    async with AsyncDaytona() as d:
        for t in tasks:
            env = T / t / "environment"
            cfg = TaskConfig.model_validate_toml((T / t / "task.toml").read_text()).environment
            kw = {}   # environment.py _sandbox_resources(), then ensure_auto's `resources or Resources()`
            if cfg.cpus is not None: kw["cpu"] = cfg.cpus
            if cfg.memory_mb is not None: kw["memory"] = cfg.memory_mb // 1024
            if cfg.storage_mb is not None: kw["disk"] = cfg.storage_mb // 1024
            res = Resources(**kw)
            svc = DaytonaSnapshotService(logger=log, environment_dir=env, dockerfile_path=env / "Dockerfile")
            name = svc.auto_snapshot_name()
            log.info("BUILD %s -> %s resources=%s", t, name, res)
            try:
                await svc._create_with_retry(d, name, res)
                log.info("OK %s %s", t, name)
            except Exception as e:
                log.error("FAIL %s %s %s: %s", t, name, type(e).__name__, str(e)[:500])

asyncio.run(main(sys.argv[1:]))
