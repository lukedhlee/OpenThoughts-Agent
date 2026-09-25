#!/usr/bin/env python3
"""Burst test: start N sandboxes from the CalibForge snapshots, then run every task's setup.sh at the same moment.

Creates are paced (the org takes about 5 creates/s) and retried on rate limits; once all sandboxes are up, every
setup.sh starts together, so the layer mirror sees the whole burst at once. setup.sh is run as harbor runs it (root,
cwd /app, the task's env) with CF_DOCKERHUB=0 by default, so only the mirror can serve layers. Every sandbox made
here carries the label purpose=cf-burst and is deleted at the end, whatever happens.

  python burst_test.py --tree <tree> --n 200 --out <dir> [--dockerhub]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import re
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_snapshots import DEFAULT_KEY_FILE, load_secret  # noqa: E402


def pct(v, q):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=4.0, help="sandbox creates per second")
    ap.add_argument("--dockerhub", action="store_true", help="allow the Docker Hub fallback")
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    tree, out = Path(a.tree), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.loads((tree / "pool.json").read_text())
    tasks = (tree / "TASKS.txt").read_text().split()
    random.Random(a.seed).shuffle(tasks)
    tasks = tasks[: a.n]
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams, DaytonaConfig
    rows = {t: {"task": t} for t in tasks}
    made = []
    async with AsyncDaytona(DaytonaConfig(api_key=load_secret("DAYTONA_API_KEY", a.key_file))) as d:
        async def create(i, t):
            await asyncio.sleep(i / a.rate)
            cfg = tomllib.loads((tree / t / "task.toml").read_text())
            label = cfg["metadata"]["calibforge_daytona_base"]
            t0 = time.monotonic()
            for attempt in range(8):
                try:
                    sb = await d.create(CreateSandboxFromSnapshotParams(
                        snapshot=pool[label]["name"], labels={"purpose": "cf-burst"}, auto_delete_interval=30,
                        auto_stop_interval=30), timeout=600)
                    made.append(sb)
                    rows[t].update(base=label, create_s=round(time.monotonic() - t0, 1), create_attempts=attempt + 1)
                    return sb
                except Exception as exc:  # noqa: BLE001
                    rows[t]["create_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    await asyncio.sleep(2 + attempt * 3 + random.random())
            return None
        try:
            t_c = time.monotonic()
            sbs = await asyncio.gather(*(create(i, t) for i, t in enumerate(tasks)))
            create_wall = time.monotonic() - t_c
            ready = [(t, sb) for t, sb in zip(tasks, sbs) if sb is not None]
            print(f"{len(ready)}/{len(tasks)} sandboxes up in {create_wall:.0f}s", flush=True)
            await asyncio.gather(*(sb.fs.upload_file((tree / t / "setup_files/setup.sh").read_bytes(),
                                                     "/setup_files/setup.sh") for t, sb in ready))

            async def setup(t, sb):
                env = dict(tomllib.loads((tree / t / "task.toml").read_text())["environment"].get("env", {}))
                env["CF_DOCKERHUB"] = "1" if a.dockerhub else "0"
                t0 = time.monotonic()
                try:
                    r = await sb.process.exec("bash /setup_files/setup.sh", cwd="/app", env=env, timeout=1200)
                    rows[t].update(setup_s=round(time.monotonic() - t0, 1), exit=r.exit_code, tail=(r.result or "")[-400:])
                    m = re.search(r"CFDELTA (\{[^}]*\})", r.result or "")
                    if m:
                        rows[t]["cfdelta"] = json.loads(m.group(1))
                except Exception as exc:  # noqa: BLE001
                    rows[t].update(setup_s=round(time.monotonic() - t0, 1), exit=None,
                                   error=f"{type(exc).__name__}: {str(exc)[:300]}")
            t_s = time.monotonic()
            await asyncio.gather(*(setup(t, sb) for t, sb in ready))
            setup_wall = time.monotonic() - t_s
        finally:
            dels = await asyncio.gather(*(d.delete(sb, timeout=120) for sb in made), return_exceptions=True)
            del_fail = [str(x)[:120] for x in dels if isinstance(x, Exception)]
    ok = [r for r in rows.values() if r.get("exit") == 0]
    setup = [r["setup_s"] for r in ok]
    summary = {
        "requested": len(tasks), "sandboxes_created": len(made), "create_wall_s": round(create_wall, 1),
        "create_s": {"p50": pct([r["create_s"] for r in rows.values() if "create_s" in r], .5),
                     "p90": pct([r["create_s"] for r in rows.values() if "create_s" in r], .9)},
        "setup_ok": len(ok), "setup_wall_s": round(setup_wall, 1),
        "setup_s": {"p50": pct(setup, .5), "p90": pct(setup, .9), "max": pct(setup, 1.0)},
        "download_s": {"p50": pct([r["cfdelta"]["download_s"] for r in ok if "cfdelta" in r], .5),
                       "p90": pct([r["cfdelta"]["download_s"] for r in ok if "cfdelta" in r], .9)},
        "mb": {"p50": pct([r["cfdelta"]["bytes"] / 1e6 for r in ok if "cfdelta" in r], .5),
               "total_gb": round(sum(r["cfdelta"]["bytes"] for r in ok if "cfdelta" in r) / 1e9, 1)},
        "by_base": collections.Counter(r.get("base") for r in ok),
        "failures": [{k: r.get(k) for k in ("task", "exit", "error", "create_error", "tail")}
                     for r in rows.values() if r.get("exit") != 0],
        "delete_failures": del_fail, "dockerhub_allowed": a.dockerhub,
    }
    (out / "rows.json").write_text(json.dumps(list(rows.values()), indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if len(ok) == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
