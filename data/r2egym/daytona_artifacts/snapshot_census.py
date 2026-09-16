#!/usr/bin/env python3
"""List every snapshot in the Daytona org, say which ones a harbor task tree owns, and count the custom quota.

Daytona names the quota that matters "custom snapshots": an org may hold 40 (server error text: "Snapshot quota
exceeded. Maximum allowed: 40"); Daytona's own base images (`general: true`, undeletable) are listed but do not count.
harbor names the snapshot it builds for a task `harbor__<12 hex>__snapshot`, where the hex is the hash of the task's
environment directory, so a tree whose tasks share one environment dir per repo builds one image per repo, and a tree
with a Dockerfile per task builds one image per task until the quota refuses the rest.

    python snapshot_census.py                       # every snapshot, one row each, plus the quota count
    python snapshot_census.py --tree pool/          # tag the rows that a task tree's environment hashes own
    python snapshot_census.py --tree a/ --tree b/   # several trees

Key: DAYTONA_API_KEY in the environment, or --key-file <path> (a file whose content, or whose DAYTONA_API_KEY=... line,
is the key). Read-only: this script never creates or deletes anything. Deleting a mistaken snapshot is
`await d.snapshot.delete(await d.snapshot.get(name))` with the exact name, one at a time, after this listing.
"""

import argparse
import asyncio
import glob
import os
import re
import sys
from pathlib import Path

CUSTOM_QUOTA = 40


def load_key(path: str | None) -> str:
    if path:
        text = Path(path).read_text().strip()
        m = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
        return m.group(1) if m else text.splitlines()[0].strip()
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        sys.exit("set DAYTONA_API_KEY or pass --key-file")
    return key


def tree_hashes(trees: list[str]) -> dict[str, str]:
    """environment-dir hash -> 'tree:first task' for every task under the given trees (needs harbor importable)."""
    if not trees:
        return {}
    from harbor.utils.container_cache import environment_dir_hash_truncated  # noqa: E402

    owned: dict[str, str] = {}
    for tree in trees:
        for env in sorted(glob.glob(os.path.join(tree, "*", "environment"))):
            try:
                h = environment_dir_hash_truncated(Path(env), truncate=12)
            except Exception:  # a task without a buildable environment dir
                continue
            owned.setdefault(h, f"{Path(tree).name}:{Path(env).parent.name}")
    return owned


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-file")
    ap.add_argument("--tree", action="append", default=[], help="task tree whose environment hashes mark owned rows")
    args = ap.parse_args()
    owned = tree_hashes(args.tree)

    from daytona import AsyncDaytona, DaytonaConfig  # noqa: E402

    async with AsyncDaytona(DaytonaConfig(api_key=load_key(args.key_file))) as d:
        page = await d.snapshot.list()
        items = getattr(page, "items", page)

    rows = []
    for s in items:
        name = s.name
        parts = name.split("__")
        h = parts[1] if name.startswith("harbor__") and len(parts) >= 3 else ""
        general = bool(getattr(s, "general", False))
        if general:
            tag = "base"
        elif h in owned:
            tag = "owned " + owned[h]
        elif h:
            tag = "harbor-other"
        else:
            tag = "other"
        rows.append(
            (
                tag,
                name,
                str(getattr(getattr(s, "state", ""), "value", getattr(s, "state", ""))),
                getattr(s, "size", None),
                str(getattr(s, "created_at", ""))[:16],
                str(getattr(s, "last_used_at", ""))[:16],
            )
        )
    rows.sort()
    print("tag | name | state | size GB | created | last used")
    for r in rows:
        print(" | ".join("" if x is None else str(x) for x in r))
    custom = [r for r in rows if r[0] != "base"]
    print(f"\ntotal snapshots: {len(rows)}; base images (do not count): {len(rows) - len(custom)}")
    print(f"custom snapshots: {len(custom)} of {CUSTOM_QUOTA} -> {CUSTOM_QUOTA - len(custom)} free")
    if owned:
        present = {r[1].split('__')[1] for r in rows if r[0].startswith('owned')}
        print(f"tree hashes present in the org: {len(present)} of {len(owned)} distinct")
        missing = sorted(set(owned) - present)
        if missing:
            print("tree hashes not built yet:", ", ".join(f"{h} ({owned[h]})" for h in missing))


if __name__ == "__main__":
    asyncio.run(main())
