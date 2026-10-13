"""Delete superseded native optimizer checkpoints of a reflection run (670 GB each); keep every HF export.

    python -m data.relay.reflection.horizon.prune_native --config <config.json> [--dry-run]

Update i's native checkpoint is only ever read to resume update i+1. Once state.json has committed a later update whose
native checkpoint is complete (metadata.json and online_state.json at its step), every earlier native directory is
dead weight: the run never replays an update. The newest committed native checkpoint is always kept, so a resume
still restores the full optimizer state. Each deletion is appended to <run>/pruned_native.log.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from data.relay.reflection.core import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = load_config(a.config)
    root = Path(c["runtime"]["output_dir"]) / c["name"]
    state = json.loads((root / "state.json").read_text())
    i = state["iteration"]
    if i < 2 or not state.get("optimizer"):
        return
    keep = Path(state["optimizer"])
    online = json.loads((keep / "online_state.json").read_text())
    meta = json.loads((keep / "metadata.json").read_text())
    if online["iteration"] != i or meta.get("step") != i or keep != root / f"update-{i:04d}" / "training" / "native":
        raise SystemExit(f"newest native checkpoint {keep} does not verify at step {i}; nothing pruned")
    for j in range(1, i):
        native = root / f"update-{j:04d}" / "training" / "native"
        if not native.exists():
            continue
        size = sum(p.stat().st_size for p in native.rglob("*") if p.is_file())
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {'would delete' if a.dry_run else 'deleted'} {native} ({size / 1e9:.0f} GB); kept {keep}"
        if not a.dry_run:
            shutil.rmtree(native)
            with open(root / "pruned_native.log", "a") as f:
                f.write(line + "\n")
        print(line)


if __name__ == "__main__":
    main()
