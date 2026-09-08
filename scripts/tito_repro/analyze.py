"""Turn probe outputs into the table that answers the question.

Reads one or more probe JSONs and reports, per turn count and transport mode:
how many trajectories SkyRL would decline, the per-turn-boundary rate that
drives it, and what the training-side repair would cost instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def rows(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(paths):
        data = json.loads(Path(path).read_text())
        turns = data.get("config", {}).get("turns")
        if turns is None:
            m = re.search(r"turns(\d+)", path)
            turns = int(m.group(1)) if m else None
        for mode in ("text", "tokens"):
            if mode not in data:
                continue
            s = data[mode]["summary"]
            out.append(
                {
                    "file": Path(path).name,
                    "turns": turns,
                    "mode": mode,
                    "n": s["n_trajectories"],
                    "declined": s["n_declined"],
                    "decline_frac": s["decline_fraction"],
                    "boundaries": s["n_turn_boundaries"],
                    "declining_boundaries": s["n_declining_turn_boundaries"],
                    "boundary_frac": s["turn_decline_fraction"],
                    "reasons": s["reasons"],
                    "replay_x": s.get("replay_multiplier_mean"),
                    "errors": data[mode].get("n_errors", 0),
                    "wall_s": data[mode].get("wall_seconds"),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="probe JSON files or globs")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    paths: List[str] = []
    for p in args.paths:
        paths.extend(glob.glob(p) or [p])
    table = rows(paths)

    header = f"{'turns':>5} {'mode':>7} {'traj':>5} {'declined':>9} {'rate':>7} {'bnd':>5} {'bnd bad':>8} {'bnd rate':>9} {'replay':>7} {'err':>4}"
    print(header)
    print("-" * len(header))
    for r in table:
        replay = f"{r['replay_x']:.1f}x" if r["replay_x"] else "-"
        print(
            f"{str(r['turns']):>5} {r['mode']:>7} {r['n']:>5} {r['declined']:>9} "
            f"{r['decline_frac']:>6.1%} {r['boundaries']:>5} {r['declining_boundaries']:>8} "
            f"{r['boundary_frac']:>8.1%} {replay:>7} {r['errors']:>4}"
        )
    for r in table:
        if r["reasons"]:
            print(f"  reasons turns={r['turns']} {r['mode']}: {r['reasons']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(table, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
