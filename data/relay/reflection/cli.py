"""python -m data.relay.reflection.cli --help"""

import argparse
import json
from pathlib import Path

from .core import (
    ARMS,
    load_config,
    read_jsonl,
    validate_sources,
    write_json,
    write_jsonl,
)
from .generation import generate_reflections
from .harbor import import_trials
from .render import render_records
from .runner import run


def plan(c):
    trajectory, reflection, mode = ARMS[c["arm"]]
    return {
        "name": c["name"],
        "arm": c["arm"],
        "trajectory_author": trajectory,
        "reflection_author": reflection,
        "mode": mode,
        "updates": c["training"]["updates"],
        "source_attempts_per_update": c["training"]["source_attempts"],
        "reflections_per_source": c["reflection"]["per_trajectory"]
        if reflection
        else 0,
        "loss": c["training"]["objective"],
        "output_dir": str(Path(c["runtime"]["output_dir"]) / c["name"]),
        "evaluation": c["evaluation"],
        "launch_readiness": "GPU two-update smoke required; no jobs submitted",
        "unconfigured_refresh_command": mode == "online"
        and not c["runtime"].get("refresh_command"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "status", "prepare", "validate-vanilla"):
        a = sub.add_parser(command)
        a.add_argument("--config", required=True)
        if command == "run":
            a.add_argument(
                "--deadline-epoch",
                type=float,
                help="start no new update after this time (epoch seconds)",
            )
    a = sub.add_parser("import-harbor")
    a.add_argument("--roots", nargs="+", required=True)
    a.add_argument("--author", choices=("teacher", "student"), required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "import-harbor":
        # Reject obvious mixed-author sources, rather than relabeling relay data as student experience.
        if any("relay_repair" in x for x in args.roots):
            raise ValueError(
                "mixed relay trajectories cannot be imported as pure teacher/student"
            )
        rows, bad = import_trials(
            args.roots, author=args.author, model=args.model, revision=args.revision
        )
        write_jsonl(args.output, rows)
        write_json(args.output + ".census.json", {"rows": len(rows), "rejections": bad})
        return
    c = load_config(args.config)
    root = Path(c["runtime"]["output_dir"]) / c["name"]
    if args.command == "plan":
        print(json.dumps(plan(c), indent=2))
    elif args.command == "status":
        print(
            (root / "state.json").read_text()
            if (root / "state.json").exists()
            else json.dumps({"phase": "not_started"})
        )
    elif args.command == "run":
        print(json.dumps(run(c, deadline=args.deadline_epoch), indent=2))
    elif args.command == "validate-vanilla":
        if c["arm"] != "vanilla":
            raise ValueError("requires vanilla arm")
        from .marin_adapter import inspect_rows

        sha, n, tokens = inspect_rows(
            c["data"]["vanilla_rows"], c["training"]["sequence_length"]
        )
        print(
            json.dumps(
                {
                    "rows": n,
                    "supervised_tokens": tokens,
                    "sha256": sha,
                    "note": "existing SFT launcher unchanged",
                }
            )
        )
    else:
        if c["arm"] in ("vanilla", "roft"):
            raise ValueError("prepare is for offline completion variants")
        rows = read_jsonl(c["data"]["source_jsonl"])
        validate_sources(rows, c)
        root.mkdir(parents=True, exist_ok=True)
        records, bad = generate_reflections(rows, c, root / "generations")
        targets, render_bad = render_records(
            records, c["student"]["tokenizer_dir"], c["training"]["sequence_length"]
        )
        write_jsonl(root / "prepared.rows.jsonl", targets)
        write_json(
            root / "prepare.census.json",
            {
                "sources": len(rows),
                "targets": len(targets),
                "generation_rejections": bad,
                "render_rejections": render_bad,
            },
        )
        print(
            json.dumps(
                {"targets": len(targets), "rows": str(root / "prepared.rows.jsonl")}
            )
        )


if __name__ == "__main__":
    main()
