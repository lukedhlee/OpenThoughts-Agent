#!/usr/bin/env python3
"""Audit r2egym tasks for overlap with a pinned SWE-bench evaluation set.

The strongest contamination signal available in these datasets is an identical
repository plus base commit.  That is intentionally reported separately from a
direct task match: two unrelated issues can start from the same code state.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


CLONE_RE = re.compile(
    r"git clone\s+(?:--\S+\s+)*https?://github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?\s+"
    r"(?:\.\s*&&\s*)?git checkout\s+([0-9a-fA-F]{7,40})",
    re.DOTALL,
)
ISSUE_RE = re.compile(r"<issue_description>\s*(.*?)\s*</issue_description>", re.DOTALL)


@dataclass(frozen=True)
class R2ETask:
    task_id: str
    repo: str
    base_commit: str
    problem_statement: str


def normalize_repo(repo: str) -> str:
    return repo.removesuffix(".git").strip().lower()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def parse_instruction(task_id: str, instruction: str) -> R2ETask:
    match = CLONE_RE.search(instruction)
    if not match:
        raise ValueError(f"{task_id}: cannot find GitHub clone/checkout command")
    owner, name, commit = match.groups()
    issue = ISSUE_RE.search(instruction)
    if not issue:
        raise ValueError(f"{task_id}: cannot find <issue_description>")
    return R2ETask(
        task_id=task_id,
        repo=normalize_repo(f"{owner}/{name}"),
        base_commit=commit.lower(),
        problem_statement=issue.group(1).strip(),
    )


def instruction_from_task_binary(task_id: str, task_binary: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.rstrip("/").endswith("instruction.md")
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{task_id}: expected one instruction.md, found {len(candidates)}"
            )
        stream = archive.extractfile(candidates[0])
        if stream is None:
            raise ValueError(f"{task_id}: instruction.md could not be read")
        return stream.read().decode("utf-8")


def read_manifest(path: Path) -> set[str]:
    """Read either a newline-delimited manifest or a JSON list/object."""
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            line.strip().rstrip("/")
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    if isinstance(parsed, list):
        values: Iterable[Any] = parsed
    elif isinstance(parsed, dict):
        values = (
            parsed.get("instance_ids")
            or parsed.get("task_names")
            or parsed.get("tasks")
            or []
        )
    else:
        raise ValueError("manifest JSON must be a list or object")
    return {
        str(item.get("instance_id") if isinstance(item, dict) else item).rstrip("/")
        for item in values
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on cluster preflight
        raise SystemExit("pyarrow is required to read parquet files") from exc
    return pq.read_table(path).to_pylist()


def audit(
    r2e_rows: list[dict[str, Any]],
    swe_rows: list[dict[str, Any]],
    instance_ids: set[str],
    *,
    direct_match_threshold: float,
) -> dict[str, Any]:
    pinned = [row for row in swe_rows if row["instance_id"] in instance_ids]
    found_ids = {row["instance_id"] for row in pinned}
    missing = sorted(instance_ids - found_ids)
    if missing:
        raise ValueError(
            f"{len(missing)} manifest instances are absent from SWE-bench parquet: "
            + ", ".join(missing[:5])
        )

    r2e_tasks: list[R2ETask] = []
    parse_errors: list[str] = []
    for row in r2e_rows:
        task_id = str(row["path"])
        try:
            instruction = instruction_from_task_binary(task_id, row["task_binary"])
            r2e_tasks.append(parse_instruction(task_id, instruction))
        except (KeyError, tarfile.TarError, UnicodeDecodeError, ValueError) as exc:
            parse_errors.append(str(exc))

    swe_by_state: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pinned:
        key = (normalize_repo(row["repo"]), str(row["base_commit"]).lower())
        swe_by_state.setdefault(key, []).append(row)

    shared: list[dict[str, Any]] = []
    for task in r2e_tasks:
        for swe in swe_by_state.get((task.repo, task.base_commit), []):
            similarity = SequenceMatcher(
                None,
                normalize_text(task.problem_statement),
                normalize_text(str(swe["problem_statement"])),
                autojunk=False,
            ).ratio()
            shared.append(
                {
                    "r2egym_task_id": task.task_id,
                    "swebench_instance_id": swe["instance_id"],
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "problem_similarity": round(similarity, 6),
                    "direct_problem_match": similarity >= direct_match_threshold,
                }
            )

    # A task can be duplicated at another base commit, so direct contamination
    # cannot be gated on identical code state. Compare every same-repository
    # pair, plus globally identical normalized statements (which also catches a
    # task copied under a renamed/forked repository).
    problem_overlaps: list[dict[str, Any]] = []
    for task in r2e_tasks:
        task_problem = normalize_text(task.problem_statement)
        for swe in pinned:
            swe_repo = normalize_repo(str(swe["repo"]))
            swe_problem = normalize_text(str(swe["problem_statement"]))
            if task.repo != swe_repo and task_problem != swe_problem:
                continue
            matcher = SequenceMatcher(
                None,
                task_problem,
                swe_problem,
                autojunk=False,
            )
            # quick_ratio is an upper bound on ratio. Most unrelated issues in
            # the same popular repository fail this linear-time gate, avoiding
            # SequenceMatcher's quadratic worst case on long issue bodies.
            if matcher.quick_ratio() < direct_match_threshold:
                continue
            similarity = matcher.ratio()
            if similarity < direct_match_threshold:
                continue
            swe_commit = str(swe["base_commit"]).lower()
            problem_overlaps.append(
                {
                    "r2egym_task_id": task.task_id,
                    "swebench_instance_id": swe["instance_id"],
                    "r2egym_repo": task.repo,
                    "swebench_repo": swe_repo,
                    "r2egym_base_commit": task.base_commit,
                    "swebench_base_commit": swe_commit,
                    "same_repo": task.repo == swe_repo,
                    "same_base_commit": task.base_commit == swe_commit,
                    "problem_similarity": round(similarity, 6),
                }
            )

    return {
        "r2egym_rows": len(r2e_rows),
        "r2egym_tasks_parsed": len(r2e_tasks),
        "r2egym_parse_errors": parse_errors,
        "swebench_manifest_size": len(instance_ids),
        "swebench_rows_found": len(pinned),
        "shared_repo_base_commit_pairs": len(shared),
        "direct_problem_matches": len(problem_overlaps),
        "direct_match_threshold": direct_match_threshold,
        "shared_pairs": shared,
        "direct_problem_match_pairs": problem_overlaps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2egym-parquet", type=Path, required=True)
    parser.add_argument("--swebench-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--direct-match-threshold", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(
        load_rows(args.r2egym_parquet),
        load_rows(args.swebench_parquet),
        read_manifest(args.manifest),
        direct_match_threshold=args.direct_match_threshold,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
