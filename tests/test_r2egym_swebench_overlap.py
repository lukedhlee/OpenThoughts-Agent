from __future__ import annotations

import io
import tarfile

import pytest

from scripts.analysis.audit_r2egym_swebench_overlap import (
    audit,
    parse_instruction,
)


def task_binary(instruction: str) -> bytes:
    output = io.BytesIO()
    payload = instruction.encode()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo("instruction.md")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def instruction(issue: str, commit: str = "a" * 40) -> str:
    return f"""\
git clone https://github.com/Org/Repo.git . && git checkout {commit}
<issue_description>
{issue}
</issue_description>
"""


def test_shared_code_state_is_not_automatically_direct_contamination() -> None:
    report = audit(
        [{"path": "r2e-1", "task_binary": task_binary(instruction("Fix the parser."))}],
        [
            {
                "instance_id": "org__repo-1",
                "repo": "org/repo",
                "base_commit": "a" * 40,
                "problem_statement": "Correct an unrelated numerical edge case.",
            }
        ],
        {"org__repo-1"},
        direct_match_threshold=0.9,
    )
    assert report["shared_repo_base_commit_pairs"] == 1
    assert report["direct_problem_matches"] == 0


def test_identical_problem_is_a_direct_match() -> None:
    problem = "Handle frobnication when the input is empty."
    report = audit(
        [{"path": "r2e-1", "task_binary": task_binary(instruction(problem))}],
        [
            {
                "instance_id": "org__repo-1",
                "repo": "Org/Repo",
                "base_commit": "A" * 40,
                "problem_statement": problem,
            }
        ],
        {"org__repo-1"},
        direct_match_threshold=0.9,
    )
    assert report["direct_problem_matches"] == 1


def test_identical_problem_at_different_commit_is_still_a_direct_match() -> None:
    problem = "Handle frobnication when the input is empty."
    report = audit(
        [
            {
                "path": "r2e-1",
                "task_binary": task_binary(instruction(problem, commit="a" * 40)),
            }
        ],
        [
            {
                "instance_id": "org__repo-1",
                "repo": "Org/Repo",
                "base_commit": "b" * 40,
                "problem_statement": problem,
            }
        ],
        {"org__repo-1"},
        direct_match_threshold=0.9,
    )
    assert report["shared_repo_base_commit_pairs"] == 0
    assert report["direct_problem_matches"] == 1
    assert report["direct_problem_match_pairs"][0]["same_base_commit"] is False


def test_parse_instruction_fails_closed() -> None:
    with pytest.raises(ValueError, match="clone/checkout"):
        parse_instruction("bad", "<issue_description>x</issue_description>")
