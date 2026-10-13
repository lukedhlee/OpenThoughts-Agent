"""Strict subprocess boundary for the native Grug online reflection updater.

Run inside an allocated training environment. This module never submits Slurm jobs.
The backend command is argv (not a shell), with {request}/{response} placeholders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


def inspect_rows(path: str, sequence_length: int) -> tuple[str, int, int]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("update batch is empty")
    tokens = 0
    identities = set()
    for row in rows:
        ids, loss = row["ids"], row["loss"]
        if len(ids) != len(loss) or not 2 <= len(ids) <= sequence_length:
            raise ValueError(
                "ids/loss must align and fit sequence_length; truncation is forbidden"
            )
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid token ids")
        if any(x not in (0, 1) for x in loss) or loss[0] != 0 or not sum(loss):
            raise ValueError(
                "binary target-token loss required; first token cannot be supervised"
            )
        identity = row.get("sid")
        if not identity or identity in identities:
            raise ValueError("unique sid required")
        identities.add(identity)
        tokens += sum(loss)
    return hashlib.sha256(raw).hexdigest(), len(rows), tokens


def run(request: dict, response_path: str) -> dict:
    iteration = request["iteration"]
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if bool(request.get("optimizer_before")) != (iteration > 1):
        raise ValueError(
            "first update requires fresh optimizer; subsequent updates require full native state"
        )
    training = request["config"]["training"]
    backend = request["config"]["backend"]
    if training["loss_reduction"] != "global_token_mean":
        raise ValueError("only global_token_mean is implemented")
    digest, count, tokens = inspect_rows(request["rows"], training["sequence_length"])
    if request.get("source_digest", digest) != digest:
        raise ValueError("source_digest mismatch")
    authors = {
        "roft": ("student", "student"),
        "teacher_teacher": ("teacher", "teacher"),
        "teacher_student": ("teacher", "student"),
        "student_teacher": ("student", "teacher"),
    }
    if request["arm"] not in authors:
        raise ValueError("native reflection updater requires a reflection arm")
    for line in Path(request["rows"]).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("trajectory_author"), row.get("reflection_author")) != authors[
            request["arm"]
        ]:
            raise ValueError("reflection row author provenance differs from arm")
        boundary = row.get("prompt_tokens")
        if type(boundary) is not int or not 0 < boundary < len(row["loss"]):
            raise ValueError("reflection row needs an explicit prompt boundary")
        if any(row["loss"][:boundary]) or not all(row["loss"][boundary:]):
            raise ValueError("only reflection tokens may receive loss")
        expected_version = iteration - 1 if request["arm"] == "roft" else 0
        if row.get("policy_version") != expected_version:
            raise ValueError("reflection targets have stale policy provenance")
    output = Path(request["output_dir"])
    if output.exists():
        raise ValueError("output_dir already exists; refusing to replay an update")
    if Path(response_path).exists():
        raise ValueError("response already exists")
    output.mkdir(parents=True)
    resolved = dict(
        request, source_digest=digest, row_count=count, supervised_tokens=tokens
    )
    input_path = output / "request.json"
    input_path.write_text(json.dumps(resolved, indent=2) + "\n")
    intermediate = output / "backend-response.json"
    command = backend.get("update_command") or [
        backend["python"],
        "-m",
        "experiments.june_tpu_67b_a2b.moe.online_reflection",
        "--request",
        "{request}",
        "--response",
        "{response}",
    ]
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("update_command must be an argv list")
    argv = [
        x.replace("{request}", str(input_path)).replace("{response}", str(intermediate))
        for x in command
    ]
    subprocess.run(argv, cwd=backend["marin_root"], check=True)
    result = json.loads(intermediate.read_text())
    expected = {
        "optimizer_steps_before": iteration - 1,
        "optimizer_steps_after": iteration,
        "weights_version": iteration,
        "source_digest": digest,
        "supervised_tokens": tokens,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"backend violated {key}: {result.get(key)!r} != {value!r}"
            )
    if not isinstance(result.get("loss"), (int, float)) or not math.isfinite(
        result["loss"]
    ):
        raise ValueError("backend did not report finite loss")
    for field, marker in (
        ("model_after", "config.json"),
        ("optimizer_after", "metadata.json"),
    ):
        if not (Path(result[field]) / marker).is_file():
            raise ValueError(f"missing {field} artifact")
    if not list(Path(result["model_after"]).glob("*.safetensors")):
        raise ValueError("HF export has no model tensors")
    if not (Path(result["optimizer_after"]) / "online_state.json").is_file():
        raise ValueError("native checkpoint missing optimizer provenance")
    destination = Path(response_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=destination.parent, delete=False
    ) as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.request).read_text()), args.response)


if __name__ == "__main__":
    main()
