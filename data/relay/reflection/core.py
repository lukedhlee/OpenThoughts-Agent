"""Dependency-light experiment contracts; all artifact writes are atomic."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ARMS = {
    "vanilla": ("teacher", None, "offline"),
    "roft": ("student", "student", "online"),
    "teacher_teacher": ("teacher", "teacher", "offline"),
    "teacher_student": ("teacher", "student", "offline"),
    "student_teacher": ("student", "teacher", "offline"),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def load_config(path):
    c = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "name",
        "arm",
        "student",
        "teacher",
        "data",
        "reflection",
        "training",
        "runtime",
        "evaluation",
        "backend",
    }
    if set(c) != required or c["schema_version"] != 1:
        raise ValueError(f"config keys must be {sorted(required)}; schema_version=1")
    if c["arm"] not in ARMS:
        raise ValueError("unknown arm")
    if not c["name"] or any(
        x not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for x in c["name"]
    ):
        raise ValueError("name must be a safe lowercase artifact slug")
    for who in ("student", "teacher"):
        for field in ("model", "revision", "tokenizer_dir"):
            if not c[who].get(field):
                raise ValueError(f"{who}.{field} required")
    if c["training"].get("objective") != (
        "actions" if c["arm"] == "vanilla" else "reflection_only"
    ):
        raise ValueError("wrong objective for arm")
    for field in ("lr", "epsilon", "grad_clip", "gradient_scale"):
        v = c["training"][field]
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"training.{field} must be finite and positive")
    for field in ("beta1", "beta2"):
        if not 0 <= c["training"][field] < 1:
            raise ValueError(f"training.{field} must be in [0,1)")
    if c["training"]["weight_decay"] < 0 or not math.isfinite(
        c["training"]["weight_decay"]
    ):
        raise ValueError("invalid weight_decay")
    for section, fields in [
        (
            "training",
            ("updates", "source_attempts", "sequence_length", "microbatch_size"),
        ),
        ("reflection", ("per_trajectory", "max_tokens")),
        ("runtime", ("timeout_sec",)),
    ]:
        for field in fields:
            if type(c[section][field]) is not int or c[section][field] <= 0:
                raise ValueError(f"{section}.{field} must be a positive integer")
    if c["arm"] == "roft" and (
        c["data"]["outcomes"] != "all" or c["training"]["passes_per_batch"] != 1
    ):
        raise ValueError("ROFT needs both outcomes and one pass, no replay")
    if c["evaluation"].get("reflection_in_context") is not False:
        raise ValueError("evaluation must not receive reflections")
    if not c["data"].get("heldout_tasks"):
        raise ValueError("explicit heldout task list required")
    return c


def validate_sources(rows, config, version=None):
    author = ARMS[config["arm"]][0]
    heldout = set(Path(config["data"]["heldout_tasks"]).read_text().splitlines())
    seen = set()
    for r in rows:
        if r["id"] in seen:
            raise ValueError("duplicate source id")
        seen.add(r["id"])
        if r["task"] in heldout:
            raise ValueError("heldout task in training")
        if r["author"] != author:
            raise ValueError(
                "trajectory author does not match arm (relay mixtures forbidden)"
            )
        model = config[author]
        if r.get("model") != model["model"]:
            raise ValueError("source model mismatch")
        if version is not None:
            if r.get("policy_version") != version:
                raise ValueError(
                    "on-policy batch must use exactly current policy version"
                )
        elif r.get("revision") != model["revision"]:
            raise ValueError("offline source revision mismatch")
        if not r.get("messages") or not any(
            m.get("role") == "assistant" for m in r["messages"]
        ):
            raise ValueError("source has no assistant experience")
    if not rows:
        raise ValueError("empty source batch")


PROMPT = """Explain this completed attempt in a short paragraph, without lists. Identify a consequential assumption or decision, the observations that support or contradict it, and a correction where appropriate. State concrete conditions under which the lesson should apply to a future task. Distinguish whether an individual decision was sound from whether the overall attempt passed. If the evidence is insufficient, acknowledge uncertainty. Do not invent unseen observations or describe yourself as having performed someone else's attempt."""


def bounded(text, limit):
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n[TRUNCATED]\n" + text[-(limit - head) :]


def reflection_prompt(row, settings):
    steps = []
    for m in row["messages"]:
        role = m.get("role", "unknown")
        if role == "system":
            continue
        # Task text is included once separately, not also as an unbounded user turn.
        fields = []
        if m.get("reasoning_content"):
            fields.append("Thought: " + bounded(m["reasoning_content"], 600))
        if m.get("tool_calls"):
            fields.append("Action: " + bounded(json.dumps(m["tool_calls"]), 600))
        if m.get("content"):
            fields.append("Observation/text: " + bounded(m["content"], 600))
        steps.append(role + ":\n" + "\n".join(fields))
    evidence = "\n\n".join(steps)
    verdict = row.get("feedback", {}).get("verdict", "unknown")
    if verdict not in ("pass", "fail", "unknown"):
        raise ValueError("invalid raw verdict")
    blocks = [
        PROMPT,
        "TASK\n" + bounded(row["task_description"], 4000),
        "RECORDED EXPERIENCE\n" + bounded(evidence, 24000),
        "FINAL ARTIFACT/PATCH\n" + bounded(row.get("patch"), 2000),
        "TERMINATION\n" + bounded(row.get("termination", "unknown"), 1000),
    ]
    if settings.get("include_feedback", True):
        blocks.append(
            "RAW VERDICT\n"
            + verdict
            + "\nTEST FEEDBACK\n"
            + bounded(row.get("feedback", {}).get("text"), 3000)
        )
    return "\n\n".join(blocks)
