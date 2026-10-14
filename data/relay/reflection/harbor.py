"""Import existing MSA2 Harbor trials without changing source artifacts."""

import json
from pathlib import Path

from .core import file_digest


def atif_messages(trial_dir, result):
    """Chat messages rebuilt from the last attempt's ATIF trajectory: agent steps are the raw replies, their
    observations the harness's answers (tool output or format error)."""
    rel = result.get("trial_relpath")
    candidates = [trial_dir.parent / rel / "agent/trajectory.json"] if rel else []
    candidates += sorted(trial_dir.glob("attempts/*/agent/trajectory.json"))[-1:]
    for traj in candidates:
        if not traj.exists():
            continue
        steps = json.loads(traj.read_text()).get("steps") or []
        msgs = []
        for step in steps:
            source = step.get("source")
            if source in ("system", "user"):
                msgs.append({"role": source, "content": step.get("message") or ""})
            elif source == "agent":
                msgs.append({"role": "assistant", "content": step.get("message") or ""})
                results = (step.get("observation") or {}).get("results") or []
                text = "\n".join(str(r.get("content", "")) for r in results)
                if text:
                    msgs.append({"role": "user", "content": text})
        return msgs
    return None


def import_trials(roots, *, author, model, revision, policy_version=0):
    rows = []
    rejected = {}
    seen = set()

    def reject(why):
        rejected[why] = rejected.get(why, 0) + 1

    for root in roots:
        for path in sorted(Path(root).glob("*/result.json")):
            result = json.loads(path.read_text())
            sid = result.get("id") or str(path.parent.resolve())
            if sid in seen:
                reject("duplicate_result")
                continue
            seen.add(sid)
            # Result metadata is the authoritative exact MSA message stream in this Harbor version.
            msgs = (
                (result.get("agent_result") or {})
                .get("metadata", {})
                .get("all_messages")
            )
            experience_source = "all_messages"
            if not msgs or not any(m.get("role") == "assistant" for m in msgs):
                # MSA keeps only the harness's error feedback in all_messages when every reply was malformed (e.g.
                # RepeatedFormatError): the model's replies survive only in the attempt's ATIF trajectory.
                msgs = atif_messages(path.parent, result)
                experience_source = "atif_trajectory"
            if not msgs or not any(m.get("role") == "assistant" for m in msgs):
                reject("no_experience")
                continue
            if experience_source != "all_messages":
                reject("recovered_from_atif")  # counted, not dropped
            task = result.get("task_name")
            task_path = (result.get("task_id") or {}).get("path")
            instruction = Path(task_path) / "instruction.md" if task_path else None
            if instruction and instruction.exists():
                task_text = instruction.read_text()
            else:
                task_text = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "user"), ""
                )
            # Preserve a raw verifier reward independently of termination/errors. Never treat missing as failure.
            vr = result.get("verifier_result") or {}
            rewards = vr.get("rewards") or {}
            reward = rewards.get("reward")
            verdict = (
                "unknown" if reward is None else ("pass" if reward == 1 else "fail")
            )
            feedback_path = path.parent / "verifier/test-stdout.txt"
            feedback = feedback_path.read_text() if feedback_path.exists() else ""
            error = result.get("exception_info") or {}
            termination = error.get("exception_type") or "completed"
            rows.append(
                {
                    "id": sid,
                    "task": task,
                    "author": author,
                    "model": model,
                    "revision": revision,
                    "policy_version": policy_version,
                    "task_description": task_text,
                    "messages": msgs,
                    "patch": "",
                    "feedback": {
                        "verdict": verdict,
                        "raw_reward": reward,
                        "text": feedback,
                    },
                    "termination": termination,
                    "experience_source": experience_source,
                    "source_path": str(path.resolve()),
                    "source_sha256": file_digest(path),
                }
            )
    return rows, rejected
