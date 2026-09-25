#!/usr/bin/env python3
"""Readout of a mini-swe-agent-host smoke job: the plan's Stage 2/3 report fields.

    PYTHONPATH=<harbor clone>/src:<mini-swe-agent 2.4.6 dir> python readout.py <jobs_dir>/<job_name>

Per job: trials, harness errors (any trial exception other than the classified agent ends), agent timeouts,
context overflows, format errors (and false ones: a rejected reply whose answer, think spans removed, holds exactly
one action block), command timeouts, submit rate, pass rate, median model calls, and whether the submit sentinel
ended every episode that printed it. Think-span counts show what the parser had to strip.
"""

import glob
import json
import os
import re
import statistics
import sys

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
from harbor.agents.mini_swe_agent_host.adapters import SUBMIT_SENTINEL, strip_think_spans  # noqa: E402

ACTION = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
THINK_MARKERS = ("<think>", "</think>", "<|start_think|>", "<|end_think|>")
AGENT_ENDS = {"AgentTimeoutError", "ContextLengthExceededError", "TurnCapExhaustedError"}


def trial_rows(job_dir: str):
    for result_path in sorted(glob.glob(f"{job_dir}/*/attempts/*/result.json")):
        attempt = os.path.dirname(result_path)
        result = json.load(open(result_path))
        trajectory_path = f"{attempt}/agent/mini-swe-agent.trajectory.json"
        trajectory = json.load(open(trajectory_path)) if os.path.exists(trajectory_path) else None
        yield attempt, result, trajectory


def main() -> None:
    job_dir = sys.argv[1]
    rows = []
    for attempt, result, trajectory in trial_rows(job_dir):
        exception = (result.get("exception_info") or {}).get("exception_type")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        messages = trajectory["messages"] if trajectory else []
        replies = [m["content"] for m in messages if m["role"] == "assistant"] + [
            m["extra"]["model_response"] for m in messages
            if m.get("extra", {}).get("interrupt_type") == "FormatError"
        ]
        format_errors = [m for m in messages if m.get("extra", {}).get("interrupt_type") == "FormatError"]
        false_format_errors = [
            m for m in format_errors
            if len(ACTION.findall(strip_think_spans(m["extra"]["model_response"]))) == 1
        ]
        actions = [a["command"] for m in messages if m["role"] == "assistant" for a in m["extra"].get("actions", [])]
        sentinel_actions = [a for a in actions if SUBMIT_SENTINEL in a]
        exit_status = trajectory["info"]["exit_status"] if trajectory else None
        ended_by_sentinel = exit_status == "Submitted" and bool(actions) and SUBMIT_SENTINEL in actions[-1]
        rows.append({
            "trial": os.path.basename(os.path.dirname(os.path.dirname(attempt))),
            "exception": exception,
            "reward": rewards.get("reward"),
            "exit_status": exit_status,
            "model_calls": len(replies),
            "format_errors": len(format_errors),
            "false_format_errors": len(false_format_errors),
            "command_timeouts": sum(
                1 for m in messages
                if m["role"] == "user" and "timed out after" in (m.get("extra", {}).get("exception_info") or "")
            ),
            "think_replies": sum(1 for r in replies if any(k in r for k in THINK_MARKERS)),
            "think_quoted_blocks": sum(
                1 for r in replies if len(ACTION.findall(r)) > len(ACTION.findall(strip_think_spans(r)))
            ),
            "sentinel_actions": len(sentinel_actions),
            "ended_by_sentinel": ended_by_sentinel,
        })

    n = len(rows)
    total_calls = sum(r["model_calls"] for r in rows)
    summary = {
        "job": job_dir,
        "trials": n,
        "harness_errors": sum(1 for r in rows if r["exception"] and r["exception"] not in AGENT_ENDS),
        "agent_timeouts": sum(1 for r in rows if r["exception"] == "AgentTimeoutError"),
        "context_exceeded": sum(1 for r in rows if r["exception"] == "ContextLengthExceededError"),
        "format_errors": sum(r["format_errors"] for r in rows),
        "format_error_rate_per_call": round(sum(r["format_errors"] for r in rows) / total_calls, 4) if total_calls else None,
        "false_format_errors": sum(r["false_format_errors"] for r in rows),
        "command_timeouts": sum(r["command_timeouts"] for r in rows),
        "submit_rate": round(sum(1 for r in rows if r["exit_status"] == "Submitted") / n, 3) if n else None,
        "pass_rate": round(sum(1 for r in rows if (r["reward"] or 0) >= 1.0) / n, 3) if n else None,
        "mean_reward": round(sum((r["reward"] or 0) for r in rows) / n, 3) if n else None,
        "median_model_calls": statistics.median(r["model_calls"] for r in rows) if rows else None,
        "sentinel_ended_every_submit": all(
            r["ended_by_sentinel"] for r in rows if r["exit_status"] == "Submitted"
        ),
        "sentinel_printed_without_ending": sum(
            1 for r in rows if r["sentinel_actions"] and not r["ended_by_sentinel"]
        ),
        "think_replies": sum(r["think_replies"] for r in rows),
        "think_quoted_blocks": sum(r["think_quoted_blocks"] for r in rows),
        "exit_statuses": {s: sum(1 for r in rows if str(r["exit_status"]) == s) for s in sorted({str(r["exit_status"]) for r in rows})},
        "exceptions": {e: sum(1 for r in rows if str(r["exception"]) == e) for e in sorted({str(r["exception"]) for r in rows})},
    }
    for r in rows:
        print(json.dumps(r))
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
