#!/usr/bin/env python3
"""Readout of a mini-swe-agent-host smoke job: the plan's Stage 2/3 report fields.

    PYTHONPATH=<harbor clone>/src:<mini-swe-agent 2.4.6 dir> python readout.py <jobs_dir>/<job_name>

Per job: trials, harness errors (any trial exception other than the classified agent ends), agent timeouts,
context overflows, format errors by cause, false format errors, command timeouts, submit rate, pass rate, median model
calls, and whether the submit sentinel ended every episode that printed it.

A format error is false when the reply's answer holds exactly one action block. The answer is the whole content when
the server's reasoning parser already split the reasoning off (the ATIF step carries reasoning_content); otherwise it is
everything after the reasoning span(s) that open the reply, or after the first closing marker when the template opened
the span. This is computed here from the recorded raw reply, independently of the agent's parser, so a parser bug
shows up as a count.
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
ACTION = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
MARKERS = (("<think>", "</think>"), ("<|start_think|>", "<|end_think|>"))
THINK_MARKERS = tuple(m for pair in MARKERS for m in pair)
AGENT_ENDS = {"AgentTimeoutError", "ContextLengthExceededError", "TurnCapExhaustedError"}


def split_reply(reply: str) -> tuple[str, str, bool]:
    """(reasoning, answer, ended_inside_reasoning) for a raw reply."""
    reasoning, text, leading = "", reply, False
    while True:
        stripped = text.lstrip()
        for open_marker, close_marker in MARKERS:
            if stripped.startswith(open_marker):
                inner, closed, rest = stripped[len(open_marker):].partition(close_marker)
                reasoning += inner
                if not closed:
                    return reasoning, "", True
                text, leading = rest, True
                break
        else:
            break
    if not leading:
        for _, close_marker in MARKERS:
            inner, closed, rest = text.partition(close_marker)
            if closed:
                return inner, rest, False
    return reasoning, text, False


def format_error_cause(reply: str, finish_reason: str | None, server_split: bool) -> str:
    reasoning, answer, unfinished = ("", reply, False) if server_split else split_reply(reply)
    blocks = len(ACTION.findall(answer))
    if blocks == 1:
        return "FALSE (one block in the answer)"
    if finish_reason == "length":
        return "truncated at the output limit"
    if unfinished:
        return "reply ended inside reasoning"
    if blocks > 1:
        stray = any(close in answer for _, close in MARKERS)
        return "several blocks (run-on past a stray end marker)" if stray else "several blocks"
    if re.search(r'"(keystrokes|commands|analysis)"\s*:', answer):
        return "Terminus-2 JSON instead of a block"
    if "```" in answer:
        return "other fence (```bash etc.)"
    return "no block"


INLINE_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def tool_format_error_cause(reply: str, finish_reason: str | None) -> str:
    """Tool mode: why a raw reply did not yield a runnable bash call."""
    reasoning, answer, unfinished = split_reply(reply)
    bodies = INLINE_TOOL_CALL.findall(answer)
    problems = []
    for body in bodies:
        try:
            call = json.loads(body)
        except json.JSONDecodeError:
            problems.append("invalid tool-call JSON")
            continue
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                problems.append("invalid arguments JSON")
                continue
        if not isinstance(call, dict) or call.get("name") != "bash":
            problems.append("unknown tool")
        elif not isinstance(arguments, dict) or "command" not in arguments:
            problems.append("missing command")
    if bodies and not problems:
        return "FALSE (well-formed bash call in the answer)"
    if finish_reason == "length":
        return "truncated at the output limit"
    if unfinished:
        return "reply ended inside reasoning"
    if problems:
        return problems[0]
    if "<tool_call>" in answer:
        return "tool call opened, never closed"
    if "```mswea_bash_command" in answer:
        return "text-mode block instead of a tool call"
    if re.search(r'"(keystrokes|commands|analysis)"\s*:', answer):
        return "Terminus-2 JSON instead of a tool call"
    if "```" in answer:
        return "fenced code instead of a tool call"
    return "prose, no tool call"


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
        atif_path = f"{attempt}/agent/trajectory.json"
        agent_steps = (
            [s for s in json.load(open(atif_path))["steps"] if s["source"] == "agent"]
            if os.path.exists(atif_path) else []
        )
        tool_mode = "HarborToolModel" in json.dumps((trajectory or {}).get("info", {}).get("config", {}))
        causes, call = [], -1
        for m in messages[2:]:
            is_format_error = m.get("extra", {}).get("interrupt_type") == "FormatError"
            if m["role"] == "assistant" or is_format_error:
                call += 1
            if is_format_error:
                finish = "length" if "finish_reason=length" in m["content"] else None
                if tool_mode:
                    causes.append(tool_format_error_cause(m["extra"]["model_response"], finish))
                    continue
                server_split = call < len(agent_steps) and bool(agent_steps[call].get("reasoning_content"))
                causes.append(format_error_cause(m["extra"]["model_response"], finish, server_split))
        if tool_mode:
            # Raw replies (tool_call blocks included) are the ATIF agent steps' messages.
            replies = [s["message"] for s in agent_steps]
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
            "format_error_causes": causes,
            "false_format_errors": sum(1 for c in causes if c.startswith("FALSE")),
            "mode": "tool" if tool_mode else "text",
            "multi_call_replies": sum(
                1 for m in messages if m["role"] == "assistant" and len(m["extra"].get("actions", [])) > 1
            ),
            "command_timeouts": sum(
                1 for m in messages
                if m["role"] in ("user", "tool")
                and "timed out after" in (m.get("extra", {}).get("exception_info") or "")
            ),
            "think_replies": sum(1 for r in replies if any(k in r for k in THINK_MARKERS)),
            "think_quoted_blocks": sum(1 for r in replies if ACTION.search(split_reply(r)[0])),
            "sentinel_actions": len(sentinel_actions),
            "ended_by_sentinel": ended_by_sentinel,
        })

    n = len(rows)
    total_calls = sum(r["model_calls"] for r in rows)
    summary = {
        "job": job_dir,
        "trials": n,
        "mode": sorted({r["mode"] for r in rows}),
        "multi_call_replies": sum(r["multi_call_replies"] for r in rows),
        "harness_errors": sum(1 for r in rows if r["exception"] and r["exception"] not in AGENT_ENDS),
        "agent_timeouts": sum(1 for r in rows if r["exception"] == "AgentTimeoutError"),
        "context_exceeded": sum(1 for r in rows if r["exception"] == "ContextLengthExceededError"),
        "format_errors": sum(r["format_errors"] for r in rows),
        "format_error_rate_per_call": round(sum(r["format_errors"] for r in rows) / total_calls, 4) if total_calls else None,
        "false_format_errors": sum(r["false_format_errors"] for r in rows),
        "format_error_causes": dict(collections.Counter(c for r in rows for c in r["format_error_causes"]).most_common()),
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
