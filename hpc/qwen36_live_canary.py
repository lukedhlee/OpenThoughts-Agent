#!/usr/bin/env python3
"""One-trial Qwen3.6 OpenCode + Apptainer + verifier live canary.

This probe allocates no Slurm nodes.  It runs one Harbor trial through an
already-live SkyRL HTTP endpoint and an already-live JURECA Apptainer bridge.
The selected r2egym task supplies the real cached environment and verifier; an
appended instruction asks the model to execute two sequential, deliberately
large bash outputs so the next model request crosses OpenCode's proactive
compaction threshold.  The original verifier still runs, so its reward may
quite honestly be zero.  That reward is evidence that the verifier path
executed, not a model-quality or reward-variance measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from hpc.qwen36_grpo_preflight import validate_bridge_status, validate_context_contract
from hpc.rl_config_utils import parse_rl_config


MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
OPENCODE_VERSION = "1.18.8"
TOOL_MARKERS = (
    "QWEN36_CANARY_TOOL_ONE_OK",
    "QWEN36_CANARY_TOOL_TWO_OK",
)
TOOL_MARKER = TOOL_MARKERS[0]
FINAL_MARKER = "CANARY_DONE"
# Window advertised to OpenCode. Harbor turns this into
# limit.context = window - output - 1024 = 15360, and OpenCode compacts
# proactively at context - output = 11264 -- far enough below the server's
# hard 28672 prompt ceiling to absorb one large tool observation.
# `opencode_config.compaction.reserved` does NOT move this threshold: OpenCode
# 1.18.8 accepts and writes the key but never reads it.
OPENCODE_CLIENT_WINDOW = 20_480
OPENCODE_SERVED_WINDOW = 32_768

_CONTEXT_RE = re.compile(
    r"prompt contains at least|maximum context length|context length exceeded|"
    r"ContextLengthExceeded|context_length_exceeded|32769 total",
    re.IGNORECASE,
)
_BRIDGE_RE = re.compile(
    r"BridgeOutage|BridgeOperation|workers?_alive|environment start timeout",
    re.IGNORECASE,
)
_TRANSPORT_RE = re.compile(
    r"Network is unreachable|connection (?:refused|reset|aborted)|ECONNREFUSED|"
    r"ENETUNREACH|APIConnectionError|NetworkError|undefined/chat/completions|"
    r"failed to fetch|fetch failed",
    re.IGNORECASE,
)
_VERIFIER_INFRA_RE = re.compile(
    r"VerifierTimeout|VerifierRuntime|VerifierInfrastructure|"
    r"RewardFileNotFound|RewardFileEmpty|VerifierOutputParse|"
    r"VerificationNotCompleted|TrialNotScored",
    re.IGNORECASE,
)


class CanaryFailure(RuntimeError):
    """A fail-loud canary result carrying a stable failure class."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind

    def __str__(self) -> str:
        return f"CANARY_FAIL[{self.kind}]: {super().__str__()}"


def classify_failure(text: str) -> str:
    """Classify known bring-up failures before falling back to AGENT_RUNTIME."""
    if _CONTEXT_RE.search(text):
        return "CONTEXT"
    if _BRIDGE_RE.search(text):
        return "BRIDGE"
    if _TRANSPORT_RE.search(text):
        return "TRANSPORT"
    if _VERIFIER_INFRA_RE.search(text):
        return "VERIFIER_INFRA"
    return "AGENT_RUNTIME"


def normalize_endpoint(endpoint: str) -> str:
    """Require an explicit OpenAI-compatible /v1 base URL."""
    value = endpoint.rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CanaryFailure("ENDPOINT", f"invalid endpoint URL: {endpoint!r}")
    if parsed.path != "/v1":
        raise CanaryFailure(
            "ENDPOINT",
            f"endpoint must end in /v1 (got path {parsed.path!r})",
        )
    return value


def fetch_bridge_status(bridge_url: str, *, timeout_sec: float) -> dict[str, Any]:
    """Fetch and decode the bridge status; HTTP 200 alone is not sufficient."""
    request = urllib.request.Request(
        f"{bridge_url.rstrip('/')}/status",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CanaryFailure("BRIDGE", f"bridge /status request failed: {exc}") from exc
    try:
        status = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryFailure("BRIDGE", "bridge /status did not return JSON") from exc
    if not isinstance(status, dict):
        raise CanaryFailure("BRIDGE", "bridge /status response is not an object")
    return status


def select_task(tasks_dir: Path) -> Path:
    """Select one task with bounded, top-level-only directory inspection."""
    if not tasks_dir.is_dir():
        raise CanaryFailure("TASK", f"tasks directory does not exist: {tasks_dir}")
    candidates: list[Path] = []
    with os.scandir(tasks_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            path = Path(entry.path)
            if all(
                candidate.exists()
                for candidate in (
                    path / "instruction.md",
                    path / "task.toml",
                    path / "environment",
                    path / "tests" / "test.sh",
                )
            ):
                candidates.append(path)
    if not candidates:
        raise CanaryFailure("TASK", f"no valid task found directly under {tasks_dir}")
    return min(candidates, key=lambda path: path.name)


def canary_instruction(markers: tuple[str, str] = TOOL_MARKERS) -> str:
    """Return a two-turn large-tool-output context-boundary probe."""
    first, second = markers
    return f"""# Pipeline canary override

This is an infrastructure canary, not the repository-fixing task above. Do not
attempt the original issue. Run exactly two bash tool calls, sequentially in two
separate assistant turns. Never combine or parallelize them.

First call:

    python -c 'print(" cat"*11000); print("{first}")'

Wait for that result. Then make this separate second call:

    python -c 'print(" cat"*11000); print("{second}")'

After the second result, reply exactly `{FINAL_MARKER}` and stop. Do not inspect
files, attempt the original task, or perform any other work.
"""


def opencode_error_text(raw_text: str) -> str:
    """Extract only OpenCode error events, avoiding false matches in reasoning."""
    errors: list[str] = []
    for line in raw_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "error":
            errors.append(json.dumps(event.get("error"), sort_keys=True))
    return "\n".join(errors)


def _tool_markers_completed(
    trajectory: Any, markers: tuple[str, str] = TOOL_MARKERS
) -> bool:
    """Require both marker calls in distinct, ordered trajectory steps."""
    if not isinstance(trajectory, dict):
        return False
    completed: list[tuple[int, str]] = []
    for step_index, step in enumerate(trajectory.get("steps") or []):
        if not isinstance(step, dict):
            continue
        calls = step.get("tool_calls") or []
        observation = step.get("observation") or {}
        results = (
            observation.get("results") or [] if isinstance(observation, dict) else []
        )
        outputs = {
            str(result.get("source_call_id")): str(result.get("content", ""))
            for result in results
            if isinstance(result, dict)
        }
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("tool_call_id"))
            arguments = json.dumps(call.get("arguments"), sort_keys=True)
            matches = [
                marker
                for marker in markers
                if marker in arguments and marker in outputs.get(call_id, "")
            ]
            if len(matches) == 1:
                completed.append((step_index, matches[0]))
    if len(completed) != len(markers):
        return False
    if [marker for _, marker in completed] != list(markers):
        return False
    return completed[0][0] < completed[1][0]


def opencode_input_tokens(raw_text: str) -> list[int]:
    """Extract ordered prompt-token counts from OpenCode step-finish events."""
    values: list[int] = []
    for line in raw_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "step_finish":
            continue
        part = event.get("part")
        tokens = part.get("tokens") if isinstance(part, dict) else None
        value = tokens.get("input") if isinstance(tokens, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    return values


def has_compaction_drop(raw_text: str) -> bool:
    """Prove a large prompt was compacted before the next model request."""
    values = opencode_input_tokens(raw_text)
    return any(
        before >= 11_264 and before - after >= 4_096
        for before, after in zip(values, values[1:])
    )


def has_final_marker(raw_text: str, marker: str = FINAL_MARKER) -> bool:
    """Require the model's final text event, not merely the injected prompt."""
    for line in raw_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part")
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and marker in str(part.get("text", ""))
        ):
            return True
    return False


def validate_canary_artifacts(
    result: Any,
    trajectory: Any,
    *,
    opencode_errors: str = "",
    opencode_raw: str = "",
    markers: tuple[str, str] = TOOL_MARKERS,
) -> float:
    """Validate exact identity, a completed tool action, and numeric reward."""
    if not isinstance(result, dict):
        raise CanaryFailure("RESULT", "Harbor result is not a JSON object")

    exception = result.get("exception_info")
    if exception:
        diagnostic = json.dumps(exception, sort_keys=True) + "\n" + opencode_errors
        raise CanaryFailure(classify_failure(diagnostic), diagnostic)
    if opencode_errors:
        raise CanaryFailure(classify_failure(opencode_errors), opencode_errors)

    agent = result.get("agent_info")
    if not isinstance(agent, dict) or agent.get("name") != "opencode":
        raise CanaryFailure("IDENTITY", f"unexpected agent identity: {agent!r}")
    if agent.get("version") != OPENCODE_VERSION:
        raise CanaryFailure(
            "IDENTITY",
            f"expected OpenCode {OPENCODE_VERSION}, got {agent.get('version')!r}",
        )
    model = agent.get("model_info")
    if not isinstance(model, dict) or model.get("name") != MODEL_REVISION:
        raise CanaryFailure(
            "IDENTITY",
            f"expected served model {MODEL_REVISION}, got {model!r}",
        )
    if model.get("provider") != "vllm":
        raise CanaryFailure("IDENTITY", f"expected vllm provider, got {model!r}")

    if not _tool_markers_completed(trajectory, markers):
        raise CanaryFailure(
            "TOOL_ACTION",
            "trajectory lacks two sequential marker-bearing tool observations",
        )
    # A ContextOverflowError anywhere in the stream means the hard ceiling was
    # crossed and OpenCode compacted REACTIVELY to recover. That still produces
    # a large prompt drop, so has_compaction_drop() alone would accept it --
    # but it is exactly the failure this canary exists to catch, and under GRPO
    # the wasted request is a real rollout defect. Reject it explicitly, even
    # when OpenCode goes on to exit zero.
    if _CONTEXT_RE.search(opencode_raw) or "ContextOverflowError" in opencode_raw:
        raise CanaryFailure(
            "COMPACTION",
            "OpenCode hit the server's hard prompt ceiling and compacted "
            "reactively; proactive compaction must fire BEFORE the overflow",
        )
    if not has_compaction_drop(opencode_raw):
        raise CanaryFailure(
            "COMPACTION",
            "OpenCode events show no >=4096-token prompt drop after crossing the "
            "proactive compaction threshold",
        )
    if not has_final_marker(opencode_raw):
        raise CanaryFailure(
            "COMPACTION", f"OpenCode never emitted final marker {FINAL_MARKER}"
        )

    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise CanaryFailure(
            "VERIFIER_INFRA",
            f"verifier did not emit a numeric reward: {verifier!r}",
        )
    reward_float = float(reward)
    if not math.isfinite(reward_float):
        raise CanaryFailure("VERIFIER_INFRA", f"non-finite verifier reward: {reward!r}")
    return reward_float


def validate_materialized_trial_contract(
    config: Any,
    *,
    endpoint: str,
    bridge_url: str,
) -> None:
    """Reject drift at the final Harbor TrialConfig boundary."""
    if not isinstance(config, dict):
        raise CanaryFailure("CONTRACT", "materialized TrialConfig is not an object")
    agent = config.get("agent")
    if not isinstance(agent, dict):
        raise CanaryFailure("CONTRACT", "materialized TrialConfig lacks agent config")
    if agent.get("name") != "opencode":
        raise CanaryFailure("CONTRACT", f"unexpected agent config: {agent!r}")
    if agent.get("model_name") != f"vllm/{MODEL_REVISION}":
        raise CanaryFailure(
            "CONTRACT",
            f"unexpected materialized model alias: {agent.get('model_name')!r}",
        )
    kwargs = agent.get("kwargs")
    if not isinstance(kwargs, dict):
        raise CanaryFailure("CONTRACT", "materialized agent kwargs are missing")
    expected_info = {
        "max_input_tokens": OPENCODE_CLIENT_WINDOW,
        "max_output_tokens": 4_096,
    }
    model_info = kwargs.get("model_info")
    if not isinstance(model_info, dict) or any(
        model_info.get(key) != value for key, value in expected_info.items()
    ):
        raise CanaryFailure(
            "CONTRACT",
            f"materialized model_info must include {expected_info}, got {model_info!r}",
        )
    if (
        kwargs.get("version") != OPENCODE_VERSION
        or kwargs.get("preinstalled") is not True
    ):
        raise CanaryFailure(
            "CONTRACT",
            "materialized OpenCode must be preinstalled at exact version "
            f"{OPENCODE_VERSION}",
        )
    compaction = kwargs.get("opencode_config", {}).get("compaction", {})
    if compaction.get("auto") is not True:
        raise CanaryFailure(
            "CONTRACT", "materialized OpenCode compaction.auto is not true"
        )
    if "reserved" in compaction:
        raise CanaryFailure(
            "CONTRACT",
            "materialized OpenCode compaction declares `reserved`, which OpenCode "
            f"{OPENCODE_VERSION} silently ignores; the threshold must come from "
            "context_budget.client_window_tokens instead",
        )
    if kwargs.get("api_base") != endpoint:
        raise CanaryFailure(
            "CONTRACT",
            f"materialized api_base is not the requested endpoint: {kwargs.get('api_base')!r}",
        )

    environment = config.get("environment")
    if not isinstance(environment, dict) or environment.get("type") != "apptainer":
        raise CanaryFailure(
            "CONTRACT", f"unexpected environment config: {environment!r}"
        )
    if environment.get("kwargs", {}).get("bridge_url") != bridge_url:
        raise CanaryFailure(
            "CONTRACT", "materialized bridge URL does not match the requested fleet"
        )


async def run_live_canary(args: argparse.Namespace) -> tuple[float, Path]:
    """Build and execute one exact Harbor trial using the live runtime imports."""
    try:
        from omegaconf import OmegaConf

        from examples.terminal_bench.harbor_config import HarborConfigBuilder
        from harbor.trial.trial import Trial
    except ImportError as exc:
        raise CanaryFailure(
            "RUNTIME",
            "could not import the pinned Harbor/SkyRL runtime; set PYTHONPATH to "
            "<harbor>/src:<MarinSkyRL>/skyrl-train:<OpenThoughts-Agent>",
        ) from exc

    parsed = parse_rl_config(str(args.rl_config))
    terminal = OmegaConf.create(parsed.terminal_bench)
    terminal.trials_dir = str(args.trials_dir)
    terminal.harbor.bridge_url = args.bridge_url
    terminal.harbor.n_concurrent_trials = 1
    # The compaction threshold is set by context_budget.client_window_tokens,
    # already materialized into model_info by parse_rl_config. Nothing to force
    # here -- and `compaction.reserved` deliberately must NOT be injected,
    # because OpenCode ignores it (see validate_materialized_trial_contract).

    builder = HarborConfigBuilder(terminal)
    config = builder.build_trial_config(
        task_path=str(args.task_path),
        trials_dir=str(args.trials_dir),
        model_name=f"hosted_vllm/{MODEL_REVISION}",
        api_base=args.endpoint,
        session_id=uuid4().hex,
        timeout_override_sec=args.agent_timeout_sec,
    )
    validate_materialized_trial_contract(
        config.model_dump(mode="json"),
        endpoint=args.endpoint,
        bridge_url=args.bridge_url,
    )

    trial_dir = args.trials_dir / config.trial_name
    trial_dir.mkdir(parents=True, exist_ok=False)
    extra_instruction = trial_dir / "canary_instruction.md"
    extra_instruction.write_text(canary_instruction())
    config.extra_instruction_paths = [extra_instruction]

    trial = await Trial.create(config)
    result_model = await trial.run()
    result = result_model.model_dump(mode="json")

    trajectory_path = trial_dir / "agent" / "trajectory.json"
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        trajectory = None
    raw_path = trial_dir / "agent" / "opencode.txt"
    try:
        raw_text = raw_path.read_text()
        errors = opencode_error_text(raw_text)
    except OSError:
        raw_text = ""
        errors = ""

    try:
        reward = validate_canary_artifacts(
            result,
            trajectory,
            opencode_errors=errors,
            opencode_raw=raw_text,
        )
    except CanaryFailure as exc:
        raise CanaryFailure(exc.kind, f"{exc.args[0]}\ntrial={trial_dir}") from exc
    return reward, trial_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-config", type=Path, required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task-path", type=Path)
    task.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://jrlogin05i:18000/v1",
        help="Endpoint as reachable from inside JURECA sandboxes; must end in /v1",
    )
    parser.add_argument("--bridge-url", default="http://10.128.1.2:9920")
    parser.add_argument("--bridge-timeout-sec", type=float, default=10.0)
    parser.add_argument("--agent-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--require-clean-bridge",
        choices=("0", "1"),
        default="1",
        help="Reject active bridge work by default; use 0 only for intentional sharing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        args.endpoint = normalize_endpoint(args.endpoint)
        try:
            validate_context_contract(args.rl_config)
        except ValueError as exc:
            raise CanaryFailure("CONTEXT_CONTRACT", str(exc)) from exc
        status = fetch_bridge_status(
            args.bridge_url, timeout_sec=args.bridge_timeout_sec
        )
        try:
            validate_bridge_status(
                status,
                require_clean=args.require_clean_bridge == "1",
            )
        except ValueError as exc:
            raise CanaryFailure("BRIDGE", str(exc)) from exc
        args.task_path = args.task_path or select_task(args.tasks_dir)
        args.trials_dir.mkdir(parents=True, exist_ok=True)
        reward, trial_dir = asyncio.run(run_live_canary(args))
    except (CanaryFailure, OSError, ValueError) as exc:
        if isinstance(exc, CanaryFailure):
            raise SystemExit(str(exc)) from exc
        raise SystemExit(f"CANARY_FAIL[CONFIG]: {exc}") from exc

    print(
        "CANARY_PASS: exact model + OpenCode 1.18.8 + tool action + verifier reward; "
        f"reward={reward:g} trial={trial_dir}"
    )
    print(
        "NOTE: this single canary reward is diagnostic only; it is not reward variance."
    )


if __name__ == "__main__":
    main()
