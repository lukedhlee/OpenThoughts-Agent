#!/usr/bin/env python3
"""Fast, allocation-free checks for the exact Qwen3.6 GRPO smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hpc.rl_config_utils import parse_rl_config


EXPECTED_WINDOW = 32_768
EXPECTED_OUTPUT = 4_096
EXPECTED_OPENCODE_CONTEXT = 27_648
EXPECTED_OPENCODE_RESERVED = 16_384
ACTIVE_ENV_STATES = ("pending", "starting", "ready", "stopping")


def validate_bridge_status(status: Any, *, require_clean: bool) -> None:
    """Reject a dead fleet, and optionally reject state left by an earlier smoke."""
    if not isinstance(status, dict):
        raise ValueError("bridge /status response must be a JSON object")
    if status.get("workers_alive") is not True:
        raise ValueError(
            "bridge worker fleet is not alive (workers_alive must be true); "
            "HTTP 200 alone is insufficient"
        )
    if not require_clean:
        return

    envs = status.get("envs")
    if not isinstance(envs, dict):
        raise ValueError("bridge /status response lacks the envs state mapping")

    dirty: list[str] = []
    for state in ACTIVE_ENV_STATES:
        count = envs.get(state)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"bridge envs.{state} must be a non-negative integer")
        if count:
            dirty.append(f"envs.{state}={count}")

    for field in ("queue_size", "active_jobs"):
        count = status.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"bridge {field} must be a non-negative integer")
        if count:
            dirty.append(f"{field}={count}")

    if dirty:
        raise ValueError(
            "dedicated smoke requires an idle bridge; stale work found: "
            + ", ".join(dirty)
            + ". Stop the stale environments/jobs or set "
            "REQUIRE_CLEAN_BRIDGE=0 only when fleet sharing is intentional."
        )


def harbor_opencode_limit(model_info: dict[str, Any]) -> dict[str, int]:
    """Mirror Harbor d93ca639 OpenCode._resolve_model_limit semantics.

    Keeping this tiny calculation local makes the launch preflight independent of
    whichever Harbor checkout happens to be importable on the Jupiter login node.
    """

    def positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    output = positive_int(model_info.get("max_output_tokens"), EXPECTED_OUTPUT)
    window = positive_int(model_info.get("max_input_tokens"), EXPECTED_WINDOW)
    output = min(output, max(1, window - 1))
    margin = min(1024, max(0, window - output - 1))
    context = max(1, window - output - margin)
    return {"context": context, "output": output}


def validate_context_contract(config_path: Path) -> None:
    """Validate the materialized Harbor/OpenCode contract used by this smoke."""
    parsed = parse_rl_config(str(config_path))
    terminal = parsed.terminal_bench
    if not isinstance(terminal, dict):
        raise ValueError("RL config did not materialize terminal_bench")

    harbor = terminal.get("harbor")
    if not isinstance(harbor, dict) or harbor.get("name") != "opencode":
        raise ValueError("Qwen3.6 smoke requires the Harbor OpenCode agent")
    if harbor.get("version") != "1.18.8":
        raise ValueError("Qwen3.6 smoke requires pinned OpenCode 1.18.8")
    compaction = harbor.get("opencode_config", {}).get("compaction", {})
    if compaction.get("auto") is not True:
        raise ValueError(
            "OpenCode compaction.auto must be true or its history can grow past "
            "the vLLM request boundary"
        )
    if compaction.get("reserved") != EXPECTED_OPENCODE_RESERVED:
        raise ValueError(
            "OpenCode compaction.reserved must be "
            f"{EXPECTED_OPENCODE_RESERVED} to absorb large tool-output jumps"
        )
    retry_exclusions = harbor.get("exclude_exceptions")
    required_fail_fast = {"ContextLengthExceededError", "NonZeroAgentExitCodeError"}
    if not isinstance(retry_exclusions, list) or not required_fail_fast.issubset(
        retry_exclusions
    ):
        raise ValueError(
            "deterministic OpenCode/context failures must bypass Harbor retry "
            f"backoff: missing {sorted(required_fail_fast - set(retry_exclusions or []))}"
        )

    model_info = terminal.get("model_info")
    expected_info = {
        "max_input_tokens": EXPECTED_WINDOW,
        "max_output_tokens": EXPECTED_OUTPUT,
    }
    if model_info != expected_info:
        raise ValueError(
            f"materialized Harbor model_info must be {expected_info}, got {model_info!r}"
        )

    limit = harbor_opencode_limit(model_info)
    expected_limit = {
        "context": EXPECTED_OPENCODE_CONTEXT,
        "output": EXPECTED_OUTPUT,
    }
    if limit != expected_limit:
        raise ValueError(
            f"Harbor/OpenCode model limit must be {expected_limit}, got {limit!r}"
        )

    if parsed.trainer.get("max_prompt_length") != EXPECTED_WINDOW - EXPECTED_OUTPUT:
        raise ValueError(
            "trainer prompt allowance is inconsistent with the smoke window"
        )
    generator = parsed.generator
    if generator.get("max_input_length") != EXPECTED_WINDOW - EXPECTED_OUTPUT:
        raise ValueError(
            "generator prompt allowance is inconsistent with the smoke window"
        )
    if generator.get("engine_init_kwargs", {}).get("max_model_len") != EXPECTED_WINDOW:
        raise ValueError("vLLM max_model_len is inconsistent with the smoke window")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-config", type=Path, required=True)
    parser.add_argument(
        "--bridge-status-file",
        type=Path,
        required=True,
        help="Path containing the JSON response from the bridge /status endpoint",
    )
    parser.add_argument("--require-clean-bridge", choices=("0", "1"), default="1")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        status = json.loads(args.bridge_status_file.read_text())
        validate_bridge_status(status, require_clean=args.require_clean_bridge == "1")
        validate_context_contract(args.rl_config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"ERROR: Qwen3.6 GRPO fast preflight failed: {exc}") from exc

    print(
        "Qwen3.6 fast preflight passed: workers_alive=true, "
        f"clean_bridge={args.require_clean_bridge == '1'}, "
        f"OpenCode limit={EXPECTED_OPENCODE_CONTEXT}+{EXPECTED_OUTPUT}, "
        f"compaction_reserved={EXPECTED_OPENCODE_RESERVED}"
    )


if __name__ == "__main__":
    main()
