"""SkyRL's full-TITO assembly checks, vendored standalone.

Mirrors ``_assemble_response_ids_tito_full`` in
``skyrl-train/skyrl_train/trajectory_runners/trajectory_processing.py``: the same
seven checks in the same order, returning the first failing reason. Kept free of
SkyRL imports so it runs against a bare vLLM on a login-free compute node.

The check that matters for MarinSkyRL#520 is PREFIX: the served-stream invariant
``prompt_token_ids[t] == prompt_token_ids[t-1] + completion_token_ids[t-1] + observation[t-1]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MISSING_STREAMS = "missing_streams"
TURN_COUNT = "turn_count"
ASSISTANT_COUNT = "assistant_count"
EMPTY_STREAM = "empty_stream"
PREFIX = "prefix"
GENERATION_PROMPT = "generation_prompt"
COMPLETION_OFFSET = "completion_offset"
OK = None

REASONS = [
    MISSING_STREAMS,
    TURN_COUNT,
    ASSISTANT_COUNT,
    EMPTY_STREAM,
    PREFIX,
    GENERATION_PROMPT,
    COMPLETION_OFFSET,
]


def check_tito(
    prompt_token_ids: Optional[Sequence[Sequence[int]]],
    completion_token_ids: Optional[Sequence[Sequence[int]]],
    n_assistant_messages: Optional[int] = None,
    generation_prompt_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return ``{"reason": <first failing reason or None>, ...details}``.

    ``generation_prompt_ids`` is optional: when omitted the generation-prompt
    check is skipped (it fixes the response/prompt boundary, which a probe that
    never assembles a training sequence does not need).
    """
    if prompt_token_ids is None or completion_token_ids is None:
        return {"reason": MISSING_STREAMS}

    n_turns = len(completion_token_ids)
    if n_turns == 0 or len(prompt_token_ids) != n_turns:
        return {"reason": TURN_COUNT, "n_prompts": len(prompt_token_ids), "n_completions": n_turns}

    if n_assistant_messages is not None and n_assistant_messages != n_turns:
        return {"reason": ASSISTANT_COUNT, "n_assistant": n_assistant_messages, "n_turns": n_turns}

    for t in range(n_turns):
        p, c = prompt_token_ids[t], completion_token_ids[t]
        if not p or not isinstance(p, list) or not c or not isinstance(c, list):
            return {"reason": EMPTY_STREAM, "turn": t}

    for t in range(1, n_turns):
        prev = list(prompt_token_ids[t - 1]) + list(completion_token_ids[t - 1])
        cur = list(prompt_token_ids[t])
        if cur[: len(prev)] != prev:
            offset = next(
                (i for i, (a, b) in enumerate(zip(prev, cur)) if a != b),
                min(len(prev), len(cur)),
            )
            return {
                "reason": PREFIX,
                "turn": t,
                "mismatch_offset": offset,
                "previous_prompt_length": len(prompt_token_ids[t - 1]),
                "previous_completion_length": len(completion_token_ids[t - 1]),
                "expected_length": len(prev),
                "actual_length": len(cur),
                # Offset measured from the start of the previous completion: a
                # positive value means the divergence lands INSIDE what the model
                # actually sampled, which is the #520 failure.
                "offset_into_prev_completion": offset - len(prompt_token_ids[t - 1]),
                "expected_ids": prev[max(0, offset - 4) : offset + 8],
                "actual_ids": cur[max(0, offset - 4) : offset + 8],
            }

    if generation_prompt_ids:
        gp = list(generation_prompt_ids)
        p0 = list(prompt_token_ids[0])
        initial_prompt_len = len(p0) - len(gp)
        if initial_prompt_len < 0 or p0[initial_prompt_len:] != gp:
            return {
                "reason": GENERATION_PROMPT,
                "expected_generation_prompt": gp,
                "actual_prompt_tail": p0[-len(gp) :],
            }

    served_full = list(prompt_token_ids[-1]) + list(completion_token_ids[-1])
    for t in range(n_turns):
        off = len(prompt_token_ids[t])
        comp = list(completion_token_ids[t])
        if served_full[off : off + len(comp)] != comp:
            return {"reason": COMPLETION_OFFSET, "turn": t, "completion_offset": off}

    return {"reason": OK}


def replay_cost(prompt_token_ids: Sequence[Sequence[int]], completion_token_ids: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Cost of the training-side repair: train each turn against its own served prefix.

    Option B in the #520 note. Instead of one linear sequence of length
    ``len(prompt[-1]) + len(completion[-1])``, it trains ``n_turns`` sequences of
    length ``len(prompt[t]) + len(completion[t])``. This is exact by construction
    -- every turn is scored against the context that was actually served -- and
    the ratio below is what it costs in forward/backward tokens.
    """
    if not prompt_token_ids:
        return {}
    linear = len(prompt_token_ids[-1]) + len(completion_token_ids[-1])
    per_turn = sum(len(p) + len(c) for p, c in zip(prompt_token_ids, completion_token_ids))
    scored = sum(len(c) for c in completion_token_ids)
    return {
        "linear_tokens": float(linear),
        "per_turn_replay_tokens": float(per_turn),
        "replay_multiplier": per_turn / max(linear, 1),
        "scored_tokens": float(scored),
        "n_turns": float(len(prompt_token_ids)),
    }
