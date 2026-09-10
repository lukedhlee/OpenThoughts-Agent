"""session_ended_census.py <trials_dir>... [--list]

Count, per trial result.json under the given directories, how each attempt's shell ended:

  agent_ended   the agent's own keystrokes ended its tmux shell (exit / C-d / reset):
                agent_result.metadata.stop_reason == "session_ended"; the verifier still ran
  harness_lost  the harness lost the shell (exec killed, timeout, bridge): exception_info is
                TmuxSessionLostError, or a RuntimeError whose message names the tmux loss
  other         everything else (completed, task_complete, context, turn cap, ...)

--list prints the trial directory of every agent_ended / harness_lost attempt.
"""
import json
import sys
from collections import Counter
from pathlib import Path

_TMUX_LOSS_PHRASES = ("no server running", "can't find session", "produced no markers")


def classify(result: dict) -> str:
    exc = result.get("exception_info") or {}
    exc_type = exc.get("exception_type") or ""
    exc_msg = (exc.get("exception_message") or "").lower()
    if exc_type == "TmuxSessionLostError" or (
        exc_type == "RuntimeError" and any(p in exc_msg for p in _TMUX_LOSS_PHRASES)
    ):
        return "harness_lost"
    meta = (result.get("agent_result") or {}).get("metadata") or {}
    if meta.get("stop_reason") == "session_ended":
        return "agent_ended"
    return "other"


def main(argv: list[str]) -> int:
    list_hits = "--list" in argv
    roots = [Path(a) for a in argv if not a.startswith("--")]
    if not roots:
        print(__doc__)
        return 2
    counts: Counter[str] = Counter()
    hits: list[tuple[str, str]] = []
    for root in roots:
        for f in root.rglob("result.json"):
            try:
                kind = classify(json.loads(f.read_text()))
            except (OSError, ValueError):
                counts["unreadable"] += 1
                continue
            counts[kind] += 1
            if kind != "other":
                hits.append((kind, str(f.parent)))
    total = sum(counts.values())
    for kind in ("agent_ended", "harness_lost", "other", "unreadable"):
        n = counts.get(kind, 0)
        print(f"{kind:13s} {n:7d}  {100.0 * n / total if total else 0:5.1f}%")
    print(f"{'total':13s} {total:7d}")
    if list_hits:
        for kind, path in sorted(hits):
            print(kind, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
