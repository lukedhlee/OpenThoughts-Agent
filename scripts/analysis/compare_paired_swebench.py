#!/usr/bin/env python3
"""Compare matched baseline and post-GRPO Harbor SWE-bench runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_trials(job_dir: Path) -> dict[str, float]:
    trials: dict[str, float] = {}
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task = result.get("task_name")
        if not task:
            raise ValueError(f"{result_path}: missing task_name")
        if result.get("exception_info") is not None:
            raise ValueError(f"{result_path}: trial has exception_info")
        try:
            reward = float(result["verifier_result"]["rewards"]["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{result_path}: missing numeric verifier reward") from exc
        if reward not in (0.0, 1.0):
            raise ValueError(f"{result_path}: expected binary reward, got {reward}")
        if task in trials:
            raise ValueError(f"{job_dir}: duplicate task {task}")
        trials[task] = reward
    return trials


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    deltas: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    lower = means[int(0.025 * samples)]
    upper = means[min(samples - 1, int(0.975 * samples))]
    return lower, upper


def compare(
    baseline: dict[str, float],
    post: dict[str, float],
    *,
    expected_tasks: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if set(baseline) != set(post):
        only_base = sorted(set(baseline) - set(post))
        only_post = sorted(set(post) - set(baseline))
        raise ValueError(
            f"task sets differ: baseline-only={only_base[:5]}, post-only={only_post[:5]}"
        )
    if len(baseline) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} paired tasks, found {len(baseline)}")
    tasks = sorted(baseline)
    deltas = [post[task] - baseline[task] for task in tasks]
    wins = sum(delta == 1 for delta in deltas)
    losses = sum(delta == -1 for delta in deltas)
    ci_low, ci_high = paired_bootstrap_ci(
        deltas, samples=bootstrap_samples, seed=seed
    )
    base_rate = sum(baseline.values()) / len(tasks)
    post_rate = sum(post.values()) / len(tasks)
    return {
        "paired_tasks": len(tasks),
        "baseline_resolved": int(sum(baseline.values())),
        "baseline_rate": base_rate,
        "post_grpo_resolved": int(sum(post.values())),
        "post_grpo_rate": post_rate,
        "absolute_delta": post_rate - base_rate,
        "absolute_delta_95pct_paired_bootstrap": [ci_low, ci_high],
        "post_only_wins": wins,
        "baseline_only_wins": losses,
        "both_resolved": sum(baseline[t] == post[t] == 1 for t in tasks),
        "neither_resolved": sum(baseline[t] == post[t] == 0 for t in tasks),
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(wins, losses),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post-grpo", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare(
        load_trials(args.baseline),
        load_trials(args.post_grpo),
        expected_tasks=args.expected_tasks,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
