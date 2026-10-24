"""Score settled mixed-model evals without treating infrastructure unknowns as failures."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from data.relay.pilot.readout import read_outcome
from data.relay.horizon.mixed_eval.plan import sha


HISTORICAL_REPORT_SHA = (
    "e3a3832d65f131fec586b6f52d6bc84e6811830030408c16fcc6506b8e56f707"
)
HISTORICAL_SOURCE_SHA = (
    "85d568a454c008aa24dadb3b11ad0b331119bcbe0d8ef468d86c1080e32e0d9e"
)


def validated_historical(path: Path) -> dict:
    assert sha(path) == HISTORICAL_REPORT_SHA, "Historical report file seal mismatch"
    data = json.loads(path.read_text())
    assert data["provenance"]["source_snapshot_sha256"] == HISTORICAL_SOURCE_SHA
    return data


def passk(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError((n, c, k))
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(root: Path, historical: Path | None = None) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    scores, coverage = {}, defaultdict(Counter)
    unknowns = []
    for group in plan["groups"]:
        tag, rep = group["tag"], group["rep"]
        for driver in group["drivers"]:
            run = driver["run"]
            family = (
                "swe"
                if driver["shard"].startswith("swe")
                else "tblite"
                if driver["shard"].startswith("tblite")
                else "tb21"
            )
            expected = {Path(item["path"]).name for item in driver["config"]["tasks"]}
            outcomes = defaultdict(list)
            for path in sorted((root / "jobs" / run).glob("*/result.json")):
                try:
                    task, reward, error = read_outcome(path)
                except (OSError, ValueError, TypeError) as parse_error:
                    task, reward, error = (
                        path.parent.name.rsplit("__", 1)[0],
                        None,
                        f"UnreadableResult:{type(parse_error).__name__}",
                    )
                if reward is not None and (
                    not isinstance(reward, (int, float)) or not math.isfinite(reward)
                ):
                    reward, error = None, "InvalidNumericReward"
                if task not in expected:
                    raise ValueError(f"Unexpected task in {run}: {task}")
                outcomes[task].append(
                    dict(path=str(path), raw_reward=reward, error=error)
                )
            for task in sorted(expected):
                key = (tag, rep, family, task)
                if key in scores:
                    raise ValueError(f"Duplicate planned slot: {key}")
                candidates = outcomes.get(task, [])
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous duplicate result files: {key}")
                result = (
                    candidates[0]
                    if candidates
                    else dict(path=None, raw_reward=None, error="MissingResult")
                )
                scores[key] = result
                count = coverage[tag, family]
                count["scheduled"] += 1
                if result["raw_reward"] is None:
                    count["unknown"] += 1
                    unknowns.append(
                        dict(tag=tag, rep=rep, family=family, task=task, **result)
                    )
                else:
                    count["scored"] += 1
                    count["full_solved"] += result["raw_reward"] >= 1
    comparisons = []
    for family in ["tb21", "swe", "tblite"]:
        tasks = sorted({key[3] for key in scores if key[2] == family})
        panel = [
            task
            for task in tasks
            if all(
                scores[tag, rep, family, task]["raw_reward"] is not None
                for tag in ["cm50", "cm100"]
                for rep in range(1, 9)
            )
        ]
        for k in [1, 3, 4]:
            values = {}
            for tag in ["cm50", "cm100"]:
                per_task = [
                    passk(
                        8,
                        sum(
                            scores[tag, rep, family, task]["raw_reward"] >= 1
                            for rep in range(1, 9)
                        ),
                        k,
                    )
                    for task in panel
                ]
                values[tag] = sum(per_task) / len(per_task) if per_task else None
            comparisons.append(
                dict(
                    family=family,
                    k=k,
                    attempts=8,
                    paired_tasks=len(panel),
                    valid_tasks=len(tasks),
                    panel=panel,
                    mix50=values["cm50"],
                    concat=values["cm100"],
                    concat_minus_mix50=values["cm100"] - values["cm50"]
                    if panel
                    else None,
                )
            )
    result = dict(
        method="Eight numeric repetitions required from BOTH mixed models per task; paired complete-case "
        "panel per benchmark, shared across k=1,3,4. Binary full solve is raw reward>=1; partial "
        "rewards retained in raw slots. Unknowns excluded, never zero. pass@k=1-C(8-c,k)/C(8,k).",
        coverage=[
            dict(tag=tag, family=family, **dict(values))
            for (tag, family), values in sorted(coverage.items())
        ],
        comparisons=comparisons,
        unknowns=unknowns,
        slots=[
            dict(tag=tag, rep=rep, family=family, task=task, **value)
            for (tag, rep, family, task), value in sorted(scores.items())
        ],
    )
    if historical is not None and historical.is_file():
        source = validated_historical(historical)
        result["historical_baselines"] = dict(
            source=str(historical),
            sha256=sha(historical),
            source_snapshot_sha256=HISTORICAL_SOURCE_SHA,
            caveat="Sealed repaired-v2 historical Vanilla/Relay results. Repairs were applied only to unknown slots; "
            "new models use repairs from the first attempt. Panels and attempt counts can differ; no "
            "unpaired significance claim or causal mixture advantage is inferred.",
            comparisons=[row for row in source["comparisons"] if row["epoch"] == 3],
        )
    return result


def markdown(result: dict) -> str:
    lines = [
        "# Mixed-model SFT evaluation",
        "",
        result["method"],
        "",
        "| Benchmark | k | Paired tasks | 50:50 | 100+100 | Difference (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparisons"]:

        def percent(value):
            return "unknown" if value is None else f"{100 * value:.1f}%"

        delta = row["concat_minus_mix50"]
        difference = "unknown" if delta is None else f"{100 * delta:+.1f}"
        lines.append(
            f"| {row['family']} | {row['k']} | {row['paired_tasks']}/{row['valid_tasks']} | "
            f"{percent(row['mix50'])} | {percent(row['concat'])} | {difference} |"
        )
    lines += [
        "",
        "| Model | Benchmark | Scored / scheduled | Unknown |",
        "|---|---|---:|---:|",
    ]
    for row in result["coverage"]:
        lines.append(
            f"| {row['tag']} | {row['family']} | {row.get('scored', 0)}/{row['scheduled']} | {row.get('unknown', 0)} |"
        )
    historical = result.get("historical_baselines")
    if historical:
        lines += [
            "",
            "## Historical Vanilla / Relay reference",
            "",
            historical["caveat"],
            "",
            "| Benchmark | k | Attempts | Tasks | Vanilla | Relay |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in historical["comparisons"]:
            data = row["repaired"]
            lines.append(
                f"| {row['family']} | {row['k']} | {row['n_attempts']} | {data['tasks']} | "
                f"{100 * data['vanilla']:.1f}% | {100 * data['relay']:.1f}% |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--historical", type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.historical)
    (args.root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.root / "results.md").write_text(markdown(result))
    print("CF_MIXED_READOUT_READY", args.root / "results.md")


if __name__ == "__main__":
    main()
