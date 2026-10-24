"""Select fresh heldout-disjoint tasks outside the old clean matched MSA2 pair."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def used_tasks(path):
    before = path.stat()
    tasks, digest, rows = set(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            tasks.add(row["task"])
            rows += 1
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed: {path}")
    if rows != 1877 or len(tasks) != 1343:
        raise ValueError(f"Unexpected old matched selection: {path}, {rows}, {len(tasks)}")
    return tasks, dict(path=str(path), rows=rows, tasks=len(tasks), sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--old-selection", required=True, type=Path)
    args = parser.parse_args()
    if (args.root / "tasks.txt").exists() or (args.root / "manifest.json").exists():
        raise ValueError("Refusing to replace a selection")
    pool = args.tree.joinpath("TASKS.txt").read_text().splitlines()
    heldout = set(args.heldout.read_text().splitlines())
    assert len(pool) == len(set(pool)) == 2613 and not set(pool) & heldout
    first, first_source = used_tasks(args.old_selection / "msaqwen_rows.jsonl")
    second, second_source = used_tasks(args.old_selection / "msarel_rows.jsonl")
    assert first == second and first <= set(pool)
    unused = sorted(set(pool) - first)
    assert len(unused) == 1270
    metadata = json.loads((args.tree / "router_tasks.json").read_text())
    by_id = {row["task_id"]: row for row in metadata}
    eligible = [task for task in unused if 0 < by_id[task]["agent_timeout_sec"] <= 1800]
    assert len(eligible) >= 64
    selected = random.Random(20261023).sample(eligible, 64)
    for task in selected:
        assert (args.tree / task / "task.toml").is_file()
        assert 0 < by_id[task]["agent_timeout_sec"] <= 1800
    payload = ("\n".join(selected) + "\n").encode()
    (args.root / "tasks.txt").write_bytes(payload)
    manifest = dict(
        pool_tasks=2613, previously_matched_tasks=1343, unused_candidates=1270,
        pilot_budget_eligible_candidates=len(eligible), pilot_max_agent_budget_sec=1800,
        selected_tasks=64, selected_task_ids=selected, seed=20261023,
        heldout_overlap=0, previous_matched_overlap=0, attempts_per_arm=1,
        arms=["control", "relay_repair"], scheduled_slots=128,
        tasks_sha256=hashlib.sha256(payload).hexdigest(), old_sources=[first_source, second_source],
        student="grug-datakit-sft-20260921", teacher="Qwen3.8-27B", harness="MSA2 2.4.6 tool mode",
        protocol="Guided MSA relay with reasoning restoration; not strict no-refeed. "
                 "Explicit 65,536 context and 16,384 output with context fit; one attempt avoids "
                 "same-task overlap in the router tokenize lookup. Future SFT may use fixed NR windows.",
        nodes=3, student_nodes=1, concurrency_per_arm=8, serve_wall="08:00:00", booked_node_hours=24,
        snapshot_policy="Only pre-existing named snapshots; no creation, deletion, or build fallback.",
    )
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("FRESH_MSA_SELECTION_OK", json.dumps({k: manifest[k] for k in ["selected_tasks", "scheduled_slots", "heldout_overlap", "previous_matched_overlap"]}), flush=True)


if __name__ == "__main__":
    main()
