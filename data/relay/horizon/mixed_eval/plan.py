"""Freeze corrected-task inputs and plan the two mixed-model eight-repetition evals."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

SHARDS = ("tb21", "swe_s0", "swe_s1", "tblite_s0", "tblite_s1")
TAGS = {"cfmix50": "cm50", "cfconcat": "cm100"}
HARNESS = "62e1c7b9"
PYTHON_DIND_TASKS = frozenset(
    {
        "book-portfolio-analysis",
        "image-tile-identification",
        "iot-device-registration-server",
        "prediction-model-evaluation",
        "sympy-bug-fix",
        "tsl-test-case-generation",
    }
)
JAVA_DIND_TASKS = frozenset({"okhttp-trailers-crash"})


def snapshot_name(environment: Path) -> str:
    """Match pinned Harbor's length-prefixed environment hash without importing its runtime."""
    digest = hashlib.sha256()
    for path in sorted(p for p in environment.rglob("*") if p.is_file()):
        name = str(path.relative_to(environment)).encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big") + name)
        digest.update(len(payload).to_bytes(4, "big") + payload)
    return f"harbor__{digest.hexdigest()[:12]}__snapshot"


BASE = Path("/scratch/11584/lukedhlee/experiments")
CANONICAL = BASE / "sft_eval/runs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = str(path.relative_to(root)).encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(payload).to_bytes(8, "big") + payload)
    return digest.hexdigest()


def task_path(original: Path) -> Path:
    name = original.name
    swe = BASE / "cf2-swe-repair-20261023/tasks" / name
    if original.parent.name == "swebench_verified_random100" and swe.is_dir():
        return swe
    grader = BASE / "cf2-grader-fixes-20261023/tasks" / name
    if name in {"query-optimize", "filter-js-from-html"}:
        return grader
    if name in {"todos-api", "mcmc-sampling-stan", "pytorch-model-recovery"}:
        return BASE / "cf2-remaining-repair-20261023/tasks" / name
    return original


def inventory() -> dict:
    templates, tasks, sources = {}, {}, {}
    for shard in SHARDS:
        path = CANONICAL / f"{shard}_cfv2_r1_cf2_20261022.yaml"
        cfg = yaml.safe_load(path.read_text())
        sources[str(path)] = sha(path)
        templates[shard] = cfg
        for item in cfg["tasks"]:
            original = Path(item["path"])
            if original.name == "git-repo-forensics":
                continue
            chosen = task_path(original)
            assert chosen.is_dir(), chosen
            assert sha(chosen / "instruction.md") == sha(original / "instruction.md"), (
                chosen
            )
            assert sha(chosen / "task.toml") == sha(original / "task.toml"), chosen
            native = original.name in PYTHON_DIND_TASKS | JAVA_DIND_TASKS
            if not native and (chosen / "environment/docker-compose.yaml").exists():
                raise ValueError(f"Unreviewed compose task: {chosen}")
            selected_snapshot = (
                "daytona-large"
                if original.name in JAVA_DIND_TASKS
                else "harbor__313e69c036ad__snapshot"
                if native
                else snapshot_name(chosen / "environment")
            )
            env_sha = tree_sha(original / "environment")
            assert tree_sha(chosen / "environment") == env_sha, chosen
            tasks[str(original)] = dict(
                provisioning="java_dind"
                if original.name in JAVA_DIND_TASKS
                else "python_dind"
                if native
                else "direct",
                existing_snapshot=selected_snapshot,
                original=str(original),
                path=str(chosen),
                environment_sha256=env_sha,
                instruction_sha256=sha(chosen / "instruction.md"),
                task_toml_sha256=sha(chosen / "task.toml"),
                tests_sha256=tree_sha(chosen / "tests"),
            )
    return dict(
        templates=templates,
        tasks=tasks,
        frozen_source_hashes=sources,
        canonical_tasks_unchanged=True,
        repaired_tasks="Only previously deployed verified copies",
    )


def build(data: dict, root: Path, harbor: Path) -> tuple[list[dict], dict]:
    from harbor_config.models.job.config import JobConfig

    groups = []
    seen = set()
    for arm, tag in TAGS.items():
        for rep in range(1, 9):
            group = dict(arm=arm, tag=tag, rep=rep, key=f"{tag}_r{rep}", drivers=[])
            for shard_index, shard in enumerate(SHARDS):
                cfg = copy.deepcopy(data["templates"][shard])
                run = f"{shard}_{tag}_r{rep}_cfmix_20261023"
                cfg["job_name"], cfg["jobs_dir"] = run, str(root / "jobs")
                cfg.pop("datasets", None)
                cfg["tasks"] = [
                    dict(item, path=data["tasks"][item["path"]]["path"])
                    for item in cfg["tasks"]
                    if Path(item["path"]).name != "git-repo-forensics"
                ]
                assert cfg["n_attempts"] == 1 and cfg["n_concurrent_trials"] == 16
                agent = cfg["agents"][0]
                assert (
                    agent["name"] == "terminus-2"
                    and agent["override_timeout_sec"] == 1800
                )
                assert agent["override_setup_timeout_sec"] == 360
                assert agent["kwargs"]["model_info"]["max_input_tokens"] == 65536
                assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384
                assert cfg["verifier"]["override_timeout_sec"] == 300
                agent["kwargs"]["api_base"] = "ENDPOINT"
                agent["kwargs"]["batch_script_stdin"] = True
                cfg["verifier"]["verify_after_terminal_error"] = True
                cfg["verifier"]["import_path"] = (
                    "data.relay.horizon.mixed_eval.runtime:MixedVerifier"
                )
                cfg["environment"].pop("type", None)
                cfg["environment"]["import_path"] = (
                    "data.relay.horizon.mixed_eval.runtime:ExistingSnapshotDaytona"
                )
                cfg["environment"]["kwargs"] = dict(harbor_root=str(harbor))
                JobConfig.model_validate(cfg)
                for item in cfg["tasks"]:
                    slot = (tag, rep, shard, Path(item["path"]).name)
                    assert slot not in seen, slot
                    seen.add(slot)
                group["drivers"].append(
                    dict(
                        run=run,
                        shard=shard,
                        node_index=shard_index,
                        bridge_port=21000 + len(groups) * len(SHARDS) + shard_index,
                        config=cfg,
                    )
                )
            groups.append(group)
    assert len(seen) == 4592, len(seen)
    assert len(groups) == 16 and sum(len(g["drivers"]) for g in groups) == 80
    report = dict(
        valid_slots=len(seen),
        models=list(TAGS),
        repetitions=8,
        tasks_per_rep=287,
        families=dict(tb21=88, swe=100, tblite=99),
        git_fixture_invalid_excluded=16,
        harbor_commit=HARNESS,
        nodes=80,
        serve_jobs=16,
        maximum_trial_concurrency=1280,
        interpretation="Compare against sealed v2 as historical repaired baselines; baseline repairs were "
        "applied only to unknown slots, so this is not an identical first-attempt harness comparison.",
        frozen_source_hashes=data["frozen_source_hashes"],
    )
    return groups, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inventory", "plan"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--harbor", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.mode == "inventory":
        print(json.dumps(inventory(), indent=2))
        return
    if args.inventory is None or args.root is None or args.harbor is None:
        parser.error("plan needs --inventory, --root and --harbor")
    data = json.loads(args.inventory.read_text())
    groups, report = build(data, args.root, args.harbor)
    if args.write:
        args.root.mkdir(parents=True, exist_ok=True)
        target = args.root / "plan.json"
        if target.exists():
            assert json.loads(target.read_text()) == dict(
                groups=groups, report=report
            ), "Plan drift"
        else:
            target.write_text(
                json.dumps(dict(groups=groups, report=report), indent=2) + "\n"
            )
    print(json.dumps(report, indent=2))
    print("CF_MIXED_EVAL_PLAN_OK: no jobs submitted")


if __name__ == "__main__":
    main()
