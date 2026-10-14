"""Unassisted MSA2 source attempts via Harbor, inside operator-provided resources."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

from .core import validate_sources, write_json, write_jsonl
from .harbor import import_trials


def infra_retry(config):
    """Harbor retries for infrastructure exceptions only, exactly the evaluation policy's list, when
    runtime.collection.infra_retries is true; otherwise none. A retried trial reruns the same task on the same policy
    version; model failures (wrong answer, context overflow, agent timeout) are never in that list."""
    if not config["runtime"]["collection"].get("infra_retries"):
        return {"max_retries": 0}
    import yaml

    policy = Path(__file__).parents[3] / config["evaluation"]["policy"]
    retry = yaml.safe_load(policy.read_text())["retry"]
    # Also seen in CalibForge collection under load (2026-10-13): a Daytona sandbox that does not start within 300 s
    # (16/64 trials of ROFT update 1) and a gateway error from the Daytona API. Both precede any model turn.
    extra = [e for e in COLLECTION_INFRA if e not in retry["include_exceptions"]]
    return dict(retry, include_exceptions=retry["include_exceptions"] + extra)


COLLECTION_INFRA = ["EnvironmentStartTimeoutError", "DaytonaBadGatewayError"]


def make_job(request):
    config = request["config"]
    runtime = config["runtime"]
    version = request["weights_version"]
    heldout = set(Path(config["data"]["heldout_tasks"]).read_text().splitlines())
    tasks = [
        x.strip()
        for x in Path(config["data"]["train_tasks"]).read_text().splitlines()
        if x.strip()
    ]
    if len(set(tasks)) != len(tasks) or heldout.intersection(tasks):
        raise ValueError("train task list duplicated or overlaps heldout")
    random.Random(config["data"]["seed"]).shuffle(tasks)
    count = request["count"]
    offset = version * count
    if count > len(tasks):
        raise ValueError("batch larger than unique training pool")
    chosen = [tasks[(offset + i) % len(tasks)] for i in range(count)]
    tree = Path(config["data"]["task_tree"]).resolve()
    for name in chosen:
        if (tree / name).resolve().parent != tree or not (
            tree / name / "task.toml"
        ).is_file():
            raise ValueError("invalid task path")
    endpoint = runtime["endpoints"]["student"]
    spec = runtime["collection"]
    out = Path(request["output_dir"])
    job = {
        "job_name": f"{config['name']}-u{version:04d}-student_only",
        "jobs_dir": str(out / "harbor"),
        "n_attempts": 1,
        "n_concurrent_trials": spec["concurrency"],
        "timeout_multiplier": 1.0,
        "quiet": True,
        "release_trial_payloads_in_memory": True,
        "retry": infra_retry(config),
        "environment": {
            "type": "daytona",
            "delete": True,
            "force_build": False,
            "kwargs": {"auto_snapshot": True},
        },
        "agents": [
            {
                "name": "mini-swe-agent-host",
                "model_name": "hosted_vllm/" + endpoint["served_model"],
                "override_timeout_sec": spec["task_timeout_sec"],
                "kwargs": {
                    "config_file": "mini.yaml",
                    "model_class": "litellm",
                    "api_base": endpoint["url"],
                    "llm_call_kwargs": {
                        "tool_choice": "none",
                        "temperature": spec["temperature"],
                        "top_p": spec["top_p"],
                        "max_tokens": spec["max_tokens"],
                    },
                    "model_info": {
                        "max_input_tokens": spec["context_length"],
                        "max_output_tokens": spec["max_tokens"],
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "extra_body": {"skip_special_tokens": False},
                    "interleaved_thinking": False,
                },
            }
        ],
        "verifier": {"override_timeout_sec": 300},
        "tasks": [{"path": str(tree / t)} for t in chosen],
    }
    return job


def run(request, response):
    import yaml

    job = make_job(request)
    out = Path(request["output_dir"])
    runtime = request["config"]["runtime"]
    p = out / "harbor.yaml"
    p.write_text(yaml.safe_dump(job, sort_keys=False))
    from harbor_config.models.job.config import JobConfig

    JobConfig.model_validate(job)
    repo = Path(runtime["collection"]["harbor_root"])
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if not sha.startswith(runtime["collection"]["harbor_revision"]):
        raise ValueError("Harbor commit differs from pinned version")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with open(out / "harbor.log", "ab") as log:
        subprocess.run(
            [runtime["collection"]["harbor_bin"], "jobs", "start", "--config", str(p)],
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    root = Path(job["jobs_dir"]) / job["job_name"]
    rows, rejected = import_trials(
        [root],
        author="student",
        model=request["config"]["student"]["model"],
        revision=request["config"]["student"]["revision"],
        policy_version=request["weights_version"],
    )
    if len(rows) != request["count"]:
        raise ValueError(
            f"collector returned {len(rows)}/{request['count']} complete experiences; rejected={rejected}"
        )
    actual = {r["task"] for r in rows}
    expected = {Path(t["path"]).name for t in job["tasks"]}
    if actual != expected:
        raise ValueError("Harbor sources do not match selected task set")
    validate_sources(rows, request["config"], request["weights_version"])
    write_jsonl(out / "collected.jsonl", rows)
    write_json(
        response,
        {
            "source_jsonl": str(out / "collected.jsonl"),
            "weights_version": request["weights_version"],
            "model": request["model"],
            "rejected": rejected,
            "job_config": str(p),
        },
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    a = ap.parse_args()
    run(json.loads(Path(a.request).read_text()), a.response)


if __name__ == "__main__":
    main()
