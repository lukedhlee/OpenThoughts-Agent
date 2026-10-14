"""Durable online/offline update controller. Running submits nothing implicitly.

Collection and serving are explicit argv adapters; training runs in an existing allocation.
Every adapter consumes --request/--response JSON, making handoff independent of shell state.
"""

from __future__ import annotations

import fcntl
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from .core import (
    digest,
    file_digest,
    read_jsonl,
    validate_sources,
    write_json,
    write_jsonl,
)
from .generation import generate_reflections
from .render import render_records


def adapter(config, name, request, directory):
    command = config["runtime"]["commands"].get(name)
    if (
        not command
        or not isinstance(command, list)
        or not all(isinstance(x, str) for x in command)
    ):
        raise ValueError(
            f"configure runtime.commands.{name} as an argv list before execution"
        )
    req = directory / (name + ".request.json")
    res = directory / (name + ".response.json")
    fingerprint = digest(request)
    if req.exists() and json.loads(req.read_text()) != request:
        raise ValueError("adapter request changed on resume")
    if res.exists():
        return json.loads(res.read_text())
    # A crashed in-flight external update is ambiguous. Never replay automatically.
    pending = directory / (name + ".inflight")
    if pending.exists():
        raise ValueError(
            f"{name} interrupted; reconcile artifacts and write response before resuming (no automatic replay)"
        )
    write_json(req, request)
    write_json(pending, {"request_digest": fingerprint})
    argv = [
        x.replace("{request}", str(req)).replace("{response}", str(res))
        for x in command
    ]
    with open(directory / (name + ".log"), "ab") as log:
        subprocess.run(
            argv,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config["runtime"]["adapter_timeout_sec"],
        )
    response = json.loads(res.read_text())
    pending.unlink()
    return response


def check_train_response(result, request):
    expected = {
        "weights_version": request["iteration"],
        "optimizer_steps_before": request["iteration"] - 1,
        "optimizer_steps_after": request["iteration"],
        "source_digest": request["source_digest"],
        "supervised_tokens": request["supervised_tokens"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"train adapter violated {key}")
    if not isinstance(result.get("loss"), (float, int)) or not math.isfinite(
        result["loss"]
    ):
        raise ValueError("nonfinite train loss")
    for key, marker in [
        ("model_after", "config.json"),
        ("optimizer_after", "metadata.json"),
    ]:
        if not (Path(result[key]) / marker).is_file():
            raise ValueError(f"missing {key} artifact")


def run(config, *, client=None, renderer=None, deadline=None):
    if config["arm"] == "vanilla":
        raise ValueError(
            "vanilla uses the existing SFT chain; use plan/validate-vanilla, not reflection run"
        )
    root = Path(config["runtime"]["output_dir"]) / config["name"]
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "owner.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run_locked(config, root, client, renderer, deadline)
        except Exception as error:
            path = root / "state.json"
            if path.exists():
                state = json.loads(path.read_text())
                if state["config_sha256"] == digest(config) and not state["completed"]:
                    state.update(
                        failure_phase=state["phase"],
                        phase="failed",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                    write_json(path, state)
            raise


def _run_locked(config, root, client, renderer, deadline=None):
    sha = digest(config)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config_sha256"] != sha:
            raise ValueError(
                "config changed; use a new experiment name/output directory"
            )
    else:
        state = {
            "config_sha256": sha,
            "iteration": 0,
            "model": config["backend"]["base_model"],
            "optimizer": None,
            "short_batches": 0,
            "phase": "ready",
            "completed": False,
        }
        write_json(root / "config.resolved.json", config)
        write_json(state_path, state)
    offline = None
    if config["arm"] != "roft":
        path = config["data"]["source_jsonl"]
        source_sha = file_digest(path)
        if state.get("offline_sha256", source_sha) != source_sha:
            raise ValueError("offline corpus changed on resume")
        state["offline_sha256"] = source_sha
        offline = read_jsonl(path)
        validate_sources(offline, config)
        random.Random(config["data"]["seed"]).shuffle(offline)
        max_updates = math.ceil(len(offline) / config["training"]["source_attempts"])
        if config["training"]["updates"] > max_updates:
            raise ValueError(
                "offline updates would replay source attempts; reduce updates or supply more sources"
            )
    while state["iteration"] < config["training"]["updates"]:
        # A deadline (epoch seconds) or a PAUSE file in the run dir stops only between committed updates, never inside an adapter, so an allocation
        # that ends leaves no ambiguous in-flight update; rerunning the same config resumes here.
        if (deadline is not None and time.time() >= deadline) or (root / "PAUSE").exists():
            state["phase"] = "paused_deadline"
            write_json(state_path, state)
            return state
        i = state["iteration"] + 1
        directory = root / f"update-{i:04d}"
        directory.mkdir(exist_ok=True)
        state["phase"] = "collecting" if config["arm"] == "roft" else "loading"
        write_json(state_path, state)
        if config["arm"] == "roft":
            served = adapter(
                config,
                "refresh",
                {
                    "model": state["model"],
                    "weights_version": i - 1,
                    "config": config,
                    "output_dir": str(directory),
                },
                directory,
            )
            if (
                served.get("model") != state["model"]
                or served.get("weights_version") != i - 1
            ):
                raise ValueError("server did not confirm current weights")
            collected = adapter(
                config,
                "collect",
                {
                    "model": state["model"],
                    "weights_version": i - 1,
                    "count": config["training"]["source_attempts"],
                    "config": config,
                    "output_dir": str(directory),
                },
                directory,
            )
            sources = read_jsonl(collected["source_jsonl"])
            if len(sources) != config["training"]["source_attempts"]:
                raise ValueError(
                    "online collector must return exactly source_attempts, not silently skip failures"
                )
            validate_sources(sources, config, i - 1)
        else:
            n = config["training"]["source_attempts"]
            sources = offline[(i - 1) * n : i * n]
            # Offline student reflections must come from the pinned base, not evolving learner weights.
        write_jsonl(directory / "sources.jsonl", sources)
        state["phase"] = "reflecting"
        write_json(state_path, state)
        kwargs = {"client": client} if client is not None else {}
        if config["arm"] == "roft":
            kwargs["expected_model"] = state["model"]
        records, rejected = generate_reflections(
            sources,
            config,
            directory / "generations",
            version=i - 1 if config["arm"] == "roft" else 0,
            **kwargs,
        )
        state["phase"] = "rendering"
        write_json(state_path, state)
        render = renderer or render_records
        rows, render_bad = render(
            records,
            config["student"]["tokenizer_dir"],
            config["training"]["sequence_length"],
        )
        write_json(
            directory / "generation_census.json",
            {
                "sources": len(sources),
                "requested": len(sources) * config["reflection"]["per_trajectory"],
                "accepted_generations": len(records),
                "accepted_rows": len(rows),
                "generation_rejections": rejected,
                "render_rejections": render_bad,
            },
        )
        if not rows:
            raise ValueError("no reflection targets; no action-token fallback allowed")
        lengths = [r["trained_tokens"] for r in rows]
        degenerate = (
            statistics.median(lengths) < 100
            or sum(n <= 10 for n in lengths) / len(lengths) > 0.1
        )
        short = state["short_batches"] + 1 if degenerate else 0
        if short >= 2:
            state["phase"] = "stopped_degenerate"
            write_json(state_path, state)
            raise ValueError("two consecutive degenerate reflection batches")
        write_jsonl(directory / "rows.jsonl", rows)
        req = {
            "arm": config["arm"],
            "iteration": i,
            "rows": str(directory / "rows.jsonl"),
            "model_before": state["model"],
            "optimizer_before": state["optimizer"],
            "output_dir": str(directory / "training"),
            "config": config,
            "source_digest": file_digest(directory / "rows.jsonl"),
            "supervised_tokens": sum(lengths),
        }
        state["phase"] = "training"
        write_json(state_path, state)
        trained = adapter(config, "train", req, directory)
        check_train_response(trained, req)
        # Persist commit only after all artifacts and the single update are verified.
        state.update(
            iteration=i,
            model=trained["model_after"],
            optimizer=trained["optimizer_after"],
            short_batches=short,
            phase="ready",
        )
        write_json(state_path, state)
    state.update(completed=True, phase="complete")
    write_json(state_path, state)
    return state
