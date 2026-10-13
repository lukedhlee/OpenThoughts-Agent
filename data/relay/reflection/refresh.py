"""Refresh operator-owned serving and require a checkpoint-specific model ID.

The configured command must stop/reload/start only this experiment's own server.
There is no implicit Slurm allocation, process killing, or public tunnel creation.
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from .core import write_json


def run(request, response):
    config = request["config"]
    command = config["runtime"].get("refresh_command")
    if not command:
        raise ValueError(
            "runtime.refresh_command must be configured for owned serving before ROFT launch"
        )
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("refresh_command must be argv")
    argv = [
        x.replace("{model}", request["model"]).replace(
            "{version}", str(request["weights_version"])
        )
        for x in command
    ]
    subprocess.run(argv, check=True)
    endpoint = config["runtime"]["endpoints"]["student"]
    with urllib.request.urlopen(
        endpoint["url"].rstrip("/") + "/models",
        timeout=config["runtime"]["timeout_sec"],
    ) as f:
        models = json.load(f)
    # vLLM exposes each served model's root. Never accept an ID-only stale-weight assertion.
    matching = [m for m in models["data"] if m["id"] == endpoint["served_model"]]
    if (
        len(matching) != 1
        or Path(matching[0].get("root", "")).resolve()
        != Path(request["model"]).resolve()
    ):
        raise ValueError(
            "served /models root does not match requested exported checkpoint"
        )
    write_json(
        response,
        {
            "model": request["model"],
            "weights_version": request["weights_version"],
            "server_models": models,
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
