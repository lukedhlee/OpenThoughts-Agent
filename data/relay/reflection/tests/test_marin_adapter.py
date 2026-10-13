import json
import subprocess
from pathlib import Path

import pytest

from data.relay.reflection.marin_adapter import inspect_rows, run


def rows(path, value=None):
    path.write_text(
        json.dumps(
            value
            or {
                "sid": "x",
                "ids": [4, 5, 6],
                "loss": [0, 1, 1],
                "prompt_tokens": 1,
                "trajectory_author": "student",
                "reflection_author": "student",
                "policy_version": 0,
            }
        )
        + "\n"
    )
    return str(path)


def request(tmp_path):
    return {
        "arm": "roft",
        "iteration": 1,
        "rows": rows(tmp_path / "rows.jsonl"),
        "model_before": str(tmp_path / "base"),
        "optimizer_before": None,
        "output_dir": str(tmp_path / "output"),
        "config": {
            "training": {
                "sequence_length": 8,
                "loss_reduction": "global_token_mean",
                "gradient_scale": 4,
            },
            "backend": {"marin_root": str(tmp_path), "python": "python"},
        },
    }


def test_shift_contract_and_truncation(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows(path)
    assert inspect_rows(str(path), 8)[1:] == (1, 2)
    with pytest.raises(ValueError, match="truncation"):
        inspect_rows(str(path), 2)
    rows(path, {"sid": "x", "ids": [4, 5], "loss": [1, 0]})
    with pytest.raises(ValueError, match="first token"):
        inspect_rows(str(path), 8)


def test_requires_full_resume_and_unique_rows(tmp_path):
    req = request(tmp_path)
    req["iteration"] = 2
    with pytest.raises(ValueError, match="subsequent"):
        run(req, str(tmp_path / "response.json"))
    path = Path(req["rows"])
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="unique sid"):
        inspect_rows(str(path), 8)


def backend_response(monkeypatch, req, steps=1):
    def fake_run(command, *, cwd, check):
        assert check and cwd == req["config"]["backend"]["marin_root"]
        source = json.loads(Path(command[-3]).read_text())
        output = Path(req["output_dir"])
        for name, marker in (("hf", "config.json"), ("native", "metadata.json")):
            (output / name).mkdir()
            (output / name / marker).write_text("{}")
        (output / "hf" / "weights.safetensors").write_bytes(b"fake")
        (output / "native" / "online_state.json").write_text("{}")
        result = {
            "optimizer_steps_before": 0,
            "optimizer_steps_after": steps,
            "weights_version": 1,
            "source_digest": source["source_digest"],
            "supervised_tokens": 2,
            "loss": 1.2,
            "model_after": str(output / "hf"),
            "optimizer_after": str(output / "native"),
        }
        Path(command[-1]).write_text(json.dumps(result))

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_validated_atomic_response_and_no_replay(tmp_path, monkeypatch):
    req = request(tmp_path)
    backend_response(monkeypatch, req)
    destination = tmp_path / "response.json"
    assert run(req, str(destination))["optimizer_steps_after"] == 1
    assert json.loads(destination.read_text())["supervised_tokens"] == 2
    with pytest.raises(ValueError, match="replay"):
        run(req, str(tmp_path / "other.json"))


def test_rejects_microbatch_optimizer_steps(tmp_path, monkeypatch):
    req = request(tmp_path)
    backend_response(monkeypatch, req, steps=256)
    destination = tmp_path / "response.json"
    with pytest.raises(ValueError, match="optimizer_steps_after"):
        run(req, str(destination))
    assert not destination.exists()


def test_refuses_action_loss_in_reflection_arm(tmp_path):
    req = request(tmp_path)
    p = Path(req["rows"])
    row = json.loads(p.read_text())
    row["prompt_tokens"] = 2
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="only reflection"):
        run(req, str(tmp_path / "response.json"))
