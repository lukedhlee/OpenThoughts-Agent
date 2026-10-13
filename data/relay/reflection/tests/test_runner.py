import copy
import json
from pathlib import Path

import pytest

from data.relay.reflection import runner
from data.relay.reflection.core import load_config, write_json, write_jsonl

from .test_core import source


def settings(tmp_path):
    heldout = tmp_path / "heldout"
    heldout.write_text("test\n")
    return {
        "schema_version": 1,
        "name": "test",
        "arm": "roft",
        "student": {
            "model": "grug",
            "revision": "pin",
            "tokenizer_dir": "unused",
            "weights_path": "base",
        },
        "teacher": {
            "model": "qwen",
            "revision": "pin",
            "tokenizer_dir": "unused",
            "weights_path": "teacher",
        },
        "data": {"outcomes": "all", "heldout_tasks": str(heldout), "seed": 0},
        "reflection": {
            "per_trajectory": 2,
            "max_tokens": 8192,
            "temperature": 0.9,
            "top_p": 1.0,
        },
        "training": {
            "lr": 1e-5,
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-8,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "gradient_scale": 4.0,
            "loss_reduction": "global_token_mean",
            "objective": "reflection_only",
            "updates": 2,
            "source_attempts": 1,
            "sequence_length": 65536,
            "microbatch_size": 16,
            "passes_per_batch": 1,
        },
        "runtime": {
            "output_dir": str(tmp_path),
            "timeout_sec": 3,
            "commands": {},
            "endpoints": {"student": {"url": "unused", "served_model": "grug"}},
        },
        "evaluation": {"reflection_in_context": False},
        "backend": {"base_model": "base"},
    }


def client(*args):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "A substantive reflection."},
            }
        ]
    }


def renderer(records, *args):
    return [
        {
            "sid": r["sid"],
            "source_id": r["source_id"],
            "task": r["task"],
            "ids": [0] + [1] * 120,
            "loss": [0] + [1] * 120,
            "n_tokens": 121,
            "trained_tokens": 120,
        }
        for r in records
    ], {}


def fake_adapters(monkeypatch, events, fail_once=None):
    def adapter(config, name, req, d):
        response = d / (name + ".response.json")
        if response.exists():
            return json.loads(response.read_text())
        events.append((name, req))
        if name == "refresh":
            result = {"model": req["model"], "weights_version": req["weights_version"]}
        elif name == "collect":
            r = source()
            r["policy_version"] = req["weights_version"]
            r["id"] = "source" + str(r["policy_version"])
            write_jsonl(d / "collected.jsonl", [r])
            result = {"source_jsonl": str(d / "collected.jsonl")}
        else:
            i = req["iteration"]
            root = Path(req["output_dir"])
            (root / "hf").mkdir(parents=True)
            (root / "native").mkdir()
            (root / "hf/config.json").write_text("{}")
            (root / "native/metadata.json").write_text("{}")
            result = {
                "model_after": str(root / "hf"),
                "optimizer_after": str(root / "native"),
                "weights_version": i,
                "optimizer_steps_before": i - 1,
                "optimizer_steps_after": i,
                "source_digest": req["source_digest"],
                "supervised_tokens": req["supervised_tokens"],
                "loss": 1.2,
            }
        write_json(response, result)
        if fail_once is not None and name == "train" and not fail_once[0]:
            fail_once[0] = True
            raise RuntimeError("interrupted after verified update response")
        return result

    monkeypatch.setattr(runner, "adapter", adapter)


def test_online_freshness_optimizer_and_no_replay(tmp_path, monkeypatch):
    c = settings(tmp_path)
    events = []
    failure = [False]
    fake_adapters(monkeypatch, events, failure)
    with pytest.raises(RuntimeError):
        runner.run(c, client=client, renderer=renderer)
    assert json.loads((tmp_path / "test/state.json").read_text())["iteration"] == 0
    result = runner.run(c, client=client, renderer=renderer)
    assert result["completed"] and result["iteration"] == 2
    training = [r for name, r in events if name == "train"]
    assert len(training) == 2
    assert training[0]["optimizer_before"] is None
    assert training[1]["optimizer_before"] == str(
        tmp_path / "test/update-0001/training/native"
    )
    assert training[1]["model_before"] == str(tmp_path / "test/update-0001/training/hf")
    assert training[0]["source_digest"] != training[1]["source_digest"]
    assert [r["weights_version"] for n, r in events if n == "refresh"] == [0, 1]
    runner.run(c, client=client, renderer=renderer)
    assert len(training) == 2
    changed = copy.deepcopy(c)
    changed["reflection"]["temperature"] = 0.8
    with pytest.raises(ValueError, match="config changed"):
        runner.run(changed, client=client, renderer=renderer)


def test_offline_source_immutability(tmp_path, monkeypatch):
    c = settings(tmp_path)
    c["arm"] = "student_teacher"
    c["runtime"]["endpoints"]["teacher"] = {"url": "unused", "served_model": "qwen"}
    c["training"]["updates"] = 1
    c["data"]["source_jsonl"] = str(tmp_path / "offline.jsonl")
    write_jsonl(c["data"]["source_jsonl"], [source()])
    events = []
    fake_adapters(monkeypatch, events)
    runner.run(c, client=client, renderer=renderer)
    assert [n for n, _ in events] == ["train"]
    r = source()
    r["messages"].append({"role": "tool", "content": "changed"})
    write_jsonl(c["data"]["source_jsonl"], [r])
    with pytest.raises(ValueError, match="corpus changed"):
        runner.run(c, client=client, renderer=renderer)


def test_config_refuses_pass_only_roft(tmp_path):
    c = settings(tmp_path)
    p = tmp_path / "config.json"
    write_json(p, c)
    assert load_config(p)["arm"] == "roft"
    c["data"]["outcomes"] = "pass"
    write_json(p, c)
    with pytest.raises(ValueError, match="both outcomes"):
        load_config(p)


def test_inflight_adapter_is_not_replayed(tmp_path):
    c = settings(tmp_path)
    c["runtime"]["commands"]["train"] = ["false"]
    write_json(tmp_path / "train.inflight", {"request_digest": "previous"})
    with pytest.raises(ValueError, match="interrupted"):
        runner.adapter(c, "train", {"x": 1}, tmp_path)
