from pathlib import Path

import pytest

from data.relay.reflection.cli import plan
from data.relay.reflection.collect import make_job
from data.relay.reflection.core import load_config

from .test_runner import settings


def test_five_checked_configs():
    root = Path(__file__).parents[1] / "configs"
    configs = [load_config(p) for p in sorted(root.glob("*.json"))]
    assert {c["arm"] for c in configs} == {
        "vanilla",
        "roft",
        "teacher_teacher",
        "teacher_student",
        "student_teacher",
    }
    assert len({plan(c)["output_dir"] for c in configs}) == 5
    assert all(c["evaluation"]["reflection_in_context"] is False for c in configs)
    roft = next(c for c in configs if c["arm"] == "roft")
    assert not plan(roft)["unconfigured_refresh_command"]
    # The ROFT smoke differs from the experiment only in its name and counts.
    smoke = load_config(root / "smoke" / "roft.json")
    for section, keys in [("training", ("updates", "source_attempts", "schedule_steps")),
                          ("reflection", ("per_trajectory", "concurrency"))]:
        for key in keys:
            smoke[section].pop(key), roft[section].pop(key)
    smoke["runtime"]["collection"].pop("concurrency"), roft["runtime"]["collection"].pop("concurrency")
    assert {**smoke, "name": None} == {**roft, "name": None}


def test_collector_selects_fresh_unassisted_tasks_and_actual_response_cap(tmp_path):
    c = settings(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    tasks = []
    for i in range(4):
        task = f"train{i}"
        tasks.append(task)
        (tree / task).mkdir()
        (tree / task / "task.toml").write_text("[agent]\ntimeout_sec=1800\n")
    train = tmp_path / "train"
    train.write_text("\n".join(tasks) + "\n")
    c["data"].update(task_tree=str(tree), train_tasks=str(train))
    c["runtime"]["collection"] = {
        "concurrency": 16,
        "task_timeout_sec": 1800,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "context_length": 65536,
    }
    req = {"config": c, "count": 2, "weights_version": 0, "output_dir": str(tmp_path)}
    a = make_job(req)
    req["weights_version"] = 1
    b = make_job(req)
    assert {t["path"] for t in a["tasks"]}.isdisjoint({t["path"] for t in b["tasks"]})
    agent = a["agents"][0]
    assert agent["kwargs"]["llm_call_kwargs"]["max_tokens"] == 16384
    assert agent["kwargs"]["interleaved_thinking"] is False
    assert "relay" not in agent["model_name"]
    train.write_text("test\n")
    with pytest.raises(ValueError, match="overlaps"):
        make_job(req)


def test_infra_retries_are_the_evaluation_policys():
    c = load_config(Path(__file__).parents[1] / "configs" / "roft.json")
    from data.relay.reflection.collect import infra_retry

    c["runtime"]["collection"]["infra_retries"] = False
    assert infra_retry(c) == {"max_retries": 0}
    c["runtime"]["collection"]["infra_retries"] = True
    retry = infra_retry(c)
    assert retry["max_retries"] > 0 and "DaytonaProcessError" in retry["include_exceptions"]
    assert not {"ContextLengthExceededError", "AgentTimeoutError"} & set(retry["include_exceptions"])
