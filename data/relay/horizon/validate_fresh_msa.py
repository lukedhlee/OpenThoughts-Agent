"""Validate the exact fresh MSA2 policy and selected task set before serving."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml
from harbor_config.models.job.config import JobConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--harbor", required=True, type=Path)
    parser.add_argument("--ota", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    tasks = (args.root / "tasks.txt").read_text().splitlines()
    assert len(tasks) == len(set(tasks)) == 64 and tasks == manifest["selected_task_ids"]
    assert hashlib.sha256((args.root / "tasks.txt").read_bytes()).hexdigest() == manifest["tasks_sha256"]
    assert manifest["heldout_overlap"] == manifest["previous_matched_overlap"] == 0
    tree = Path(f"/scratch/11584/{Path.home().name}/relay/calibforge_tree")
    template = (args.ota / "data/relay/pilot/relay_msa_fresh_0921.yaml").read_text()
    replacements = dict(JOB_NAME="fresh_policy_probe", JOBS_DIR=str(args.root / "jobs"),
                        API_BASE="http://127.0.0.1:23750/v1", CONC="8", MAX_INPUT="65536",
                        MAX_OUTPUT="16384", AGENT_MULT="8.0")
    for key, value in replacements.items():
        template = template.replace(f"__{key}__", value)
    config = yaml.safe_load(template)
    config["tasks"] = [dict(path=str(tree / task)) for task in tasks]
    parsed = JobConfig.model_validate(config)
    assert parsed.n_attempts == 1 and parsed.n_concurrent_trials == 8
    kwargs = config["agents"][0]["kwargs"]
    assert config["agents"][0]["name"] == "mini-swe-agent-host"
    assert kwargs["model_class"] == "litellm" and kwargs["config_file"] == "mini.yaml"
    assert kwargs["llm_kwargs"]["fit_max_tokens_to_context"] is True
    assert kwargs["llm_call_kwargs"] == dict(tool_choice="none", max_tokens=16384)
    assert kwargs["model_info"]["max_input_tokens"] == 65536
    assert kwargs["model_info"]["max_output_tokens"] == 16384
    assert kwargs["interleaved_thinking"] is True
    assert config["environment"]["kwargs"] == {}
    assert config["environment"]["import_path"] == "data.relay.horizon.fresh_msa_runtime:ExistingCalibforgeSnapshot"
    fit_source = args.harbor / "src/harbor/llms/lite_llm.py"
    assert "fit_max_tokens_to_context" in fit_source.read_text()
    model = Path(f"/scratch/11584/{Path.home().name}/models/grug-datakit-sft-20260921")
    model_config = json.loads((model / "config.json").read_text())
    assert model_config["model_type"] == "grug_moe" and model_config["qk_mult"] == 1.75
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert all((model / part).is_file() for part in set(index["weight_map"].values()))
    (args.root / "validated_policy.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (args.root / "policy_gate.json").write_text(json.dumps(dict(
        passed=True, tasks=64, arms=2, attempts=1, scheduled_slots=128,
        public_context=65536, max_output=16384, context_fit=True,
        snapshot_operations="none", harness="mini-swe-agent2.4.6 tools",
    ), indent=2) + "\n")
    print("FRESH_MSA_POLICY_OK 64 tasks x2 arms x1 attempt; fit65536/16384; snapshots existing only")


if __name__ == "__main__":
    main()
