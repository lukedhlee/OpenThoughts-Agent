from pathlib import Path
from types import SimpleNamespace

import yaml

from hpc.rl_config_utils import build_skyrl_hydra_args, parse_rl_config


ROOT = Path(__file__).resolve().parents[2]
CANARY_CONFIG = (
    ROOT / "hpc/skyrl_yaml/jupiter/5node_qwen3_6_35b_a3b_r2egym_grpo_canary.yaml"
)
CANARY_LAUNCHER = (
    ROOT / "hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo_canary.sh"
)
EXPECTED_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


def load_canary() -> dict:
    return yaml.safe_load(CANARY_CONFIG.read_text())


def test_canary_is_a_standalone_intermediate_profile() -> None:
    raw = load_canary()
    launcher = CANARY_LAUNCHER.read_text()

    assert "defaults" not in raw and "base_config" not in raw
    assert set(
        ("terminal_bench", "trainer", "generator", "rollout", "container")
    ) <= set(raw)
    assert "intermediate" in CANARY_CONFIG.read_text().lower()
    assert (
        "not replace the established six-node final gate" in CANARY_CONFIG.read_text()
    )
    assert "export NUM_NODES=5" in launcher
    assert "export MAX_STEPS=1" in launcher
    assert "export CKPT_INTERVAL=1" in launcher
    assert "export HF_SAVE_INTERVAL=1" in launcher
    assert "5node_qwen3_6_35b_a3b_r2egym_grpo_canary.yaml" in launcher


def test_canary_collects_exactly_64_trajectories_in_32_groups() -> None:
    raw = load_canary()
    trainer = raw["trainer"]
    generator = raw["generator"]
    harbor = raw["terminal_bench"]["harbor"]

    assert trainer["train_batch_size"] == 32
    assert trainer["policy_mini_batch_size"] == 32
    assert trainer["fully_async"] == {
        "num_parallel_generation_workers": 32,
        "max_buffered_groups": 32,
    }
    assert generator["n_samples_per_prompt"] == 2
    assert trainer["train_batch_size"] * generator["n_samples_per_prompt"] == 64
    assert harbor["n_concurrent_trials"] == 32


def test_canary_keeps_four_policy_nodes_and_uses_one_rollout_node() -> None:
    raw = load_canary()
    trainer = raw["trainer"]
    placement = trainer["placement"]
    generator = raw["generator"]

    assert placement["policy_num_nodes"] == 4
    assert placement["ref_num_nodes"] == 4
    assert placement["policy_num_gpus_per_node"] == 4
    assert placement["ref_num_gpus_per_node"] == 4
    assert placement["colocate_policy_ref"] is True
    assert placement["colocate_all"] is False
    assert trainer["policy"]["fsdp_config"]["fsdp_size"] == 4
    assert trainer["policy"]["fsdp_config"]["expert_model_parallel_size"] == 4
    assert generator["num_inference_engines"] == 4
    assert generator["inference_engine_tensor_parallel_size"] == 1

    parsed = parse_rl_config(str(CANARY_CONFIG))
    args = build_skyrl_hydra_args(
        parsed,
        {"job_name": "qwen36-canary", "num_nodes": 5},
        SimpleNamespace(gpus_per_node=4),
    )
    assert "trainer.placement.policy_num_nodes=4" in args
    assert "generator.num_inference_engines=4" in args
    assert "++trainer.fully_async.max_buffered_groups=32" in args


def test_canary_retains_exact_safe_pipeline_contract() -> None:
    raw = load_canary()
    trainer = raw["trainer"]
    generator = raw["generator"]
    harbor = raw["terminal_bench"]["harbor"]

    assert trainer["max_steps"] == 1
    assert trainer["ckpt_interval"] == 1
    assert trainer["hf_save_interval"] == 1
    assert EXPECTED_REVISION in trainer["policy"]["model"]["path"]
    assert trainer["policy"]["model"]["path"] == trainer["ref"]["model"]["path"]
    assert generator["model_dtype"] == "bfloat16"
    assert "quantization" not in generator
    assert "fp8" not in str(generator).lower()
    assert harbor["environment_type"] == "apptainer"
    assert harbor["bridge_url"].startswith("${oc.env:APPTAINER_BRIDGE_URL")
    assert harbor["opencode_config"]["compaction"]["auto"] is True
    assert "reserved" not in harbor["opencode_config"]["compaction"]
    assert harbor["opencode_config"]["compaction"]["auto"] is True
    assert {"ContextLengthExceededError", "NonZeroAgentExitCodeError"}.issubset(
        harbor["exclude_exceptions"]
    )

    parsed = parse_rl_config(str(CANARY_CONFIG))
    assert parsed.trainer["max_prompt_length"] == 28_672
    assert parsed.generator["engine_init_kwargs"]["max_model_len"] == 32_768
    assert parsed.terminal_bench["model_info"] == {
        "max_input_tokens": 20_480,
        "max_output_tokens": 4_096,
    }
