from pathlib import Path
from types import SimpleNamespace

import yaml

from hpc.rl_config_utils import build_skyrl_hydra_args, parse_rl_config


# The model is staged verbatim from the hub — there is no conversion script to
# import these from, so the contract is pinned here and asserted against the
# configs that consume it.
EXPECTED_REPO = "Qwen/Qwen3.6-35B-A3B"
EXPECTED_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"


ROOT = Path(__file__).resolve().parents[2]
RL_CONFIG = (
    ROOT
    / "hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml"
)
SERVE_CONFIG = (
    ROOT
    / "hpc/datagen_yaml/qwen3_6_35b_a3b_swebench80k_jupiter.yaml"
)
EVAL_CONFIG = (
    ROOT
    / "hpc/harbor_yaml/eval/configs/eval_opencode_apptainer_qwen3_6_swebench100_ctx80k.yaml"
)
GRPO_LAUNCHER = (
    ROOT
    / "hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo.sh"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_pinned_origin_is_the_unquantized_release() -> None:
    """Both arms must resolve to the plain release at one exact revision, or
    the paired SWE-Bench comparison is meaningless."""
    assert EXPECTED_REPO == "Qwen/Qwen3.6-35B-A3B"
    assert len(EXPECTED_REVISION) == 40 and EXPECTED_REVISION.isalnum()

    launcher = GRPO_LAUNCHER.read_text()
    assert EXPECTED_REVISION in launcher
    assert "fp8_origin_provenance" not in launcher
    assert "dequantize" not in launcher.lower()

    # Every consumer must point at the same pinned revision.
    for path in (RL_CONFIG, SERVE_CONFIG):
        assert EXPECTED_REVISION in path.read_text(), path


def test_vlm_shell_is_unwrapped_in_memory_not_on_disk() -> None:
    """We stage the multimodal shell and let SkyRL unwrap at load time. The flag
    also gates the weight-sync name mapping, so policy and rollout engines must
    agree on it."""
    extra_env = load_yaml(RL_CONFIG)["container"]["extra_env"]
    assert str(extra_env["SKYRL_QWEN3_5_VLM_UNWRAP"]) == "1", (
        "unwrap must be enabled; the staged checkpoint is the multimodal shell"
    )


def test_grpo_smoke_geometry_and_text_checkpoint_namespace() -> None:
    config = load_yaml(RL_CONFIG)
    trainer = config["trainer"]
    policy = trainer["policy"]
    ref = trainer["ref"]
    placement = trainer["placement"]
    generator = config["generator"]
    env = config["container"]["extra_env"]

    assert trainer["strategy"] == "fsdp2"
    assert trainer["max_steps"] == 1
    assert trainer["algorithm"]["advantage_estimator"] == "grpo"
    assert trainer["algorithm"]["use_tis"] is True
    assert policy["model"]["path"] == ref["model"]["path"]
    assert EXPECTED_REVISION in policy["model"]["path"]
    assert policy["fsdp_config"]["fsdp_size"] == 4
    assert policy["fsdp_config"]["expert_model_parallel_size"] == 4
    assert placement["policy_num_nodes"] == 4
    assert placement["policy_num_gpus_per_node"] == 4
    assert placement["colocate_policy_ref"] is True
    assert placement["colocate_all"] is False
    assert generator["model_dtype"] == "bfloat16"
    assert generator["inference_engine_tensor_parallel_size"] == 1
    assert generator["num_inference_engines"] == 8
    assert generator["weight_sync_backend"] == "nccl"
    assert generator["engine_init_kwargs"]["enable_auto_tool_choice"] is True
    assert generator["engine_init_kwargs"]["tool_call_parser"] == "qwen3_coder"
    assert "quantization" not in generator
    assert "fuse_weights" not in generator
    # Both policy and vLLM load the multimodal shell straight from the hub and
    # unwrap in memory. The same flag gates the weight-sync name mapper, so it
    # rewrites model.* to model.language_model.* for the engines in lockstep.
    assert env["SKYRL_QWEN3_5_VLM_UNWRAP"] == "1"
    assert env["SKYRL_GDN_FLASHQLA"] == "1"
    assert env["SKYRL_GDN_FLASHQLA_REQUIRED"] == "1"
    assert env["APPTAINER_BRIDGE_URL"] == "http://10.128.1.2:9920"
    assert env["PYTHONPATH"].startswith(
        "/e/scratch/reformo/lee27/pydeps/qwen36-flashqla-0.1.2:"
    )
    assert "qwen36-flashqla-0.1.2-sm90" in env["TILELANG_CACHE_DIR"]
    # Unset means SkyRL auto-detects GDN from layer_types and masks broken FLA.
    assert "SKYRL_GDN_MASK_FLA" not in env


def test_grpo_context_budget_is_single_source_of_truth() -> None:
    raw = load_yaml(RL_CONFIG)
    budget = raw["context_budget"]

    assert budget == {
        "request_window_tokens": 32_768,
        "max_new_tokens_per_turn": 4_096,
        "max_turns": 999_999,
        "client_prompt_overhead_tokens": 1,
    }

    parsed = parse_rl_config(str(RL_CONFIG))
    assert parsed.trainer["max_prompt_length"] == 28_672
    assert parsed.generator["max_turns"] == 999_999
    assert parsed.generator["sampling_params"]["max_generate_length"] == 4_096
    assert parsed.generator["engine_init_kwargs"]["max_model_len"] == 32_768
    assert parsed.terminal_bench["harbor"]["max_episodes"] == 999_999
    assert "max_turns" not in parsed.terminal_bench["harbor"]
    assert parsed.terminal_bench["model_info"] == {
        "max_input_tokens": 28_671,
        "max_output_tokens": 4_096,
    }
    # OpenCode budgets message content before Qwen's chat template inserts one
    # special token. The server-side request must still fit exactly in 32 KiB.
    assert (
        parsed.terminal_bench["model_info"]["max_input_tokens"]
        + budget["client_prompt_overhead_tokens"]
        + parsed.terminal_bench["model_info"]["max_output_tokens"]
        == budget["request_window_tokens"]
    )


def test_grpo_launcher_requires_compiled_vllm_and_gpu_smoked_flashqla() -> None:
    launcher = GRPO_LAUNCHER.read_text()

    assert "import vllm._C" in launcher
    assert "get_fsdp_wrap_policy(wrap_probe) is not None" in launcher
    assert '"$FLASHQLA_LAYER/gpu_sm90_smoke.ok"' in launcher
    assert '"Qwen3_5MoeGatedDeltaNet" in gdn._FLASHQLA_GDN_TYPES' in launcher
    assert '"flash-qla": "0.1.2"' in launcher


def test_grpo_explicitly_disables_hf_hub_upload() -> None:
    parsed = parse_rl_config(str(RL_CONFIG))
    args = build_skyrl_hydra_args(
        parsed,
        {"job_name": "qwen36-smoke", "num_nodes": 6},
        SimpleNamespace(gpus_per_node=4),
    )

    assert parsed.trainer["hf_hub_repo_id"] is None
    assert not any("trainer.hf_hub_repo_id" in arg for arg in args)
    assert "++generator.engine_init_kwargs.enable_auto_tool_choice=true" in args
    assert "++generator.engine_init_kwargs.tool_call_parser=qwen3_coder" in args


def test_grpo_rollouts_use_pinned_opencode_without_compaction() -> None:
    harbor = load_yaml(RL_CONFIG)["terminal_bench"]["harbor"]

    assert harbor["name"] == "opencode"
    assert harbor["version"] == "1.18.8"
    assert harbor["preinstalled"] is True
    assert harbor["prompt_template_path"].endswith(
        "/eval/jureca/opencode_swebench_prompt.md.j2"
    )
    assert harbor["opencode_config"] == {
        "autoupdate": False,
        "compaction": {"auto": False},
    }
    for terminus_only_key in (
        "strict_json_parser",
        "interleaved_thinking",
        "extra_body",
    ):
        assert terminus_only_key not in harbor


def test_swebench_arms_share_one_served_id_and_81920_context() -> None:
    serve = load_yaml(SERVE_CONFIG)
    evaluation = load_yaml(EVAL_CONFIG)
    server = serve["vllm_server"]
    agent = evaluation["agents"][0]

    assert server["max_model_len"] == 81_920
    assert server["tensor_parallel_size"] == 1
    assert server["data_parallel_size"] == 4
    extra = server["extra_args"]
    assert extra[extra.index("--dtype") + 1] == "bfloat16"
    assert extra[extra.index("--kv-cache-dtype") + 1] == "fp8"
    served_id = extra[extra.index("--served-model-name") + 1]
    assert agent["model_name"] == f"vllm/{served_id}"
    info = agent["kwargs"]["model_info"]
    assert info["max_input_tokens"] + info["max_output_tokens"] == 81_920
    assert agent["kwargs"]["opencode_config"]["compaction"]["auto"] is False
    assert evaluation["n_attempts"] == 1
    assert "swebench-verified-random-100-a2e51e9" in evaluation["datasets"][0]["path"]
