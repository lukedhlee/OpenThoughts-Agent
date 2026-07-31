from pathlib import Path

import yaml

from scripts.huggingface.dequantize_qwen3_6_fp8_for_grpo import (
    EXPECTED_REPO,
    EXPECTED_REVISION,
    MINIMUM_BF16_CHECKPOINT_BYTES,
)


ROOT = Path(__file__).resolve().parents[2]
RL_CONFIG = (
    ROOT
    / "hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_fp8_origin_r2egym_grpo.yaml"
)
SERVE_CONFIG = (
    ROOT
    / "hpc/datagen_yaml/qwen3_6_35b_a3b_fp8_origin_bf16_swebench80k_jupiter.yaml"
)
EVAL_CONFIG = (
    ROOT
    / "hpc/harbor_yaml/eval/configs/eval_opencode_apptainer_qwen3_6_swebench100_ctx80k.yaml"
)
GRPO_LAUNCHER = (
    ROOT
    / "hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_fp8_origin_grpo.sh"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_exact_fp8_origin_and_bf16_artifact_gate() -> None:
    assert EXPECTED_REPO == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert EXPECTED_REVISION == "95a723d08a9490559dae23d0cff1d9466213d989"
    assert MINIMUM_BF16_CHECKPOINT_BYTES >= 55_000_000_000


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
    assert "quantization" not in generator
    assert "fuse_weights" not in generator
    # Both policy and vLLM load the already-unwrapped text checkpoint. The VLM
    # shell mapper would rewrite model.* to the wrong model.language_model.* keys.
    assert env["SKYRL_QWEN3_5_VLM_UNWRAP"] == "0"
    assert env["SKYRL_GDN_FLASHQLA"] == "1"
    assert env["SKYRL_GDN_FLASHQLA_REQUIRED"] == "1"
    assert env["PYTHONPATH"].startswith(
        "/e/scratch/reformo/lee27/pydeps/qwen36-flashqla-0.1.2:"
    )
    assert "qwen36-flashqla-0.1.2-sm90" in env["TILELANG_CACHE_DIR"]
    # Unset means SkyRL auto-detects GDN from layer_types and masks broken FLA.
    assert "SKYRL_GDN_MASK_FLA" not in env


def test_grpo_launcher_requires_compiled_vllm_and_gpu_smoked_flashqla() -> None:
    launcher = GRPO_LAUNCHER.read_text()

    assert "import vllm._C" in launcher
    assert '"$FLASHQLA_LAYER/gpu_sm90_smoke.ok"' in launcher
    assert '"Qwen3_5MoeGatedDeltaNet" in gdn._FLASHQLA_GDN_TYPES' in launcher
    assert '"flash-qla": "0.1.2"' in launcher


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
