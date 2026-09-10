#!/usr/bin/env python3
"""Clone a band-wave experiment (band_r3_s0 template) into a Snowball pass@K probe.
Swaps the policy to the Snowball Stage-3 checkpoint, the venv to `snowball`, and applies the
report's grug training constraints (FSDP2 EP=1, no packing, no grouped GEMM, frozen router) plus the
inference geometry that fits 134 GB of bf16 weights on 96 GB GH200s (TP=1 x DP=4 x EP=4 per engine).
Usage: make_snowball_probe.py --name snowball_probe_val --val-dir <task dir> --k 8 --conc 128 [--attn-backend X]
Prints the sbatch path. Submit from the OTA repo root: cd $OTA && sbatch <path>."""
import argparse, json, os, re, shutil
# 2026-09-07: run dirs move to /e/data1/mmlaion (Luke). Opt-in override so probes already in flight keep the old default.
EXP = os.environ.get("SNOWBALL_EXP", "/e/fscratch/reformo/lee27/experiments")
MODEL = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
VENV = "/e/project1/transfernetx/lee27/code/envs/snowball"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--template", default="band_r3_s0")
ap.add_argument("--val-dir", required=True); ap.add_argument("--k", type=int, default=8); ap.add_argument("--conc", type=int, default=128)
ap.add_argument("--attn-backend", default=None, help="vLLM attention backend engine kwarg; omit for auto")
ap.add_argument("--moe-backend", default=None, help="vLLM moe_backend engine kwarg; omit for default")
ap.add_argument("--wall", default="05:00:00"); ap.add_argument("--lr", default="1e-5")
ap.add_argument("--max-in", type=int, default=24576, help="max input tokens (context budget for the harness)")
ap.add_argument("--max-out", type=int, default=8192, help="max output tokens per turn; max_in + max_out must be <= 32768")
ap.add_argument("--max-model-len", type=int, default=32768, help="vLLM max_model_len; >32768 adds the grug hf_overrides (RoPE table is sized from max_seq_len, a frozen copy of max_position_embeddings)")
ap.add_argument("--nodes", type=int, default=6, help="sbatch nodes: 4 policy + engines (each DP4 engine = 1 node)")
ap.add_argument("--engines", type=int, default=2, help="generator.num_inference_engines (TP1xDP4xEP4 each); nodes must be 4 + engines")
ap.add_argument("--parity", action="store_true", help="SFT-format parity: skip_special_tokens=false via harbor extra_body (think markers kept in history; needs harbor lukedhlee/terminus2-think-parity) + chat_template_content_format=string (needs MarinSkyRL lukedhlee/jupiter-parity64k)")
ap.add_argument("--summarize", action="store_true", help="enable terminus-2 context summarization")
ap.add_argument("--model", default=None, help="policy/ref checkpoint dir (HF format, grug_moe); default = the Stage-3 base. Use an exported GRPO checkpoint for the transfer re-probe")
ap.add_argument("--eval-timeout", type=int, default=None, help="agent wall timeout (s) for EVAL trials; MarinSkyRL defaults eval to 900 s regardless of harbor.override_timeout_sec (training rollouts use the 1800 s override)")
ap.add_argument("--verifier-timeout", type=int, default=600, help="harbor verifier_override_timeout_sec for probe trials (default 600; the template value 120 nulls ~0.4 %% of attempts and far more under sandbox-pool load)")
a = ap.parse_args()
if a.model: MODEL = a.model.rstrip("/"); assert os.path.isfile(f"{MODEL}/config.json"), MODEL
assert a.max_in + a.max_out <= a.max_model_len, (a.max_in, a.max_out, a.max_model_len)
assert a.nodes == 4 + a.engines, (a.nodes, a.engines)
src, dst = f"{EXP}/{a.template}", f"{EXP}/{a.name}"
assert a.name != a.template and os.path.isdir(src)
if os.path.exists(dst): shutil.rmtree(dst)
for sub in ("configs", "sbatch", "logs"): os.makedirs(f"{dst}/{sub}")
c = json.loads(open(f"{src}/configs/{a.template}_rl_config.json").read().replace(a.template, a.name))
old_model = c["model_path"]; c["model_path"] = MODEL
drop_prefixes = (
    "data.val_data=", "generator.eval_n_samples_per_prompt=", "++terminal_bench_config.harbor.n_concurrent_trials=",
    "++generator.engine_init_kwargs.moe_backend=", "++generator.engine_init_kwargs.attention_backend=",
    "++generator.engine_init_kwargs.max_model_len=", "++generator.engine_init_kwargs.served_model_name=",
    "trainer.policy.model.path=", "trainer.ref.model.path=", "trainer.policy.optimizer_config.lr=",
    "generator.max_input_length=", "trainer.max_prompt_length=", "++terminal_bench_config.model_info.max_input_tokens=",
    "generator.sampling_params.max_generate_length=", "++terminal_bench_config.model_info.max_output_tokens=",
    "generator.inference_engine_tensor_parallel_size=", "generator.inference_engine_expert_parallel_size=", "generator.num_inference_engines=",
    "trainer.use_sample_packing=", "trainer.attn_backend=", "trainer.flash_attn=",
    "trainer.policy.fsdp_config.fsdp_size=", "trainer.ref.fsdp_config.fsdp_size=",
    "trainer.policy.fsdp_config.expert_model_parallel_size=", "trainer.ref.fsdp_config.expert_model_parallel_size=",
    "trainer.policy.fsdp_config.moe_grouped_gemm=", "trainer.ref.fsdp_config.moe_grouped_gemm=",
    "trainer.policy.fsdp_config.ep_comm_backend=", "trainer.ref.fsdp_config.ep_comm_backend=",
    "trainer.project_name=", "trainer.max_steps=", "trainer.eval_before_train=", "generator.n_samples_per_prompt=",
)
args = [x for x in c["skyrl_hydra_args"] if not x.startswith(drop_prefixes)]
args = [x.replace(old_model, MODEL) for x in args]
args += [
    f"trainer.policy.model.path={MODEL}", f"trainer.ref.model.path={MODEL}",
    f"trainer.policy.optimizer_config.lr={a.lr}",
    # grug training constraints (report POLICY.md): FSDP2, EP=1, no packing, no grouped GEMM/mm, CP=1, frozen bias (branch default)
    "trainer.strategy=fsdp2", "trainer.use_sample_packing=false", "trainer.flash_attn=true", "trainer.attn_backend=flash_attention_2",
    "trainer.policy.fsdp_config.fsdp_size=16", "trainer.ref.fsdp_config.fsdp_size=16",
    "trainer.policy.fsdp_config.expert_model_parallel_size=1", "trainer.ref.fsdp_config.expert_model_parallel_size=1",
    "trainer.policy.fsdp_config.moe_grouped_gemm=false", "trainer.ref.fsdp_config.moe_grouped_gemm=false",
    "++trainer.policy.fsdp_config.use_grouped_mm=false", "++trainer.policy.fsdp_config.context_parallel_size=1",
    "trainer.policy.fsdp_config.moe_router_replay=false",
    "++trainer.policy.grug_query_bias_update_mode=frozen",
    # inference geometry: 2 engines x (TP1 x DP4 x EP4) = 8 GPUs = 2 nodes; SkyRL requires dp*tp == ep
    "generator.inference_engine_tensor_parallel_size=1", "++generator.inference_engine_data_parallel_size=4",
    "generator.inference_engine_expert_parallel_size=4", f"generator.num_inference_engines={a.engines}",
    # context geometry: Snowball max_position_embeddings=32768 -> 24576 in / 8192 out (the SFT eval contract)
    f"++generator.engine_init_kwargs.max_model_len={a.max_model_len}", f"generator.max_input_length={a.max_in}", f"trainer.max_prompt_length={a.max_in}",
    f"++terminal_bench_config.model_info.max_input_tokens={a.max_in}", f"generator.sampling_params.max_generate_length={a.max_out}",
    f"++terminal_bench_config.model_info.max_output_tokens={a.max_out}",
    f"++generator.engine_init_kwargs.served_model_name={os.path.basename(MODEL)}",
    # probe mechanics: eval-only at step 0 (kill at timing/step), K samples on the val dir
    "trainer.project_name=jupiter-snowball-r2egym", "trainer.max_steps=1", "trainer.eval_before_train=true",
    "generator.n_samples_per_prompt=4",
    f'data.val_data=["{a.val_dir}"]', f"generator.eval_n_samples_per_prompt={a.k}",
    f"++terminal_bench_config.harbor.n_concurrent_trials={a.conc}",
]
if a.max_model_len > 32768: args.append(f"++generator.engine_init_kwargs.hf_overrides={{max_position_embeddings:{a.max_model_len},max_seq_len:{a.max_model_len}}}")
if a.parity: args += ["++terminal_bench_config.harbor.extra_body.skip_special_tokens=false", "++generator.engine_init_kwargs.chat_template_content_format=string"]
if a.summarize: args = [x for x in args if not x.startswith("++terminal_bench_config.harbor.enable_summarize=")] + ["++terminal_bench_config.harbor.enable_summarize=true"]
if a.eval_timeout: args = [x for x in args if not x.startswith("++terminal_bench_config.harbor.eval_timeout_override_sec=")] + [f"++terminal_bench_config.harbor.eval_timeout_override_sec={a.eval_timeout}"]
if a.attn_backend: args.append(f"++generator.engine_init_kwargs.attention_backend={a.attn_backend}")
if a.moe_backend: args.append(f"++generator.engine_init_kwargs.moe_backend={a.moe_backend}")
args = [x for x in args if not x.lstrip("+").startswith("terminal_bench_config.harbor.verifier_override_timeout_sec=")]
args.append(f"++terminal_bench_config.harbor.verifier_override_timeout_sec={a.verifier_timeout}")  # 2026-09-05: template had 120 s; the verifier audit found the pre-pytest scaffolding, not test runtime, trips it (0.4 % of attempts, 190 nulls on the KL 0.05 s24 probe during the pool crunch); training arms use 150-600
c["skyrl_hydra_args"] = args
json.dump(c, open(f"{dst}/configs/{a.name}_rl_config.json", "w"), indent=2)
sb = open(f"{src}/sbatch/{a.template}_rl.sbatch").read().replace(a.template, a.name)
sb = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={a.wall}", sb, flags=re.M)
sb = re.sub(r"^#SBATCH --nodes=.*$", f"#SBATCH --nodes={a.nodes}", sb, flags=re.M)
sb = sb.replace("export NUM_INFERENCE_ENGINES=24", f"export NUM_INFERENCE_ENGINES={a.nodes*4}").replace("export POLICY_NUM_NODES=6", f"export POLICY_NUM_NODES={a.nodes}")
sb = sb.replace('RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"', f'export DCFT_RL_ENV={VENV}\nexport RL_PYTHON={VENV}/bin/python\nexport NCCL_PXN_DISABLE=1\nexport LD_LIBRARY_PATH="${{LD_LIBRARY_PATH//envs\\/rl-fa/envs\\/snowball}}"\nexport VLLM_CACHE_ROOT=/e/fscratch/reformo/lee27/cache/vllm XDG_CACHE_HOME=/e/fscratch/reformo/lee27/cache/xdg TRITON_CACHE_DIR=/e/fscratch/reformo/lee27/cache/triton TORCHINDUCTOR_CACHE_DIR=/e/fscratch/reformo/lee27/cache/inductor\nRL_ENV_DIR="${{RL_ENV_DIR:-$WORKDIR/envs/rl}}"', 1)
assert f"DCFT_RL_ENV={VENV}" in sb and f"RL_PYTHON={VENV}/bin/python" in sb
out = f"{dst}/sbatch/{a.name}_rl.sbatch"; open(out, "w").write(sb); print(out)
