#!/usr/bin/env python3
"""GH200 fused_moe Triton tile tuning for Snowball at EP4 (E=64 local experts, N=1280, K=2560, top-k 4, bf16).

Wraps the fork's benchmarks/kernels/benchmark_moe.py; the grug_moe config keys are not in its
get_model_params table, so the shapes are pinned here (256 experts / 4 = 64 per rank at TP1xEP4).
  --tune            search the Triton tile space and write E=64,N=1280,device_name=NVIDIA_GH200_120GB.json
  --batch-size ...  benchmark the config the loader currently picks (honours VLLM_TUNED_CONFIG_FOLDER)
"""
import argparse, sys

sys.path.insert(0, "/e/project1/transfernetx/lee27/code/src/marin_vllm/benchmarks/kernels")
import benchmark_moe as bm  # noqa: E402

bm.get_model_params = lambda config: (256, 4, 1280, 2560)
ap = argparse.ArgumentParser()
ap.add_argument("--save-dir", default=".")
ap.add_argument("--batch-size", type=int, nargs="+", default=None)
ap.add_argument("--tune", action="store_true")
a = ap.parse_args()
args = argparse.Namespace(
    model="/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888",
    tp_size=4, enable_expert_parallel=True, dtype="auto", use_deep_gemm=False,
    save_dir=a.save_dir, seed=0, batch_size=a.batch_size, tune=a.tune,
    trust_remote_code=False, model_prefix=None,
)
bm.main(args)
