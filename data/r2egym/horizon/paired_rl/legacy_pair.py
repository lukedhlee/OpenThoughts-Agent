#!/usr/bin/env python3
"""Prepare or launch the bounded base/vanilla-SFT GRPO pair on the proven Horizon stack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRATCH = Path("/scratch/11584/lukedhlee")
ROOT = SCRATCH / "experiments/paired-sft-rl-20261026/legacy"
OTA = Path.home() / "snowball/ota-rl"
RUNTIME = Path.home() / "snowball/ota-rl-runtime"
PYTHON = Path.home() / "snowball/envs/snowball/bin/python"
GUARD = SCRATCH / "experiments/kimi-only-fp32-capacity2-20261025/bin/sbatch"
ARMS = {
    "rl_base_1026": SCRATCH / "models/grug-datakit-sft-20260921",
    "rl_v7_1026": SCRATCH / "experiments/cf2-epochs57-20261022/training/fdv/export-step959-hf-bf16",
}


def run(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, **kwargs)


def model_metadata(path):
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        assert (path / shard).stat().st_size > 1000, shard
    tokens = {}
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        tokens[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    return {"path": str(path), "config": config, "weight_keys": sorted(index["weight_map"]), "tokenizers": tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    ROOT.mkdir(exist_ok=True)
    (ROOT / "bin").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    guard_link = ROOT / "bin/sbatch"
    if not guard_link.exists():
        guard_link.symlink_to(GUARD)
    env = dict(os.environ, FP32_REPLAY_ROOT=str(ROOT), OMP_NUM_THREADS="1")
    env["PATH"] = str(ROOT / "bin") + ":" + env["PATH"]
    env.pop("DCFT_PRIVATE", None)
    models = {name: model_metadata(path) for name, path in ARMS.items()}
    base, sft = models.values()
    assert base["weight_keys"] == sft["weight_keys"], "tensor names differ"
    for key in ["model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "qk_mult"]:
        assert base["config"].get(key) == sft["config"].get(key), key
    assert base["tokenizers"]["tokenizer.json"] == sft["tokenizers"]["tokenizer.json"], "tokenizer vocabulary differs"
    # Reuse the historical task-level separation. These CF identities are a new source,
    # so report exact-ID intersection without claiming a content-level contamination audit.
    train = set((SCRATCH / "rl_pool/train.txt").read_text().splitlines())
    heldout = set((SCRATCH / "rl_pool/heldout.txt").read_text().splitlines())
    assert len(train) == 1104 and len(heldout) == 123 and not train & heldout
    selected = SCRATCH / "experiments/relay-fixed-design-20261021/final/fdv/fdv_selected.jsonl"
    cf = {json.loads(line)["task"] for line in selected.open()}
    assert not cf & (train | heldout)
    draft = SCRATCH / "hf_hub/models--laion--snowball-64k-eagle3-draft-r2egym/snapshots/4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    assert (draft / "config.json").is_file()
    for name, model in ARMS.items():
        command = [sys.executable, OTA / "data/r2egym/horizon/rl/make_arm.py",
                   "--name", name, "--recipe", "arm", "--model", model,
                   "--tree", SCRATCH / "rl_pool/train_tree", "--val-tree", SCRATCH / "rl_pool/heldout_tree",
                   "--policy-nodes", "4", "--engines", "8", "--batch", "64", "--seats", "256",
                   "--coords", "16", "--steps", "30", "--wall", "08:00:00", "--draft", draft,
                   "--ckpt-interval", "6", "--hf-save-interval", "6", "--project", "snowball-strategy-legacy-20261026",
                   "--out", ROOT, "--set", "trainer.eval_before_train=true",
                   "--set", "trainer.eval_interval=1000000", "--set", "generator.eval_n_samples_per_prompt=8"]
        arm = ROOT / name
        # Never regenerate a live run's config.
        if not (arm / "jobs.txt").exists():
            print(run(command), flush=True)
        config = arm / "configs" / (name + "_rl_config.json")
        print(run([PYTHON, OTA / "data/r2egym/horizon/rl/validate_hydra_args.py", config], cwd="/tmp", env=env), flush=True)
        run(["bash", "-n", arm / "sbatch" / (name + "_rl.sbatch")])
    manifest = {"models": models, "train_tasks": len(train), "heldout_tasks": len(heldout),
                "cf_tasks": len(cf), "exact_cf_id_overlap": 0,
                "content_overlap_audit": "not established by names alone",
                "runtime_sha": run(["git", "-C", RUNTIME, "rev-parse", "HEAD"]).strip(),
                "skyrl_sha": run(["git", "-C", Path.home()/"snowball/marinskyrl-marin", "rev-parse", "HEAD"]).strip(),
                "nodes_per_arm": 12, "wall_hours_per_arm": 8, "maximum_node_hours": 192,
                "steps": 30, "samples_per_task": 8, "batch_groups": 64,
                "selection": "uniform shared train pool, no model-dependent curriculum",
                "baseline_eval": "123 heldout tasks x8 before learning in each arm",
                "comparison": "same steps and maximum per-rollout budget, realized rollout tokens and compute must also be reported",
                "new_upstream_runtime": "independent, not required for these jobs"}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PREPARE_OK", flush=True)
    if args.launch:
        for name in ARMS:
            arm = ROOT / name
            if (arm / "jobs.txt").exists():
                print(name, "already submitted", (arm / "jobs.txt").read_text().strip(), flush=True)
                continue
            env["TUNNEL_PORTS"] = "18226,18227" if name == "rl_base_1026" else "18228,18229"
            output = run(["bash", OTA / "data/r2egym/horizon/rl/launch_arm.sh", arm], cwd=RUNTIME, env=env)
            (ROOT / "logs" / (name + "-launch.log")).write_text(output)
            print(output, flush=True)


if __name__ == "__main__":
    main()
