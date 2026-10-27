#!/usr/bin/env python3
"""One-shot owner for paired R2E+TMax pre-RL probes. Never launches training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import tomllib

S = Path("/scratch/11584/lukedhlee")
ROOT = S / "experiments/paired-sft-rl-20261026/mixed"
OTA = Path.home() / "snowball/ota-rl"
PYTHON = Path.home() / "snowball/envs/snowball/bin/python"
RUNTIME = Path.home() / "snowball/ota-rl-runtime"
TMAX = S / "experiments/tmax"
ARMS = {
    "probe_mix_base_1026": S / "models/grug-datakit-sft-20260921",
    "probe_mix_v7_1026": S / "experiments/cf2-epochs57-20261022/training/fdv/export-step959-hf-bf16",
}


def call(command, **kwargs):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def status(message):
    ROOT.mkdir(exist_ok=True)
    (ROOT / "STATUS").write_text(message + "\n")
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    ROOT.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    status("PREPARING_MIXED_PROBES")
    # The earlier read-only artifact and Hydra checks passed for both exact models.
    prior = json.loads((ROOT.parent / "legacy/manifest.json").read_text())
    assert {m["path"] for m in prior["models"].values()} == set(map(str, ARMS.values()))
    gate = module_at("tmax_gate", OTA / "data/tmax/daytona/gate.py")
    gate.HARBOR_SRC = str(Path.home() / "snowball/harbor-rl/src")
    gate.HARBOR = str(PYTHON.parent / "harbor")
    key = gate.load_secret("DAYTONA_API_KEY", None)
    assert gate.snapshot_active("harbor__05b5bff9013d__snapshot", key) == "active"
    passed_file = TMAX / "gate/jobs_full/tmax_nop/gate_pass.txt"
    eligible = sorted({x.split("/")[-1] for x in passed_file.read_text().splitlines() if x})
    assert len(eligible) == 3031
    pilot_file = ROOT / "tmax-pilot20.txt"
    if not pilot_file.exists():
        pilot_file.write_text("\n".join(sorted(random.Random(20261026).sample(eligible, 20))) + "\n")
    pilot_dir = ROOT / "gate/tmax_nop"
    if not (pilot_dir / "gate_summary.json").exists():
        status("TMAX_CURRENT_ENVIRONMENT_PILOT")
        if not (pilot_dir / "result.json").exists():
            rc = gate.run(argparse.Namespace(tree=str(TMAX/"tree_gate"), tasks=str(pilot_file),
                jobs=str(ROOT/"gate"), conc=20, name="tmax_nop", harbor=gate.HARBOR, key_file=None))
            print("pilot run exit", rc, flush=True)
        gate.report_dir(pilot_dir)
    pilot = json.loads((pilot_dir / "gate_summary.json").read_text())
    assert pilot["trials"] == 20 and pilot["pass"] >= 18, pilot.get("classes")
    failed = {x["task"].split("/")[-1] for x in pilot.get("failed", []) if x.get("task")}
    eligible = [x for x in eligible if x not in failed]
    # Model-independent split fixed before any student rollouts. No held-out task enters a screen.
    ranked = sorted(eligible, key=lambda x: hashlib.sha256(("20261026:"+x).encode()).hexdigest())
    nheld = round(len(ranked) * .1)
    t_hold, t_train = set(ranked[:nheld]), set(ranked[nheld:])
    r_train = set((S / "rl_pool/train.txt").read_text().split())
    r_hold = set((S / "rl_pool/heldout.txt").read_text().split())
    assert len(r_train) == 1104 and len(r_hold) == 123 and not r_train & r_hold
    train = {x: S/"rl_pool/train_tree"/x for x in r_train}
    train.update({x: TMAX/"tree_relay"/x for x in t_train})
    held = {x: S/"rl_pool/heldout_tree"/x for x in r_hold}
    held.update({x: TMAX/"tree_relay"/x for x in t_hold})
    assert not set(train) & set(held)
    task_rows = []
    def link_and_read(item):
        task, source, tree, split = item
        assert (source/"tests/test.sh").is_file(), source
        dest = tree/task
        if not dest.is_symlink():
            dest.symlink_to(source, target_is_directory=True)
        cfg = tomllib.loads((source/"task.toml").read_text())
        return [task, cfg.get("task", {}).get("name", task), "tmax" if task.startswith("task_") else "r2egym", split, str(source)]
    for label, mapping in [("train", train), ("heldout", held)]:
        tree = ROOT / (label + "_tree")
        tree.mkdir(exist_ok=True)
        # NFS metadata is high-latency. Bound concurrent independent file operations,
        # rather than serializing thousands of readlink/stat calls on the login node.
        with ThreadPoolExecutor(max_workers=16) as pool:
            task_rows.extend(pool.map(link_and_read, [(task,source,tree,label) for task,source in sorted(mapping.items())]))
        assert {x.name for x in tree.iterdir()} == set(mapping)
        (ROOT / (label + ".txt")).write_text("\n".join(sorted(mapping)) + "\n")
        print(label, "tree ready", len(mapping), flush=True)
    with (ROOT / "task_map.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["task", "task_label", "source", "split", "path"])
        w.writerows(task_rows)
    draft = S/"hf_hub/models--laion--snowball-64k-eagle3-draft-r2egym/snapshots/4bdb47c08e5b5190bea3c7a93c3e14470230e469"
    for name, model in ARMS.items():
        arm = ROOT / name
        if not (arm/"jobs.txt").exists():
            call([sys.executable, OTA/"data/r2egym/horizon/rl/make_arm.py", "--name", name,
                "--recipe", "arm", "--model", model, "--tree", ROOT/"train_tree",
                "--val-tree", ROOT/"heldout_tree", "--policy-nodes", "4", "--engines", "8",
                "--batch", "64", "--seats", "448", "--coords", "16", "--steps", "1",
                "--probe", "8", "--wall", "12:00:00", "--draft", draft, "--out", ROOT,
                "--project", "snowball-strategy-mixed-20261026",
                "--set", "terminal_bench_config.harbor.override_storage_mb=10240"])
        cfg = arm/"configs"/(name+"_rl_config.json")
        call([PYTHON, OTA/"data/r2egym/horizon/rl/validate_hydra_args.py", cfg], cwd="/tmp")
        call(["bash", "-n", arm/"sbatch"/(name+"_rl.sbatch")])
    manifest = dict(r2egym_train=len(r_train), r2egym_heldout=len(r_hold), tmax_train=len(t_train),
        tmax_heldout=len(t_hold), attempts_per_task_per_model=8, attempts_per_model=len(train)*8,
        total_attempts=len(train)*16, models={k:str(v) for k,v in ARMS.items()},
        tmax_historical_gate=str(passed_file), tmax_current_pilot=pilot,
        task_ids_sha256=hashlib.sha256((ROOT/"train.txt").read_bytes()).hexdigest(),
        purpose="Pre-RL screen only, zero training epochs. Held-out tree is not screened.",
        selection="No model-dependent selection before both screens. Common mixed-success union after eight scored attempts per model. Unknown infra outcomes remain unknown.",
        source_mix="Native task counts, identical for both models. A training mixture weight has not yet been frozen.",
        maximum_nodes=24, maximum_reserved_node_hours=288, runtime="existing FSDP2 a03b2773",
        benchmark_overlap="Historical TMax strict filter excluded TB-Hard and checked TB2.1/CalibForge instructions. Re-audit other eval panels before RL training.")
    (ROOT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    status("PROBES_PREPARED")
    if not args.launch:
        return
    (ROOT/"bin").mkdir(exist_ok=True)
    guard = ROOT/"bin/sbatch"
    if not guard.exists():
        guard.symlink_to(S/"experiments/kimi-only-fp32-capacity2-20261025/bin/sbatch")
    env = dict(os.environ, FP32_REPLAY_ROOT=str(ROOT), OMP_NUM_THREADS="1")
    env["PATH"] = str(ROOT/"bin")+":"+env["PATH"]
    env.pop("DCFT_PRIVATE", None)
    # Two arms x448 plus current started count must fit before either job is submitted.
    count = subprocess.check_output([str(PYTHON), str(OTA/"data/r2egym/horizon/rl/sandbox_count.py")], text=True)
    import re
    m = re.search(r"'started': (\d+)", count)
    assert m and int(m.group(1)) + 896 <= 1200, count
    for name in ARMS:
        arm = ROOT/name
        if (arm/"jobs.txt").exists():
            continue
        env["TUNNEL_PORTS"] = "18230,18231" if "base" in name else "18232,18233"
        with (ROOT/"logs"/(name+"-launch.log")).open("w") as f:
            call(["bash", OTA/"data/r2egym/horizon/rl/launch_arm.sh", arm], cwd=RUNTIME, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(name, "submitted", (arm/"jobs.txt").read_text().strip(), flush=True)
    status("BOTH_PROBES_SUBMITTED")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        status("PROBE_SETUP_FAILED: " + repr(exc))
        raise
