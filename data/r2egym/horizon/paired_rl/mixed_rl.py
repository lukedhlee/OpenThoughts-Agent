#!/usr/bin/env python3
"""Freeze each arm's own learnable band from its mixed pre-RL probe, render the SFT and base GRPO arms, launch them.

    mixed_rl.py band --arm v7|base    independent readout of that arm's probe, its learnable band, natural-mix tree
                                      (refuses to overwrite a frozen band or to read a probe still in the queue)
    mixed_rl.py prepare --arm v7|base|vk|rk [--seats N]  render one arm (vk/rk = CF+Kimi, on band_v7) with the shared settings and verify the generated config
    mixed_rl.py launch --arm v7|base  preflight + submit one arm through the booking guard (--suffix r1: retry dir)
    mixed_rl.py heldout --name N --model DIR --ports A,B [--launch]   held-out K=8 probe (start model or RL export)
    mixed_rl.py export --arm X --step N                               keep step N's model shards (hardlinks) + submit HF consolidation
    mixed_rl.py heldout-report N [N ...]                              per-task held-out outcomes and pass@1

Band rule (Luke 2026-10-26): no common band, SFT RL and base RL are independent. Each arm trains on the tasks its own
start model solved 1-6 of 8 times, among tasks it scored 8 of 8 (first 8 raw outcomes in completion order, invalids kept
as invalid, never as zero), minus eval-overlap exclusions. Held-out tasks are never probed or trained on. Both arms use the
same recipe and layout. Sources mix in their natural task proportion (uniform over band tasks).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import collections
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

S = Path("/scratch/11584/lukedhlee")
MIXED = S / "experiments/paired-sft-rl-20261026/mixed"
ROOT = MIXED / "rl"
OTA = Path.home() / "snowball/ota-rl"
HZ = OTA / "data/r2egym/horizon"
RUNTIME = Path.home() / "snowball/ota-rl-runtime"
PYTHON = Path.home() / "snowball/envs/snowball/bin/python"
GUARD = S / "experiments/kimi-only-fp32-capacity2-20261025/bin/sbatch"
DRAFT = S / "hf_hub/models--laion--snowball-64k-eagle3-draft-r2egym/snapshots/4bdb47c08e5b5190bea3c7a93c3e14470230e469"
PROBES = {"base": "probe_mix_base_1026", "v7": "probe_mix_v7_1026"}
ARMS = {  # launch order: v7, then vk and rk, then base
    "v7": ("rl_mix_v7_1026", S / "experiments/cf2-epochs57-20261022/training/fdv/export-step959-hf-bf16", "18234,18235"),
    "base": ("rl_mix_base_1026", S / "models/grug-datakit-sft-20260921", "18236,18237"),
    # Luke 2026-10-26: CF + Kimi SFT checkpoints, no probes of their own, trained on the V7 arm's frozen band
    "vk": ("rl_mix_vkimi_1026", S / "experiments/vanilla-kimi-fp32-capacity2-20261021/training/vanilla_kimi/export-step630-hf-bf16", "18238,18239"),
    "rk": ("rl_mix_rkimi_1026", S / "experiments/relay-kimi-fp32-capacity2-20261021/training/relay_kimi/export-step723-hf-bf16", "18240,18241"),
}
BAND_OF = {"v7": "v7", "base": "base", "vk": "v7", "rk": "v7"}
# Luke 2026-10-26: the three CF arms run concurrently at equal seats; base runs alone after its probe at 448.
SEATS = {"v7": 320, "vk": 320, "rk": 320, "base": 448}
MAX_ORG = 1500  # Luke 2026-10-26: the org takes 1,500 started sandboxes for this plan (launch_arm.sh default 1,200)
BAND = (1, 6)  # Luke 2026-10-26: learnable = 1-6 solved of 8 (7/8 tasks excluded as near-solved)
COPIES = 16  # build_band_tree.sh default: one dataloader epoch outlasts 30 steps x 64 groups
ARM_FLAGS = ["--recipe", "arm", "--policy-nodes", "4", "--engines", "8", "--batch", "64",
             "--coords", "16", "--steps", "30", "--wall", "08:00:00", "--ckpt-interval", "6", "--hf-save-interval", "6"]
# The generated config must carry these (the recipe as the brief states it); anything else drifting is a refusal.
EXPECT = {
    "trainer.algorithm.advantage_estimator": "rloo_n", "trainer.algorithm.use_kl_loss": "false",
    "trainer.algorithm.kl_loss_coef": "0.0", "trainer.algorithm.eps_clip_low": "0.2", "trainer.algorithm.eps_clip_high": "0.2",
    "trainer.algorithm.use_tis": "true", "trainer.algorithm.tis_imp_ratio_cap": "2.0",
    "trainer.policy.optimizer_config.lr": "5e-7", "trainer.policy.optimizer_config.num_warmup_steps": "3",
    "trainer.fully_async.max_staleness_steps": "2", "generator.n_samples_per_prompt": "8",
    "generator.sampling_params.temperature": "1.0", "generator.sampling_params.top_p": "1.0",
    "generator.sampling_params.top_k": "-1", "generator.engine_init_kwargs.max_model_len": "65536",
    "trainer.policy.grug_query_bias_update_mode": "frozen", "trainer.max_steps": "30", "trainer.train_batch_size": "64",
    "trainer.ckpt_interval": "6", "trainer.hf_save_interval": "6", "trainer.eval_before_train": "false",
    "terminal_bench_config.harbor.override_storage_mb": "10240",
    "trainer.seed": "42",
}


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hydra(cfg):
    out = {}
    for a in cfg["skyrl_hydra_args"]:
        k, _, v = a.lstrip("+").partition("=")
        out[k] = v
    return out


def outcomes(run, tasks, screen):
    """Per task: first 8 raw outcomes in completion order from the probe's single eval session."""
    masks = screen.masks_of(run)
    sessions = sorted(p.name for p in (run / "trials/eval_sessions").iterdir())
    assert sessions == [run.name + "_eval_step0"], f"{run.name}: unexpected eval sessions {sessions}"
    paths = collections.defaultdict(list)
    for p in (run / "trials/eval_sessions" / sessions[0]).glob("*/result.json"):
        task = re.sub(r"__r\d+$", "", p.parent.name.rsplit("__", 1)[0])
        assert task in tasks, f"{run.name}: {task} is not in the probed task list"
        paths[task].append(p)
    rows = {}
    for task in tasks:
        ps = sorted(paths.get(task, []), key=lambda p: p.stat().st_mtime)
        scores, inval = [], collections.Counter()
        for p in ps[:8]:
            _, reward, ex = screen.read_outcome(p)
            if reward in (0, 1) and ex not in masks:
                scores.append(int(reward))
            else:
                inval[ex or "unscored"] += 1
        rows[task] = dict(raw=min(8, len(ps)), extra=max(0, len(ps) - 8), scored=len(scores), solved=sum(scores),
                          invalid=sum(inval.values()), exceptions=dict(inval))
    return rows


def build_tree(tasks, src, dst, copies):
    """build_band_tree.sh semantics (<task>, <task>__r1..__r{copies-1}, each a symlink to the resolved source task dir),
    with the NFS metadata calls run 32 at a time: the serial script took ~14 min for 4,500 links under load."""
    assert not dst.exists(), dst
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.mkdir(parents=True)
    def one(t):
        tgt = os.path.realpath(src / t)
        assert os.path.isfile(os.path.join(tgt, "task.toml")), f"no task {t} under {src}"
        for k in range(copies):
            os.symlink(tgt, tmp / (t if k == 0 else f"{t}__r{k}"))
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(one, tasks))
    tmp.rename(dst)
    log(f"{dst}: {len(tasks)} tasks x {copies} = {len(os.listdir(dst))} entries")


def band(args):
    """One arm's own learnable band (Luke 10-26: no common band, the arms are independent)."""
    out = Path(args.out) if args.dry_run else ROOT / f"band_{args.arm}"
    if not args.dry_run and (out / "FROZEN").exists():
        raise SystemExit(f"band already frozen: {(out / 'FROZEN').read_text()}")
    out.mkdir(parents=True, exist_ok=True)
    screen = module_at("screen", HZ / "rl/screen_report.py")
    train = (MIXED / "train.txt").read_text().split()
    held = set((MIXED / "heldout.txt").read_text().split())
    assert len(train) == 3832 and not held & set(train)
    source = {r["task"]: r["source"] for r in csv.DictReader((MIXED / "task_map.tsv").open(), delimiter="\t")}
    exclude_file = MIXED / "overlap_audit/exclude.txt"
    assert exclude_file.exists() or args.dry_run, "run the eval-overlap audit first"
    exclude = set(exclude_file.read_text().split()) if exclude_file.exists() else set()
    probe = PROBES[args.arm]
    if not args.dry_run:
        job = (MIXED / probe / "jobs.txt").read_text().split()[-1]
        assert not subprocess.check_output(["squeue", "-h", "-j", job], text=True).strip(), f"{probe} ({job}) is still in the queue"
    rows = outcomes(MIXED / probe, train, screen)
    complete = [t for t in train if rows[t]["scored"] == 8]
    learnable = sorted(t for t in complete if BAND[0] <= rows[t]["solved"] <= BAND[1])
    chosen = [t for t in learnable if t not in exclude]
    by = lambda ts: dict(collections.Counter(source[t] for t in ts))
    summary = dict(
        arm=args.arm, probe=probe,
        rule=f"this model's own tasks with {BAND[0]}-{BAND[1]} solved of 8, among tasks it scored 8 of 8 (first 8 raw outcomes in "
             "completion order, invalids stay invalid), minus the eval-overlap exclusions; held-out never probed",
        tasks=by(train), fully_scored=by(complete), learnable=by(learnable), excluded_overlap=sorted(set(learnable) & exclude),
        band=by(chosen), all0=by([t for t in complete if rows[t]["solved"] == 0]),
        all8=by([t for t in complete if rows[t]["solved"] == 8]),
        seven_of_8=by([t for t in complete if rows[t]["solved"] == 7]),
        per_source={s: dict(fully_scored=sum(source[t] == s for t in complete),
                            pass1=sum(rows[t]["solved"] for t in complete if source[t] == s) /
                            max(1, 8 * sum(source[t] == s for t in complete)),
                            pass8=sum(rows[t]["solved"] > 0 for t in complete if source[t] == s) /
                            max(1, sum(source[t] == s for t in complete)),
                            invalid_attempts=sum(rows[t]["invalid"] for t in train if source[t] == s),
                            missing_attempts=sum(8 - rows[t]["raw"] for t in train if source[t] == s),
                            tasks_not_fully_scored=sum(source[t] == s and rows[t]["scored"] < 8 for t in train),
                            extra_outcomes=sum(rows[t]["extra"] for t in train if source[t] == s))
                    for s in ("r2egym", "tmax")})
    with (out / f"{probe}.per_task.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "source", "raw", "scored", "solved", "invalid", "extra", "exceptions"])
        for t in train:
            r = rows[t]
            w.writerow([t, source[t], r["raw"], r["scored"], r["solved"], r["invalid"], r["extra"], json.dumps(r["exceptions"])])
    # Natural mix (Luke 2026-10-27): every band task gets the same copy count, so each source's share is its task share.
    per = {s: [t for t in chosen if source[t] == s] for s in ("r2egym", "tmax")}
    viable = {s: ts for s, ts in per.items() if ts}
    copies = {s: COPIES for s in viable}
    entries = {s: len(viable[s]) * copies[s] for s in viable}
    summary.update(copies=copies, entries=entries,
                   entry_share={s: entries[s] / sum(entries.values()) for s in entries},
                   missing_source=[s for s in per if not per[s]])
    brief = {k: summary[k] for k in ("fully_scored", "learnable", "band", "copies", "entry_share", "missing_source")}
    if args.dry_run:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log("DRY_RUN", json.dumps(brief))
        return
    tree = out / "tree"
    assert not tree.exists(), tree
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    for s, ts in viable.items():
        (out / f"band_{s}.txt").write_text("\n".join(ts) + "\n")
        build_tree(ts, MIXED / "train_tree", parts / s, copies[s])
    tree.mkdir()
    for s in viable:
        for e in (parts / s).iterdir():
            e.rename(tree / e.name)
        (parts / s).rmdir()
    parts.rmdir()
    names = sorted(os.listdir(tree))
    assert len(names) == sum(entries.values())
    assert not {re.sub(r"__r\d+$", "", n) for n in names} & held
    (out / "band.txt").write_text("\n".join(chosen) + "\n")
    summary["band_sha256"] = hashlib.sha256((out / "band.txt").read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "FROZEN").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} band {len(chosen)} tasks sha256 {summary['band_sha256']}\n")
    log("BAND_FROZEN", json.dumps(brief))


def model_metadata(path):
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        assert (path / shard).stat().st_size > 1000, shard
    return {"path": str(path), "config": config, "weight_keys": sorted(index["weight_map"]),
            "tokenizer_sha256": hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()}


def git_sha(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def arm_name(args):
    return ARMS[args.arm][0] + (f"_{args.suffix}" if args.suffix else "")


def prepare(args):
    """Render one arm. A retry gets its own run dir (--suffix r1) so partial attempts are never pooled."""
    args.seats = args.seats or SEATS[args.arm]
    band_dir = ROOT / f"band_{BAND_OF[args.arm]}"
    frozen = band_dir / "FROZEN"
    assert frozen.exists(), f"freeze {band_dir.name} first"
    tree = band_dir / "tree"
    _, model, ports = ARMS[args.arm]
    name = arm_name(args)
    models = {k: model_metadata(m) for k, (_, m, _) in ARMS.items() if k in ("base", args.arm)}
    assert models["base"]["weight_keys"] == models[args.arm]["weight_keys"], "tensor names differ"
    for key in ["model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size"]:
        assert models["base"]["config"].get(key) == models[args.arm]["config"].get(key), key
    assert models["base"]["tokenizer_sha256"] == models[args.arm]["tokenizer_sha256"], "tokenizer differs"
    arm = ROOT / name
    if not (arm / "jobs.txt").exists():  # never re-render a submitted arm
        subprocess.run([sys.executable, HZ / "rl/make_arm.py", "--name", name, "--model", model, "--tree", tree,
                        "--val-tree", MIXED / "heldout_tree", *ARM_FLAGS, "--seats", str(args.seats), "--draft", DRAFT, "--out", ROOT,
                        "--project", "snowball-strategy-mixed-rl-20261026",
                        "--set", "terminal_bench_config.harbor.override_storage_mb=10240"], check=True)
    cfg_path = arm / "configs" / f"{name}_rl_config.json"
    subprocess.run([PYTHON, HZ / "rl/validate_hydra_args.py", cfg_path], cwd="/tmp", check=True,
                   env=dict(os.environ, OMP_NUM_THREADS="1"))
    subprocess.run(["bash", "-n", arm / "sbatch" / f"{name}_rl.sbatch"], check=True)
    h = hydra(json.loads(cfg_path.read_text()))
    want_all = dict(EXPECT, **{"terminal_bench_config.harbor.n_concurrent_trials": str(args.seats)})
    bad = {key: (h.get(key), want) for key, want in want_all.items() if h.get(key) != want}
    assert not bad, f"{name}: recipe drift {bad}"
    assert h["trainer.epochs"] != "0" and h["trainer.eval_before_train"] == "false", "probe setting in a training config"
    assert str(DRAFT) in h["generator.engine_init_kwargs.speculative_config"] and "num_speculative_tokens:3" in \
        h["generator.engine_init_kwargs.speculative_config"]
    assert h["trainer.policy.model.path"] == str(model) and json.loads(h["data.train_data"]) == [str(tree.resolve())]
    diff = {}
    for k, (n, _, _) in ARMS.items():  # arms may differ only in names, paths, their band tree and seats (throughput)
        o_path = ROOT / n / "configs" / f"{n}_rl_config.json"
        if k == args.arm or not o_path.exists():
            continue
        o = hydra(json.loads(o_path.read_text()))
        diff[k] = sorted(x for x in set(h) | set(o) if h.get(x) != o.get(x))
        allowed = {"trainer.run_name", "generator.engine_init_kwargs.served_model_name", "terminal_bench_config.trials_dir",
                   "trainer.ckpt_path", "trainer.export_path", "trainer.policy.model.path", "trainer.ref.model.path",
                   "data.train_data", "terminal_bench_config.harbor.n_concurrent_trials"}
        assert set(diff[k]) <= allowed, f"{args.arm} vs {k} differ beyond names/paths/band/seats: {set(diff[k]) - allowed}"
    manifest = dict(
        arm=args.arm, name=name, model=str(model), tunnel_ports=ports, band=str(tree.parent / "summary.json"),
        band_frozen=frozen.read_text().strip(), tokenizer_sha256=models[args.arm]["tokenizer_sha256"],
        make_arm_flags=ARM_FLAGS, seats=args.seats, band_from=BAND_OF[args.arm], draft=str(DRAFT), diff_vs_other_arm=diff, recipe_checked=EXPECT,
        shas=dict(ota_scripts=git_sha(OTA), runtime=git_sha(RUNTIME), marinskyrl=git_sha(Path.home() / "snowball/marinskyrl-marin"),
                  harbor_overlay=git_sha(Path.home() / "snowball/harbor-rl"), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        heldout_protocol="separate K=8 probe on mixed/heldout_tree for the start model and for steps 12 and 30",
        budget="12 nodes x 8 h = 96 requested node-h per arm; one same-size SFT retry only for a diagnosed infra failure")
    (arm / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("PREPARE_OK", name, json.dumps(manifest["shas"]))


def launch(args):
    name = arm_name(args)
    ports = ARMS[args.arm][2]
    arm = ROOT / name
    assert (arm / "manifest.json").exists(), "prepare first"
    if (arm / "jobs.txt").exists():
        raise SystemExit(f"{name} already submitted: {(arm / 'jobs.txt').read_text().split()} (a retry uses --suffix r1)")
    # Luke 2026-10-26 launch order: V7 first; the CF+Kimi arms once V7's startup is healthy; base after its probe
    before = {"v7": [], "vk": ["v7"], "rk": ["v7"], "base": ["v7"]}[args.arm]
    for k in before:
        assert (ROOT / ARMS[k][0] / "jobs.txt").exists(), f"{ARMS[k][0]} launches before {args.arm}"
    env = guarded_env(ROOT, ports)
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / f"{name}-launch-{time.strftime('%Y%m%dT%H%M%S')}.log").open("w") as f:
        subprocess.run(["bash", HZ / "rl/launch_arm.sh", arm], cwd=RUNTIME, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    log(name, "submitted", (arm / "jobs.txt").read_text().split()[-1])


def guarded_env(root, ports):
    (root / "bin").mkdir(parents=True, exist_ok=True)
    if not (root / "bin/sbatch").exists():
        (root / "bin/sbatch").symlink_to(GUARD)
    env = dict(os.environ, FP32_REPLAY_ROOT=str(root), OMP_NUM_THREADS="1", TUNNEL_PORTS=ports, MAX_ORG=str(MAX_ORG))
    env["PATH"] = str(root / "bin") + ":" + env["PATH"]
    env.pop("DCFT_PRIVATE", None)
    return env


def heldout(args):
    """Held-out probe: the same eval-only protocol as the pre-RL screens (K=8, temperature 1) on mixed/heldout_tree,
    for a start model or an exported RL checkpoint. Independent of band selection."""
    root = ROOT / "heldout"
    root.mkdir(parents=True, exist_ok=True)
    run = root / args.name
    model = Path(args.model)
    assert (model / "config.json").is_file(), model
    if not (run / "jobs.txt").exists():
        subprocess.run([sys.executable, HZ / "rl/make_arm.py", "--name", args.name, "--recipe", "arm", "--model", model,
                        "--tree", MIXED / "heldout_tree", "--policy-nodes", "4", "--engines", "8", "--batch", "64",
                        "--seats", str(args.seats), "--coords", "16", "--steps", "1", "--probe", "8", "--wall", args.wall,
                        "--draft", DRAFT, "--out", root, "--project", "snowball-strategy-mixed-rl-20261026",
                        "--set", "terminal_bench_config.harbor.override_storage_mb=10240"], check=True)
    cfg = run / "configs" / f"{args.name}_rl_config.json"
    subprocess.run([PYTHON, HZ / "rl/validate_hydra_args.py", cfg], cwd="/tmp", check=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    h = hydra(json.loads(cfg.read_text()))
    assert h["trainer.epochs"] == "0" and h["generator.eval_n_samples_per_prompt"] == "8" and \
        h["generator.sampling_params.temperature"] == "1.0" and h["generator.sampling_params.top_p"] == "1.0"
    if not args.launch:
        log("HELDOUT_PREPARED", run)
        return
    if (run / "jobs.txt").exists():
        raise SystemExit(f"{args.name} already submitted")
    with (root / f"{args.name}-launch-{time.strftime('%Y%m%dT%H%M%S')}.log").open("w") as f:
        subprocess.run(["bash", HZ / "rl/launch_arm.sh", run], cwd=RUNTIME, env=guarded_env(root, args.ports),
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    log(args.name, "submitted", (run / "jobs.txt").read_text().split()[-1])


def heldout_report(args):
    screen = module_at("screen", HZ / "rl/screen_report.py")
    held = (MIXED / "heldout.txt").read_text().split()
    source = {r["task"]: r["source"] for r in csv.DictReader((MIXED / "task_map.tsv").open(), delimiter="\t")}
    out = {}
    for name in args.names:
        rows = outcomes(ROOT / "heldout" / name, held, screen)
        complete = [t for t in held if rows[t]["scored"] == 8]
        out[name] = {s: dict(tasks=sum(source[t] == s for t in held), fully_scored=sum(source[t] == s for t in complete),
                             pass1=sum(rows[t]["solved"] for t in complete if source[t] == s) / max(1, 8 * sum(source[t] == s for t in complete)),
                             invalid=sum(rows[t]["invalid"] for t in held if source[t] == s),
                             missing=sum(8 - rows[t]["raw"] for t in held if source[t] == s)) for s in ("r2egym", "tmax")}
        with (ROOT / "heldout" / f"{name}.per_task.tsv").open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["task", "source", "raw", "scored", "solved", "invalid"])
            for t in held:
                w.writerow([t, source[t], rows[t]["raw"], rows[t]["scored"], rows[t]["solved"], rows[t]["invalid"]])
    print(json.dumps(out, indent=1))


def export(args):
    """Keep one RL checkpoint's model shards (hardlinks, so max_ckpts_to_keep=2 rotation cannot delete them; optimizer
    state is not kept) and submit the offline FSDP2 -> HF consolidation (one CPU node) through the booking guard.
    In-run HF exports need hf_hub_repo_id, which these arms leave unset on purpose (no uploads)."""
    run, model, _ = ARMS[args.arm]
    run = run + (f"_{args.suffix}" if args.suffix else "")
    ckpts = ROOT / run / run / "checkpoints"
    src = ckpts / f"global_step_{args.step}" / "policy"
    keep = ROOT / run / "keep" / f"global_step_{args.step}" / "policy"
    if not keep.exists():
        assert int((ckpts / "latest_ckpt_global_step.txt").read_text()) >= args.step, "checkpoint not written yet"
        tmp = keep.with_name("policy.partial")
        subprocess.run(["rm", "-rf", tmp], check=True)
        tmp.mkdir(parents=True)
        for f in src.iterdir():
            if f.name.startswith("optim_"):
                continue
            subprocess.run(["cp", "-al", f, tmp / f.name], check=True)
        n = len([f for f in tmp.iterdir() if f.name.startswith("model_world_size_")])
        assert n == 16, f"expected 16 model shards, found {n}"
        tmp.rename(keep)
        log("KEPT", keep)
    dst = ROOT / "exports" / f"{run}_step{args.step}" / "model"
    if (dst / "model.safetensors.index.json").exists():
        log("EXPORT_EXISTS", dst)
        return
    (ROOT / "exports" / "logs").mkdir(parents=True, exist_ok=True)
    env = guarded_env(ROOT, "")
    env.update(RUN=run, STEP=str(args.step), BASE=str(model), SRC=str(keep), DST=str(dst))
    out = subprocess.run(["sbatch", "--export=ALL", "-J", f"export_{run}_s{args.step}", "-t", args.wall,
                          "-o", str(ROOT / "exports" / "logs" / "%x_%j.out"), HZ / "rl/export_hf.sbatch"],
                         env=env, capture_output=True, text=True, check=True)
    log("EXPORT_SUBMITTED", run, args.step, out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("band")
    b.add_argument("--arm", choices=list(ARMS), required=True)
    b.add_argument("--dry-run", action="store_true", help="write summary + per-task TSVs to --out only: no tree, no freeze")
    b.add_argument("--out", default="")
    for cmd in ("prepare", "launch"):
        x = sub.add_parser(cmd)
        x.add_argument("--arm", choices=list(ARMS), required=True)
        x.add_argument("--suffix", default="", help="r1 for the one allowed same-size retry (record why in STATUS)")
        x.add_argument("--seats", type=int, default=0, help="concurrent sandboxes (throughput only); default SEATS[arm]")
    h = sub.add_parser("heldout")
    h.add_argument("--name", required=True)
    h.add_argument("--model", required=True)
    h.add_argument("--seats", type=int, default=192)
    h.add_argument("--wall", default="03:00:00")
    h.add_argument("--ports", required=True)
    h.add_argument("--launch", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--arm", choices=list(ARMS), required=True)
    e.add_argument("--step", type=int, required=True)
    e.add_argument("--suffix", default="")
    e.add_argument("--wall", default="02:00:00")
    r = sub.add_parser("heldout-report")
    r.add_argument("names", nargs="+")
    a = ap.parse_args()
    {"band": band, "prepare": prepare, "launch": launch, "heldout": heldout, "export": export, "heldout-report": heldout_report}[a.cmd](a)


if __name__ == "__main__":
    main()
