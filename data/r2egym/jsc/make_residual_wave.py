#!/usr/bin/env python3
"""make_residual_wave.py --tables <pass8_pass8_table.csv...> --all-tasks <train task dir> --prefix snowball_pool60k_r1 [--shards 8]
   [--conc 256 --nodes 8 --engines 4 --wall 08:00:00 --bridge-margin 600] [--submit-cmd]
Residual pool wave: every task in <all-tasks> that is NOT fully sampled (8 scored attempts) in the given tables is
re-run — includes tasks with no artifacts at all. Builds shard task dirs (copies), generates the probe configs with the
parity/64k contract, patches trainer.eval_batch_size to the whole shard (no batch drain) and exports the bridge exec
margin. Prints the pool_launch/pool_watch commands."""
import argparse, csv, json, os, re, shutil, subprocess, sys
EXP = "/e/fscratch/reformo/lee27/experiments"; TASKS = "/e/fscratch/reformo/lee27/tasks"; C = "/e/project1/transfernetx/lee27/code/snowball"
ap = argparse.ArgumentParser(); ap.add_argument("--tables", nargs="+", required=True); ap.add_argument("--all-tasks", default=f"{TASKS}/r2egym-raw-v3-train")
ap.add_argument("--prefix", required=True); ap.add_argument("--shards", type=int, default=8); ap.add_argument("--conc", type=int, default=256)
ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--engines", type=int, default=4); ap.add_argument("--wall", default="08:00:00")
ap.add_argument("--bridge-margin", type=int, default=600); ap.add_argument("--k", type=int, default=8); a = ap.parse_args()
full = set()
for t in a.tables:
    for r in csv.DictReader(open(t)):
        if r["full"] == "True": full.add(r["task"])
alltasks = sorted(d for d in os.listdir(a.all_tasks) if os.path.isdir(f"{a.all_tasks}/{d}"))
resid = [t for t in alltasks if t not in full]
print(f"tasks total {len(alltasks)}, fully sampled {len(full)}, residual {len(resid)}")
shards = [resid[i::a.shards] for i in range(a.shards)]
for i, tasks in enumerate(shards):
    d = f"{TASKS}/{a.prefix}-s{i}"
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
    for t in tasks: shutil.copytree(f"{a.all_tasks}/{t}", f"{d}/{t}")
    name = f"{a.prefix}_s{i}"
    gen = [sys.executable, f"{C}/make_snowball_probe.py", "--name", name, "--val-dir", d, "--k", str(a.k), "--conc", str(a.conc),
           "--max-in", "61440", "--max-out", "4096", "--max-model-len", "65536", "--nodes", str(a.nodes), "--engines", str(a.engines),
           "--parity", "--eval-timeout", "1800", "--wall", a.wall]
    subprocess.check_output(gen, text=True)
    cp = f"{EXP}/{name}/configs/{name}_rl_config.json"; c = json.load(open(cp))
    args = [x for x in c["skyrl_hydra_args"] if not x.startswith("trainer.eval_batch_size=")] + [f"trainer.eval_batch_size={-(-len(tasks) // 4)}"]  # 4 batches = one per fanout coordinator (each holds n_concurrent/4 slots); one giant batch would run on a single coordinator
    c["skyrl_hydra_args"] = args; json.dump(c, open(cp, "w"), indent=2)
    sp = f"{EXP}/{name}/sbatch/{name}_rl.sbatch"; s = open(sp).read()
    s = s.replace("export NCCL_PXN_DISABLE=1\n", f"export NCCL_PXN_DISABLE=1\nexport HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC={a.bridge_margin}\n", 1)
    assert f"HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC={a.bridge_margin}" in s; open(sp, "w").write(s)
    print(f"{name}: {len(tasks)} tasks -> {d}; eval_batch_size={-(-len(tasks)//4)}; sbatch {sp}")
open(f"{EXP}/{a.prefix}_residual_tasks.txt", "w").write("\n".join(resid) + "\n")
print(f"launch: tmux new -d -s pool_launch_r1 \"bash {C}/pool_launch.sh {a.prefix} {a.shards} 150\" && tmux new -d -s pool_watch_r1 \"bash {C}/pool_watch.sh {a.prefix} {a.shards}\"")
