#!/usr/bin/env python3
"""make_tt_wave.py — shard a task tree and generate Snowball pass@8 probe jobs under the 64k parity contract
(the `snowball_pool60k` recipe: 61,440 in / 4,096 out, max_model_len 65,536, summarize off, eval 1,800 s, conc 256,
8 nodes = 4 policy + 4 engines, --parity), pointed at a bridge of our choosing.

Runs on the Jupiter login node (python3.9-safe). Adapted by diff from code/snowball/make_residual_wave.py:
  - takes the task tree + an optional allowlist instead of pass@8 tables;
  - `--bridge-url` inserts `export APPTAINER_BRIDGE_URL=<url>` after the DCFT_RL_ENV line (the dotenv sets 9922;
    the generator has no bridge flag; the swesmith_sample6 probe did the same by hand);
  - `--verifier-timeout` (default 600) replaces the template's 120 s (sympy suites run minutes; pool_trace_synthesis
    already raised the RL arms to 600).
Everything else is the residual-wave code path: eval_batch_size = ceil(n/4) (one batch per fan-out coordinator),
HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC exported, pool_launch.sh / pool_watch.sh for submit + table + scancel.

Usage:
  make_tt_wave.py --tasks /e/fscratch/reformo/lee27/tasks/r2egym-tt-raw --allow allowlist.txt --prefix tt60k \
      --shards 6 --bridge-url http://10.128.1.2:9924 [--conc 256 --nodes 8 --engines 4 --wall 10:00:00]
"""
import argparse, json, os, shutil, subprocess, sys
EXP = "/e/fscratch/reformo/lee27/experiments"; TASKS = "/e/fscratch/reformo/lee27/tasks"; C = "/e/project1/transfernetx/lee27/code/snowball"
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True); ap.add_argument("--allow", default=None, help="file of task dir names to keep (default: all)")
ap.add_argument("--prefix", required=True); ap.add_argument("--shards", type=int, default=6); ap.add_argument("--conc", type=int, default=256)
ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--engines", type=int, default=4); ap.add_argument("--wall", default="10:00:00")
ap.add_argument("--bridge-margin", type=int, default=600); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--bridge-url", required=True); ap.add_argument("--verifier-timeout", type=int, default=600)
a = ap.parse_args()
alltasks = sorted(d for d in os.listdir(a.tasks) if os.path.isdir(os.path.join(a.tasks, d)))
if a.allow:
    keep = {l.strip() for l in open(a.allow) if l.strip() and not l.startswith("#")}
    tasks = [t for t in alltasks if t in keep]
    print("tasks in tree %d, allowlisted %d, kept %d" % (len(alltasks), len(keep), len(tasks)))
else:
    tasks = alltasks; print("tasks %d (no allowlist)" % len(tasks))
shards = [tasks[i::a.shards] for i in range(a.shards)]
for i, ts in enumerate(shards):
    d = "%s/%s-s%d" % (TASKS, a.prefix, i)
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
    for t in ts: shutil.copytree(os.path.join(a.tasks, t), os.path.join(d, t))
    name = "%s_s%d" % (a.prefix, i)
    if subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "lee27"), "-n", name, "-o", "%i"], capture_output=True, text=True).stdout.strip():
        sys.exit("refusing: a job named %s is in squeue (make_snowball_probe.py rm-rf's the run dir)" % name)
    gen = [sys.executable, C + "/make_snowball_probe.py", "--name", name, "--val-dir", d, "--k", str(a.k), "--conc", str(a.conc),
           "--max-in", "61440", "--max-out", "4096", "--max-model-len", "65536", "--nodes", str(a.nodes), "--engines", str(a.engines),
           "--parity", "--eval-timeout", "1800", "--wall", a.wall]
    subprocess.check_output(gen, text=True)
    cp = "%s/%s/configs/%s_rl_config.json" % (EXP, name, name); c = json.load(open(cp))
    ebs = -(-len(ts) // 4)
    args = [x for x in c["skyrl_hydra_args"] if not x.startswith(("trainer.eval_batch_size=", "++terminal_bench_config.harbor.verifier_override_timeout_sec="))]
    args += ["trainer.eval_batch_size=%d" % ebs, "++terminal_bench_config.harbor.verifier_override_timeout_sec=%d" % a.verifier_timeout]
    c["skyrl_hydra_args"] = args; json.dump(c, open(cp, "w"), indent=2)
    sp = "%s/%s/sbatch/%s_rl.sbatch" % (EXP, name, name); s = open(sp).read()
    ins = "export NCCL_PXN_DISABLE=1\nexport HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=%d\nexport APPTAINER_BRIDGE_URL=%s\n" % (a.bridge_margin, a.bridge_url)
    assert s.count("export NCCL_PXN_DISABLE=1\n") == 1
    s = s.replace("export NCCL_PXN_DISABLE=1\n", ins, 1)
    assert ("APPTAINER_BRIDGE_URL=%s" % a.bridge_url) in s and ("HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=%d" % a.bridge_margin) in s
    open(sp, "w").write(s)
    print("%s: %d tasks -> %s; eval_batch_size=%d; verifier %ds; bridge %s; sbatch %s" % (name, len(ts), d, ebs, a.verifier_timeout, a.bridge_url, sp))
open("%s/%s_tasks.txt" % (EXP, a.prefix), "w").write("\n".join(tasks) + "\n")
print('launch: tmux new -d -s pool_launch_%s "bash %s/pool_launch.sh %s %d 150" && tmux new -d -s pool_watch_%s "bash %s/pool_watch.sh %s %d"'
      % (a.prefix, C, a.prefix, a.shards, a.prefix, C, a.prefix, a.shards))
