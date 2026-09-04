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

  - `--daytona`: environment backend daytona instead of the apptainer bridge (same recipe otherwise). Edits the generated
    files only (found read-only 2026-09-04): hydra `harbor.environment_type=daytona` + `"harbor_env": "daytona"`; the sbatch
    gets the preset-SOCKS preamble so the launcher's `_setup_proxy` block builds a proxychains conf pointing at the
    microsocks on jpbl-s01-02 (10.128.1.2:7011; JSC TOTP blocks compute->login `ssh -D`, the currease probe proved this
    route) instead of skipping; `PYTHONPATH=<harbor-hook>/src` puts the setup-files-hook harbor (branch
    lukedhlee/terminus2-think-parity-bridgewait-setuphook) in front of the venv's copy for THIS job only (the shared venv
    is untouched); Daytona infra errors (sandbox start timeout, conflict, rate limit) are masked like bridge outages.
    DAYTONA_API_KEY comes from the template's sbatch (must be the org holding the prebuilt snapshots).

Usage:
  make_tt_wave.py --tasks /e/fscratch/reformo/lee27/tasks/r2egym-tt-raw --allow allowlist.txt --prefix tt60k \
      --shards 6 --bridge-url http://10.128.1.2:9924 [--conc 256 --nodes 8 --engines 4 --wall 10:00:00]
  make_tt_wave.py --tasks /e/fscratch/reformo/lee27/tasks/r2egym-tt-daytona --allow allowlist_daytona.txt --prefix ttd60k \
      --shards 6 --daytona [--socks-env /e/fscratch/reformo/lee27/keys/socks5_currease.env --harbor-src .../harbor-hook/src]
"""
import argparse, json, os, shutil, subprocess, sys
EXP = "/e/fscratch/reformo/lee27/experiments"; TASKS = "/e/fscratch/reformo/lee27/tasks"; C = "/e/project1/transfernetx/lee27/code/snowball"
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True); ap.add_argument("--allow", default=None, help="file of task dir names to keep (default: all)")
ap.add_argument("--prefix", required=True); ap.add_argument("--shards", type=int, default=6); ap.add_argument("--conc", type=int, default=256)
ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--engines", type=int, default=4); ap.add_argument("--wall", default="10:00:00")
ap.add_argument("--bridge-margin", type=int, default=600); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--bridge-url", default=None, help="apptainer bridge URL (required unless --daytona)"); ap.add_argument("--verifier-timeout", type=int, default=600)
ap.add_argument("--daytona", action="store_true", help="environment backend daytona (see module doc)")
ap.add_argument("--socks-env", default="/e/fscratch/reformo/lee27/keys/socks5_currease.env", help="--daytona: file exporting SOCKS_USER/SOCKS_PASS for the microsocks")
ap.add_argument("--socks-host", default="10.128.1.2"); ap.add_argument("--socks-port", default="7011")
ap.add_argument("--harbor-src", default="/e/project1/transfernetx/lee27/code/harbor-hook/src", help="--daytona: harbor checkout with the setup_files/setup.sh hook, prepended to PYTHONPATH")
ap.add_argument("--daytona-key-env", default="/e/fscratch/reformo/lee27/keys/daytona_eval.env", help="--daytona: file with DAYTONA_API_KEY for the org holding the prebuilt snapshots (the template's baked key was found invalid 2026-09-04); sourced AFTER the template's export")
ap.add_argument("--daytona-mask", default="EnvironmentStartTimeoutError,DaytonaConflictError,DaytonaRateLimitError,SandboxBuildFailedError", help="--daytona: exception names appended to harbor.mask_exceptions")
a = ap.parse_args()
if not a.daytona and not a.bridge_url: sys.exit("--bridge-url is required unless --daytona")
if a.daytona:
    for f in (a.socks_env, a.daytona_key_env, a.harbor_src + "/harbor/trial/trial.py"):
        if not os.path.exists(f): sys.exit("--daytona: missing %s" % f)
    if "_run_setup_script" not in open(a.harbor_src + "/harbor/trial/trial.py").read(): sys.exit("--daytona: %s has no setup_files hook" % a.harbor_src)
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
    cp = "%s/%s/configs/%s_rl_config.json" % (EXP, name, name)
    # MarinSkyRL schema after the 2026-09-03 upstream merge: the template still emits pre-merge keys (generator.vllm_stats_interval,
    # rollout.fanout.*); the other session's fix_merged_keys.py rewrites them (tt60k attempt 1: all 6 shards died in hydra on this)
    if os.path.exists(C + "/fix_merged_keys.py"): print(subprocess.check_output([sys.executable, C + "/fix_merged_keys.py", cp], text=True).strip())
    c = json.load(open(cp))
    ebs = -(-len(ts) // 4)
    args = [x for x in c["skyrl_hydra_args"] if not x.startswith(("trainer.eval_batch_size=", "++terminal_bench_config.harbor.verifier_override_timeout_sec="))]
    args += ["trainer.eval_batch_size=%d" % ebs, "++terminal_bench_config.harbor.verifier_override_timeout_sec=%d" % a.verifier_timeout]
    if a.daytona:
        env_key = "++terminal_bench_config.harbor.environment_type="
        assert sum(x.startswith(env_key) for x in args) == 1, "expected one environment_type hydra arg"
        args = [env_key + "daytona" if x.startswith(env_key) else x for x in args]
        mk = "++terminal_bench_config.harbor.mask_exceptions="
        masks = [x for x in args if x.startswith(mk)]; assert len(masks) == 1, "expected one mask_exceptions hydra arg"
        cur = masks[0][len(mk):].strip("[]"); extra = ['"%s"' % m for m in a.daytona_mask.split(",") if m and m not in cur]  # quoted like the template's entries
        args = [mk + "[" + ",".join([cur] + extra if cur else extra) + "]" if x.startswith(mk) else x for x in args]
        c["harbor_env"] = "daytona"
    c["skyrl_hydra_args"] = args; json.dump(c, open(cp, "w"), indent=2)
    sp = "%s/%s/sbatch/%s_rl.sbatch" % (EXP, name, name); s = open(sp).read()
    if a.daytona:
        ins = ("export NCCL_PXN_DISABLE=1\nexport HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=%d\n"
               "# --- daytona environment: preset SOCKS (microsocks on jpbl-s01-02) for _setup_proxy; setup-files-hook harbor first on PYTHONPATH ---\n"
               "set -a; source %s; source %s; set +a\n"
               "export PROXYCHAINS_SOCKS5_PRESET_HOST=%s\nexport PROXYCHAINS_SOCKS5_PRESET_PORT=%s\n"
               "export PROXYCHAINS_SOCKS5_PRESET_AUTH=\"$SOCKS_USER $SOCKS_PASS\"\n"
               "export PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH}\n"
               "unset APPTAINER_BRIDGE_URL\n") % (a.bridge_margin, a.socks_env, a.daytona_key_env, a.socks_host, a.socks_port, a.harbor_src)
        assert "DAYTONA_API_KEY" in s, "template sbatch exports no DAYTONA_API_KEY"
    else:
        ins = "export NCCL_PXN_DISABLE=1\nexport HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=%d\nexport APPTAINER_BRIDGE_URL=%s\n" % (a.bridge_margin, a.bridge_url)
    assert s.count("export NCCL_PXN_DISABLE=1\n") == 1
    s = s.replace("export NCCL_PXN_DISABLE=1\n", ins, 1)
    assert ("HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=%d" % a.bridge_margin) in s
    assert a.daytona or ("APPTAINER_BRIDGE_URL=%s" % a.bridge_url) in s
    open(sp, "w").write(s)
    print("%s: %d tasks -> %s; eval_batch_size=%d; verifier %ds; env %s; sbatch %s" % (name, len(ts), d, ebs, a.verifier_timeout, "daytona (socks %s:%s)" % (a.socks_host, a.socks_port) if a.daytona else "apptainer " + a.bridge_url, sp))
open("%s/%s_tasks.txt" % (EXP, a.prefix), "w").write("\n".join(tasks) + "\n")
print('launch: tmux new -d -s pool_launch_%s "bash %s/pool_launch.sh %s %d 150" && tmux new -d -s pool_watch_%s "bash %s/pool_watch.sh %s %d"'
      % (a.prefix, C, a.prefix, a.shards, a.prefix, C, a.prefix, a.shards))
