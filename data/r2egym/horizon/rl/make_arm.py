#!/usr/bin/env python3
"""make_arm.py — render (never submit) a Horizon Snowball RL arm: config JSON + sbatch, by diff from Jupiter's arms.

The scientific recipe comes from a Jupiter reference config (refs/); this script changes only what Horizon needs and what
the command line asks for, and asserts every anchor it touches (a template that drifted fails here, not in the job).

  --recipe gate5   hydra args of 62ft5sky (Stage-3 step 1888, apptainer, KL 0.01, no draft) = the port-gate-5 reference
  --recipe arm     hydra args of snowball_ttband_ota3d517u_b (the 09-21 SFT->GRPO arm, SWE .241 -> .307; newstack: KL 0, verifier 2400 s, preserve-on-timeout off,
                   staleness 2, grouped_mm, lr 5e-7, warmup 3, EAGLE-3 draft) = what an RL arm from an SFT export ran on

Both use the sbatch of snowball_ttband_ota3d517u_b (the post-migration template) with the Horizon deltas:
  - SBATCH header: debug / CCR24067, 144 CPUs, no Jupiter excludes; modules -> CC=gcc CXX=g++ (Triton JIT under nvc fails)
  - env: ~/snowball/envs/snowball + ~/snowball/jupiter.local.env, then DCFT/PYTHONPATH reset to this job checkout;
    compile caches node-local (/tmp); NCCL_SOCKET_IFNAME=ib (compute IPoIB ports are ibs2/ibP*)
  - Daytona: key read from ~/.config/otagent/daytona_eval.env at run time; harbor environment_type=daytona +
    auto_snapshot (the 11 restored per-repo snapshots; launch_arm.sh refuses a tree whose hashes are not all ACTIVE);
    create pacing HARBOR_DAYTONA_CREATE_RATE=5 split over the coordinators; Daytona infra errors masked (build_ota_darm.sh)
  - transport: one socks_connect_bridge.py per node over the login-side `ssh -R` tunnels (tunnel.sh, started by
    launch_arm.sh), HTTPS_PROXY=127.0.0.1:18946 on every node, HTTP_PROXY unset (vLLM traffic is plain http, in-cluster)
  - no artifact store (trials on /dev/shm, pruned by shm_prune.sh); a step stopper ends the job once step STEPS is logged
    (and, if checkpoints are on, written): the fully-async trainer keeps generating past max_steps.

Usage (login node, stdlib python3):
  python3 make_arm.py --name rl_gate5 --recipe gate5 --model <hf dir> --tree <task tree> --steps 3 --seats 128
  python3 make_arm.py --name rl_h9 --recipe arm --model <export dir> --tree <train tree> --steps 24 --seats 512 \
      --draft <eagle3 dir|none> --ckpt-interval 6 --hf-save-interval 12
"""
import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"
REF_CFG = {"gate5": REFS / "ref_62ft5sky_rl_config.json", "arm": REFS / "ref_ota3d517u_b_rl_config.json"}
REF_SB = REFS / "ref_ota3d517u_b_rl.sbatch"
REF_SB_NAME = "snowball_ttband_ota3d517u_b"
REF_SB_HASH = "3bc3d165cf20"
S = "/scratch/11584/lukedhlee"
HOMEDIR = os.path.expanduser("~")
SB = f"{HOMEDIR}/snowball"
DAYTONA_INFRA = ["DaytonaError", "DaytonaRateLimitError", "DaytonaTimeoutError", "DaytonaNotFoundError",
                 "DaytonaConflictError", "DaytonaSandboxStopError", "SandboxBuildFailedError", "SetupScriptError",
                 "DaytonaBadRequestError"]   # rl_h9 10-01: "declarative builds are not allowed" scored as false zeros
HARBOR_OVERLAY = f"{HOMEDIR}/snowball/harbor-rl"   # marin-community/harbor lukedhlee/daytona-snapshot-readonly (dcf609bc + read-only snapshots)
HARBOR_OVERLAY_SHA = "a20612bf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--recipe", choices=["gate5", "arm"], required=True)
    ap.add_argument("--model", required=True, help="HF-format model dir (config.json inside)")
    ap.add_argument("--served-name", default="", help="default: basename of --model (probe-served-model-name rule)")
    ap.add_argument("--tree", required=True, help="harbor task tree: task dirs only (symlinks into r2egym-daytona-v3 are fine)")
    ap.add_argument("--val-tree", default="", help="default: --tree (in-run eval is off either way)")
    ap.add_argument("--policy-nodes", type=int, default=4)
    ap.add_argument("--engines", type=int, default=8, help="one DP4-EP4 engine per node")
    ap.add_argument("--batch", type=int, default=64, help="groups per step (x n_samples 8)")
    ap.add_argument("--seats", type=int, default=128)
    ap.add_argument("--coords", type=int, default=16)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--wall", default="08:00:00")
    ap.add_argument("--draft", default=None, help="arm: EAGLE-3 draft dir, or 'none' to serve without it")
    ap.add_argument("--ckpt-interval", type=int, default=0, help="0 = no checkpoints")
    ap.add_argument("--hf-save-interval", type=int, default=0, help="0 = no HF exports")
    ap.add_argument("--project", default="horizon-snowball-rl")
    ap.add_argument("--probe", type=int, default=0,
                    help="K > 0: eval-only screen (Jupiter's refresh_screen probe): K attempts per tree entry, no training step; "
                         "trials kept on /scratch for the per-task readout (screen_report.py); the job stops at eval step 0")
    ap.add_argument("--drop", action="append", default=[], help="hydra key to remove (e.g. an arg the installed stack rejects)")
    ap.add_argument("--set", action="append", default=[], help="extra hydra override key=value (replaces if present)")
    ap.add_argument("--ota", default=f"{SB}/ota-rl", help="checkout of this branch: the Horizon scripts (bridge, shm_prune)")
    ap.add_argument("--runtime", default=f"{SB}/ota-rl-runtime",
                    help="OTA checkout the RL runner imports (WORKDIR): lukedhlee/horizon-rl = f3edec45, the commit port gate 1 "
                         "verified as Jupiter's RL stack, + the horizon cluster entry")
    ap.add_argument("--out", default=f"{S}/experiments/rl")
    a = ap.parse_args()

    # not realpath: harbor names the model after basename(policy path) in every request, and vLLM 400s any request
    # whose name differs from served_model_name (gate-5 job 39625: an HF snapshot hash vs the served name, every trial
    # BadRequestError, masked, no step in 1 h). A symlink named after the model keeps both equal.
    model = os.path.abspath(a.model)
    assert os.path.isfile(f"{model}/config.json"), f"no config.json under {model}"
    tree = os.path.realpath(a.tree)
    val_tree = os.path.realpath(a.val_tree) if a.val_tree else tree
    tasks = sorted(os.listdir(tree))
    assert len(tasks) >= 16, f"tree {tree} has {len(tasks)} entries"
    for t in tasks[:5]:
        assert os.path.isfile(f"{tree}/{t}/tests/test.sh") and os.path.isfile(f"{tree}/{t}/environment/Dockerfile"), \
            f"{tree}/{t} is not a harbor task dir"
    served = a.served_name or os.path.basename(model)
    assert served == os.path.basename(model), f"served name {served} != basename of the model path {model}: every LLM call would 400"
    nodes = a.policy_nodes + a.engines
    fsdp = 4 * a.policy_nodes
    run = f"{a.out}/{a.name}"
    os.makedirs(f"{run}/configs", exist_ok=True)
    os.makedirs(f"{run}/sbatch", exist_ok=True)
    os.makedirs(f"{run}/logs", exist_ok=True)
    trials = f"{run}/trials" if a.probe else f"/dev/shm/otagent_trials/{a.name}/trace_jobs"

    # ------------------------------------------------------------------ config
    c = json.load(open(REF_CFG[a.recipe]))
    src = c["job_name"]
    src_exp = c["experiments_dir"]
    c = json.loads(json.dumps(c).replace(src_exp, run).replace(src, a.name))
    args = c["skyrl_hydra_args"]

    def idx(prefix):
        return [k for k, x in enumerate(args) if x.lstrip("+").startswith(prefix)]

    def setk(prefix, val, add="++"):
        i = idx(prefix)
        assert len(i) <= 1 or prefix in ("trainer.run_name=", "trainer.logger=", "trainer.strategy=",
                                          "trainer.enable_db_registration=", "trainer.policy.fsdp_config.moe_router_replay="), (prefix, i)
        if not i:
            args.append(add + prefix + val)
            return
        for k in i:
            args[k] = args[k][: args[k].index(prefix)] + prefix + val

    def getk(prefix):
        i = idx(prefix)
        assert i, prefix
        return args[i[-1]].split("=", 1)[1]

    def dropk(prefix):
        for k in reversed(idx(prefix)):
            args.pop(k)

    old_model = getk("trainer.policy.model.path=")
    assert getk("trainer.ref.model.path=") == old_model
    setk("trainer.policy.model.path=", model)
    setk("trainer.ref.model.path=", model)
    setk("generator.engine_init_kwargs.served_model_name=", served)
    c["model_path"] = model
    # geometry
    setk("trainer.placement.policy_num_nodes=", str(a.policy_nodes))
    setk("trainer.placement.ref_num_nodes=", str(a.policy_nodes))
    setk("trainer.policy.fsdp_config.fsdp_size=", str(fsdp))
    setk("trainer.ref.fsdp_config.fsdp_size=", str(fsdp))
    setk("generator.num_inference_engines=", str(a.engines))
    assert getk("generator.inference_engine_data_parallel_size=") == "4"
    assert getk("generator.inference_engine_expert_parallel_size=") == "4"
    assert getk("generator.inference_engine_tensor_parallel_size=") == "1"
    setk("trainer.train_batch_size=", str(a.batch))
    setk("trainer.policy_mini_batch_size=", str(a.batch))
    c["num_nodes"] = nodes
    c["cpus_per_node"] = 144
    c["cluster_name"] = "horizon"
    # data
    setk("data.train_data=", json.dumps([tree]))
    setk("data.val_data=", json.dumps([val_tree]))
    c.update(train_data=[tree], val_data=[val_tree], train_data_sources=[tree], val_data_sources=[val_tree])
    # steps, eval, checkpoints
    setk("trainer.max_steps=", str(a.steps))
    setk("trainer.eval_before_train=", "false")
    setk("trainer.eval_interval=", "0")
    setk("trainer.ckpt_interval=", str(a.ckpt_interval if a.ckpt_interval else 10**6))
    setk("trainer.hf_save_interval=", str(a.hf_save_interval if a.hf_save_interval else 10**6))
    setk("trainer.resume_mode=", "latest" if a.ckpt_interval else "none")
    setk("trainer.project_name=", a.project)
    # seats / coordinators / trials dir
    setk("terminal_bench_config.harbor.n_concurrent_trials=", str(a.seats))
    setk("trajectory_runner.process_pool.num_coordinators=", str(a.coords))
    setk("terminal_bench_config.trials_dir=", trials)
    c["trials_dir"] = trials
    # Daytona backend (build_ota_darm.sh)
    setk("terminal_bench_config.harbor.environment_type=", "daytona")
    setk("terminal_bench_config.harbor.auto_snapshot=", "true")
    mask = json.loads(getk("terminal_bench_config.harbor.mask_exceptions="))
    if a.recipe == "gate5":  # + the transport names the newstack arms mask (62ft5sky predates them)
        bmask = [x for x in json.load(open(REF_CFG["arm"]))["skyrl_hydra_args"] if "harbor.mask_exceptions=" in x][0]
        mask += [m for m in json.loads(bmask.split("=", 1)[1]) if m not in mask]
    mask += [m for m in DAYTONA_INFRA if m not in mask]
    setk("terminal_bench_config.harbor.mask_exceptions=", json.dumps(mask, separators=(",", ":")))
    # verifier budget + preserve-on-timeout: the 09-07 fix (rl-verifier-timeouts-false-zeros); gate5's 150 s predates it
    setk("terminal_bench_config.harbor.verifier_override_timeout_sec=", "2400")
    setk("terminal_bench_config.harbor.preserve_logprobs_on_timeout=", "false")
    setk("terminal_bench_config.harbor.zero_exceptions=", '["TmuxSessionEndedError"]')
    # spec decode
    if a.recipe == "arm":
        assert a.draft, "--draft <dir|none> is required for an arm"
        if a.draft == "none":
            dropk("generator.engine_init_kwargs.speculative_config=")
        else:
            d = os.path.realpath(a.draft)
            assert os.path.isfile(f"{d}/config.json"), d
            setk("generator.engine_init_kwargs.speculative_config=", "{method:eagle3,model:%s,num_speculative_tokens:3}" % d)
    else:
        assert not idx("generator.engine_init_kwargs.speculative_config="), "62ft5sky had no draft"
    if a.probe:  # refresh_screen.build(): eval-before-train only (EvaluationCallback needs eval_interval > 0), zero epochs
        setk("trainer.eval_interval=", "9999")
        setk("trainer.eval_before_train=", "true")
        setk("trainer.epochs=", "0")
        setk("generator.eval_n_samples_per_prompt=", str(a.probe))
        setk("trainer.eval_batch_size=", str(min(32, len(tasks))))
        setk("trajectory_runner.process_pool.eval_spread_coordinators=", "true")
        setk("generator.sampling_params.top_p=", "1.0")
        setk("generator.sampling_params.top_k=", "-1")
        setk("data.val_data=", json.dumps([tree]))
        c.update(val_data=[tree], val_data_sources=[tree])
    for k in a.drop:
        assert idx(k + "="), f"--drop {k}: not in the config"
        dropk(k + "=")
    for kv in a.set:
        k, v = kv.split("=", 1)
        setk(k + "=", v)
    # launcher fields
    c.update(harbor_env="daytona", needs_ssh_tunnel=False, artifact_store_enabled=False, artifact_store_image=None,
             artifact_store_mount=None, export_path=f"{run}/{a.name}/exports", checkpoints_dir=None)
    c["skyrl_hydra_args"] = args
    assert "/e/" not in json.dumps(c).replace("/dev/", ""), [x for x in args if "/e/" in x]
    cfg = f"{run}/configs/{a.name}_rl_config.json"
    json.dump(c, open(cfg, "w"), indent=2)

    # ------------------------------------------------------------------ sbatch
    b = open(REF_SB).read()
    jexp = f"/e/fscratch/reformo/lee27/experiments/{REF_SB_NAME}"

    def sub1(old, new):
        nonlocal b
        assert b.count(old) == 1, ("anchor", old[:80], b.count(old))
        b = b.replace(old, new)

    def subre(pat, new, count=1):
        nonlocal b
        b, n = re.subn(pat, new, b, flags=re.M)
        assert n == count, ("regex anchor", pat, n)

    sub1("#SBATCH --time=08:00:00\n", f"#SBATCH --time={a.wall}\n")
    # TACC preloads XALT (libxalt_init.so) into every process; it prints an NVML stub warning on stdout, which lands in
    # every $(...) capture (job 39606: the Daytona key read back as 978 characters). Drop it before anything runs.
    sub1("set -eo pipefail\n", "set -eo pipefail\nunset LD_PRELOAD   # Horizon: XALT pollutes command substitutions\n")
    sub1("#SBATCH --nodes=20\n", f"#SBATCH --nodes={nodes}\n")
    sub1("#SBATCH --cpus-per-task=288\n", "#SBATCH --cpus-per-task=144\n")
    sub1("#SBATCH --mail-type=END,TIME_LIMIT,FAIL\n#SBATCH --mail-user=\n", "")
    sub1("#SBATCH -p booster\n#SBATCH --account laionize\n", "#SBATCH -p debug\n#SBATCH --account CCR24067\n")
    subre(r"^#SBATCH --exclude=.*\n", "")
    sub1("module load GCC/14.3.0\nmodule load nvidia-compilers/25.9-CUDA-13\n",
         "# Horizon: no modules; the login env's nvidia/26.9 rides along but sets CC=nvc, which Triton's JIT rejects\n"
         "export CC=gcc CXX=g++\n")
    # env files: the Horizon overlay (jupiter.local.env is the name build_snowball_env.sh gives it), then this checkout
    sub1('_OT_DOTENV_DIR="${DCFT:-}/hpc/dotenv"\nif [ -n "${DCFT:-}" ] && [ -f "$_OT_DOTENV_DIR/jupiter.env" ]; then\n'
         '  source "$_OT_DOTENV_DIR/jupiter.env"\nfi\n'
         '# Gitignored per-user overlay (<cluster>.local.env); later wins. Mirrors hpc.set_environment.\n'
         'if [ -f "$_OT_DOTENV_DIR/jupiter.local.env" ]; then\n  source "$_OT_DOTENV_DIR/jupiter.local.env"\nfi\n',
         f'# Horizon: the per-user overlay build_snowball_env.sh wrote (paths, venv, LD_LIBRARY_PATH); it points DCFT at\n'
         f'# ~/snowball/OpenThoughts-Agent and turns proxychains on, so both are reset right after.\n'
         f'source {SB}/jupiter.local.env\n'
         f'export DCFT="$WORKDIR"\n'
         f'unset PROXYCHAINS_BIN_OVERRIDE PROXYCHAINS_SOCKS5_PRESET_HOST PROXYCHAINS_SOCKS5_PRESET_PORT PROXYCHAINS_SOCKS5_PRESET_AUTH\n')
    keyf = f"{HOMEDIR}/.config/otagent/daytona_eval.env"
    sub1('DAYTONA_API_KEY_OVERRIDE="dtn_REDACTED"\n',
         '# Horizon: key read from the key file at run time (never written into this file)\n'
         f'DAYTONA_API_KEY_OVERRIDE="$(grep -m1 -E \'^(export )?DAYTONA_API_KEY=\' {keyf} | cut -d= -f2- | sed -E \'s/[[:space:]]+#.*$//\' | tr -d "\\"\' ")"\n'
         f'[ -n "$DAYTONA_API_KEY_OVERRIDE" ] || {{ echo "FATAL: no DAYTONA_API_KEY in {keyf}" >&2; exit 96; }}\n')
    sub1("export DCFT_RL_ENV=/e/project1/transfernetx/lee27/code/envs/snowball-v2\n"
         "export APPTAINER_BRIDGE_URL=http://10.128.1.2:9930\n"
         "export RL_PYTHON=/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python\n",
         f"export DCFT_RL_ENV={SB}/envs/snowball\nunset APPTAINER_BRIDGE_URL\nexport RL_PYTHON={SB}/envs/snowball/bin/python\n")
    sub1('export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//envs\\/rl-fa/envs\\/snowball-v2}"\n', "")
    sub1("export VLLM_CACHE_ROOT=/e/fscratch/reformo/lee27/cache/vllm XDG_CACHE_HOME=/e/fscratch/reformo/lee27/cache/xdg "
         "TRITON_CACHE_DIR=/e/fscratch/reformo/lee27/cache/triton TORCHINDUCTOR_CACHE_DIR=/e/fscratch/reformo/lee27/cache/inductor\n",
         "# Horizon: compile caches node-local (a shared NFS cache broke multi-node starts, ops.md 09-29)\n"
         "_LC=/tmp/rlcache_${SLURM_JOB_ID}; mkdir -p $_LC\n"
         "export VLLM_CACHE_ROOT=$_LC/vllm XDG_CACHE_HOME=$_LC/xdg TRITON_CACHE_DIR=$_LC/triton TORCHINDUCTOR_CACHE_DIR=$_LC/inductor\n"
         # Daytona create pacing (build_ota_darm.sh): 5/s org budget, split over this arm's coordinators
         f"export HARBOR_DAYTONA_CREATE_RATE=${{HARBOR_DAYTONA_CREATE_RATE:-5}} HARBOR_DAYTONA_CREATE_SHARES={a.coords}\n"
         "export RES_OPTIONS='timeout:1 attempts:2' LITELLM_LOCAL_MODEL_COST_MAP=True\n")
    # Horizon nodes carry 4 InfiniBand HCAs (mlx5_0/1/4/5, 800 Gb/s) and 2 RoCE ones (mlx5_2/3); unpinned, NCCL paired an
    # IB device with a RoCE one across nodes and the first policy->engine weight broadcast died ("Remote IB device is
    # incompatible", jobs 39609/39610). Same names on every node checked (28 of 28).
    subre(r'^export NCCL_SOCKET_IFNAME="ib0"\n', 'export NCCL_SOCKET_IFNAME="ib"\nexport NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_4,mlx5_5"\n')
    subre(r'^export FLASHINFER_WORKSPACE_BASE="/e/fscratch/reformo/lee27/cache/flashinfer"\n',
          'export FLASHINFER_WORKSPACE_BASE="$_LC/flashinfer"\n', count=2)
    subre(r'^export WANDB_PROJECT="jupiter-r2egym-grpo"\n', f'export WANDB_PROJECT="{a.project}"\n', count=2)
    subre(r"^export OT_AGENT_RAY_LOG_DIR=.*\n", f"export OT_AGENT_RAY_LOG_DIR={S}/experiments/_ray_logs\n")
    subre(r"^export RAY_object_spilling_config=.*\n",
          "export RAY_object_spilling_config='{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"%s/ray_spill\"}}'\n" % S)
    sub1('ARTIFACT_STORE_ENABLED="1"\n', 'ARTIFACT_STORE_ENABLED="0"\n')
    subre(r'^ARTIFACT_STORE_IMAGE=.*\n', 'ARTIFACT_STORE_IMAGE=""\n')
    subre(r'^ARTIFACT_STORE_MOUNT=.*\n', 'ARTIFACT_STORE_MOUNT=""\n')
    # the tmpfs trials dir lives inside the artifact-store `if`: move it out
    shm = (f'SHM_TRIALS={trials}; mkdir -p "$SHM_TRIALS"; echo "trials_dir on tmpfs: $SHM_TRIALS ($(df -h /dev/shm | tail -n 1))"\n'
           f'( bash /e/project1/transfernetx/lee27/code/snowball/shm_prune.sh "$SHM_TRIALS" 10 70 60 ) &\n')
    sub1(shm.replace(trials, f"/dev/shm/otagent_trials/{REF_SB_NAME}/trace_jobs"), "")
    sub1("\n_setup_proxy\n", "\n" + (
        "# TACC preloads XALT (libxalt_init.so), which prints an NVML stub warning into every captured command output\n"
        "unset LD_PRELOAD\n" +
        (f'mkdir -p {trials}; echo "probe trials_dir (kept): {trials}"\n' if a.probe else
         f'SHM_TRIALS={trials}; mkdir -p "$SHM_TRIALS"; echo "trials_dir on tmpfs: $SHM_TRIALS ($(df -P /dev/shm | tail -n 1))"\n'
         f'( bash {a.ota}/data/r2egym/horizon/rl/shm_prune.sh "$SHM_TRIALS" 10 70 60 ) &\n') +
        "# --- Horizon: Daytona egress = one CONNECT bridge per node over the login-side ssh -R tunnels (launch_arm.sh starts\n"
        "# tunnel.sh per port); no proxychains. Every node's bridge listens on its own 127.0.0.1:18946.\n"
        "unset PROXYCHAINS_BIN_OVERRIDE PROXYCHAINS_CONF_FILE LD_PRELOAD SSH_KEY HTTP_PROXY http_proxy ALL_PROXY all_proxy\n"
        f"GW_DIR={run}/gateway/$SLURM_JOB_ID; mkdir -p \"$GW_DIR\"\n"
        "TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}\n"
        "srun --overlap --nodes=\"$SLURM_NNODES\" --ntasks=\"$SLURM_NNODES\" --ntasks-per-node=1 --cpus-per-task=1 --gres=none --export=ALL \\\n"
        f"  bash {a.ota}/data/r2egym/horizon/rl/node_bridge.sh \"$GW_DIR\" \"$TUNNEL_PORTS\" 18946 &\n"
        "for _i in $(seq 1 90); do [ \"$(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)\" -ge \"$SLURM_NNODES\" ] && break; sleep 10; done\n"
        "if [ \"$(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)\" -lt \"$SLURM_NNODES\" ]; then\n"
        "  echo \"FATAL: bridge ready on $(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)/$SLURM_NNODES nodes after 15 min (tunnel.sh running on the login node?)\" >&2\n"
        "  tail -n 5 \"$GW_DIR\"/*.log >&2; exit 95\nfi\n"
        "export HTTPS_PROXY=http://127.0.0.1:18946 https_proxy=http://127.0.0.1:18946\n"
        "export NO_PROXY=localhost,127.0.0.1,.horizon.tacc.utexas.edu no_proxy=localhost,127.0.0.1,.horizon.tacc.utexas.edu\n"
        "_ok=0; for _i in $(seq 1 12); do curl -sf -o /dev/null --max-time 30 -H \"Authorization: Bearer $DAYTONA_API_KEY\" https://app.daytona.io/api/api-keys/current && { _ok=1; break; }; sleep 5; done\n"
        "[ $_ok = 1 ] || { echo \"FATAL: Daytona API not reachable through the bridge (12 tries)\" >&2; exit 98; }\n"
        "echo \"horizon rl: bridges ready on $SLURM_NNODES nodes; Daytona API reachable through 127.0.0.1:18946\"\n"))
    sub1('setup_container_runtime "apptainer" "$WORKDIR" || exit $?\n', 'setup_container_runtime "daytona" "$WORKDIR" || exit $?\n')
    sub1("export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}\n",
         f"export PYTHONPATH={SB}/src/marin_vllm_eagle3${{PYTHONPATH:+:$PYTHONPATH}}\n"
         "# --- Horizon: harbor overlay = the venv's dcf609bc + read-only auto-snapshots (never create/delete a snapshot, never\n"
         "# fall through to a declarative build); the venv's editable harbor-marin is shared by running jobs and stays untouched\n"
         f"export PYTHONPATH={HARBOR_OVERLAY}/src:$PYTHONPATH HARBOR_DAYTONA_SNAPSHOT_READONLY=1\n"
         f"[ \"$(git -C {HARBOR_OVERLAY} rev-parse --short=8 HEAD)\" = {HARBOR_OVERLAY_SHA} ] && [ -z \"$(git -C {HARBOR_OVERLAY} status --short)\" ] || {{ echo \"FATAL: harbor overlay is not a clean {HARBOR_OVERLAY_SHA}\" >&2; exit 97; }}\n"
         f"_HI=$(cd /tmp && \"$RL_PYTHON\" -c 'import harbor.environments.daytona.snapshots as s; print(s.__file__, hasattr(s.DaytonaSnapshotService, \"_ensure_auto_readonly\"))' 2>&1 | tail -n 1)\n"
         f"case \"$_HI\" in \"{HARBOR_OVERLAY}/src/\"*\" True\") echo \"harbor overlay: $_HI\";; *) echo \"FATAL: harbor overlay not imported: $_HI\" >&2; exit 97;; esac\n")
    launch = re.search(r'\n"\$RL_PYTHON" -m hpc\.rl_launch_utils --config "[^"]+" &\n', b)
    assert launch
    ckpt_wait = (f'  for _i in $(seq 1 60); do [ "$(cat {run}/{a.name}/checkpoints/latest_ckpt_global_step.txt 2>/dev/null)" = {a.steps} ] && break; sleep 30; done\n'
                 if a.ckpt_interval and a.steps % a.ckpt_interval == 0 else "")
    if a.hf_save_interval and a.steps % a.hf_save_interval == 0:   # and the step's HF export (any dir naming the step)
        ckpt_wait += (f'  for _i in $(seq 1 80); do find {run}/{a.name}/exports -maxdepth 3 -name config.json -path "*{a.steps}*" 2>/dev/null | grep -q . && break; sleep 30; done\n'
                      f'  echo "horizon rl: export of step {a.steps}: $(find {run}/{a.name}/exports -maxdepth 3 -name config.json -path "*{a.steps}*" 2>/dev/null | head -1)"\n')
    stop_on = "kind=eval step=0 " if a.probe else f"kind=train step={a.steps} "
    b = b.replace(launch.group(0), (
        f"\n# --- Horizon: stop once '{stop_on.strip()}' is logged (and its checkpoint written); the fully-async trainer keeps going ---\n"
        f"DS_LOG={run}/logs/${{SLURM_JOB_NAME}}_${{SLURM_JOB_ID}}.out\n"
        f"( while ! grep -q 'WANDB_MIRROR {stop_on}' \"$DS_LOG\" 2>/dev/null; do sleep 30; done\n"
        f"  echo \"horizon rl: {stop_on.strip()} logged $(date +%T)\"\n" + ("" if a.probe else ckpt_wait) +
        ("  echo \"horizon rl: stopping $(date +%T) (probe: at once; SkyRL's eval_on_train_end starts a 2nd eval that would orphan sandboxes)\"\n"
         "  scancel \"$SLURM_JOB_ID\" ) &\n" if a.probe else
         "  echo \"horizon rl: stopping $(date +%T)\"; sleep 120\n"
         "  scancel -s USR1 -b \"$SLURM_JOB_ID\"; sleep 240; scancel \"$SLURM_JOB_ID\" ) &\n") +
        f'\n"$RL_PYTHON" -m hpc.rl_launch_utils --config "{cfg}" &\n'), 1)
    b = b.replace(f"{jexp}/{REF_SB_NAME}", f"{run}/{a.name}").replace(jexp, run).replace(REF_SB_NAME, a.name)
    b = b.replace(f"{a.name}-{REF_SB_HASH}", a.name)
    bad = [l for l in b.splitlines() if re.search(r"(?<![\w/])/e/(fscratch|project1|data1|scratch)", l) and not l.lstrip().startswith("#")
           and "_setup_proxy" not in l]
    # the (now uncalled) _setup_proxy function body still names JSC paths; anything else is a missed substitution
    leftovers = [l for l in bad if not re.search(r"proxychains|PROXYCHAINS|bfeuer|synthlaion|dc-agent-shared|LOGIN_NODE|SSH_KEY|^\s*echo ", l)]
    assert not leftovers, leftovers[:10]
    assert "dtn_" not in b
    sbf = f"{run}/sbatch/{a.name}_rl.sbatch"
    open(sbf, "w").write(b)
    json.dump(dict(runtime=os.path.realpath(a.runtime), ota=os.path.realpath(a.ota)), open(f"{run}/configs/checkouts.json", "w"))
    print(json.dumps(dict(config=cfg, sbatch=sbf, runtime=a.runtime, nodes=nodes, policy_nodes=a.policy_nodes, engines=a.engines,
                          seats=a.seats, coords=a.coords, steps=a.steps, model=model, served=served, tree=tree,
                          ntasks=len(tasks), recipe=a.recipe, probe=a.probe), indent=1))


if __name__ == "__main__":
    main()
