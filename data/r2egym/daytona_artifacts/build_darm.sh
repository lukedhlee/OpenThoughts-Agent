#!/bin/bash
# build_darm.sh [dst=snowball_ttband_daytona1k_a] — build (never submit) the 1,000-seat Daytona RL arm for the steps/hour
# comparison with apptainer: the wave-1 control arm snowball_ttband_ns_c_lr5e7_c geometry unchanged (20 nodes = 8 policy
# fsdp 32 + 12 engine nodes DP4 EP4 EAGLE-3, batch 64 x 8, staleness 2, 16 coordinators), rollouts on Daytona at 1,024 seats
# (64 per coordinator; the control arm ran 768 on apptainer), 4 steps then self-stop. Derived from build_dsmoke.sh: every
# scientific setting is the control arm's; what changes is the step count, the task tree, the seats and the sandbox backend.
#   - backend: harbor environment_type daytona + auto_snapshot, Daytona infra names masked, key from the key file at run time,
#     egress through ONE ASYNC LOOPBACK GATEWAY PER NODE to the login-node microsocks (no proxychains: it blocks the
#     coordinator event loop on every reconnect, research/2026-09-15_daytona_proxy_tokenization.md), no bridge.
#   - code under test first on PYTHONPATH and as SKYRL_HOME (the launcher runs the entrypoint with cwd $SKYRL_HOME/skyrl-train):
#     harbor 9940bdfd (lukedhlee/snowball-r2egym tip: create pacing + tokenize budget + plain $ prompt) and MarinSkyRL f616e800 (lukedhlee/daytona-create-shares), each one commit
#     on top of the lukedhlee/snowball-r2egym tips the shared venv installs; the job exits 97 if either is not what imports.
#   - tasks: all 1,000 training-pool tasks that pass the Daytona gate (the control arm trained on the same 1,003-task pool).
# Runs on the Jupiter login node with python3 (stdlib) only.
set -euo pipefail
SRC=snowball_ttband_ns_c_lr5e7_c; DST=${1:-snowball_ttband_daytona1k_a}
NODES=20; POL=8; ENG=12; FSDP=32; BATCH=64; STEPS=${STEPS:-4}; WALL=${WALL:-02:00:00}
# Seats per arm and the org-wide create budget split. Two arms at once: SEATS=512 SHARES=32 for each (16 coordinators x 2 jobs),
# so the two jobs together stay at the 5 creates/s we measured and under one user's ~1,000 concurrent sandboxes (the cap is per person).
SEATS=${SEATS:-1024}; COORD=${COORD:-16}; SHARES=${SHARES:-$COORD}
E=/e/fscratch/reformo/lee27/experiments; M=/e/data1/mmlaion/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks
OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; C=/e/project1/transfernetx/lee27/code/snowball
DS_HARBOR=/e/fscratch/reformo/lee27/code/harbor-pacing; DS_SKYRL=/e/fscratch/reformo/lee27/code/marinskyrl-shares
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env; SOCKSF=/e/fscratch/reformo/lee27/keys/socks5_currease.env
TREE=$T/r2egym-daytona-v3-train1000; DTREE=$T/r2egym-daytona-v3; ALLOW=$T/allowlist_r2egym_daytona_v3.txt
GWPY=/e/fscratch/reformo/lee27/experiments/daytona_rl_arm1k/async_socks_connect_proxy.py   # the gateway (OTA data/r2egym/daytona_artifacts, 264629ba)
GWDEPS=/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps        # python-socks 3.1.1, pure python
[ -f $GWPY ] && [ -d $GWDEPS/python_socks ] || { echo "gateway script or its deps missing"; exit 1; }
case "$DST" in *snowball_ttband*) ;; *) echo "dst must contain snowball_ttband (store_reaper)"; exit 1;; esac
squeue -h -u "$USER" -n "$DST" -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
[ "$(git -C $DS_HARBOR rev-parse --short=8 HEAD)" = 9940bdfd ] && [ -z "$(git -C $DS_HARBOR status --short)" ] || { echo "harbor-pacing is not a clean 9940bdfd"; exit 1; }
[ "$(git -C $DS_SKYRL rev-parse --short=8 HEAD)" = f616e800 ] && [ -z "$(git -C $DS_SKYRL status --short)" ] || { echo "marinskyrl-shares is not a clean f616e800"; exit 1; }

# --- 1. task tree: every training-pool task on the Daytona allowlist (1,000), copied like the smoke's 64 ---
if [ -f $TREE/TASKS.txt ] && [ "$(wc -l < $TREE/TASKS.txt)" = 1000 ]; then echo "task tree $TREE exists (1000 tasks)"; else
python3 - "$T" "$DTREE" "$ALLOW" "$TREE" <<'PY'
import os, re, shutil, sys
T, dtree, allow_f, out = sys.argv[1:5]
pool = {re.sub(r"__r\d+$", "", d) for d in os.listdir(T + "/r2egym-tt-v2-train-basecurr-x16")}
allow = {l.strip() for l in open(allow_f) if l.strip()}
pick = sorted(pool & allow)
assert len(pick) == 1000, len(pick)
if os.path.isdir(out): shutil.rmtree(out)
os.makedirs(out)
for t in pick: shutil.copytree(f"{dtree}/{t}", f"{out}/{t}", symlinks=True)
open(out + "/TASKS.txt", "w").write("\n".join(pick) + "\n")
print("task tree %s: %d tasks" % (out, len(pick)))
PY
fi

# --- 2. run dir (data1, symlinked from fscratch like every new-stack arm) ---
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
CFG=$E/$DST/configs/${DST}_rl_config.json; SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $CFG
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $SB

# --- 3. config: size, tree, backend ---
python3 - "$CFG" "$SRC" "$TREE" "$NODES:$POL:$ENG:$FSDP:$BATCH:$SEATS:$COORD:$STEPS" <<'PY'
import json, sys
p, src, tree, geo = sys.argv[1:5]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
NODES, POL, ENG, FSDP, BATCH, SEATS, COORD, STEPS = geo.split(":")
assert not any(src in x for x in a), "src name survived the sed"
def idx(prefix): return [k for k, x in enumerate(a) if x.lstrip("+").startswith(prefix)]
def setk(prefix, val):
    i = idx(prefix); assert len(i) == 1, (prefix, i)
    a[i[0]] = a[i[0]][: a[i[0]].index(prefix)] + prefix + val
def getk(prefix):
    i = idx(prefix); assert len(i) == 1, (prefix, i); return a[i[0]].split("=", 1)[1]
# size: the control arm's own batch (64 prompts x 8) and geometry; only the step count, seats and backend move
setk("trainer.epochs=", "1"); setk("trainer.max_steps=", STEPS)
setk("trainer.train_batch_size=", BATCH); setk("trainer.policy_mini_batch_size=", BATCH)
setk("trainer.placement.policy_num_nodes=", POL); setk("trainer.placement.ref_num_nodes=", POL)
setk("trainer.policy.fsdp_config.fsdp_size=", FSDP); setk("trainer.ref.fsdp_config.fsdp_size=", FSDP)
setk("generator.num_inference_engines=", ENG)
setk("terminal_bench_config.harbor.n_concurrent_trials=", SEATS)          # 1,024 seats on Daytona (control arm: 768 on apptainer)
setk("trajectory_runner.process_pool.num_coordinators=", COORD)          # 16 coordinators x 64 seats, the ramp's coordinator shape
setk("trainer.resume_mode=", "none")                                     # 4 steps from base, never resumes
setk("trainer.ckpt_interval=", "999"); setk("trainer.hf_save_interval=", "999"); setk("trainer.max_ckpts_to_keep=", "1")
setk("data.train_data=", '["%s"]' % tree); setk("data.val_data=", '["%s"]' % tree)   # in-run eval stays off (eval_interval 9999)
# backend: Daytona
setk("terminal_bench_config.harbor.environment_type=", "daytona")
assert not idx("terminal_bench_config.harbor.auto_snapshot="); a.append("++terminal_bench_config.harbor.auto_snapshot=true")
mask = json.loads(getk("terminal_bench_config.harbor.mask_exceptions="))
daytona_infra = ["DaytonaError", "DaytonaRateLimitError", "DaytonaTimeoutError", "DaytonaNotFoundError", "DaytonaConflictError",
                 "DaytonaSandboxStopError", "SandboxBuildFailedError", "SetupScriptError"]
setk("terminal_bench_config.harbor.mask_exceptions=", json.dumps(mask + [m for m in daytona_infra if m not in mask], separators=(",", ":")))
# settings the smoke must carry unchanged from the arm (asserted, not set)
must = {"trainer.algorithm.use_kl_loss=": "false", "trainer.fully_async.max_staleness_steps=": "2",
        "trainer.policy.optimizer_config.lr=": "5e-7", "terminal_bench_config.harbor.verifier_override_timeout_sec=": "2400",
        "terminal_bench_config.harbor.preserve_logprobs_on_timeout=": "false", "generator.n_samples_per_prompt=": "8",
        "trainer.algorithm.use_tis=": "true", "trainer.eval_interval=": "9999", "trainer.eval_before_train=": "false",
        "terminal_bench_config.harbor.zero_exceptions=": '["TmuxSessionEndedError"]'}
for k, v in must.items(): assert getk(k) == v, (k, getk(k), v)
assert "method:eagle3" in getk("generator.engine_init_kwargs.speculative_config="), "spec decode missing"
assert getk("terminal_bench_config.trials_dir=").startswith("/dev/shm/otagent_trials/")
c.update(num_nodes=int(NODES), harbor_env="daytona", train_data=[tree], val_data=[tree], train_data_sources=[tree], val_data_sources=[tree])
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print("config:", p)
PY
python3 $C/fix_merged_keys.py $CFG
# compose against the code under test (validate_hydra_args.py hardcodes the shared checkout)
sed "s#/e/project1/transfernetx/lee27/code/marinskyrl-marin/skyrl-train#$DS_SKYRL/skyrl-train#" $C/validate_hydra_args.py > $E/$DST/configs/validate_hydra_args_dsmoke.py
grep -q "$DS_SKYRL/skyrl-train/skyrl_train/config" $E/$DST/configs/validate_hydra_args_dsmoke.py
OMP_NUM_THREADS=1 /e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python $E/$DST/configs/validate_hydra_args_dsmoke.py $CFG

# --- 4. artifact store (same bookkeeping as the arms; the mount path hashes the image path) ---
IMG=$E/$DST/$DST/artifact_store.img; SRC_IMG=$E/$SRC/$SRC/artifact_store.img
OLDH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$SRC_IMG')).name.rsplit('-', 1)[-1])")
NEWH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$IMG')).name.rsplit('-', 1)[-1])")
sed -i "s/$OLDH/$NEWH/g" $SB $CFG
grep -q "ARTIFACT_STORE_MOUNT=\"/tmp/otagent-artifact-stores/$DST-$NEWH\"" $SB
(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import ensure_image; ensure_image(Path('$IMG'), size='1T', inode_count=50_000_000)") >/dev/null
[ -f $IMG ]

# --- 5. sbatch ---
python3 - "$SB" "$DST" "$NODES" "$WALL" "$KEYF" "$SOCKSF" "$DS_HARBOR" "$DS_SKYRL" "$STEPS" "$E/$DST" "$GWPY" "$GWDEPS" "$SHARES" <<'PY'
import re, sys
sb, dst, nodes, wall, keyf, socksf, dsh, dss, steps, run, gwpy, gwdeps, shares = sys.argv[1:14]
b = open(sb).read()
def sub1(old, new):
    global b
    assert b.count(old) == 1, ("anchor count", old, b.count(old)); b = b.replace(old, new)
assert b.count("#SBATCH --nodes=%s\n" % nodes) == 1, "control arm node count"
sub1("#SBATCH --time=12:00:00\n", "#SBATCH --time=%s\n" % wall)
# the arm's sbatch carries a baked (stale) key; read Luke's key from the key file at run time instead, never into this file
key_block = ('# Daytona 1k arm: key read from the key file at run time (a trailing "# comment" on the line is stripped)\n'
     'DAYTONA_API_KEY_OVERRIDE="$(grep -m1 -E \'^(export )?DAYTONA_API_KEY=\' %s | cut -d= -f2- | sed -E \'s/[[:space:]]+#.*$//\' | tr -d "\\"\' ")"\n'
     '[ -n "$DAYTONA_API_KEY_OVERRIDE" ] || { echo "FATAL: no DAYTONA_API_KEY in %s" >&2; exit 96; }\n') % (keyf, keyf)
b, n = re.subn(r'^DAYTONA_API_KEY_OVERRIDE="dtn_[0-9a-f]+"\n', lambda m: key_block, b, flags=re.M)
assert n == 1 and "dtn_" not in b, "key override line"
sub1("export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000\n", "export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000\n" + (
    "# --- Daytona 1k arm: sandbox backend ---\n"
    "# harbor 9940bdfd paces sandbox creates at this org-wide rate; MarinSkyRL f616e800 gives each rollout coordinator 1/num_coordinators of it\n"
    "export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES=%s\n"
    "# SOCKS credentials for the per-node gateway (started below in place of _setup_proxy); never proxychains on this job\n"
    "set -a; source %s; set +a\n"
    "unset APPTAINER_BRIDGE_URL\n") % (shares, socksf))
# keep the control arm's tmpfs pruner (1,024 seats x staleness 2 would not fit unpruned) and add a 10-minute snapshot of the
# small result files to the run dir for the phase-timing comparison
sub1('( while true; do find "$SHM_TRIALS" -mindepth 1 -maxdepth 1 -type d -mmin +90 -exec rm -rf {} + 2>/dev/null; sleep 300; done ) &\n',
     '( while true; do find "$SHM_TRIALS" -mindepth 1 -maxdepth 1 -type d -mmin +90 -exec rm -rf {} + 2>/dev/null; sleep 300; done ) &\n'
     '( while true; do sleep 600; (cd "$SHM_TRIALS" && find . \\( -name "*result.json" -o -name exception.txt \\) -print0 | tar --null -czf %s/trials_meta.tgz.tmp -T - && mv %s/trials_meta.tgz.tmp %s/trials_meta.tgz) 2>/dev/null; done ) &\n' % (run, run, run))
gw_dir = run + "/gateway"
sub1("\n_setup_proxy\n", "\n"
    "# --- Daytona 1k arm: transport = one async loopback gateway per node, no proxychains (research/2026-09-15_daytona_proxy_tokenization.md) ---\n"
    "# proxychains switches sockets to blocking mode while it negotiates SOCKS and stalls the whole coordinator event loop on every\n"
    "# reconnect (tokenization timeouts at 256 agents). The gateway accepts HTTP CONNECT on loopback and opens the SOCKS tunnel\n"
    "# asynchronously; the Daytona SDK picks it up from HTTPS_PROXY. Engines and Ray are plain http and stay direct (HTTP_PROXY unset).\n"
    "unset PROXYCHAINS_BIN_OVERRIDE PROXYCHAINS_SOCKS5_PRESET_HOST PROXYCHAINS_SOCKS5_PRESET_PORT PROXYCHAINS_SOCKS5_PRESET_AUTH PROXYCHAINS_CONF_FILE LD_PRELOAD SSH_KEY HTTP_PROXY http_proxy\n"
    "GW_DIR=%s; mkdir -p \"$GW_DIR\"; rm -f \"$GW_DIR\"/ready-* \"$GW_DIR\"/*.log\n"
    "srun --overlap --nodes=\"$SLURM_NNODES\" --ntasks=\"$SLURM_NNODES\" --ntasks-per-node=1 --cpus-per-task=1 --gres=none --export=ALL \\\n"
    "  bash -c 'PYTHONPATH=%s exec \"$RL_PYTHON\" -u %s --ready \"%s/ready-$(hostname -s)\" > \"%s/$(hostname -s).log\" 2>&1' &\n"
    "for _i in $(seq 1 120); do [ \"$(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)\" -ge \"$SLURM_NNODES\" ] && break; sleep 1; done\n"
    "if [ \"$(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)\" -lt \"$SLURM_NNODES\" ]; then echo \"FATAL: gateway ready on $(ls \"$GW_DIR\"/ready-* 2>/dev/null | wc -l)/$SLURM_NNODES nodes\" >&2; cat \"$GW_DIR\"/*.log; exit 95; fi\n"
    "export HTTPS_PROXY=http://127.0.0.1:18946 https_proxy=http://127.0.0.1:18946 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1\n"
    "if ! curl -sf -o /dev/null --connect-timeout 20 -x http://127.0.0.1:18946 -H \"Authorization: Bearer $DAYTONA_API_KEY\" https://app.daytona.io/api/api-keys/current; then\n"
    "  echo \"FATAL: Daytona API not reachable through the gateway\" >&2; exit 98\n"
    "fi\n"
    "echo \"daytona 1k arm: gateways ready on $SLURM_NNODES nodes; Daytona API reachable through 127.0.0.1:18946; HTTPS_PROXY set, HTTP_PROXY unset\"\n"
    % (gw_dir, gwdeps, gwpy, gw_dir, gw_dir))
sub1('setup_container_runtime "apptainer" "$WORKDIR" || exit $?\n', 'setup_container_runtime "daytona" "$WORKDIR" || exit $?\n')
eagle = "export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}\n"
sub1(eagle, eagle + (
    "# --- Daytona 1k arm: code under test shadows the venv's editable installs (harbor-marin b964a5f6, marinskyrl-marin 20032472) ---\n"
    "DS_HARBOR=%s   # 9940bdfd = lukedhlee/snowball-r2egym tip: create pacing, tokenize budget, plain $ prompt\n"
    "DS_SKYRL=%s   # f616e800 = lukedhlee/snowball-r2egym + create shares per coordinator\n"
    "# the launcher runs the entrypoint with cwd $SKYRL_HOME/skyrl-train, which wins over PYTHONPATH (gotchas 2026-09-13)\n"
    "export SKYRL_HOME=$DS_SKYRL RL_REPO_DIR=$DS_SKYRL\n"
    "export PYTHONPATH=$DS_HARBOR/src:$DS_SKYRL/skyrl-train:$DS_SKYRL/skyrl-gym:$DS_SKYRL:$PYTHONPATH\n"
    "DS_IMPORTS=$(cd \"$DS_SKYRL/skyrl-train\" && \"$RL_PYTHON\" -c 'import inspect, harbor, marinskyrl, skyrl_train, harbor.environments.daytona.utils as u, skyrl_train.trajectory_runners.harbor.rollout_dispatcher as r; print(harbor.__file__, u.__file__, marinskyrl.__file__, r.__file__, \"pace=%%s\" %% hasattr(u, \"pace_sandbox_create\"), \"shares=%%s\" %% (\"HARBOR_DAYTONA_CREATE_SHARES\" in inspect.getsource(r)), \"rate_per_process=%%s\" %% u.daytona_create_rate())' 2>&1 | tail -n 1) || true\n"
    "echo \"daytona 1k arm imports: $DS_IMPORTS\"\n"
    "case \"$DS_IMPORTS\" in \"$DS_HARBOR/src/harbor/__init__.py $DS_HARBOR/src/harbor/environments/daytona/utils.py $DS_SKYRL/marinskyrl/\"*\" $DS_SKYRL/skyrl-train/skyrl_train/trajectory_runners/harbor/rollout_dispatcher.py pace=True shares=True\"*) ;;\n"
    "  *) echo \"FATAL: the code under test is not what imports\" >&2; exit 97;;\n"
    "esac\n") % (dsh, dss))
launch = re.search(r'\n"\$RL_PYTHON" -m hpc\.rl_launch_utils --config "[^"]+" &\n', b); assert launch, "launch line"
b = b.replace(launch.group(0), (
    "\n# --- Daytona 1k arm: stop once step %s is logged (the fully-async trainer keeps generating past max_steps) ---\n"
    "DS_LOG=%s/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out\n"
    "( while ! grep -q 'WANDB_MIRROR kind=train step=%s ' \"$DS_LOG\" 2>/dev/null; do sleep 30; done\n"
    "  echo \"daytona 1k arm: step %s logged $(date +%%T), stopping\"; sleep 60\n"
    "  # signal FIRST: on 1826380 the tar of ~3 GB of result.json ran for >20 min and the scancel never came (gotchas 2026-09-16)\n"
    "  scancel -s USR1 -b \"$SLURM_JOB_ID\"\n"
    "  ( cd \"$SHM_TRIALS\" && find . \\( -name '*result.json' -o -name exception.txt \\) -print0 | tar --null -cf %s/trials_meta_stop.tar -T - ) 2>/dev/null &\n"
    "  sleep 240; scancel \"$SLURM_JOB_ID\" ) &\n") % (steps, run, steps, steps, run) + launch.group(0), 1)
open(sb, "w").write(b)
print("sbatch:", sb)
PY
bash -n $SB
grep -q "^#SBATCH --job-name=$DST$" $SB && grep -q "^#SBATCH --nodes=$NODES$" $SB
echo "--- sbatch diff vs $SRC (names/hashes normalised):"
diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $SB) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/sbatch/${SRC}_rl.sbatch | sed 's/dtn_[0-9a-f]*/dtn_<redacted>/') || true
echo "--- hydra diff vs $SRC:"
diff <(python3 -c "import json;print('\n'.join(json.load(open('$CFG'))['skyrl_hydra_args']))" | sed "s/$DST/NAME/g") \
     <(python3 -c "import json;print('\n'.join(json.load(open('$E/$SRC/configs/${SRC}_rl_config.json'))['skyrl_hydra_args']))" | sed "s/$SRC/NAME/g") || true
echo "built, not submitted. submit: cd $OTA && DCFT=\$PWD sbatch $SB"
