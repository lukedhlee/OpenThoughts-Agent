#!/bin/bash
# build_ota_darm.sh <dst> <model> <tree> — build (never submit) a 12-node Daytona RL arm from an SFT export on its own
# learnable band. The scientific recipe is the Kimi SFT->RL arm's (snowball_ttband_tailsft_l453_12n_a: 12 nodes = 4 policy
# fsdp 16 + 8 engine nodes DP4 EP4 EAGLE-3, batch 64 x 8, lr 5e-7, staleness 2, 16 coordinators, 24 steps); what changes is
# the model, the task tree, the sandbox backend (Daytona, from build_darm.sh) and the seat count. The in-run val441 eval is
# OFF (trainer.eval_interval=0): the Kimi arms lingered 40 min in it after step 24 and the policy eval is the verdict anyway.
# Two arms run at once at SEATS=512/448 SHARES=32 so the pair stays under one user's ~1,000 concurrent sandboxes and the
# 5 creates/s org budget. Runs on the Jupiter login node with python3 (stdlib) only.
#   build_ota_darm.sh snowball_ttband_otastd_12n_a <export-step630-hf-bf16 dir> /e/fscratch/reformo/lee27/tasks/screen_otastd_20260919_learnable_x16
set -euo pipefail
SRC=snowball_ttband_tailsft_l453_12n_a
DST=${1:?dst}; MODEL=${2:?model dir}; TREE=${3:?learnable task tree}
NODES=12; POL=4; ENG=8; FSDP=16; BATCH=64; STEPS=${STEPS:-24}; WALL=${WALL:-08:00:00}
SEATS=${SEATS:-512}; COORD=${COORD:-16}; SHARES=${SHARES:-32}; ACCOUNT=${ACCOUNT:-open-sci-mm}
E=/e/fscratch/reformo/lee27/experiments; M=/e/data1/mmlaion/lee27/experiments
OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; C=/e/project1/transfernetx/lee27/code/snowball
DS_HARBOR=/e/fscratch/reformo/lee27/code/harbor-pacing; DS_SKYRL=/e/fscratch/reformo/lee27/code/marinskyrl-shares
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env; SOCKSF=/e/fscratch/reformo/lee27/keys/socks5_currease.env
GWPY=/e/fscratch/reformo/lee27/experiments/daytona_rl_arm1k/async_socks_connect_proxy.py
GWDEPS=/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps
[ -f $GWPY ] && [ -d $GWDEPS/python_socks ] || { echo "gateway script or its deps missing"; exit 1; }
[ -f $MODEL/config.json ] || { echo "no config.json under $MODEL"; exit 1; }
[ -d $TREE ] && [ "$(ls $TREE | wc -l)" -ge 16 ] || { echo "task tree $TREE missing or empty"; exit 1; }
for t in $(ls $TREE | head -3); do [ -f $TREE/$t/tests/required_tests.json ] || { echo "$TREE/$t is not a Daytona task (no tests/required_tests.json)"; exit 1; }; done
case "$DST" in *snowball_ttband*) ;; *) echo "dst must contain snowball_ttband (store_reaper)"; exit 1;; esac
squeue -h -u "$USER" -n "$DST" -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
[ "$(git -C $DS_HARBOR rev-parse --short=8 HEAD)" = 9940bdfd ] && [ -z "$(git -C $DS_HARBOR status --short)" ] || { echo "harbor-pacing is not a clean 9940bdfd"; exit 1; }
[ "$(git -C $DS_SKYRL rev-parse --short=8 HEAD)" = f616e800 ] && [ -z "$(git -C $DS_SKYRL status --short)" ] || { echo "marinskyrl-shares is not a clean f616e800"; exit 1; }
NTASK=$(ls $TREE | sed -E 's/__r[0-9]+$//' | sort -u | wc -l)
echo "tree $TREE: $(ls $TREE | wc -l) entries, $NTASK tasks"

# --- 1. run dir (data1, symlinked from fscratch like every new-stack arm) ---
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
CFG=$E/$DST/configs/${DST}_rl_config.json; SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $CFG
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $SB

# --- 2. config: model, tree, backend, seats, eval off ---
python3 - "$CFG" "$SRC" "$TREE" "$MODEL" "$NODES:$POL:$ENG:$FSDP:$BATCH:$SEATS:$COORD:$STEPS" <<'PY'
import json, os, sys
p, src, tree, model, geo = sys.argv[1:6]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
NODES, POL, ENG, FSDP, BATCH, SEATS, COORD, STEPS = geo.split(":")
assert not any(src in x for x in a), "src name survived the sed"
def idx(prefix): return [k for k, x in enumerate(a) if x.lstrip("+").startswith(prefix)]
def setk(prefix, val):
    i = idx(prefix); assert len(i) == 1, (prefix, i)
    a[i[0]] = a[i[0]][: a[i[0]].index(prefix)] + prefix + val
def getk(prefix):
    i = idx(prefix); assert len(i) == 1, (prefix, i); return a[i[0]].split("=", 1)[1]
old_model = getk("trainer.policy.model.path=")
assert getk("trainer.ref.model.path=") == old_model and c["model_path"] == old_model
assert getk("generator.engine_init_kwargs.served_model_name=") == os.path.basename(old_model)
# model: the SFT export under test (policy, ref, served name, launcher field)
setk("trainer.policy.model.path=", model); setk("trainer.ref.model.path=", model)
setk("generator.engine_init_kwargs.served_model_name=", os.path.basename(model)); c["model_path"] = model
# geometry asserted equal to the Kimi arm (12 nodes), steps set
for k, v in {"trainer.placement.policy_num_nodes=": POL, "trainer.placement.ref_num_nodes=": POL, "trainer.policy.fsdp_config.fsdp_size=": FSDP,
             "trainer.ref.fsdp_config.fsdp_size=": FSDP, "generator.num_inference_engines=": ENG, "trainer.train_batch_size=": BATCH,
             "trainer.policy_mini_batch_size=": BATCH}.items(): assert getk(k) == v, (k, getk(k), v)
assert c["num_nodes"] == int(NODES)
setk("trainer.max_steps=", STEPS)
setk("terminal_bench_config.harbor.n_concurrent_trials=", SEATS)
setk("trajectory_runner.process_pool.num_coordinators=", COORD)
setk("trainer.resume_mode=", "latest")                                   # fresh dir = starts from the export; a requeue resumes
# tree: the SFT export's own learnable band; val = same tree, eval OFF (the Kimi arms lingered 40 min in val441 after step 24)
setk("data.train_data=", '["%s"]' % tree); setk("data.val_data=", '["%s"]' % tree)
setk("trainer.eval_interval=", "0")
# backend: Daytona (as build_darm.sh)
setk("terminal_bench_config.harbor.environment_type=", "daytona")
assert not idx("terminal_bench_config.harbor.auto_snapshot="); a.append("++terminal_bench_config.harbor.auto_snapshot=true")
mask = json.loads(getk("terminal_bench_config.harbor.mask_exceptions="))
daytona_infra = ["DaytonaError", "DaytonaRateLimitError", "DaytonaTimeoutError", "DaytonaNotFoundError", "DaytonaConflictError",
                 "DaytonaSandboxStopError", "SandboxBuildFailedError", "SetupScriptError"]
setk("terminal_bench_config.harbor.mask_exceptions=", json.dumps(mask + [m for m in daytona_infra if m not in mask], separators=(",", ":")))
must = {"trainer.algorithm.use_kl_loss=": "false", "trainer.fully_async.max_staleness_steps=": "2",
        "trainer.policy.optimizer_config.lr=": "5e-7", "terminal_bench_config.harbor.verifier_override_timeout_sec=": "2400",
        "terminal_bench_config.harbor.preserve_logprobs_on_timeout=": "false", "generator.n_samples_per_prompt=": "8",
        "trainer.algorithm.use_tis=": "true", "trainer.eval_before_train=": "false", "trainer.ckpt_interval=": "6", "trainer.hf_save_interval=": "12",
        "terminal_bench_config.harbor.zero_exceptions=": '["TmuxSessionEndedError"]', "generator.max_input_length=": "49152"}
for k, v in must.items(): assert getk(k) == v, (k, getk(k), v)
assert "method:eagle3" in getk("generator.engine_init_kwargs.speculative_config="), "spec decode missing"
assert getk("terminal_bench_config.trials_dir=").startswith("/dev/shm/otagent_trials/")
c.update(harbor_env="daytona", train_data=[tree], val_data=[tree], train_data_sources=[tree], val_data_sources=[tree])
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print("config:", p)
PY
python3 $C/fix_merged_keys.py $CFG
sed "s#/e/project1/transfernetx/lee27/code/marinskyrl-marin/skyrl-train#$DS_SKYRL/skyrl-train#" $C/validate_hydra_args.py > $E/$DST/configs/validate_hydra_args_ds.py
grep -q "$DS_SKYRL/skyrl-train/skyrl_train/config" $E/$DST/configs/validate_hydra_args_ds.py
OMP_NUM_THREADS=1 /e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python $E/$DST/configs/validate_hydra_args_ds.py $CFG

# --- 3. artifact store ---
IMG=$E/$DST/$DST/artifact_store.img; SRC_IMG=$E/$SRC/$SRC/artifact_store.img
OLDH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$SRC_IMG')).name.rsplit('-', 1)[-1])")
NEWH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$IMG')).name.rsplit('-', 1)[-1])")
sed -i "s/$OLDH/$NEWH/g" $SB $CFG
grep -q "ARTIFACT_STORE_MOUNT=\"/tmp/otagent-artifact-stores/$DST-$NEWH\"" $SB
(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import ensure_image; ensure_image(Path('$IMG'), size='1T', inode_count=50_000_000)") >/dev/null
[ -f $IMG ]

# --- 4. sbatch: account, wall, key, backend, gateway, code under test, step stopper ---
python3 - "$SB" "$DST" "$NODES" "$WALL" "$KEYF" "$SOCKSF" "$DS_HARBOR" "$DS_SKYRL" "$STEPS" "$E/$DST" "$GWPY" "$GWDEPS" "$SHARES" "$ACCOUNT" <<'PY'
import re, sys
sb, dst, nodes, wall, keyf, socksf, dsh, dss, steps, run, gwpy, gwdeps, shares, account = sys.argv[1:15]
b = open(sb).read()
def sub1(old, new):
    global b
    assert b.count(old) == 1, ("anchor count", old, b.count(old)); b = b.replace(old, new)
assert b.count("#SBATCH --nodes=%s\n" % nodes) == 1, "node count"
sub1("#SBATCH --time=12:00:00\n", "#SBATCH --time=%s\n" % wall)
sub1("#SBATCH --account reformo\n", "#SBATCH --account %s\n" % account)
key_block = ('# OTA Daytona arm: key read from the key file at run time (a trailing "# comment" on the line is stripped)\n'
     'DAYTONA_API_KEY_OVERRIDE="$(grep -m1 -E \'^(export )?DAYTONA_API_KEY=\' %s | cut -d= -f2- | sed -E \'s/[[:space:]]+#.*$//\' | tr -d "\\"\' ")"\n'
     '[ -n "$DAYTONA_API_KEY_OVERRIDE" ] || { echo "FATAL: no DAYTONA_API_KEY in %s" >&2; exit 96; }\n') % (keyf, keyf)
b, n = re.subn(r'^DAYTONA_API_KEY_OVERRIDE="dtn_[0-9a-f]+"\n', lambda m: key_block, b, flags=re.M)
assert n == 1 and "dtn_" not in b, "key override line"
sub1("export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000\n", "export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000\n" + (
    "# --- OTA Daytona arm: sandbox backend ---\n"
    "export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES=%s\n"
    "set -a; source %s; set +a\n"
    "unset APPTAINER_BRIDGE_URL\n") % (shares, socksf))
gw_dir = run + "/gateway"
sub1("\n_setup_proxy\n", "\n"
    "# --- OTA Daytona arm: transport = one async loopback gateway per node, no proxychains (build_darm.sh) ---\n"
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
    "echo \"ota daytona arm: gateways ready on $SLURM_NNODES nodes; Daytona API reachable through 127.0.0.1:18946; HTTPS_PROXY set, HTTP_PROXY unset\"\n"
    % (gw_dir, gwdeps, gwpy, gw_dir, gw_dir))
sub1('setup_container_runtime "apptainer" "$WORKDIR" || exit $?\n', 'setup_container_runtime "daytona" "$WORKDIR" || exit $?\n')
eagle = "export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}\n"
sub1(eagle, eagle + (
    "# --- OTA Daytona arm: code under test shadows the venv's editable installs ---\n"
    "DS_HARBOR=%s   # 9940bdfd: create pacing, tokenize budget, plain $ prompt\n"
    "DS_SKYRL=%s   # f616e800: create shares per coordinator\n"
    "export SKYRL_HOME=$DS_SKYRL RL_REPO_DIR=$DS_SKYRL\n"
    "export PYTHONPATH=$DS_HARBOR/src:$DS_SKYRL/skyrl-train:$DS_SKYRL/skyrl-gym:$DS_SKYRL:$PYTHONPATH\n"
    "DS_IMPORTS=$(cd \"$DS_SKYRL/skyrl-train\" && \"$RL_PYTHON\" -c 'import inspect, harbor, marinskyrl, skyrl_train, harbor.environments.daytona.utils as u, skyrl_train.trajectory_runners.harbor.rollout_dispatcher as r; print(harbor.__file__, u.__file__, marinskyrl.__file__, r.__file__, \"pace=%%s\" %% hasattr(u, \"pace_sandbox_create\"), \"shares=%%s\" %% (\"HARBOR_DAYTONA_CREATE_SHARES\" in inspect.getsource(r)), \"rate_per_process=%%s\" %% u.daytona_create_rate())' 2>&1 | tail -n 1) || true\n"
    "echo \"ota daytona arm imports: $DS_IMPORTS\"\n"
    "case \"$DS_IMPORTS\" in \"$DS_HARBOR/src/harbor/__init__.py $DS_HARBOR/src/harbor/environments/daytona/utils.py $DS_SKYRL/marinskyrl/\"*\" $DS_SKYRL/skyrl-train/skyrl_train/trajectory_runners/harbor/rollout_dispatcher.py pace=True shares=True\"*) ;;\n"
    "  *) echo \"FATAL: the code under test is not what imports\" >&2; exit 97;;\n"
    "esac\n") % (dsh, dss))
launch = re.search(r'\n"\$RL_PYTHON" -m hpc\.rl_launch_utils --config "[^"]+" &\n', b); assert launch, "launch line"
b = b.replace(launch.group(0), (
    "\n# --- OTA Daytona arm: stop once step %s is logged and its checkpoint is written (the fully-async trainer keeps generating past max_steps) ---\n"
    "DS_LOG=%s/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out\n"
    "( while ! grep -q 'WANDB_MIRROR kind=train step=%s ' \"$DS_LOG\" 2>/dev/null; do sleep 30; done\n"
    "  echo \"ota daytona arm: step %s logged $(date +%%T)\"\n"
    "  for _i in $(seq 1 60); do [ \"$(cat %s/%s/checkpoints/latest_ckpt_global_step.txt 2>/dev/null)\" = %s ] && break; sleep 30; done\n"
    "  echo \"ota daytona arm: stopping $(date +%%T)\"; sleep 120\n"
    "  scancel -s USR1 -b \"$SLURM_JOB_ID\"; sleep 240; scancel \"$SLURM_JOB_ID\" ) &\n") % (steps, run, steps, steps, run, dst, steps) + launch.group(0), 1)
open(sb, "w").write(b)
print("sbatch:", sb)
PY
bash -n $SB
grep -q "^#SBATCH --job-name=$DST$" $SB && grep -q "^#SBATCH --nodes=$NODES$" $SB && grep -q "^#SBATCH --account $ACCOUNT$" $SB
! grep -q "dtn_" $SB
echo "--- sbatch diff vs $SRC (names/hashes normalised):"
diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $SB) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/sbatch/${SRC}_rl.sbatch | sed 's/dtn_[0-9a-f]*/dtn_<redacted>/') || true
echo "--- hydra diff vs $SRC:"
diff <(python3 -c "import json;print('\n'.join(json.load(open('$CFG'))['skyrl_hydra_args']))" | sed "s/$DST/NAME/g") \
     <(python3 -c "import json;print('\n'.join(json.load(open('$E/$SRC/configs/${SRC}_rl_config.json'))['skyrl_hydra_args']))" | sed "s/$SRC/NAME/g") || true
echo "built, not submitted. submit: cd $OTA && DCFT=\$PWD sbatch $SB"
