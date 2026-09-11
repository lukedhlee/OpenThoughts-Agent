#!/bin/bash
# clone_newstack_arm.sh <dst_name> [submit=0] [extra hydra args...] — a FRESH arm (from the Stage-3 base) on the tt-v2 split
# Env: NODES=40|20 (default 40; 20 = 8 policy + 12 generator, fsdp 32, 12 engines, 1,056 seats / 16 coordinators),
#      SPEC=0|1 (default 0; 1 = serve with the adapted EAGLE-3 draft: the marin_vllm_eagle3 tree on PYTHONPATH +
#      generator.engine_init_kwargs.speculative_config, as in snowball_ttband_sdon_5n_a; acceptance length 2.5 healthy).
# on the NEW stack (venv snowball-v2, harbor-marin, marinskyrl-marin): clones snowball_ttband_migsmoke_v2_c (the 6-node
# migration smoke, 2026-09-10) and grows it to the old fresh arm's size (snowball_ttband_v2train_fulldist_gmm_seats1584_x16_a):
# 40 nodes = 16 policy + 24 generator, 1,584 seats, batch 64 x 8, 24 coordinators, ckpt every 6, HF export every 12;
# trains 3 epochs of r2egym-tt-v2-train-basecurr-x16 (1,003 tasks -> ~48 steps); val = r2egym-tt-v2-val441 with in-run eval OFF
# (held-out scores come from external probes on the exports, same as the old arm). Keeps the smoke's clean settings
# (verifier 2400, preserve_logprobs_on_timeout false, connect timeout 120) and adds the two scrollback knobs the recipe
# now carries. Derived from clone_fresh_arm.sh (store image + mount hash + mmlaion symlink).
# 2026-09-11 re-sync onto marin main (harbor e05e2532, MarinSkyRL 4db3ab39): drops the retired
# generator.engine_init_kwargs.chat_template_content_format arg (MarinSkyRL #545 lets vLLM own rendering; the key would now
# reach vLLM's engine kwargs), and carries the audited error policy: the TITO prefix-mismatch ValueError and raw OpenAI transport
# errors are masked (they were trained as reward 0), the dead ContextLengthExceededError mask entry and the removed
# TmuxSessionLostError class are gone, and TmuxSessionEndedError (upstream's new name, classified infrastructure) is scored 0
# so a shell the model killed stays a failure.
set -euo pipefail
SRC=snowball_ttband_migsmoke_v2_c; DST=$1; SUBMIT=${2:-0}; shift 2 2>/dev/null || shift $#; EXTRA=("$@")
NODES=${NODES:-40}; SPEC=${SPEC:-0}
case "$NODES" in 40) POL=16; ENG=24; FSDP=64; SEATS=1584; COORD=24;; 20) POL=8; ENG=12; FSDP=32; SEATS=1056; COORD=16;; *) echo "NODES must be 40 or 20"; exit 1;; esac
DRAFT=/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3; EAGLE_TREE=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3
[ "$SPEC" = 1 ] && { [ -d $DRAFT ] && [ -d $EAGLE_TREE/vllm ] || { echo "draft or eagle3 vllm tree missing"; exit 1; }; }
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; C=/e/project1/transfernetx/lee27/code/snowball
case "$DST" in *"$SRC"*) echo "dst name must not contain the src name"; exit 1;; esac
case "$DST" in *snowball_ttband*) ;; *) echo "dst must contain snowball_ttband (store_reaper)"; exit 1;; esac
[ -d $T/r2egym-tt-v2-train-basecurr-x16 ] && [ -d $T/r2egym-tt-v2-val441 ] || { echo "task trees missing"; exit 1; }
[ -d $E/$DST ] && squeue -h -u $USER -n $DST -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
M=/e/data1/mmlaion/lee27/experiments; mkdir -p $M
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed -i "s/^#SBATCH --nodes=6$/#SBATCH --nodes=$NODES/; s/^#SBATCH --time=04:00:00$/#SBATCH --time=12:00:00/" $SB
grep -q "^#SBATCH --nodes=$NODES$" $SB && grep -q "^#SBATCH --time=12:00:00$" $SB || { echo "sbatch nodes/time sed failed"; exit 1; }
if [ "$SPEC" = 1 ]; then
  # the EAGLE-3-capable vLLM tree shadows the venv vLLM (same commit + the grugmoe draft support), as sdon_5n_a ran it
  sed -i "/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120$/i export PYTHONPATH=$EAGLE_TREE\${PYTHONPATH:+:\$PYTHONPATH}" $SB
  grep -q "^export PYTHONPATH=$EAGLE_TREE" $SB || { echo "eagle3 PYTHONPATH sed failed"; exit 1; }
  EXTRA+=("++generator.engine_init_kwargs.speculative_config={method:eagle3,model:$DRAFT,num_speculative_tokens:3}")
fi
# the two scrollback knobs the recipe carries (MarinSkyRL b3a288bb): read whole test logs, same window on both backends
sed -i "/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120$/a export HARBOR_TMUX_CAPTURE_BUDGET_CHARS=400000\nexport HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000" $SB
grep -q "^export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000$" $SB || { echo "scrollback env sed failed"; exit 1; }
python3 - $E/$DST/configs/${DST}_rl_config.json $SRC "$T" "$NODES:$POL:$ENG:$FSDP:$SEATS:$COORD" "${EXTRA[@]}" <<'PY'
import json, sys
p, src, T, geo, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
NODES, POL, ENG, FSDP, SEATS, COORD = geo.split(":")
assert not any(src in x for x in a), "src name survived the sed"
def setk(prefix, val):
    i = [k for k, x in enumerate(a) if x.lstrip("+").startswith(prefix)]; assert len(i) == 1, (prefix, i)
    a[i[0]] = a[i[0]][: a[i[0]].index(prefix)] + prefix + val
val = "%s/r2egym-tt-v2-val441" % T
setk("data.val_data=", '["%s"]' % val); c["val_data"] = [val]; c["val_data_sources"] = [val]
train = "%s/r2egym-tt-v2-train-basecurr-x16" % T   # 1,003 tasks: 728 train + 275 base-learnable rest (curriculum_build.py, 2026-09-11)
setk("data.train_data=", '["%s"]' % train); c["train_data"] = [train]
if "train_data_sources" in c: c["train_data_sources"] = [train]
# a fresh arm that RESUMES on chain restart (the smoke ran resume_mode=none), KL off (Luke 2026-09-11)
setk("trainer.resume_mode=", "latest")
setk("trainer.algorithm.use_kl_loss=", "false"); setk("trainer.algorithm.kl_loss_coef=", "0.0")
setk("trainer.epochs=", "3")                              # 3 epochs of 1,003 tasks / 64 = ~48 steps; max_steps 200 stays as the cap
setk("trainer.max_steps=", "200")
setk("trainer.train_batch_size=", "64"); setk("trainer.policy_mini_batch_size=", "64")
setk("trainer.placement.policy_num_nodes=", POL); setk("trainer.placement.ref_num_nodes=", POL)
setk("trainer.policy.fsdp_config.fsdp_size=", FSDP); setk("trainer.ref.fsdp_config.fsdp_size=", FSDP)
setk("generator.num_inference_engines=", ENG)             # (NODES - POL) nodes x 4 GPUs / (4 dp x 1 tp)
setk("generator.eval_n_samples_per_prompt=", "8")
setk("terminal_bench_config.harbor.n_concurrent_trials=", SEATS)
setk("trajectory_runner.process_pool.num_coordinators=", COORD)   # 66 trials per coordinator, the proven ceiling
# re-sync 2026-09-11: retired engine kwarg + audited error policy (see header)
a[:] = [x for x in a if not x.lstrip("+").startswith("generator.engine_init_kwargs.chat_template_content_format=")]
assert not any("chat_template_content_format" in x for x in a), "content-format arg survived"
mask = ["BridgeOutageError", "BridgeOperationError", "VerifierInfrastructureError", "EnvironmentStartTimeoutError",
        "NetworkError", "ConnectionError", "RewardFileNotFoundError", "RewardFileEmptyError", "AgentEnvironmentTimeoutError",
        "ConnectionResetError", "BridgeOperationTimeoutError", "RuntimeError", "VerifierTimeoutError", "TrialNotScoredError",
        "VerificationNotCompletedError", "ValueError", "APIConnectionError", "BadRequestError"]
setk("terminal_bench_config.harbor.mask_exceptions=", json.dumps(mask, separators=(",", ":")))
zero_key = "terminal_bench_config.harbor.zero_exceptions="
if not any(x.lstrip("+").startswith(zero_key) for x in a):
    a.append("++" + zero_key + json.dumps(["TmuxSessionEndedError"], separators=(",", ":")))
else:
    setk(zero_key, json.dumps(["TmuxSessionEndedError"], separators=(",", ":")))
c["num_nodes"] = int(NODES)
for x in extra:
    pre = x.split("=")[0].lstrip("+") + "="; i = [k for k, y in enumerate(a) if y.lstrip("+").startswith(pre)]
    assert len(i) <= 1, (x, i)
    if i: a[i[0]] = x
    else: a.append(x)
    print("extra:", x, "(replaced)" if i else "(appended)")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
keys = ("resume", "train_data", "val_data", "staleness", "optimizer_config.lr", "run_name", "epochs", "max_steps", "train_batch", "use_tis",
        "num_inference_engines", "n_concurrent_trials", "ckpt_interval", "hf_save_interval", "eval_interval", "grouped_mm", "num_coordinators",
        "verifier_override", "preserve_logprobs", "skip_special", "collect_rollout", "policy_num_nodes", "fsdp_size",
        "mask_exceptions", "zero_exceptions", "passthrough_exceptions", "speculative_config")
print("model_path:", c["model_path"], "num_nodes:", c["num_nodes"]); print("arm hydra args:", *[x for x in a if any(k in x for k in keys)], sep="\n   ")
PY
python3 $C/fix_merged_keys.py $E/$DST/configs/${DST}_rl_config.json
/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python $C/validate_hydra_args.py $E/$DST/configs/${DST}_rl_config.json
grep -q "job-name=$DST" $SB
IMG=$E/$DST/$DST/artifact_store.img; SRC_IMG=$E/$SRC/$SRC/artifact_store.img
OLDH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$SRC_IMG')).name.rsplit('-', 1)[-1])")
NEWH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$IMG')).name.rsplit('-', 1)[-1])")
sed -i "s/$OLDH/$NEWH/g" $SB $E/$DST/configs/${DST}_rl_config.json
grep -q "ARTIFACT_STORE_MOUNT=\"/tmp/otagent-artifact-stores/$DST-$NEWH\"" $SB
(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import ensure_image; ensure_image(Path('$IMG'), size='1T', inode_count=50_000_000)") >/dev/null
[ -f $IMG ] && echo "artifact store: $IMG mount hash $OLDH -> $NEWH"
echo "--- config diff vs smoke c (names/hashes normalised):"
diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $E/$DST/configs/${DST}_rl_config.json) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/configs/${SRC}_rl_config.json) || true
echo "--- sbatch diff vs smoke c:"; diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $SB) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/sbatch/${SRC}_rl.sbatch) || true
[ "$SUBMIT" = 1 ] && cd $OTA && DCFT=$PWD sbatch $SB || echo "not submitted (SUBMIT=$SUBMIT): $SB"
