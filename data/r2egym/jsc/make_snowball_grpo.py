#!/usr/bin/env python3
"""Generate a Snowball GRPO run (64k parity harness) from the probe generator's output, then patch for training.
Usage: make_snowball_grpo.py --name <n> --train-dir <band task dir> [--val-dir …] [--lr 1e-5] [--batch 64] [--n 8]
       [--fsdp 64] [--engines 8] [--conc-per-engine 64] [--eval-interval 6] [--eval-k 8] [--ckpt-interval 6]
       [--max-steps 200] [--warmup 0] [--wall 12:00:00] [--resume none|latest] [--submit]
Geometry: policy nodes = fsdp/4, engine nodes = engines (TP1xDP4xEP4 each); sbatch nodes = policy + engines.
Prints the sbatch path (and submits from the OTA root with --submit)."""
import argparse, json, os, re, subprocess, sys
EXP = "/e/fscratch/reformo/lee27/experiments"; C = "/e/project1/transfernetx/lee27/code/snowball"
OTA = "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--train-dir", required=True)
ap.add_argument("--val-dir", default="/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val")
ap.add_argument("--lr", default="1e-5"); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--fsdp", type=int, default=64); ap.add_argument("--engines", type=int, default=8)
ap.add_argument("--conc-per-engine", type=int, default=64)
ap.add_argument("--eval-interval", type=int, default=6); ap.add_argument("--eval-k", type=int, default=8)
ap.add_argument("--ckpt-interval", type=int, default=6); ap.add_argument("--hf-save-interval", type=int, default=12); ap.add_argument("--max-steps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=0); ap.add_argument("--wall", default="12:00:00")
ap.add_argument("--resume", default="none"); ap.add_argument("--max-in", type=int, default=61440); ap.add_argument("--max-out", type=int, default=4096)
ap.add_argument("--bridge-margin", type=int, default=600, help="HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC exported in the sbatch (harbor default 180)")
ap.add_argument("--submit", action="store_true")
a = ap.parse_args()
assert a.fsdp % 4 == 0 and os.path.isdir(a.train_dir) and os.path.isdir(a.val_dir), (a.fsdp, a.train_dir, a.val_dir)
assert a.batch * a.n >= a.fsdp, "mini-batch × n must cover every dp rank"
pol = a.fsdp // 4; nodes = pol + a.engines; conc = a.conc_per_engine * a.engines
gen = [sys.executable, f"{C}/make_snowball_probe.py", "--name", a.name, "--val-dir", a.val_dir, "--k", str(a.eval_k), "--conc", str(conc),
       "--max-in", str(a.max_in), "--max-out", str(a.max_out), "--max-model-len", "65536", "--nodes", str(4 + a.engines),
       "--engines", str(a.engines), "--parity", "--eval-timeout", "1800", "--wall", a.wall, "--lr", a.lr]
print(subprocess.check_output(gen, text=True).strip())
cp = f"{EXP}/{a.name}/configs/{a.name}_rl_config.json"; c = json.load(open(cp)); args = c["skyrl_hydra_args"]
drop = ("trainer.max_steps=", "trainer.eval_before_train=", "generator.n_samples_per_prompt=", "trainer.train_batch_size=",
        "trainer.policy_mini_batch_size=", "trainer.eval_interval=", "trainer.ckpt_interval=", "trainer.hf_save_interval=",
        "trainer.resume_mode=", "data.train_data=", "trainer.policy.fsdp_config.fsdp_size=", "trainer.ref.fsdp_config.fsdp_size=",
        "trainer.placement.policy_num_nodes=", "trainer.placement.ref_num_nodes=", "trainer.max_ckpts_to_keep=",
        "trainer.policy.optimizer_config.num_warmup_steps=", "trainer.eval_batch_size=")
args = [x for x in args if not x.startswith(drop)]
args += [
    f"trainer.max_steps={a.max_steps}", "trainer.eval_before_train=true", f"generator.n_samples_per_prompt={a.n}",
    f"trainer.train_batch_size={a.batch}", f"trainer.policy_mini_batch_size={a.batch}",
    f"trainer.eval_interval={a.eval_interval}", f"trainer.ckpt_interval={a.ckpt_interval}", f"trainer.hf_save_interval={a.hf_save_interval}",
    "trainer.max_ckpts_to_keep=2", f"trainer.resume_mode={a.resume}", f'data.train_data=["{a.train_dir}"]',
    f"trainer.policy.fsdp_config.fsdp_size={a.fsdp}", f"trainer.ref.fsdp_config.fsdp_size={a.fsdp}",
    f"trainer.placement.policy_num_nodes={pol}", f"trainer.placement.ref_num_nodes={pol}",
    f"trainer.policy.optimizer_config.num_warmup_steps={a.warmup}",
    f"trainer.eval_batch_size={-(-len([d for d in os.listdir(a.val_dir) if os.path.isdir(os.path.join(a.val_dir, d))]) // 4)}",  # 4 eval batches = one per fanout coordinator (n_concurrent/4 slots each); a single batch runs on one coordinator
]
c["skyrl_hydra_args"] = args; c["num_nodes"] = nodes
c["train_data"] = [a.train_dir]; c["train_data_sources"] = [a.train_dir]
json.dump(c, open(cp, "w"), indent=2)
sp = f"{EXP}/{a.name}/sbatch/{a.name}_rl.sbatch"; s = open(sp).read()
s = re.sub(r"^#SBATCH --nodes=.*$", f"#SBATCH --nodes={nodes}", s, flags=re.M)
s = re.sub(r"^export NUM_INFERENCE_ENGINES=.*$", f"export NUM_INFERENCE_ENGINES={nodes*4}", s, flags=re.M)
s = re.sub(r"^export POLICY_NUM_NODES=.*$", f"export POLICY_NUM_NODES={nodes}", s, flags=re.M)
# wider bridge-job margin for the batched exec (harbor lukedhlee/terminus2-think-parity-bridgewait): RL rollouts cannot retry a
# BridgeOperationTimeoutError, and the JURECA per-node worker queue held short execs > 3 min at ~2k live sandboxes (09-03)
s = s.replace("export NCCL_PXN_DISABLE=1\n", f"export NCCL_PXN_DISABLE=1\nexport HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC={a.bridge_margin}\n", 1)
assert f"HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC={a.bridge_margin}" in s
assert f"--nodes={nodes}" in s and f"NUM_INFERENCE_ENGINES={nodes*4}" in s and f"POLICY_NUM_NODES={nodes}" in s
open(sp, "w").write(s)
print(f"{a.name}: nodes={nodes} (policy {pol} @ fsdp {a.fsdp} + engines {a.engines}), rollouts/step={a.batch*a.n}, conc={conc}, lr={a.lr}, warmup={a.warmup}, eval every {a.eval_interval} (K={a.eval_k}), ckpt every {a.ckpt_interval} (keep 2, ~375 GB each), hf-export request every {a.hf_save_interval}, max_steps={a.max_steps}, resume={a.resume}")
print(sp)
if a.submit:
    out = subprocess.check_output(["sbatch", sp], cwd=OTA, env={**os.environ, "DCFT": OTA}, text=True).strip(); print(out)
