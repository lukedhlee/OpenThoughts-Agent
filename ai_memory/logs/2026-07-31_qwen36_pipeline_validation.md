# 2026-07-31 — Qwen3.6 pipeline-validation execution log

Continuation of `handoff.md`'s pipeline-validation run. The objective remains one honest GRPO step with
real reward variance and a finite update, not model quality.

## Git and checkout reconciliation

- Pushed OpenThoughts-Agent `lukedhlee/qwen3-6-r2egym-grpo` and MarinSkyRL
  `lukedhlee/apptainer-bridge-rl` to Luke-owned forks. No PRs.
- Raised Harbor JURECA worker `#SBATCH --time` and `CHAIN_TIME` from 6h to 24h, committed and pushed as
  `6b25eb16` on `lukedhlee/apptainer-opencode-bridge`. `MAX_CHAIN=0` remains the default.
- Audited Jupiter's bundle-derived and dirty trees. The three old Harbor commits at `a981a449` are
  patch-identical to canonical commits. Preserved divergent heads plus binary dirty patches/manifests
  under `/e/scratch/reformo/lee27/reconcile_backups_20260731` before any reset.
- Fetch + hard-reset, never pull: Jupiter execution checkouts now track the Luke forks. The historical
  dirty OpenThoughts-Agent and MarinSkyRL trees were left untouched.
- Replaced the launcher's plain-copy `/e/scratch/reformo/lee27/MarinSkyRL-apptainer-bridge` safely by
  moving it into the reconciliation archive and renaming the clean git checkout into that exact path.
- Cleaned four unused internal bridge tmux services on ports 9922/9924/9926/9928. The required bridge
  remains bound only to `10.128.1.2:9920`.

## Code fixes discovered during execution

- FlashQLA wheel metadata validation now canonicalizes underscores to hyphens in the installer and GRPO
  preflight (`760011d3` plus later integrated launcher changes).
- Training rollouts switched from Terminus2 to OpenCode 1.18.8 with preinstalled binary, pinned prompt,
  autoupdate off, and compaction off. Terminus-only keys were removed.
- MarinSkyRL now exposes `version`, `preinstalled`, `prompt_template_path`, and `opencode_config` through
  its Harbor schema; a Linux runtime probe built the final `TrialConfig` and verified those values plus
  the 28,672/4,096 `model_info` limits. Branch head after contract-test fixes: `cd641b1`.
- The GRPO wrapper no longer uses `find` to count GPFS tasks; it performs a bounded one-level `scandir`.
- The wrapper loads Jupiter CUDA 13 modules before its compiled-vLLM login-node preflight.

OpenThoughts-Agent execution branch head after these fixes: `a36f0408`.

## FlashQLA and checkpoint gates

- First installer attempt exposed the wheel-name normalization bug and cleaned its partial layer.
- FlashQLA 0.1.2 isolated layer installed at
  `/e/scratch/reformo/lee27/pydeps/qwen36-flashqla-0.1.2`.
- Job `1139040` compiled the real GH200 kernels but failed because the smoke supplied a BF16 gate while
  FlashQLA backward requires FP32 `dg`. Transformers Qwen3.6 actually produces FP32 gates from
  `a.float()`, so the smoke was corrected to match.
- Job `1139108` passed GH200 BF16 forward/backward with finite gradients and wrote
  `gpu_sm90_smoke.ok` (`torch=2.11.0+cu128`, `flash-qla=0.1.2`, `tilelang=0.1.9`, SM90).
- Staged `Qwen/Qwen3.6-35B-A3B` revision
  `995ad96eacd98c81ed38be0c5b274b04031597b0` verbatim and atomically at the required Jupiter path.
  Validation: plain `qwen3_5_moe`, no quantization config, 26/26 referenced safetensor shards present,
  GDN kernel dimension 4.
- Full launcher preflight passes: compiled vLLM, exact checkpoint, FlashQLA marker, 3,328 task dirs,
  bridge health, and `MAX_STEPS=1`. No RL job submitted yet.

## Current blocker / exact next action

The Jupiter bridge is healthy at internal-only `10.128.1.2:9920`, but JURECA's internal listener
`jrlogin04i:9920` is closed because the Jupiter→JURECA ControlMaster expired. The passphrase-protected
key exists with mode 600; the ControlMaster socket does not. One interactive TOTP establishment is
required on Jupiter. After that: recreate the internal-only reverse listener, submit two 24h JURECA
CPU worker nodes (16 workers/node, no chain), verify `workers_alive`, then submit the six-node one-step
GRPO smoke. Combined allocation will be 8 nodes, under the operator's hard 16-node ceiling.

## Later execution — launch defects and first real Qwen3.6 allocation

- MFA cleared. Jupiter ControlMaster PID `2944931`, internal reverse forward, bridge tmux
  `r2egym_bridge_9920`, and JURECA worker job `15486097` (2 nodes, 24h, 16 workers/node) became live.
  JURECA can curl the bridge through `jrlogin06i:9920`; listeners remain internal-only.
- Launch attempts exposed and fixed, in order: caller paths overwritten by cluster dotenv; generic
  Conda activation pointing at another user's tree; offline runtime paths/Ray spilling; literal
  `${APPTAINER_BRIDGE_URL}` causing Bash bad substitution; unquoted JSON in `container.extra_env`;
  and distributed W&B actors forcing online `mode="shared"` despite `WANDB_MODE=offline`.
- Final pushed heads after those corrections: OpenThoughts-Agent `fc262339`, MarinSkyRL `c0a7098`,
  Harbor `6b25eb16`. Focused OpenThoughts tests: 12 pass; Marin AST/import checks and Ruff pass.
- Attempt `1141941` formed a 6-node/24-GPU Ray cluster and proved the corrected W&B offline path. It
  recognized the exact plain Qwen3.6 checkpoint as the Qwen3.5/3.6 VLM shell, selected its text tower,
  and started eight TP1 vLLM 0.22.0 engines with `quantization=None`. Each loaded all 26 shards and
  ~64.69 GiB of weights.
- After inference initialization, policy construction failed before any rollout with
  `ModuleNotFoundError: torchtitan`. Root cause: grouped-GEMM MoE + Torch EP needs SkyRL's `ep` extra,
  but the borrowed `envs/rl-megatron` had only base/vLLM dependencies.
- Installed exact TorchTitan `a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41` plus locked `tomli` and
  `tyro` into the actual Jupiter RL venv. Both `torchtitan.distributed.expert_parallel` and
  `skyrl_train.models.layers.moe` import successfully.
- Replacement six-node smoke `1144362` submitted with identical one-step geometry. It waited in the
  Jupiter queue, then failed in 51 seconds before Ray with exit 127. This was a direct-resubmission
  error: the rendered sbatch depends on inherited `RL_ENV_DIR`, but it was submitted directly without
  that export. Activation therefore fell back to the nonexistent execution-checkout `envs/rl`, warned
  and continued. The runner's absolute Python path still started the driver, but RayCluster invokes
  bare `ray`; with no venv on `PATH`, all six steps exited 127. The correct venv and Ray 2.51.1 are
  intact. JURECA workers, bridge, and ControlMaster remained healthy and were never used by this job.

Next: make the rendered venv path self-contained and activation fail-fast, then launch through the
wrapper rather than directly resubmitting an old sbatch. Follow the new job through policy/ref load,
FlashQLA engagement, initial weight sync, 32 OpenCode
rollouts via JURECA, reward variance, finite update, second sync, checkpoint, and HF export. Once the
smoke no longer uses the path, run the bounded one-second tunnel disconnect/reconnect test and restore
production keepalives.
