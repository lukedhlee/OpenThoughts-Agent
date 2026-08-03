# JURECA agentic RL (Daytona) bring-up — plan + state

How the JURECA agentic-RL runtime was finally made to import (the borrowed env + the `examples`
PYTHONPATH shim + the pyoverlay), after the fresh-venv flash-attn build wall and the discovery that the
co-lead's own path is Apptainer, not Daytona.
Read for the runtime recipe and the launcher anatomy. ⚠ Sections are layered oldest-first; the ✅
RUNTIME UNBLOCKED section supersedes BOTTOM LINE, OPEN DECISION, and Next below it.

Goal: smoke-test **agentic** RL (Harbor drives agent rollouts in **cloud Daytona** sandboxes →
verifier reward → SkyRL GRPO/RLOO update) on JURECA, for lee27. Distinct from the sandbox-free
gsm8k A/B (see [[jureca_grpo_vs_rloo_plan]]).

## Path decision (resolved by elimination)
- **dc-agent NOT required** and NOT available (private, `gh` 404). It's Marianna's Apptainer-DinD
  JURECA orchestrator.
- OTA's own launcher `python -m hpc.launch --job_type rl` + **MarinSkyRL fork + Harbor + cloud Daytona**
  is the viable route. Since we can't do Marianna's Apptainer route (no dc-agent), **cloud Daytona is
  the route** — which is what "using daytona" means anyway.

## Anatomy (SLURM path)
`hpc.launch --job_type rl` → `hpc/rl_launch_utils.py:launch_rl_job` → parses skyrl yaml
(`entrypoint: examples.terminal_bench.entrypoints.main_tbench`), extracts tasks from `--train_data`,
**pre-builds Daytona snapshots on the login node** (`hpc/snapshot_manager.py`, caps: 10 new/launch,
60 org — HARD), renders `hpc/sbatch_rl/universal_rl.sbatch`, submits. In-job: proxychains egress →
Ray cluster → SkyRL TerminalBenchGenerator drives `n_concurrent_trials` Harbor terminus-2 episodes,
each a cloud Daytona sandbox (create→agent tmux tool-calls vs colocated vLLM→verifier reward→delete)
→ GRPO/RLOO FSDP update → weight sync → repeat.

## 4 gaps vs our gsm8k setup (all real)
1. **Framework**: public NovaSky SkyRL (our gsm8k venv) has NO `terminal_bench` entrypoint. Need
   **MarinSkyRL (`marin-community/MarinSkyRL`, penfever/working) + Harbor (`marin-community/harbor`)**,
   newer torch/vLLM → **fresh venv** (NOT the gsm8k skyrl-venv).
2. **No jureca skyrl config** → AUTHORED `hpc/skyrl_yaml/jureca/smoke.yaml` (see below).
3. **`jureca` HPC entry was wrong** (westai0007/dc-hwai/H100) → FIXED to synthlaion/dc-gpu-devel/A100.
4. **Egress unverified** for RL on JURECA (shared SOCKS proxy 10.14.0.53:1080 + proxychains_preload
   `/p/scratch/laionize/raj3/proxychains-ng/libproxychains4.so`) → test on a devel node.

## Guardrails (Daytona)
Shared org (populated — /api/sandbox returned 30KB). ≤6 RUNNING RL jobs/cluster; snapshot caps HARD
(10 new/launch, 60 org) — clean stale, never raise; enable_db_registration:false (in the config);
tiny task set with few unique envs so prebuild stays under caps.

## Progress (2026-07-14)
- **Branch**: `lukedhlee/jureca-rl-test` off `origin/penfever/working` (main was 245 behind).
  No push access to `open-thoughts/OpenThoughts-Agent` → pushed to **fork** `lukedhlee/OpenThoughts-Agent`
  (remote `fork`). Migrate later via cross-fork PR or once granted collaborator access. `origin` stays upstream.
- **Config edits committed** (commit b80d3127, pushed to fork):
  - `hpc/hpc.py` jureca entry → account=synthlaion, partition=dc-gpu-devel (2h; prod=dc-gpu/24h),
    gpus_type=A100 40GB, cpus_per_node=128, total_partition_nodes=180. (NO --partition/--account CLI
    override exists for the RL launcher → had to edit the entry.)
  - `hpc/skyrl_yaml/jureca/smoke.yaml` NEW — 1-node/4xA100-40GB Qwen3-0.6B, colocate_all, 3 steps,
    n_concurrent_trials=16, TIS off, trace_upload off, logger console. FIRST DRAFT — GPU smoke validates it.
- **Daytona key VALID** (from Mac: /api/snapshots + /api/sandbox → 200).
- **JURECA→GitHub auth**: port 22 blocked; SSH-over-443 works. Generated `~/.ssh/id_ed25519_github` on
  JURECA + `~/.ssh/config` routes github.com→ssh.github.com:443. **PENDING: Luke adds the pubkey to
  https://github.com/settings/ssh/new** (pubkey: ssh-ed25519 AAAAC3...X1Q7 lee27@jureca-github).

## Step 2 progress (2026-07-14) — runtime build WALL
- GitHub auth WORKS (pubkey added; SSH-over-443). Cloned on JURECA under code/: MarinSkyRL@d75c3f8,
  harbor@4ac18dc, OpenThoughts-Agent@b80d3127 (our fork branch).
- OTA RL runtime model: conda env via `--rl_use_conda --rl_conda_env`; MarinSkyRL on PYTHONPATH via
  SKYRL_HOME (resolve_rl_repo.sh probes SkyRL|MarinSkyRL); entrypoint run as `-m examples.terminal_bench...`.
- **Marianna's env (`/p/project1/ccstdl/envs/marianna/py3.12`) ALMOST works**: torch 2.10+cu128, vllm 0.18,
  ray 2.51, `daytona` SDK, and our harbor 0.1.24 (via PYTHONPATH=harbor/src) ALL import; our MarinSkyRL
  skyrl_train/skyrl_gym import. TWO blockers: (1) NO harbor installed (fixed via PYTHONPATH); (2) a rogue
  `examples` pkg in her site-packages (basic_example.py + __init__.py) SHADOWS MarinSkyRL's namespace
  `examples/` (no __init__.py) → `examples.terminal_bench...` entrypoint won't resolve. Read-only env → can't strip it.
- **Fresh venv build FAILED**: `uv sync --extra vllm` in MarinSkyRL/skyrl-train dies on `flash-attn==2.6.3`
  (`match-runtime=true` + no static metadata). MarinSkyRL vllm extra pins torch==2.11.0+cu128 and
  source-builds flash-attn + transformer-engine — prod avoids this with a PREBUILT wheel cache (gpu-rl Docker).
  Partial venv left at /p/scratch/synthlaion/lee27/envs/agentic-venv (incomplete).

## MINED Marianna's JURECA agentic RL recipe (2026-07-14) — reframes everything
From `/p/project1/laionize/marianna/dc_agent/bash-scripts/run_rl_bridge_*.sh`:
- **Uses the APPTAINER BRIDGE, not cloud Daytona.** env class
  `harbor.environments.apptainer_bridge.harbor_env:BridgeApptainerEnvironment` + `APPTAINER_BRIDGE_URL`
  (http://10.128.1.1:9920) + `sif_cache`. Cloud Daytona (app.daytona.io via proxychains) is used only for
  her **eval** (`eval_swebench.sbatch`), NOT RL.
- Runtime: mamba env `/e/data1/datasets/playground/ot/envs/py3.12` (+ `py3.12-flashrl`), SkyRL = **BenSkyRL**
  (`/e/project1/jureap59/marianna/ot/BenSkyRL`), harbor = **harbor_patched** (her fork w/ the bridge env),
  PYTHONPATH=harbor_patched:...:SKYRL_HOME/skyrl-train:.../examples. Proxychains at jureap59.
- **lee27 has NO ACCESS** to her env / BenSkyRL / harbor_patched (all under `jureap59` + `/e/` mounts →
  "NO ACCESS"). So NO reuse shortcut. The `/p/project1/laionize/marianna/dc_agent/synthlaion-grant/run_rl_gsm8k.sh`
  is readable (SKYRL_HOME=.../dc_agent/SkyRL shared, DCFT_SCRATCH=/p/scratch/synthlaion/dc-agent-shared) but
  it's NON-agentic gsm8k (no harbor) — not the agentic runtime.

## ✅ RUNTIME UNBLOCKED (2026-07-14) — route (0), NO Marianna contact
The agentic entrypoint now imports end-to-end. Validated recipe (import-level; GPU smoke still pending):
- **Runtime python:** `/p/project1/ccstdl/envs/marianna/py3.12/bin/python` — world-readable (0777, mmlaion;
  lee27 in mmlaion). Has torch 2.10+cu128, vLLM 0.18, **flash_attn 2.8.3 (compiled → no build wall)**, daytona SDK.
- **PYTHONPATH (in order):** `$SHIM:$OV:$R:$H`
  - `SHIM=/p/scratch/synthlaion/lee27/pythonpath` — `examples/__init__.py` shim: a REGULAR pkg (out-ranks the
    rogue `examples` regular pkg in the borrowed env's site-packages) whose `__path__` redirects to MarinSkyRL's
    real examples tree. Beats the shadow WITHOUT editing the read-only env or the marin-community checkout.
  - `OV=/p/scratch/synthlaion/lee27/pyoverlay` — `pip install --target` overlay for pure-Python harbor deps
    missing from the env: `universal_pathlib` (upath) + fsspec + pathlib-abc. Add more here as discovered.
  - `R=/p/project1/synthlaion/lee27_jureca/code/MarinSkyRL/skyrl-train` (SKYRL_HOME)
  - `H=/p/project1/synthlaion/lee27_jureca/code/harbor/src`
- **Verified:** `HF_HUB_OFFLINE=1 PYTHONPATH=$SHIM:$OV:$R:$H python -c "import examples.terminal_bench.entrypoints.main_tbench"` → OK.
- **Caveat:** it's Marianna's env — could move/vanish; fine for a smoke, build our own (route a) for anything durable.
- **NEXT:** wire SHIM+OV onto the launcher's PYTHONPATH (launcher sets SKYRL_HOME via resolve_rl_repo.sh — check
  how to prepend extra paths), run with `--rl_use_conda --rl_conda_env /p/project1/ccstdl/envs/marianna/py3.12`;
  then egress test (ssh-tunnel) → snapshot prebuild dry-run (≤10 new) → GPU agentic smoke (needs GPU + explicit go).

## BOTTOM LINE (agentic on JURECA) — SUPERSEDED by the ✅ section above
No shortcut exists for us: her runtime is inaccessible, JURECA agentic = Apptainer (not Daytona), and our
fresh cloud-Daytona build hit the flash-attn wall. Realistic paths: (a) BUILD our own compiled runtime
(prebuilt flash-attn wheel + torch2.11 + vllm + stock harbor daytona backend + proxychains egress) — heavy
but self-contained + matches the user's "cloud Daytona" ask; (b) COORDINATE with Marianna for read access to
her env/BenSkyRL/harbor_patched or a shareable SIF + confirm intended path (Apptainer vs Daytona). The
sandbox-free gsm8k A/B is a complete standalone win proving the cluster + RL pipeline.

## OPEN DECISION — how to get the agentic runtime on JURECA (superseded by BOTTOM LINE)
1. **Mine Marianna's RL launch recipe** (`/p/project1/laionize/marianna/dc_agent/bash-scripts/
   run_rl_easyband_continue_jureca.sh` + her env/SIF) — she runs agentic RL on JURECA; her setup is ground
   truth (which env, SKYRL_HOME, SIF vs conda, how she beats flash-attn + the examples import). Highest leverage.
2. **Clone Marianna's conda env** to scratch (needs conda/mamba), `pip uninstall` the stray examples pkg +
   `pip install -e harbor` — reuses her compiled flash-attn/torch2.10/vllm0.18. Risk: torch 2.10 vs MarinSkyRL's
   preferred 2.11 (likely OK for a colocated smoke w/o context-parallel).
3. **Build the prod wheel-cache venv** (prebuilt flash-attn/TE wheels, port the gpu-rl Docker recipe) — correct
   but heavy.

## Next (superseded — see OPEN DECISION above)
1. Verify `ssh -T git@github.com` on JURECA greets lukedhlee.
2. Clone under `/p/project1/synthlaion/lee27_jureca/code/`: MarinSkyRL (penfever/working), harbor,
   and the OTA fork branch lukedhlee/jureca-rl-test. Build a FRESH agentic venv on
   `/p/scratch/synthlaion/lee27/envs/` (torch 2.9-2.11 / vLLM 0.16-0.20 per the fork).
3. `python -c "import harbor; import examples.terminal_bench.entrypoints.main_tbench"` must pass.
4. Egress test on a devel node (proxychains → api.daytona.io + HF).
5. Snapshot prebuild dry-run for a tiny task set (≤10 new).
6. GPU smoke: `hpc.launch --job_type rl --rl_config hpc/skyrl_yaml/jureca/smoke.yaml
   --model_path Qwen/Qwen3-0.6B --train_data <tiny-repo> --num_nodes 1`. SPENDS GPU + shared sandboxes
   → explicit go + check org headroom first.

## Open questions
- Tiny agentic task dataset with few unique envs (scoping named a 4-task r2egym subset w/ existing snapshots).
- Are marin-community/MarinSkyRL + harbor public or private (affects clone auth)?
- torch/vLLM version resolution for the fresh venv on A100-40GB/cu12.x.
