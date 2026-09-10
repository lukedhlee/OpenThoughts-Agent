# Snowball R2E-Gym — Jupiter/Slurm operational layer

The shell and Python the Snowball agentic-RL arms were actually driven with on the JSC
clusters (Jupiter, JURECA, JUWELS), mirrored from the cluster tree `code/snowball/`,
frozen 2026-09-09. Operator scripts, not a library: they assume JSC paths, Slurm, and
the arm directory layout under `$EXPERIMENTS/<arm>/`.

**The canonical, declarative recipe is not here.** It lives in MarinSkyRL as
`cloud/iris/configs/snowball_r2egym_*.yaml`. This directory is the imperative layer
that predates it, kept on this branch temporarily so the migration has a reference for
what the runs really did. Prefer the YAML for anything new.

## What lives here

**Probe and arm builders** — turn a checkpoint plus a task list into a runnable job.
`make_snowball_probe.py` (eval-only probe config), `make_snowball_grpo.py` (standard
GRPO), `make_tt_wave.py` (TaskTrove wave, `--no-tis` for the eval-only masked-group
crash), `probe_ckpt.sh` / `probe_ckpt_v2.sh` (probe a checkpoint end to end),
`clone_fresh_arm.sh` and `clone_resume_v2.sh` (new arm from a template, with the
2026-09-07 connect-timeout and verifier-timeout fixes applied),
`validate_hydra_args.py` and `fix_merged_keys.py` (config sanity), `set_hydra.py`,
`patch_arm.py`, `build_band.py`, `tt_split.py`, `tt_allowlist.py`, `curriculum_build.py`.

**Fleet and bridge** — the Apptainer worker fleet and the login-node bridge that fronts
it. `bridge_restart.sh`, `pool_launch.sh` / `pool_watch.sh` / `pool_gate.sh`,
`juwels_bridge2_setup.sh`, `jureca_bridge_setup.sh`, `jureca_fleet_launch.sh`,
`engine_health.sh`, `envgate*.sh` (environment gating), `chain_rebal.sh`,
`rl_reaper.sh`, `store_reaper.sh`, `wb_sync_loop.sh` (offline W&B sync).

**Readout and analysis** — `hist_extract.py` (schema in `hist_extract_SCHEMA.md`),
`hist_readout.py`, `pass8_fast.py`, `paired_delta.py`, `heldout_compare.py`,
`v2val_compare.py` / `v2rest_compare.py`, `wf_*.py` / `wf2_*.py` (workflow-prompt
probes), `turn_*.py` (turn timing and splits), `census_*.py`, `arm_metrics.py`.

**Serving and build** — `serve_*.sbatch`, `build_marin_vllm.sh`,
`build_vllm_next.sbatch`, `nccl_test.sbatch`, `pyspy_*.sh` (live-hang capture).

## Subdirectories

- `analysis/` — token/budget scripts and `parity/` (served-stream vs re-render);
  `audit/` — sandbox audit protocol.
- `bench/` — the serving benchmark harness (`bench_client*.py`, `bench_serve.sbatch`,
  `prof_*.py`, `tune_moe*`). Result data and replay corpora are deliberately not
  mirrored; only the harness and the tuned MoE config in `tuned_h200/`.
- `tools/` — migration harnesses, both read-only and runnable off-cluster.
  `tito_measure/tito_match.py` decides from harbor artifacts alone whether the trainer
  trains on the ids vLLM sampled. `lossharness/` (`run.sh`, `loss_equiv.py`,
  `compare.py`) diffs the policy-loss surface of two MarinSkyRL revisions on CPU. Both
  take their paths as documented parameters; see each one's header.
- `fixtures/` — frozen configs from the 2026-09-09 migration baseline, showing how a
  real arm (`arm_rl_config.json`) and a real probe (`band_r3_s0_rl_config.json`) were
  configured.
- `fleet/juwels_workers.sbatch` — the JUWELS worker-fleet submit script as it was
  working at the freeze, including the proxychains path for compute-node egress.

## Caveats

Paths inside these scripts are JSC-absolute (`/e/project1/...`, `/p/scratch/...`) and
resolve nowhere else. Several are generational (`patch_probe_gen2..8.py`); the highest
number is the one that was in use. No credentials are committed — scripts needing a
Daytona or HF key source it from a key file named on the command line.
