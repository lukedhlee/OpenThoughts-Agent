# Snowball agentic RL on Horizon: status and how to get to a run

**Verdict (2026-09-29, still true 09-30): port gate 5 has not run. It is blocked on the R2E-Gym Daytona snapshots, and
bringing them back is Luke's call.** Since 09-30 the task tree, the Jupiter arm configs and their step-1 values are on
Horizon, so once the snapshots are back only the Horizon launcher (step 3) remains. All 11 per-repo pool snapshots (`harbor__<hash>__snapshot`, list in
`ai_memory/active/snowball-r2egym/state.md` 09-18/19) are missing from the only org our key reaches. That org holds
54 snapshots and is full. The trainer-side numerics were checked without sandboxes and look healthy (§ Proxy).

## What gate 5 needs, and what is ready

Gate 5 needs the same config and checkpoint as a Jupiter reference, and a logged step-1 `policy/tis/log_ratio_abs_mean`
that is no bigger than Jupiter's.

- **Reference (updated 2026-09-30).** Jupiter's arm configs, sbatch files and step-1 values are now on Horizon in
  `/scratch/11584/lukedhlee/rl_refs/` (README.md has the table; values read from each arm's W&B offline file on Jupiter).
  - **Daytona, the closest match:** `snowball_ttband_ota3d1034u_b` step 1 = **0.02772** and `snowball_ttband_ota3d517u_b`
    = **0.02754**. Their init checkpoints are the OTA SFT arm-D exports
    (`snowball-ota-sft/ota3_if_rstsucc/lr1e-4-sched2/export-step{1034,517}-hf-bf16`), which exist only on Jupiter, so
    one of them has to come over through HF (a Jupiter-side upload) before this reference can be matched exactly.
  - **Apptainer:** `62ft5sky` = `snowball_ttband_v2train_fulldist_gmm_seats1584_x16_a`, from Stage-3 step 1888 (on
    Horizon), full sampler, grouped_mm, staleness 2, lr 5e-7, step 1 = **0.03531**. Same checkpoint available here, but a
    different sandbox backend.
  - Plan: match a Daytona "b" arm once its export is on HF. Until then, the fallback is 62ft5sky's config and checkpoint
    on Daytona. The metric compares trainer and rollout log-probs of the same tokens, so the backend should not move it.
- **Stack.** It is identical to Jupiter's (port gate 1): `~/snowball/envs/snowball`, MarinSkyRL a03b2773, harbor
  dcf609bc, vLLM fa50698a + EAGLE-3 overlay. Serving passed gates 2 and 3. The relay driver already reaches Daytona from
  compute nodes through the login-side `ssh -R` tunnels and `socks_connect_bridge.py` (`data/relay/horizon/DRIVER.md`),
  and it is the transport an RL arm would reuse in place of Jupiter's per-node async gateway.
- **Not built yet.** There is no Horizon RL launcher. On Jupiter, an arm is a config JSON plus an sbatch, cloned from a
  previous arm (`data/r2egym/jsc/clone_newstack_arm.sh`, `data/r2egym/daytona_artifacts/build_ota_darm.sh`). Those
  templates live only on Jupiter (`$E/<arm>/configs/*_rl_config.json`). The declarative form of the same recipe is
  MarinSkyRL `cloud/iris/configs/snowball_r2egym_arm_b.yaml`.

## Steps to a gate-5 run, once the pool is back

1. **Pool.** Luke restores the 11 snapshots from the public images (`data/r2egym/daytona_artifacts/RECOVERY.md`,
   `ghcr.io/lukedhlee/r2egym/<repo>`), after freeing 11 of the org's 40 custom slots. Then verify them read-only with
   `snapshot_census.py`.
2. **Task tree (on Horizon already).** The Daytona task tree `r2egym-daytona-v3` (3,035 tasks) is inside the recovery
   bundle: `~/.local/share/otagent/r2egym-recovery/v3-20260918/task_tree.tar.gz`. Unpack it under `/scratch` and take the
   training pool of the reference run.
3. **Config (on Horizon already).** Take the reference arm's `configs/*_rl_config.json` and `sbatch/` from
   `/scratch/11584/lukedhlee/rl_refs/<arm>/`, then change only the Horizon parts:
   - paths come from `~/snowball/jupiter.local.env`;
   - Daytona egress goes through `tunnel.sh` + `socks_connect_bridge.py`, not the Jupiter gateway;
   - add `CC=gcc CXX=g++` and a node-local `SERVE_CACHE_LOCAL=/tmp`;
   - keep engines node-local DP groups (`skyrl-dp-engines-must-be-node-local`), check the `DP rank -> node` lines, and keep
     4 GPUs per engine node as on Jupiter;
   - use at most 128 seats, because the relay run uses up to ~770 of the org's ~1,000 sandboxes;
   - set `max_steps` to 3, turn off in-run eval, and take no checkpoints.
4. **Run and read.** Run it and read `policy/tis/log_ratio_abs_mean` at step 1. Pass if it is ≤ the reference value.
   Size: a 12-node layout (4 policy + 8 engine) takes ~2–3 h to step 3 at 128 seats, which is ~30 node-h.

## Proxy (done, informational): trainer vs rollout log-probs on GB200 without sandboxes

`rl_logprob_parity.py` + `rl_logprob_parity.sbatch` (1 node, ~6 min, of which 5 min is model load) run MarinSkyRL's own
`HFModelWrapper.forward` with the arm's trainer settings (GrugMoe, flash_attention_2, bf16, native grouped_mm, no
packing, `training_strategy=fsdp2`). They score the 96 gate-2 R2E-Gym turns (Stage-3 step 1888, fixed completions,
142,817 tokens) with four processes, one full model per GB200, and compare against vLLM's log-probs of the same tokens.
The measure is mean |Δ log p|, the `tis/log_ratio_abs_mean` formula, with a 95 % bootstrap over turns (job 36712):

| trainer (Horizon) vs | mean \|Δ log p\| |
|---|---|
| Horizon vLLM (gate-2 serve, rep 0 / rep 1) | 0.0304 [0.0286, 0.0321] / 0.0303 |
| Jupiter vLLM at sampling time (the probe's rollout log-probs) | 0.0309 [0.0289, 0.0327] |
| context: Horizon vLLM against itself on a rerun | 0.0247 |

The gap sits at the level of Jupiter's step-1 value (0.0353 on 62ft5sky's own rollouts), and only 0.1 % of tokens have
|Δ| > 0.69. So the Blackwell trainer path (grouped_mm and FA2 on sm_100) runs and matches the engine. The token sets
differ, so this is not the gate.
