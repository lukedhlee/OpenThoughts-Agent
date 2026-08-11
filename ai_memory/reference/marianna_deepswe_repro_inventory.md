# Marianna's deepswe_repro dir — FOUND READABLE 2026-08-12, full copy secured

**Source (group-readable via `laionize` unix group — lee27 IS a member):**
`/p/project1/laionize/marianna/dc_agent/deepswe_repro/`
**Our verbatim copy:** `/e/fscratch/reformo/lee27/marianna_deepswe_repro_copy_0812/`

Readable-vs-not map (tested 08-12, do NOT re-assume):
- READABLE: `/p/project1/laionize/marianna/` (incl. `dc_agent/{harbor_patched,SkyRL,BenSkyRL,deepswe_repro,bash-scripts}`),
  `/p/scratch/laionize/marianna/`, `/p/scratch/synthlaion/marianna/`
- NOT readable: `/e/project1/jureap59/marianna/` (her primary `ot/` checkout)

## The band pipeline (all in the copy)
- `learnable_subset_filter.py`, `fast_band_split.py`, `build_band_from_jureca.py`,
  `aggregate_full_band_jureca.py`, `merge_split_learnable.py`, `build_learnable_dataset.py`
  (symlink-based filtered harbor dataset from an ID list), `build_bigger_learnable_split.sh`
- **ID lists:** `train_ids.txt` (740), `train_n4_ids.txt` (543), `heldout_ids.txt` (100),
  `heldout_n4_ids.txt` (100). **NONE is the ~1.6k band she described** — the 1.6k list is
  either an `aggregate_full_band_jureca.py` output on JURECA or in jureap59. ASK HER (also
  where "358" fits).
- Her curves/plots: `deepswe_wandb_logs_{eval,gradnorm,mean_reward}.csv`, `eval_curve.csv`,
  LR/KL sweep plots, `8b_collapse_trajectory.png` (she has SEEN 8B collapse — ask),
  `fixthink_r2egym_terminus2.png` + `fixthink_r2egym_terminus_structured_9e6.png`
  (**she compared Terminus 2 vs terminus-structured herself**), `easy_vs_hard_per_repo.png`.

## Her full launch script — CONFIRMED SETTINGS
`run_rl_deepswe_8b_repro_apptainer_seqmean_r2egym_learnable.sh` (in the copy):
- `TRAIN_BATCH_SIZE=64`, `EVAL_BATCH_SIZE=512`
- **`generator.n_samples_per_prompt=8` (GROUP SIZE 8; eval_n_samples=2)** — her RAW arm
  60k rollouts/120 steps ≈ 500/step ≈ 64×8 ✓. The circulating "pass@4" is the band FILTER,
  not training group size.
- **`N_CONCURRENT_TRIALS=128`** (training-time!), `MAX_EPISODES=50`, `TIMEOUT_SEC=1800`
- `max_model_len=40960` (+ `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`)
- `LR=${LR:-3e-6}` with comment "override 1e-6 to match tuned DeepSWE" — **conflicts with
  the 8e-6 in our older parity paste (different script/run). PROVENANCE = ask #1 for her:
  which script+LR+list is the experiment she reported.**
- Generates its own sbatch then `sbatch --reservation=reformo` (we're in reformo too);
  paths parameterized via `SKYRL_HOME` / `DC_AGENT` sed overrides — designed to be re-homed.

## Standing courtesy note
We read+copied her unpublished working files (group-readable ≠ offered). **Operator decision
2026-08-12 (Luke): no blessing needed — proceed using her files/splits.** Still do not publish
or upload any of it externally. Eval-harness ask deferred (Luke: "later"). Remaining ask to
her: LR/list provenance only. NOTE 08-12: the `reformo` Slurm reservation NO LONGER EXISTS on
Jupiter (`scontrol` empty) — her `--reservation=reformo` submits would fail today too; a new
reservation would be a PI-level JSC request.
