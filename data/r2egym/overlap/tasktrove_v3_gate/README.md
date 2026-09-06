# TaskTrove r2egym v3 — gate results and task lists (2026-09-04)

Source: `laion/r2egym-patched-full-oracle-v3` (3,328 tasks; 3,035 active in TaskTrove v4.12, byte-identical).
Gate: a task is valid when the pristine checkout (no fix) scores 0 and the gold patch scores 1.
Run twice: on R2E-Gym's original per-task images (JSC apptainer, all 3,328) and on TaskTrove's own
flattened images with the setup preamble run by the harness (Daytona, 2,476 gated).

| file | rows | what |
|---|---|---|
| `wheel_only_pandas_numpy_active.tsv` | 421 | every active pandas and numpy task; their tests import the pre-installed wheel, so any patch scores 1 |
| `daytona_pass_with_no_fix.tsv` | 453 | tasks whose pristine run scored 1 on TaskTrove's images (the 412 pandas/numpy that completed + sporadic pillow, orange3, coveragepy, datalad, sympy) |
| `gold_patch_fails_on_r2egym_images.tsv` | 26 | tasks whose gold patch scores 0 on R2E-Gym's per-task images (25) plus one whose verifier never returned |
| `empty_issue_text_removed_in_v4.12.tsv` | 293 | matplotlib and moto tasks with an empty `<issue_description>` (already dropped from TaskTrove v4.12) |
| `raw_set_gated_not_in_tasktrove.tsv` | 2,748 | R2E-Gym-V1 tasks that pass the same gate on per-task images and are not in TaskTrove (upstream image + commit) |
| `gate_report_r2egym_images_jsc.md` | | per-repo table + per-task status, R2E-Gym images |
| `gate_report_tasktrove_images_daytona.md` | | per-repo table + per-task status, TaskTrove images |
| `pass8_paired_jsc_vs_daytona.md` | | same model and recipe on 1,905 tasks valid in both environments |

Task ids are TaskTrove paths (`r2egym-v1-NNNNN`). `upstream_docker_image` is the R2E-Gym-V1 per-task
image the task maps to (mapping in `../tasktrove_v3_upstream_map.tsv`, verified by byte-identical hidden tests).
