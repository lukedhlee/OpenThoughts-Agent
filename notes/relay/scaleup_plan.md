# Relay scale-up: more tasks, finetuned student (plan, not launched)

Written 2026-09-28 ~22:00 PT after Luke asked to run the relay recipe on the rest of CalibForge, TMax and SWE-like tasks
with a finetuned Snowball as the student. Nothing is submitted; the first batch waits for Luke's go and the node-hour line.

## Verdict

**Start with the rest of CalibForge, served with the finetuned student. It is the only pool that runs on Daytona today
without new snapshots.** About 2,660 unused tasks remain on the three CalibForge snapshots we already hold. TMax needs a
conversion and snapshot slots, and every SWE-like pool needs 8–15 snapshot slots in an eval org that is full (40/40).

## Pools

| pool | tasks usable | runs on | snapshot slots | SWE-bench Verified risk | ready? |
|---|---|---|---|---|---|
| CalibForge, unused | ~2,660 (of 3,088 unstaged; ≤1,800 s budgets, covered bases, no TB2 copies) | Daytona, existing 3 snapshots | 0 | none | yes, after `build_tree.py --all` and mirroring the new layers |
| TMax (`laion/TMax-15K-Harbor`) | 14,601 | needs the CalibForge pattern (one ubuntu 22.04 snapshot + setup replay) | 1–2 | TB-Hard is drawn from TMax (spends it as a held-out set) | no: validation never run, 600 s budgets, internet on, no reference solutions, GLM-5.2 solves ~92 % |
| R2E-Gym Daytona v3 | 3,020 gated; ~1,750 without sympy | Daytona + apptainer | 11 (none held now) | **sympy overlaps: 14 of our random-100 eval tasks are sympy; 3 R2E tasks share a base commit with eval tasks (06762/sympy-21847, 07797/23413, 06565/16597)** | after freeing 11 slots and dropping sympy |
| SWE-smith | ~2,500 on 15 repos | Daytona | ~15 | clean | no snapshots built |
| SWE-rebench V2 (untraced) | ~1,540 | Daytona via TaskTrove images | ~15 | clean | no snapshots built |

## Cost

From the 09-26/28 relay runs (8 nodes: 4 student, 4 teacher): **15–20 node-h per 1,000 episodes, 16–21 per 1,000 usable
rows** (a pass or a failure whose verifier ran). The final SFT set used fewer rows than generated because of the 1:1
match with the Qwen-alone arm and the 2-per-task cap, not because of generation.

## Finetuned student

- Swapping the student is one variable (`STUDENT_MODEL` in serve_relay.sbatch). The relay-SFT exports carry 09-21's
  tokenizer, chat template and config, so the router's reasoning cap and the template stay right.
- What is tuned to 09-21 and needs an early check: the EAGLE-3 draft (acceptance unmeasured on the finetuned model:
  speed only), the takeover triggers and run gates (calibrated on 09-21: a better student triggers fewer takeovers, so
  fewer trained teacher turns per episode), and autofix (09-21's format quirks; the SFT models reject 15–20 % of turns
  vs 89 %).
- Which student: the better of A (relay SFT) and MIX (relay + Qwen-alone) after MIX's evals on 09-29.

## First batch (proposed)

1. `build_tree.py --all` for the unused CalibForge tasks; `mirror_upload.py` for their layers (or accept the Docker Hub
   fallback).
2. **Pilot: 1,000 tasks × 2 tries ≈ 2,000 episodes, ~30–40 node-h**, with the existing early gates plus two new ones:
   takeover rate and draft acceptance on the finetuned student.
3. If the pilot holds: the remaining ~1,600 tasks × 2, ~50–65 node-h more.
4. Next pool: R2E-Gym v3 without sympy once 11 snapshot slots are free.
