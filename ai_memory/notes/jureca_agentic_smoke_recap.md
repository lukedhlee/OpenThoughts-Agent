# JURECA Stream B — agentic smoke recap (parked)

What the dense-model (0.6B→1.7B→8B) JURECA agentic GRPO smokes actually proved: cloud Daytona worked,
Harbor `terminus-2` + MarinSkyRL `terminal_bench` was the harness, and reward rose from
thinking-length shaping rather than verifier pass-rate.
Read for what plumbing is already proven on JURECA; this stream is parked and is not the MoE work.

**What ran:** dense Qwen3 **0.6B → 1.7B → 8B** agentic GRPO on JURECA A100. Plumbing proven; verifier/pass-rate lift was still open. **Not** the Vista MoE workstream.

## Train tasks (agentic)

Local Harbor task subsets under:
`/p/project1/synthlaion/lee27_jureca/bench_subsets/agentic_grpo_smoke_20260714/`

| Subset | Tasks | Role |
|--------|-------|------|
| **`easy4`** | `swe__psf__requests-2317`, `tb2__fix-git`, `v2__jq-data-processing`, `v2__broken-python` | 1.7B baseline/GRPO (mix of SWE / TB2 / OT-v2) |
| **`fast4`** | duplicated fast tasks e.g. `tb2__fix-git-a/b`, `v2__broken-python-a/…` | Final **8B** job `15428130` / `ag_grpo_fast4_q8_thinklen` |

Not full SWE-Bench / TB2. Early `r2egym_1unique_16` failed reward validity (`tests/test.sh` missing).

## Daytona?

**Yes — cloud Daytona worked** on `dc-gpu-devel` (direct egress verified 2026-07-14; old SOCKS proxy failed). Sandboxes created, Harbor trials ran, GRPO steps completed. 8B reward rose under `thinking_length` shaping (`0.545 → 0.768`), not verifier pass-rate.

## Harness?

**Yes — Harbor agent `terminus-2`**, driven by MarinSkyRL’s `terminal_bench` entrypoint (`examples.terminal_bench.entrypoints.main_tbench`).

Loop: Harbor episode in a **cloud Daytona** sandbox → model tools via colocated vLLM → task verifier reward → MarinSkyRL GRPO update.

(Marianna’s JURECA prod path uses Apptainer bridge, not Daytona; our smokes used **cloud Daytona**.)

## Framework / repo changes?

Mostly **config + runtime wiring**, not a new harness:

| Repo | What we changed |
|------|-----------------|
| **OpenThoughts-Agent** | Branch `lukedhlee/jureca-daytona-grpo` (earlier `jureca-rl-test`): `hpc/hpc.py` jureca entry; `hpc/skyrl_yaml/jureca/smoke*.yaml` |
| **MarinSkyRL** | Branch `lukedhlee/jureca-vllm018-render` (+ uncommitted `SKYRL_FORCE_CPU_INIT_WEIGHTS` patch noted in session) |
| **Harbor** | Stock `marin-community/harbor` on PYTHONPATH (not Marianna’s apptainer-bridge fork) |
| **Runtime** | Borrowed `/p/project1/ccstdl/envs/marianna/py3.12` + **PYTHONPATH shim** (rogue `examples` shadow) + pyoverlay for missing harbor deps |

Public NovaSky SkyRL alone is **not** enough (no `terminal_bench`). Need MarinSkyRL + Harbor.
