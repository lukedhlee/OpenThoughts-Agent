# JURECA setup plan — Marianna RL stack → lee27

The RL co-lead's JURECA stack: her fork refs, tree layout, split `dc-cpu`/`dc-gpu`/router mode, and the
inode-safe lee27 layout + env strategy for working alongside it.
Read for her fork refs and JURECA design, or before deciding whether to reuse vs clone her env.

Updated from Marianna’s handoff + partial read of her launch script on jrlogin12.

## What she pointed at

| Piece | Location |
|-------|----------|
| Harbor fork | https://github.com/marianna13/harbor/tree/marianna/beam (`064ec1a`) |
| SkyRL fork | https://github.com/marianna13/SkyRL/tree/marianna/fp8 (`b07f04a`) |
| Launch example | `/p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh` |
| dc-agent branch | `marianna/jsc` on `mlfoundations/dc-agent` (**private / not visible to our `gh` yet**) |
| Conda | `/p/project1/ccstdl/envs/marianna/py3.12/` |

## What the launch script already implies (from your `cat` on JURECA)

Her `dc_agent` tree is a **working JURECA agentic/filter pipeline**, not a bare clone:

- `ROOT/.../harbor_patched/` — Harbor with Apptainer bridge (`jureca_workers.sbatch`, `vllm_router*.py`)
- `bash-scripts/` — filter + RL helpers (`run_full_r2egym_filter_jureca.sh`, `vllm_node_jureca.sbatch`, `run_rl_easyband_continue_jureca.sh`)
- **Split mode:** `dc-cpu` Apptainer worker pool + `dc-gpu` vLLM pool + login-node router
- Paths for staging / SIF cache / HF cache / jobs / HF outs (exact vars in script header — confirm with explore script)
- Flow: `infra` → `submit` → `status` → `aggregate` → then RL on banded dataset

So she has **already set up a lot on JURECA** under `/p/project1/laionize/marianna/dc_agent/`.

## Recommended lee27 layout (inode-safe)

```text
/p/project1/ccstdl/lee27_jureca/          # “real home” (or laionize if PI prefers)
  code/
    dc-agent/          # marianna/jsc (or her tree as remote)
    harbor/            # marianna/beam — editable install into env
    SkyRL/             # marianna/fp8 — editable install into env
/p/scratch/synthlaion/lee27/              # or ccstdl/laionize scratch you have
  cache/               # HF / pip / apptainer / torch
  experiments/         # SLURM logs, job dirs, filter runs
  sif/                 # local sif if not using shared sif_cache
  envs/                # ONLY if you must fork a personal conda
```

**Env strategy (preferred):**
1. **Reuse** `/p/project1/ccstdl/envs/marianna/py3.12/` for first bring-up (read/activate only).
2. If you need write access or divergent deps: clone env to  
   `/p/scratch/<project>/lee27/envs/py3.12-lee` (or ask Marianna for a shared `envs/lee27` under `ccstdl`) — **don’t** duplicate full conda on `/p/project1/laionize`.
3. Editable-install **your** harbor/SkyRL/dc-agent checkouts into that env.

## Setup phases

### Phase 0 — Access / ControlMaster
- Interactive: `ssh jureca` + TOTP once.
- Config now has `ControlMaster` / `ControlPersist 8h` → later agent SSH skips MFA while socket lives.
- Run: `bash ai_memory/scripts/explore_marianna_jureca.sh` **on the login node** (or paste its contents).

### Phase 1 — Inventory (don’t reinvent)
Confirm on disk:
- Exact `ROOT`, `ACCOUNT`, `PARTITION`, `STAGING_BASE`, `SIF_CACHE`, `HF_*`, `JOBS_DIR`
- Whether `dc_agent` is a git checkout of `marianna/jsc` and whether harbor/SkyRL are remotes or `harbor_patched` vendored
- Whether you are in JuDoor projects: `ccstdl`, `laionize`, `synthlaion`

### Phase 2 — Your code trees (Mac → JURECA)
On Mac (ground truth for *your* work):
```bash
# example — adjust parent dir
mkdir -p ~/Documents/jsc-rl && cd ~/Documents/jsc-rl
git clone -b marianna/beam https://github.com/marianna13/harbor.git
git clone -b marianna/fp8  https://github.com/marianna13/SkyRL.git
# dc-agent once you have access:
# git clone -b marianna/jsc git@github.com:mlfoundations/dc-agent.git
```
On JURECA: clone same branches under `$JSC_HOME/code/` (or rsync from Mac after commit). **Edit locally, push, pull on cluster** — same OT-Agent discipline.

### Phase 3 — Env
```bash
source /p/project1/ccstdl/envs/marianna/py3.12/bin/activate
# verify: python -V; python -c "import torch; print(torch.__version__)"
cd $JSC_HOME/code/harbor && pip install -e .   # only if env is writable / your clone
cd $JSC_HOME/code/SkyRL && pip install -e .    # follow her SkyRL layout (skyrl-train etc.)
```
If env is not group-writable: ask Marianna to chmod shared env, or `conda create --clone` into scratch.

### Phase 4 — Smoke, don’t scale
1. Read-only: dry-run / `status` against her existing run dirs if allowed.
2. Tiny job: 1 shard / 1 GPU devel partition (`dc-gpu-devel` + `synthlaion` per LAION doc) before full `infra` (20+10 nodes).
3. Point caches at `$JSC_SCRATCH/cache`; logs at `$JSC_SCRATCH/experiments`.

### Phase 5 — RL after filter
Her aggregate path hands off to `run_rl_easyband_continue_jureca.sh` with banded HF datasets — wire your `$DATASET` / model after a successful mini filter.

## Decisions to lock with Marianna / PI
1. Which **compute account** should lee27 charge (`synthlaion` vs `laionize` vs `westai0066`)?
2. OK to **activate her conda** as-is, or clone a lee27 env?
3. Access to private **`dc-agent` `marianna/jsc`** (and any sif_cache / staging she uses).
4. Should you work **inside** `/p/project1/laionize/marianna/...` (dangerous for inodes / ownership) or a **lee27 sibling tree**?

## Default recommendation
- **Code:** your own clones under `/p/project1/ccstdl/lee27_jureca/code/` from her three branches.  
- **Env:** activate hers first; clone only if needed.  
- **Heavy I/O:** `/p/scratch/<your-project>/lee27/...`.  
- **Don’t** install a second full stack under `/p/project1/laionize/marianna/`.

## Next action
On your open `[lee27@jrlogin12]` session, run:

```bash
bash -s <<'EOF'
# or scp/curl the explore script — shortest: 
head -n 100 /p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh
ls /p/project1/laionize/marianna/dc_agent
ls /p/project1/laionize/marianna/dc_agent/bash-scripts
EOF
```

Better: copy `ai_memory/scripts/explore_marianna_jureca.sh` to JURECA and run it; paste the output here.
