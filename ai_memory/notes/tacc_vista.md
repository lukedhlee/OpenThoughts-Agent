# TACC Vista — Luke quick ref

## Login (your account)
```bash
ssh vista          # lukedhlee@vista.tacc.utexas.edu
# Password + TACC token (TOTP from authenticator / TACC token app)
# ControlMaster 4h already in ~/.ssh/config → later ssh/scp reuse MFA
```
- Alias: `vista` (also `vista-gh` = ProxyJump to a compute host once on Vista).
- Docs: https://docs.tacc.utexas.edu/hpc/vista/
- Lab ops (penfever / OT-Agent paths): `.claude/ops/tacc/ops.md`
- **Not JuDoor-style IP whitelist.** New cafe/home IP does not block Vista; failures are usually MFA / account / allocation.
- Login help: https://accounts.tacc.utexas.edu/login_support · Portal: https://portal.tacc.utexas.edu/

## Important: two accounts
| Account | Role |
|---------|------|
| **`lukedhlee`** | Your login (SSH config) |
| **`penfever` / CCR24067** | Lab OT-Agent tree on Vista (`/scratch/10635/penfever/…`) — see ops.md |

You may not have write access to penfever’s scratch; confirm allocation (`showq` / `projects` / `bbalance`) after login.

## After login — orient
```bash
whoami; hostname; pwd
echo "HOME=$HOME SCRATCH=$SCRATCH WORK=$WORK"
projects          # or: taccinfo
bbalance          # SU balance if on a charged account
squeue -u $USER
```

## Interactive GPU (GH200, aarch64)
```bash
# prefer production partitions; gh-dev = 2h max, 1 running job
idev -A <YOUR_ALLOC> -p gg -m 1400    # long interactive
idev -A <YOUR_ALLOC> -p gh-dev -m 120 # short debug
# or:
srun -A <YOUR_ALLOC> -p gh -N 1 -n 1 -t 02:00:00 --pty bash -l
```
Partitions: `gh` / `gg` (prod), `gh-dev` (dev). **1 GPU/node** — request nodes, not gres.

## QOS node caps (live 2026-07-15, `sacctmgr`)
| Partition / QOS | MaxWall | MaxJobs | MaxNodes / job | Notes |
|-----------------|---------|---------|----------------|-------|
| `gh-dev` / `qdevelopment` | **2h** | **1** | **8** | 20-node pool; good for load/serve smokes |
| `gh` / `qgh` | 48h | 20 | **64** | Needed for full 24-GPU MoE RL (24 Vista nodes) |

Luke: `$SCRATCH=/scratch/11584/lukedhlee`, account `CCR24067`, also in group `G-827553` (penfever tree readable).

## Don’ts on login node
Source of truth: [TACC Vista Common Pitfalls](https://docs.google.com/document/d/1URcWe8mLQF8HMNre7vZwgk6TiRNxFxVBwK_HCcRXQdM/edit) + `.claude/ops/tacc/ops.md`.
- Login OK: `git pull`, `squeue`/`sbatch`, light shell. **Not OK:** `uv`/pip builds, vLLM load, big HF downloads/uploads, `/tmp` package builds (doc: `/tmp` write not allowed on login → use compute).
- Heavy work → `idev`/`srun`/`sbatch` on `gh-dev`/`gh`/`gg`. Caches on `$SCRATCH`, never `$HOME`.
- TPS smoke job pattern: `sbatch -p gh-dev` (compute), not login-local python.

## Primary repo = OpenThoughts-Agent (not dc-agent)
- **SoT / work tree:** `$SCRATCH/OpenThoughts-Agent` (Luke). Do **not** use `dc-agent` / `dc_agent` as the primary codebase.
- Remotes: `origin` = `open-thoughts/OpenThoughts-Agent`, `fork` = `lukedhlee/OpenThoughts-Agent`.
- Related clones (also under Luke `$SCRATCH`): `MarinSkyRL/`, `harbor/` — readable; penfever’s copies are often mode `700`.

## Git auth on Vista
Public HTTPS fetch of `open-thoughts/*` works **without** a token today. GitHub **SSH key on Vista is NOT registered** (`ssh -T git@github.com` → Permission denied). No `GH_TOKEN` / `GITHUB_TOKEN` in `$SCRATCH/keys.env` yet.

**GitHub SSH: DONE** (Vista `~/.ssh/id_ed25519` registered as `lukedhlee`). Remotes on `$SCRATCH/OpenThoughts-Agent` (+ MarinSkyRL/harbor) use `git@github.com:...`.

## Lab OT-Agent preamble (Luke tree)
```bash
cd $SCRATCH/OpenThoughts-Agent
source hpc/dotenv/tacc.env
source $SCRATCH/keys.env
source $SCRATCH/miniconda3/bin/activate otagent   # symlink → penfever miniconda OK
```
Full map: `.claude/ops/tacc/{ops.md,ENVIRONMENT_MAP.md}`.
