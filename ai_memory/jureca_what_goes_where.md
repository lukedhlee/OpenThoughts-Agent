# JURECA / JSC — what goes where (`lee27`)

Rule of thumb (same idea as TACC `$HOME` vs `$SCRATCH`):
- **`$HOME`** = tiny / fragile  
- **`/p/project1/...`** = durable code & light project files (inode-sensitive)  
- **`/p/scratch/...`** = heavy / many files / short-lived (wiped ~90d)  
- **`/p/data1/...`** = big shared datasets & published checkpoints  

Replace `<PROJECT>` with your active budget (`synthlaion` for JURECA A100, `laionize` if that’s where your group tree is, `westai0066` for DC-HWAI H100). Prefer **one** “real home” project and stick to it.

## Recommended layout

| What | Put it here | Why |
|------|-------------|-----|
| **SSH / tiny login home** | `$HOME` (`/p/home/jusers/lee27/jureca`) | Only dotfiles. Almost nothing else. |
| **“Real” home / personal workspace** | `/p/project1/<PROJECT>/lee27_jureca/` → symlink as `~/lee27_jureca` | Durable; still watch **inode** count |
| **Git repos / code** | `~/lee27_jureca/code/` (→ under `/p/project1/...`) | Durable, shared FS, not wiped |
| **Critical / shared conda·mamba envs** | `/p/project1/<PROJECT>/shared/envs/` **or** group paths like Marianna’s | Prefer **shared** + modules/containers; private conda = inode bomb. Heavy multi-env → **scratch** |
| **Marianna RL env (existing)** | `/p/project1/ccstdl/envs/marianna/py3.12/` | Reuse if compatible; don’t duplicate |
| **Personal throwaway venv** | `/p/scratch/<PROJECT>/lee27/envs/` | Scratch OK; reinstallable |
| **Big caches** (HF, pip, torch, wandb local, uv) | `/p/scratch/<PROJECT>/lee27/cache/` + symlink `~/.cache` → here | Caches are huge + many files; never in `$HOME` or project root long-term |
| **Experiment logs / SLURM `.out`** | `/p/scratch/<PROJECT>/lee27/experiments/<run>/` | High churn, many small files |
| **Checkpoints (training, recoverable)** | Prefer `/p/data1/mmlaion/...` (or group data project); interim → scratch | Project paths fill inodes; scratch expires |
| **Final / shared model artifacts** | `/p/data1/mmlaion/` or `/p/data1/cstdl/` (after joining) | Durable shared data FS |
| **Large datasets (>~1TB or shared)** | `/p/data1/{mmlaion,datasets,cstdl}/` as **tars/webdataset** | Never unpack millions of files on project |
| **Small temp datasets (<~500GB–1TB)** | `/p/scratch/<PROJECT>/lee27/data/` | Scratch; expect wipe |
| **SIF / Apptainer images** | `/p/scratch/<PROJECT>/lee27/sif/` or group sif_cache | Large binaries; not `$HOME` |
| **Group scripts you only read** | e.g. `/p/project1/laionize/marianna/...` | Existing group tree — don’t dump your caches there |

## Env vars to set (once in `~/.bashrc` on JURECA)

```bash
export JSC_PROJECT=synthlaion   # or laionize / westai0066
export JSC_HOME=/p/project1/${JSC_PROJECT}/lee27_jureca
export JSC_SCRATCH=/p/scratch/${JSC_PROJECT}/lee27
export HF_HOME=${JSC_SCRATCH}/cache/hf
export HF_HUB_CACHE=${HF_HOME}/hub
export TORCH_HOME=${JSC_SCRATCH}/cache/torch
export XDG_CACHE_HOME=${JSC_SCRATCH}/cache
export PIP_CACHE_DIR=${JSC_SCRATCH}/cache/pip
export WANDB_DIR=${JSC_SCRATCH}/wandb
export TMPDIR=${JSC_SCRATCH}/tmp
mkdir -p "$JSC_HOME" "$JSC_SCRATCH"/{cache,envs,experiments,data,sif,tmp,wandb}
```

## One-time symlink setup

```bash
mkdir -p /p/project1/${JSC_PROJECT}/lee27_jureca
ln -sfn /p/project1/${JSC_PROJECT}/lee27_jureca ~/lee27_jureca
mkdir -p /p/scratch/${JSC_PROJECT}/lee27/cache
# move existing cache off tiny home if present:
[ -d ~/.cache ] && [ ! -L ~/.cache ] && mv ~/.cache /p/scratch/${JSC_PROJECT}/lee27/cache/dot_cache
ln -sfn /p/scratch/${JSC_PROJECT}/lee27/cache/dot_cache ~/.cache
```

## Known shared envs (group)
- Marianna (RL co-lead): **`/p/project1/ccstdl/envs/marianna/py3.12/`**
  - Under `ccstdl` project (good — shared env on project, not private conda forest in `$HOME`)
  - Prefer **reusing / extending** this (or other shared envs) over installing a new personal mamba tree
  - Before activating: `module purge` if the env expects a clean module state; ask Marianna which modules (if any) to load first
  - If you update a shared env, fix group perms after (`find … chmod 775/664`) so others aren’t locked out — see LAION home-handling note

## Explicit don’ts
- Don’t install conda under `$HOME` or flood `/p/project1/laionize/` with personal caches.
- Don’t unpack ImageNet-style trees into project — use webdataset/tars on `/p/data1`.
- Don’t treat scratch as permanent (≈90-day wipe).
- Don’t run GPU jobs expecting outbound internet (login/JUDAC only).
- Don’t clone Marianna’s env into a second full copy under your user unless necessary — inode cost.

## Quota check
```bash
jutil project dataquota -p synthlaion
jutil project dataquota -p laionize
```
Watch **file/inode** columns, not just TB.
