# JSC / JURECA hazards & paths (LAION)

Sources: [LAION JSC access](https://iffmd.fz-juelich.de/WPHnb-DoTAKN3BN0afc9MQ), [Home handling](https://iffmd.fz-juelich.de/EgRn0takSoaWym8N9qoq4g)

## Hard hazards (can ban / lock everyone)
- **Inode / file-count caps** on project dirs: ~**3M files total for ALL users** per compute project (`/p/project1/<project>/`). Hitting the cap **locks the project for everyone**; can get you banned.
- Scratch also has a high file cap (~4M mentioned). Prefer **few large tars / webdataset**, not millions of small files.
- Put `ulimit -c 0` in bashrc/sbatch so core dumps don’t explode file counts.
- Watch quota: `jutil project dataquota -p <ACCOUNT>` (e.g. `synthlaion`, `laionize`, `reformo`). Clean up before ~3M.
- Conda/mamba installs **many files** → put envs on **scratch** or use **shared** envs / modules / containers. Don’t proliferate private conda trees on `/p/project1`.

## Path map (JUWELS / JURECA / JUSUF / DC-HWAI) — like TACC home vs scratch
| Use for | Path | Notes |
|---------|------|--------|
| Tiny real `$HOME` | `/p/home/jusers/<user>/…` | **Very small** — do not dump code/caches/datasets here |
| “Real” home / code | `/p/project1/<project>/$USER_<machine>/` then `ln -s` into `~` | e.g. `/p/project1/ccstdl/$USER_jureca` → `~/…` |
| Caches | move `~/.cache` (and `.vscode` etc.) under that project dir + symlink | Same pattern as home |
| Temp / many files / small data | `/p/scratch/<project>/$USER/` | Wiped ~**90 days**; OK for >100k files if needed |
| Large / shared datasets & checkpoints | `/p/data1/{cstdl,datasets,mmlaion}/` | Join those JuDoor data projects; use tars/webdataset |
| Scratch datasets | `/p/scratch/<project>/` | **≤ ~1TB** per personal stash; shared scratch ~90TB total — big data → `/p/data1` |

**JURECA compute budget** (from LAION doc): account **`synthlaion`** (A100); H100 DC-HWAI often **`westai0066`**.  
Marianna path of interest is under **`/p/project1/laionize/`** (project tree — keep file counts low there).

## Do / don’t
- **Don’t** run training or dump HF caches in `$HOME`.
- **Don’t** unpack huge datasets into millions of files on `/p/project1`.
- **Don’t** store >1TB on scratch; use `/p/data1/...`.
- **Do** stage data on login/JUDAC first — **compute nodes have no internet**.
- **Do** prefer modules/shared envs over private conda forests.
- **JUPITER only:** replace `/p/...` → `/e/...`; `/p` data is not usable the same way there.

## Quick home setup (once per machine)
```bash
# example project ccstdl — swap for synthlaion/laionize if that’s your space
mkdir -p /p/project1/ccstdl/${USER}_jureca
ln -s /p/project1/ccstdl/${USER}_jureca ~
mkdir -p /p/scratch/ccstdl/$USER
# move caches off tiny home:
[ -d ~/.cache ] && mv ~/.cache /p/project1/ccstdl/${USER}_jureca/.cache
mkdir -p /p/project1/ccstdl/${USER}_jureca/.cache
ln -sfn /p/project1/ccstdl/${USER}_jureca/.cache ~/.cache
```
