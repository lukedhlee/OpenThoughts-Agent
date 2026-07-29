# Jupiter WandB (`lee27`)

## Rule
**Compute nodes have no internet** → train with `WANDB_MODE=offline`, sync from a **login node**.

## Secrets
- Source of truth (Mac): `~/.config/otagent/secrets.env` (`WANDB_API_KEY`, `WANDB_ENTITY=lukeleeai`)
- On Jupiter (mode 600): `$JSC_SCRATCH/keys/secrets.env` and `~/.config/otagent/secrets.env`
- Env pointer: `DC_AGENT_SECRET_ENV=$JSC_SCRATCH/keys/secrets.env`
- Never commit; never source someone else’s keys.

## Runtime env (from `$JSC_HOME/env.sh`)
```bash
source /e/project1/reformo/lee27/env.sh
# sets WANDB_MODE=offline, WANDB_DIR=$JSC_SCRATCH/wandb, WANDB_PROJECT=jupiter-moe-gsm8k-grpo
set -a; source $DC_AGENT_SECRET_ENV; set +a
```

## After a job finishes — sync from login
```bash
source /e/project1/reformo/lee27/env.sh
set -a; source $DC_AGENT_SECRET_ENV; set +a
export WANDB_MODE=online
# needs wandb CLI in whatever env you install
wandb sync $WANDB_DIR/offline-run-*
# or: bash scripts/wandb/jupiter_sync_offline.sh
```

## Project naming
- Vista canary used `vista-moe-gsm8k-grpo`
- Jupiter canary default: `jupiter-moe-gsm8k-grpo` (entity `lukeleeai`)
