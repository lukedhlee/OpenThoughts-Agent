# Vista secrets (Luke)

Where Luke's secrets live on the Mac and on Vista, which env var the sbatch path reads, and the rule
against sourcing another user's keys.
Read before wiring a job's env, or when WandB/Daytona identity looks wrong.

- **Local SoT (do not commit):** `~/.config/otagent/secrets.env` on the Mac.
- **On Vista (mode 600):** `$SCRATCH/keys.env` and `$HOME/.config/otagent/secrets.env`.
- RL/sbatch reads `DC_AGENT_SECRET_ENV=$SCRATCH/keys.env` (also set in gsm8k MoE yaml `container.extra_env`).
- WandB: entity from secrets (`lukeleeai`); project forced to `vista-moe-gsm8k-grpo` for this canary.
- Never source penfever `keys.env` for Luke jobs (wrong WandB/Daytona identity).
- Optional later: `GH_TOKEN` for HTTPS git push/private repos (or prefer registering Vista `~/.ssh/id_ed25519.pub` on GitHub — see `tacc_vista.md`).
