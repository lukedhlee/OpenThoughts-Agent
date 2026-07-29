# JUDAC SSH (JuDoor)

Data-access / transfer login at JSC ([access docs](https://apps.fz-juelich.de/jsc/hps/judac/access.html)). Same rules as JURECA: ed25519 + `from=` + MFA TOTP. **Keys are per-system** — uploading for JURECA does **not** cover JUDAC.

## Facts
- User: `lee27` · Alias: `ssh judac` · Key: `~/.ssh/id_ed25519_jsc` (reuse JURECA key)
- JuDoor: https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/judac/add_ssh_key  
  (or JuDoor → systems → **JUDAC** → Manage SSH-keys)
- Host: `judac.fz-juelich.de` · optional IPv4: `ssh judac-ipv4`
- Activation: ≤15 min after upload
- MFA: `JSC TOTP Verification code:` after pubkey (interactive)
- ControlMaster: same as JURECA (`~/.ssh/cm/`, 8h) after one login

## Setup checklist
1. JuDoor → **JUDAC** Manage SSH-keys (not JURECA).
2. Paste `from="…"` line (script below; keep prior IPs comma-separated).
3. Wait ≤15m → `ssh judac` + TOTP once.
4. Use for git/HF downloads, scp/rsync, staging — compute nodes have no internet.

## IP / paste line
```bash
ai_memory/scripts/jureca_from_clause.sh   # same key; paste into JUDAC page
# Prefer: from="CURRENT,121.162.145.92,61.253.228.193,211.54.32.170" <pubkey>
```

## Why JUDAC vs JURECA login
- **JUDAC**: data transfer, git, HF, builds that need net; shared `$HOME`/`$PROJECT`/`$SCRATCH`/`$DATA` views.
- **JURECA login**: submit Slurm / run near GPUs. Stage on JUDAC, then use paths from JURECA jobs.
