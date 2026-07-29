# Jupiter SSH (JuDoor) — `lee27`

## Status (2026-07-22)
- Maintenance banner in lab ops (`Jun 23 → Jul 12`) is **stale** — login nodes answer again.
- Local alias ready: `ssh jupiter` → `lee27@login02.jupiter.fz-juelich.de` (IPv4).
- **Blocked:** JuDoor key not uploaded for **JUPITER** yet → `Permission denied (publickey)`.
- JURECA/JUDAC key uploads do **not** cover Jupiter (per-system).

## Facts
- User: `lee27` · Alias: `ssh jupiter` · Key: `~/.ssh/id_ed25519_jsc`
- JuDoor: https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/jupiter/add_ssh_key  
  (or JuDoor → Systems → **JUPITER** → Manage SSH-keys)
- Docs: https://apps.fz-juelich.de/jsc/hps/jupiter/access.html
- Logins: `login01`…`login10.jupiter.fz-juelich.de` or `login.jupiter.fz-juelich.de`
- This Mac: **no IPv6 route** to JSC → SSH config forces `AddressFamily inet`
- Activation: ≤15 min after JuDoor upload
- MFA: `JSC TOTP Verification code:` after pubkey (interactive; agents can't enter TOTP)
- ControlMaster: `~/.ssh/cm/`, 8h — after one interactive login, agent SSH skips TOTP

## Setup checklist
1. Confirm Jupiter project membership in JuDoor (Systems list shows **JUPITER**).
2. Paste `from="…"` line on **JUPITER** Manage SSH-keys (not JURECA/JUDAC).
3. Wait ≤15m → in a real Terminal: `ssh jupiter` + TOTP once.
4. Then continue path/env bring-up (`/e/...` not `/p/...`).

## IP / paste line
```bash
ai_memory/scripts/jureca_from_clause.sh
# Current (2026-07-28 evening): 118.235.5.36
# Prefer keeping prior cafe/home IPs comma-separated, e.g.:
# from="118.235.5.36,61.72.135.209,211.54.32.170,121.162.145.92,61.253.228.193" <pubkey>
```

## After first login — quick sanity
```bash
hostname; whoami; groups
jutil user projects
ls /e/project1 /e/scratch /e/data1 2>/dev/null | head
```

## Path note
Jupiter uses **`/e/...`** (not `/p/...`). Lab OT-Agent stack lives under `jureap59/feuer1` — personal `lee27` layout TBD after we see which projects you have.