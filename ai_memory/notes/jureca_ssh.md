# JURECA SSH (JuDoor)

## Facts
- User: `lee27` · Host alias: `ssh jureca` · Key: `~/.ssh/id_ed25519_jsc`
- JuDoor key page: https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/jureca/add_ssh_key
- Login: `lee27@jureca.fz-juelich.de` (or `jureca-ipv4` / `ssh -4` for IPv4)
- New/updated keys take **≤15 min** to activate
- Keys: **ed25519 only** (no RSA). Private key stays on Mac only
- **MFA:** after pubkey, JuDoor prompts `JSC TOTP Verification code:` — must be interactive (agents can't enter TOTP)
- SSH config: `Host jureca` must use **only** `id_ed25519_jsc` + `IdentitiesOnly yes` (don't let `Host *` also offer `id_ed25519` — that can look like plain `Permission denied (publickey)`)

## The `from=` trap
- Every JuDoor key **must** include `from="IP-or-range"` — cafe/home IP changes → login fails until you update `from=`
- Same private key is fine; re-paste pubkey with new `from=` (comma-separate multiple IPs/ranges)
- No allow-from-anywhere. Prefer VPN/fixed egress if available

## Quick workflow (IP changed)
```bash
ai_memory/scripts/jureca_from_clause.sh          # print paste line
# → JuDoor → Manage SSH-keys (JURECA) → paste → wait ≤15m → ssh jureca
```
- Current (2026-07-28): `118.235.5.36` (also keep `61.72.135.209`, cafe/home IPs comma-separated)

## After login — original check target
`/p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh`

## Agent reuse (optional)
After one interactive `ssh jureca` + TOTP, enable ControlMaster so later commands skip MFA:
`ControlMaster auto` / `ControlPersist 8h` / `ControlPath ~/.ssh/cm/%r@%h:%p` on the `jureca` Host block.
