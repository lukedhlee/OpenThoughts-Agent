# JURECA SSH (JuDoor)

Getting `ssh jureca` to work: ed25519-only keys, the `from=` IP trap, the `IdentitiesOnly` config
requirement, TOTP, and ControlMaster reuse.
Read when a JURECA login breaks — most often after a cafe/home IP change.

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
- **IPv6 trap (2026-08-10):** Mac can connect to `jureca` via IPv6 while JuDoor `from=` is IPv4-only → looks like pubkey deny even with the right IP listed. SSH config forces `AddressFamily inet`. Quick check: `ssh -4 jureca`.

## Quick workflow (IP changed)
```bash
ai_memory/scripts/jureca_from_clause.sh          # print paste line
# → JuDoor → Manage SSH-keys (JURECA) → paste → wait ≤15m → ssh jureca
```
- Current (2026-08-10): `135.180.125.128` — keep prior IPs comma-separated if you still use those networks
- Also JUWELS: same key line, **separate** JuDoor upload — see `ai_memory/notes/juwels_ssh.md` (`ssh juwels` / `ssh juwels-booster`)

## After login — original check target
`/p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh`

## Agent reuse (optional)
After one interactive `ssh jureca` + TOTP, enable ControlMaster so later commands skip MFA:
`ControlMaster auto` / `ControlPersist 8h` / `ControlPath ~/.ssh/cm/%r@%h:%p` on the `jureca` Host block.
