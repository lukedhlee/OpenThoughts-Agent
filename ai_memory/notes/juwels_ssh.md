# JUWELS SSH (JuDoor)

## Facts
- User: `lee27` · Key: `~/.ssh/id_ed25519_jsc` (same as JURECA/JUPITER/JUDAC)
- Aliases: `ssh juwels` → Cluster login; `ssh juwels-booster` → Booster login
- Docs: https://apps.fz-juelich.de/jsc/hps/juwels/access.html
- JuDoor key page: https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/juwels/add_ssh_key
- **Separate upload** — JURECA/JUPITER/JUDAC keys do **not** unlock JUWELS
- Keys: **ed25519 only** + `from="IP"`; MFA TOTP after pubkey (interactive)
- New/updated keys: **≤15 min**

## Cluster vs Booster
| Alias | Login host | Use for |
|-------|------------|---------|
| `juwels` / `juwels-cluster` / `juwels01`… | `juwels-cluster.fz-juelich.de` | Cluster partitions |
| `juwels-booster` / `juwels21`… | `juwels-booster.fz-juelich.de` | Booster (A100) partitions |

Submit jobs from the **matching** login side (Cluster↔Booster cross-submit is flaky).

## First login / new WiFi
```bash
ai_memory/scripts/jureca_from_clause.sh   # print from= + pubkey line
# → JuDoor → Manage SSH-keys (JUWELS) → paste → wait ≤15m
ssh juwels          # or: ssh juwels-booster
# enter JSC TOTP when prompted
```
- Current (2026-08-12): `136.152.209.31` — keep prior IPs comma-separated if useful

## Notes for our work
- Apptainer-bridge workers: handoff prefers **JURECA / JUSUF over JUWELS** when we have accounts there; JUWELS is still useful if you have a project on it.
- ControlMaster already on the `juwels*` Host blocks → after one interactive login, agents can reuse the socket for ~8h.
- First connect may prompt to accept host key for `juwels-cluster.fz-juelich.de` (compare fingerprint on JuDoor).
