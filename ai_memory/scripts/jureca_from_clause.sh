#!/usr/bin/env bash
# Print current public IP + a JuDoor-ready from= paste line for JSC
# (JURECA / JUDAC / JUPITER / JUWELS).
# Keys are uploaded PER SYSTEM on JuDoor — paste the same line on each Manage SSH-keys page.
# Usage:
#   ./jureca_from_clause.sh              # IP + paste line (uses ~/.ssh/id_ed25519_jsc.pub)
#   ./jureca_from_clause.sh --ip-only    # just the IP
#   ./jureca_from_clause.sh --extra '10.0.0.0/8,*.uni.edu'  # append more from= entries

set -euo pipefail

KEY_PUB="${JSC_PUBKEY:-$HOME/.ssh/id_ed25519_jsc.pub}"
IP_ONLY=0
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ip-only) IP_ONLY=1; shift ;;
    --extra) EXTRA="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

fetch_ip() {
  local url
  for url in \
    "https://api.ipify.org" \
    "https://ifconfig.me/ip" \
    "https://icanhazip.com"
  do
    if ip=$(curl -fsS --max-time 5 "$url" 2>/dev/null | tr -d '[:space:]'); then
      if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || [[ "$ip" =~ : ]]; then
        printf '%s\n' "$ip"
        return 0
      fi
    fi
  done
  echo "ERROR: could not fetch public IP" >&2
  return 1
}

IP=$(fetch_ip)

if [[ "$IP_ONLY" -eq 1 ]]; then
  printf '%s\n' "$IP"
  exit 0
fi

if [[ ! -f "$KEY_PUB" ]]; then
  echo "ERROR: missing pubkey: $KEY_PUB" >&2
  echo "Generate with: ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519_jsc -C jsc-lee27" >&2
  exit 1
fi

PUB=$(tr -d '\n' < "$KEY_PUB")
FROM="$IP"
if [[ -n "$EXTRA" ]]; then
  FROM="${IP},${EXTRA}"
fi

echo "public_ip=$IP"
echo "pubkey=$KEY_PUB"
echo "--- paste into JuDoor Manage SSH-keys (separate upload per system) ---"
echo "from=\"${FROM}\" ${PUB}"
echo "JUPITER: https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/jupiter/add_ssh_key"
echo "JUDAC:   https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/judac/add_ssh_key"
echo "JURECA:  https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/jureca/add_ssh_key"
echo "JUWELS:  https://judoor.fz-juelich.de/account/a/JSC_LDAP/lee27/system/juwels/add_ssh_key"
