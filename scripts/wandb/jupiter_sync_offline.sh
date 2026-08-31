#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: jupiter_sync_offline.sh [options]

Sync offline WandB runs from a Jupiter login node (compute has no internet).

Options:
  --root DIR        Root to search for offline-run-* dirs
                    default: $WANDB_DIR, then $DCFT/experiments
  --match TEXT      Only sync runs whose path contains TEXT
  --dry-run         Print runs that would be synced
  -h, --help        Show this help

Environment:
  JUPITER_ENV       Optional env file sourced first (default: hpc/dotenv/jupiter.env
                    relative to this repo)
  OTAGENT_SECRETS   Secrets file with WANDB_API_KEY (default: $DC_AGENT_SECRET_ENV,
                    then ~/secrets.env)
  WANDB_CLI         Optional wandb executable path/command

This script never prints WANDB_API_KEY. Run it on a login node, not inside a
compute job.
USAGE
}

dry_run=0
match_text=""
root_arg=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) root_arg="${2:?}"; shift 2 ;;
    --match) match_text="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JUPITER_ENV="${JUPITER_ENV:-$REPO_ROOT/hpc/dotenv/jupiter.env}"

if [[ -f "$JUPITER_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$JUPITER_ENV"
fi
OTAGENT_SECRETS="${OTAGENT_SECRETS:-${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}}"
OTAGENT_SECRETS="${OTAGENT_SECRETS/#\~/$HOME}"
if [[ -f "$OTAGENT_SECRETS" ]]; then
  set +x
  # shellcheck disable=SC1090
  source "$OTAGENT_SECRETS"
fi

wandb_root="${root_arg:-${WANDB_DIR:-${DCFT:-$REPO_ROOT}/experiments}}"
export WANDB_MODE=online

if [[ ! -d "$wandb_root" ]]; then
  echo "WandB root does not exist: $wandb_root" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" && "$dry_run" -eq 0 ]]; then
  echo "WANDB_API_KEY is not set. Check $OTAGENT_SECRETS" >&2
  exit 1
fi

if [[ "$dry_run" -eq 0 ]]; then
  if [[ -n "${WANDB_CLI:-}" ]]; then
    read -r -a wandb_cmd <<< "$WANDB_CLI"
  elif command -v wandb >/dev/null 2>&1; then
    wandb_cmd=(wandb)
  elif [[ -n "${DCFT_RL_ENV:-}" && -x "$DCFT_RL_ENV/bin/wandb" ]]; then
    wandb_cmd=("$DCFT_RL_ENV/bin/wandb")
  else
    echo "Could not find wandb CLI. Set WANDB_CLI=/path/to/wandb." >&2
    exit 1
  fi
fi

echo "Searching for offline WandB runs under: $wandb_root"
mapfile -d '' runs < <(find "$wandb_root" -maxdepth 6 -type d -name 'offline-run-*' -print0 | sort -z)

selected=()
for run_dir in "${runs[@]}"; do
  if [[ -n "$match_text" && "$run_dir" != *"$match_text"* ]]; then
    continue
  fi
  selected+=("$run_dir")
done

if [[ "${#selected[@]}" -eq 0 ]]; then
  echo "No matching offline-run-* directories found."
  exit 0
fi

for run_dir in "${selected[@]}"; do
  if [[ "$dry_run" -eq 1 ]]; then
    echo "Would sync: $run_dir"
  else
    echo "Syncing: $run_dir"
    "${wandb_cmd[@]}" sync "$run_dir"
  fi
done
