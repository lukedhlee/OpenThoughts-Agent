#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: jureca_sync_offline.sh [options]

Sync offline WandB runs from a JURECA login node.

Options:
  --root DIR        Root to search for offline-run-* dirs
                    default: $WANDB_DIR, then /p/scratch/synthlaion/lee27/wandb
  --match TEXT      Only sync runs whose path contains TEXT
  --dry-run         Print runs that would be synced
  -h, --help        Show this help

Environment:
  JURECA_ENV        Optional env file, default /p/project1/synthlaion/lee27_jureca/env.sh
  OTAGENT_SECRETS   Optional secrets file, default ~/.config/otagent/secrets.env
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
    --root)
      root_arg="${2:?missing value for --root}"
      shift 2
      ;;
    --match)
      match_text="${2:?missing value for --match}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

JURECA_ENV="${JURECA_ENV:-/p/project1/synthlaion/lee27_jureca/env.sh}"
OTAGENT_SECRETS="${OTAGENT_SECRETS:-$HOME/.config/otagent/secrets.env}"

if [[ -f "$JURECA_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$JURECA_ENV"
fi

if [[ -f "$OTAGENT_SECRETS" ]]; then
  set +x
  # shellcheck disable=SC1090
  source "$OTAGENT_SECRETS"
fi

wandb_root="${root_arg:-${WANDB_DIR:-/p/scratch/synthlaion/lee27/wandb}}"

if [[ ! -d "$wandb_root" ]]; then
  echo "WandB root does not exist: $wandb_root" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && "$dry_run" -eq 0 ]]; then
  echo "WANDB_API_KEY is not set. Check $OTAGENT_SECRETS or export it before syncing." >&2
  exit 1
fi

if [[ "$dry_run" -eq 0 ]]; then
  if [[ -n "${WANDB_CLI:-}" ]]; then
    read -r -a wandb_cmd <<< "$WANDB_CLI"
  elif command -v wandb >/dev/null 2>&1; then
    wandb_cmd=(wandb)
  elif [[ -x /p/project1/ccstdl/envs/marianna/py3.12/bin/wandb ]]; then
    wandb_cmd=(/p/project1/ccstdl/envs/marianna/py3.12/bin/wandb)
  else
    echo "Could not find wandb CLI. Set WANDB_CLI=/path/to/wandb." >&2
    exit 1
  fi
fi

echo "Searching for offline WandB runs under: $wandb_root"
if [[ -n "$match_text" ]]; then
  echo "Filtering paths containing: $match_text"
fi

mapfile -d '' runs < <(find "$wandb_root" -maxdepth 5 -type d -name 'offline-run-*' -print0 | sort -z)

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
