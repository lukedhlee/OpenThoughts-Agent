#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: jupiter_sync_offline.sh [options]

Sync offline WandB runs from a Jupiter login node (compute has no internet).

Options:
  --root DIR        Root to search for offline-run-* dirs
                    default: $WANDB_DIR, then /e/scratch/reformo/lee27/wandb
  --match TEXT      Only sync runs whose path contains TEXT
  --dry-run         Print runs that would be synced
  -h, --help        Show this help
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

JUPITER_ENV="${JUPITER_ENV:-/e/project1/reformo/lee27/env.sh}"
OTAGENT_SECRETS="${OTAGENT_SECRETS:-/e/scratch/reformo/lee27/keys/secrets.env}"

if [[ -f "$JUPITER_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$JUPITER_ENV"
fi
if [[ -f "$OTAGENT_SECRETS" ]]; then
  set +x
  # shellcheck disable=SC1090
  source "$OTAGENT_SECRETS"
fi

wandb_root="${root_arg:-${WANDB_DIR:-/e/scratch/reformo/lee27/wandb}}"
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
  if command -v wandb >/dev/null 2>&1; then
    wandb_cmd=(wandb)
  elif [[ -x /e/scratch/reformo/lee27/miniforge3/envs/otagent/bin/wandb ]]; then
    wandb_cmd=(/e/scratch/reformo/lee27/miniforge3/envs/otagent/bin/wandb)
  elif [[ -x /e/scratch/reformo/lee27/miniforge3/bin/wandb ]]; then
    wandb_cmd=(/e/scratch/reformo/lee27/miniforge3/bin/wandb)
  else
    echo "Could not find wandb CLI. pip install wandb in otagent/base env." >&2
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
