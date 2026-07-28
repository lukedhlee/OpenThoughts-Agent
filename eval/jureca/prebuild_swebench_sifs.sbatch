#!/bin/bash
#SBATCH --job-name=prebuild_swebench_sifs
#SBATCH --account=synthlaion
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/synthlaion/lee27/dc_agent_eval/logs/prebuild_swebench_sifs_%j.out

# Pull published SWE-bench images directly into the task-name cache expected by
# Harbor's Apptainer worker. JURECA cannot build Dockerfiles rootlessly, and the
# runtime bridge supplies OpenCode/uv plus writable log directories separately.
#
# Usage:
#   sbatch prebuild_swebench_sifs.sbatch DATASET [SIF_CACHE] [MAX_PARALLEL]

set -euo pipefail

DATASET="${1:?Usage: $0 DATASET [SIF_CACHE] [MAX_PARALLEL]}"
SIF_CACHE="${2:-/p/scratch/synthlaion/lee27/sif}"
MAX_PARALLEL="${3:-4}"
PROXY_HOST="${PROXY_HOST:-10.14.0.53}"
PROXY_PORT="${PROXY_PORT:-1080}"

if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: dataset does not exist: $DATASET" >&2
    exit 2
fi
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || (( MAX_PARALLEL > 8 )); then
    echo "ERROR: MAX_PARALLEL must be between 1 and 8" >&2
    exit 2
fi

mkdir -p "$SIF_CACHE"
JOB_TMP="/p/scratch/synthlaion/lee27/apptainer_tmp/prebuild_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMP"
cleanup() {
    case "$JOB_TMP" in
        /p/scratch/synthlaion/lee27/apptainer_tmp/prebuild_*)
            rm -rf -- "$JOB_TMP"
            ;;
    esac
}
trap cleanup EXIT

export APPTAINER_TMPDIR="$JOB_TMP"
export APPTAINER_CACHEDIR="$JOB_TMP/cache"
export ALL_PROXY="socks5h://${PROXY_HOST}:${PROXY_PORT}"
export all_proxy="$ALL_PROXY"
export HTTPS_PROXY="$ALL_PROXY"
export HTTP_PROXY="$ALL_PROXY"
export https_proxy="$ALL_PROXY"
export http_proxy="$ALL_PROXY"

command -v apptainer >/dev/null
echo "dataset=$DATASET"
echo "sif_cache=$SIF_CACHE"
echo "parallel=$MAX_PARALLEL"
echo "proxy=socks5h://${PROXY_HOST}:${PROXY_PORT}"
curl --fail --silent --show-error --max-time 20 \
    --proxy "$ALL_PROXY" https://hub.docker.com/ >/dev/null

STATUS_FILE="$JOB_TMP/status.tsv"
: > "$STATUS_FILE"

pull_task() {
    local task_dir="$1"
    local task_name dockerfile base_image target tmp_target
    task_name="$(basename "$task_dir")"
    dockerfile="$task_dir/environment/Dockerfile"
    [[ -f "$dockerfile" ]] || dockerfile="$task_dir/Dockerfile"
    if [[ ! -f "$dockerfile" ]]; then
        printf '%s\tskipped\tno Dockerfile\n' "$task_name" >> "$STATUS_FILE"
        return 0
    fi

    base_image="$(
        awk 'toupper($1) == "FROM" {
            for (i = 2; i <= NF; i++) {
                if ($i !~ /^--platform=/) { print $i; exit }
            }
        }' "$dockerfile"
    )"
    if [[ -z "$base_image" ]]; then
        printf '%s\tfailed\tno base image\n' "$task_name" >> "$STATUS_FILE"
        return 1
    fi

    target="$SIF_CACHE/${task_name}.sif"
    if [[ -s "$target" ]]; then
        printf '%s\tcached\t%s\n' "$task_name" "$target" >> "$STATUS_FILE"
        return 0
    fi

    tmp_target="$JOB_TMP/${task_name}.$$.sif"
    echo "[$task_name] pulling docker://$base_image"
    if apptainer pull --disable-cache "$tmp_target" "docker://$base_image" &&
       apptainer inspect "$tmp_target" >/dev/null; then
        mv "$tmp_target" "$target"
        printf '%s\tbuilt\t%s\n' "$task_name" "$target" >> "$STATUS_FILE"
        return 0
    fi

    rm -f -- "$tmp_target"
    printf '%s\tfailed\t%s\n' "$task_name" "$base_image" >> "$STATUS_FILE"
    return 1
}
export -f pull_task
export SIF_CACHE JOB_TMP STATUS_FILE

running=0
failures=0
for task_dir in "$DATASET"/*/; do
    pull_task "${task_dir%/}" &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
        wait -n || failures=$((failures + 1))
        running=$((running - 1))
    fi
done
while (( running > 0 )); do
    wait -n || failures=$((failures + 1))
    running=$((running - 1))
done

awk -F '\t' '
    { counts[$2]++ }
    END {
        for (key in counts) printf "%s=%d\n", key, counts[key]
    }
' "$STATUS_FILE" | sort

if (( failures > 0 )); then
    echo "ERROR: $failures image pull(s) failed" >&2
    awk -F '\t' '$2 == "failed"' "$STATUS_FILE" >&2
    exit 1
fi
