#!/bin/bash
# Stage a pinned FlashInfer AOT cache on every allocated node.
#
# This hook is deliberately opt-in.  When FLASHINFER_AOT_ARCHIVE is unset it
# changes nothing and FlashInfer continues to use the per-job node-local JIT
# workspace configured by triton_cache.sh.  Enabling it requires all four of:
#
#   FLASHINFER_AOT_ARCHIVE         read-only .tar.gz archive on shared storage
#   FLASHINFER_AOT_ARCHIVE_SHA256  sha256 of that selective archive
#   FLASHINFER_AOT_SO_SHA256       sha256 of fused_moe_90.so in the archive
#   FLASHINFER_AOT_CACHE_KEY       filesystem-safe identity for the full stack
#
# Archive layout (the archive root is added to PYTHONPATH):
#
#   flashinfer_jit_cache/__init__.py
#   flashinfer_jit_cache/_build_meta.py
#   flashinfer_jit_cache/jit_cache/fused_moe_90/fused_moe_90.so
#
# The archive is read once by sbcast, not independently by every node from
# GPFS.  Each node verifies and extracts its own copy under /tmp.  The normal
# writable FLASHINFER_WORKSPACE_BASE remains a separate per-job /tmp fallback,
# so cache misses never JIT into shared GPFS.

_flashinfer_aot_die() {
    echo "FATAL: [flashinfer_aot] $*" >&2
    return 1
}

_flashinfer_aot_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        _flashinfer_aot_die "neither sha256sum nor shasum is available"
    fi
}

_flashinfer_aot_validate_sha() {
    local label="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[[:xdigit:]]{64}$ ]]; then
        _flashinfer_aot_die "$label must be a 64-character SHA-256 digest"
    fi
}

_flashinfer_aot_stage_node() {
    local archive="${FLASHINFER_AOT_NODE_ARCHIVE:?missing node archive}"
    local root="${FLASHINFER_AOT_NODE_ROOT:?missing node root}"
    local archive_sha="${FLASHINFER_AOT_ARCHIVE_SHA256:?missing archive hash}"
    local so_sha="${FLASHINFER_AOT_SO_SHA256:?missing shared-object hash}"
    local so_rel="flashinfer_jit_cache/jit_cache/fused_moe_90/fused_moe_90.so"
    local package_rel="flashinfer_jit_cache"
    local actual_sha=""
    local members=""
    local expected_members=""

    actual_sha="$(_flashinfer_aot_sha256 "$archive")" || return 1
    if [[ "$actual_sha" != "$archive_sha" ]]; then
        _flashinfer_aot_die "archive hash mismatch on $(hostname): expected $archive_sha, got $actual_sha"
        return 1
    fi

    # Reject path traversal and any unexpected payload before asking tar to
    # write anything to node-local disk. The selective artifact must remain
    # exactly three files, never the full 956-entry/8.2-GiB wheel.
    if ! tar -tzf "$archive" | awk '
        /(^|\/)\.\.($|\/)/ || /^\// { bad=1 }
        END { exit bad ? 1 : 0 }
    '; then
        _flashinfer_aot_die "archive contains an absolute or parent-traversing path"
        return 1
    fi
    members="$(tar -tzf "$archive" | sed '/\/$/d' | LC_ALL=C sort)" || return 1
    expected_members="$(printf '%s\n' \
        "flashinfer_jit_cache/__init__.py" \
        "flashinfer_jit_cache/_build_meta.py" \
        "$so_rel" | LC_ALL=C sort)"
    if [[ "$members" != "$expected_members" ]]; then
        _flashinfer_aot_die "archive must contain exactly the three pinned FlashInfer AOT files"
        return 1
    fi

    rm -rf "$root"
    mkdir -p "$root"
    tar -xzf "$archive" -C "$root"

    if [[ ! -f "$root/$package_rel/__init__.py" || ! -f "$root/$package_rel/_build_meta.py" ]]; then
        _flashinfer_aot_die "archive does not contain the expected importable $package_rel package"
        return 1
    fi
    if [[ ! -f "$root/$so_rel" ]]; then
        _flashinfer_aot_die "archive is missing $so_rel"
        return 1
    fi
    actual_sha="$(_flashinfer_aot_sha256 "$root/$so_rel")" || return 1
    if [[ "$actual_sha" != "$so_sha" ]]; then
        _flashinfer_aot_die "shared-object hash mismatch on $(hostname): expected $so_sha, got $actual_sha"
        return 1
    fi

    # Validate the package selected by PYTHONPATH, its exact compatible version,
    # and the FlashInfer-generated JitSpec.  is_aot plus the exact aot_path proves
    # that runtime discovery points at this node-local .so, not a shared/JIT copy.
    PYTHONPATH="$root:${PYTHONPATH:-}" python - "$root" "$root/$so_rel" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected_so = pathlib.Path(sys.argv[2]).resolve()

import flashinfer_jit_cache
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

package_path = pathlib.Path(flashinfer_jit_cache.__file__).resolve().parent
if package_path != root / "flashinfer_jit_cache":
    raise SystemExit(
        f"flashinfer_jit_cache resolved outside node-local staging root: {package_path}"
    )

version = str(getattr(flashinfer_jit_cache, "__version__", ""))
expected_version = "0.6.11.post2+cu130"
if not version.startswith(expected_version):
    raise SystemExit(
        f"flashinfer_jit_cache version mismatch: expected {expected_version!r}, got {version!r}"
    )

spec = gen_cutlass_fused_moe_sm90_module()
if not bool(spec.is_aot):
    raise SystemExit("fused_moe_90 JitSpec did not select an AOT artifact")
actual_so = pathlib.Path(spec.aot_path).resolve()
if actual_so != expected_so:
    raise SystemExit(
        f"fused_moe_90 AOT path mismatch: expected {expected_so}, got {actual_so}"
    )
print(f"[flashinfer_aot] verified version={version} aot_path={actual_so}")
PY
    local verify_status=$?
    if [[ "$verify_status" -ne 0 ]]; then
        return "$verify_status"
    fi
    rm -f "$archive"
}

setup_flashinfer_aot_cache() {
    local variables=(
        FLASHINFER_AOT_ARCHIVE
        FLASHINFER_AOT_ARCHIVE_SHA256
        FLASHINFER_AOT_SO_SHA256
        FLASHINFER_AOT_CACHE_KEY
    )
    local supplied=0
    local name=""
    local script_path="${BASH_SOURCE[0]}"

    for name in "${variables[@]}"; do
        [[ -n "${!name:-}" ]] && supplied=$((supplied + 1))
    done
    if [[ "$supplied" -eq 0 ]]; then
        return 0
    fi
    if [[ "$supplied" -ne "${#variables[@]}" ]]; then
        for name in "${variables[@]}"; do
            if [[ -z "${!name:-}" ]]; then
                echo "FATAL: [flashinfer_aot] $name is required when AOT staging is enabled" >&2
            fi
        done
        return 1
    fi

    _flashinfer_aot_validate_sha "FLASHINFER_AOT_ARCHIVE_SHA256" "$FLASHINFER_AOT_ARCHIVE_SHA256" || return 1
    _flashinfer_aot_validate_sha "FLASHINFER_AOT_SO_SHA256" "$FLASHINFER_AOT_SO_SHA256" || return 1
    if [[ ! "$FLASHINFER_AOT_CACHE_KEY" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]{0,191}$ ]]; then
        _flashinfer_aot_die "FLASHINFER_AOT_CACHE_KEY is not filesystem-safe"
        return 1
    fi
    if [[ ! -r "$FLASHINFER_AOT_ARCHIVE" || ! -f "$FLASHINFER_AOT_ARCHIVE" ]]; then
        _flashinfer_aot_die "archive is not a readable regular file: $FLASHINFER_AOT_ARCHIVE"
        return 1
    fi
    case "$FLASHINFER_AOT_ARCHIVE" in
        *.tar.gz|*.tgz) ;;
        *)
            _flashinfer_aot_die "FLASHINFER_AOT_ARCHIVE must be a gzip-compressed tar (.tar.gz or .tgz)"
            return 1
            ;;
    esac
    if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_NNODES:-}" ]]; then
        _flashinfer_aot_die "AOT staging requires a Slurm allocation"
        return 1
    fi
    if ! command -v sbcast >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
        _flashinfer_aot_die "sbcast and srun are required for multi-node staging"
        return 1
    fi
    if ! command -v python >/dev/null 2>&1; then
        _flashinfer_aot_die "the activated RL environment does not provide python"
        return 1
    fi

    local actual_archive_sha=""
    actual_archive_sha="$(_flashinfer_aot_sha256 "$FLASHINFER_AOT_ARCHIVE")" || return 1
    if [[ "$actual_archive_sha" != "$FLASHINFER_AOT_ARCHIVE_SHA256" ]]; then
        _flashinfer_aot_die "source archive hash mismatch: expected $FLASHINFER_AOT_ARCHIVE_SHA256, got $actual_archive_sha"
        return 1
    fi

    local user_key="${USER:-unknown}"
    local node_archive="/tmp/flashinfer_aot_${user_key}_${SLURM_JOB_ID}_${FLASHINFER_AOT_CACHE_KEY}.tar.gz"
    local node_root="/tmp/flashinfer_aot_${user_key}_${SLURM_JOB_ID}/${FLASHINFER_AOT_CACHE_KEY}"

    # AOT is read-only; any uncovered kernels must compile into the existing
    # per-job node-local fallback.  Force that invariant even if a yaml or shell
    # environment accidentally supplied a writable shared-filesystem path.
    export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_${user_key}_${SLURM_JOB_ID}"
    mkdir -p "$FLASHINFER_WORKSPACE_BASE"

    echo "[flashinfer_aot] broadcasting one pinned archive to ${SLURM_NNODES} node(s)"
    sbcast --force "$FLASHINFER_AOT_ARCHIVE" "$node_archive" || return 1
    srun \
        --nodes="$SLURM_NNODES" \
        --ntasks="$SLURM_NNODES" \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        env \
        "FLASHINFER_AOT_NODE_ARCHIVE=$node_archive" \
        "FLASHINFER_AOT_NODE_ROOT=$node_root" \
        "FLASHINFER_AOT_ARCHIVE_SHA256=$FLASHINFER_AOT_ARCHIVE_SHA256" \
        "FLASHINFER_AOT_SO_SHA256=$FLASHINFER_AOT_SO_SHA256" \
        "PYTHONPATH=${PYTHONPATH:-}" \
        bash "$script_path" --stage-node || return 1

    # The batch shell runs on one allocated node, where the same absolute /tmp
    # path has just been validated.  Ray propagates this value to every worker.
    export PYTHONPATH="$node_root:${PYTHONPATH:-}"
    export FLASHINFER_AOT_ROOT="$node_root"
    echo "[flashinfer_aot] active root: $FLASHINFER_AOT_ROOT"
    echo "[flashinfer_aot] writable fallback: ${FLASHINFER_WORKSPACE_BASE:-<unset>}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == "--stage-node" ]]; then
    _flashinfer_aot_stage_node
elif [[ "${FLASHINFER_AOT_MANUAL:-0}" != "1" ]]; then
    setup_flashinfer_aot_cache
fi
