#!/bin/bash
# Turn-count sweep inside a serving allocation.
#
# A trajectory declines if ANY turn boundary breaks the served-stream prefix
# invariant, so the decline rate should compound with turn count while the
# per-boundary rate stays roughly flat. This measures both.
set -uo pipefail
SCRATCH="${SCRATCH:-/scratch/11584/lukedhlee}"
DCFT="${DCFT:-$SCRATCH/OpenThoughts-Agent-tito}"
OUT_DIR="${OUT_DIR:-$SCRATCH/experiments/tito_repro/sweep}"
TRAJECTORIES="${TRAJECTORIES:-32}"
MAX_TOKENS="${MAX_TOKENS:-384}"
TURN_LIST="${TURN_LIST:-2 4 8 16}"

mkdir -p "$OUT_DIR"
for T in $TURN_LIST; do
    echo "=== turns=$T ==="
    bash "$DCFT/scripts/tito_repro/drive_probe.sh" \
        --trajectories "$TRAJECTORIES" --turns "$T" --max-tokens "$MAX_TOKENS" \
        --out "$OUT_DIR/turns${T}.json" "$@"
done
echo "sweep written to $OUT_DIR"
