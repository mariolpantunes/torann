#!/usr/bin/env bash
# Bring a finished sweep back and report it.
#
# Usage: cluster/fetch.sh TAG [--aa examples/out/cluster/ab_aa.json]
#
# Results land in examples/out/cluster/, never in examples/out/ next to the
# laptop's own runs: a cluster number and a laptop number are different CPUs
# and must not be compared, only reported side by side.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

TAG="${1:?tag}"; shift || true
DEST="$LOCAL_TORANN/examples/out/cluster"
mkdir -p "$DEST"
ROOT=$(ssh_remote "eval echo $REMOTE_ROOT")

rsync -az --info=stats1 \
    "$REMOTE:$ROOT/torann/examples/out/ab_${TAG}*.json" "$DEST/" || {
        echo "nothing to fetch for $TAG (still running?)" >&2; exit 1; }
rsync -az "$REMOTE:$ROOT/logs/" "$DEST/logs/" 2>/dev/null || true

MERGED="$DEST/ab_${TAG}.json"
[ -f "$MERGED" ] || { echo "merged file missing: $MERGED"; ls -1 "$DEST"; exit 1; }
echo "== $MERGED"
python "$LOCAL_TORANN/examples/bench_ab.py" report "$MERGED" "$@"
