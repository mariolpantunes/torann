#!/usr/bin/env bash
# Sync, build the arms, run a sharded sweep, merge it, and fetch the result.
#
# Usage:
#   cluster/submit.sh ARM_A ARM_B [options]
#     --spec FILE     arms to build first (default: none, arms must exist)
#     --suites "W1 W4"    default "W1 W2 W3 W4 W5"
#     --cases REGEX   subset of case ids
#     --pairs N       default 7
#     --shards N      default: one per node in $NODES
#     --name TAG      output tag (default: ARM_A-vs-ARM_B)
#     --no-sync       skip rsync (the tree on the cluster is already right)
#     --wait          block until the merge job finishes, then fetch
#
# The array is submitted with one task per shard and `--nodelist`, so tasks
# land on the nodes chosen in config.sh; the merge job runs afterok and
# refuses any case whose two arms did not meet on one machine.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

ARM_A="${1:?arm A}"; ARM_B="${2:?arm B}"; shift 2
SPEC="" SUITES="W1 W2 W3 W4 W5" CASES="" PAIRS=7 SHARDS="" TAG="" SYNC=1 WAIT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --spec) SPEC="$2"; shift 2;;
        --suites) SUITES="$2"; shift 2;;
        --cases) CASES="$2"; shift 2;;
        --pairs) PAIRS="$2"; shift 2;;
        --shards) SHARDS="$2"; shift 2;;
        --name) TAG="$2"; shift 2;;
        --no-sync) SYNC=0; shift;;
        --wait) WAIT=1; shift;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done
TAG="${TAG:-${ARM_A}-vs-${ARM_B}}"
NODE_COUNT=$(awk -F, '{print NF}' <<<"$NODES")
SHARDS="${SHARDS:-$NODE_COUNT}"

[ "$SYNC" = 1 ] && "$(dirname "${BASH_SOURCE[0]}")/sync.sh"
ROOT=$(ssh_remote "eval echo $REMOTE_ROOT")
OUT_PREFIX="$ROOT/torann/examples/out/ab_${TAG}"

if [ -n "$SPEC" ]; then
    ssh_remote "mkdir -p '$ROOT/logs'"
    scp -q "$SPEC" "$REMOTE:$ROOT/arms.spec"
    BUILD_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable -p '$PARTITION' -w '${NODES%%,*}' \
        -c 8 -t '$TIME_BUILD' --export=ALL,BENCH_ROOT='$ROOT',ARMS_SPEC='$ROOT/arms.spec' \
        '$ROOT/torann/cluster/build_arms.sbatch'")
    echo "build job $BUILD_ID"
    DEP="--dependency=afterok:$BUILD_ID"
else
    DEP=""
fi

ARRAY_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable $DEP -p '$PARTITION' \
    --nodelist='$NODES' --array=0-$((SHARDS - 1))%$NODE_COUNT -c '$CPUS_PER_NODE' --mem=0 -t '$TIME_SWEEP' \
    --export=ALL,BENCH_ROOT='$ROOT',ARM_A='$ARM_A',ARM_B='$ARM_B',SUITES='$SUITES',\
PAIRS='$PAIRS',SHARDS='$SHARDS',CASES='$CASES',OUT_PREFIX='$OUT_PREFIX',RAYON_THREADS='$RAYON_THREADS' \
    '$ROOT/torann/cluster/sweep.sbatch'")
echo "sweep array $ARRAY_ID ($SHARDS shards over $NODES)"

MERGE_ID=$(ssh_remote "cd '$ROOT/logs' && sbatch --parsable --dependency=afterok:$ARRAY_ID \
    -p '$PARTITION' -n1 -c2 -t 30 --job-name=torann-merge --output=%x-%j.out \
    --wrap=\"'$ROOT/venv/bin/python' '$ROOT/torann/examples/bench_ab.py' merge ${OUT_PREFIX}.shard*.json --out ${OUT_PREFIX}.json\"")
echo "merge job $MERGE_ID -> ${OUT_PREFIX}.json"

cat <<EOF

watch:   ssh $REMOTE squeue -u \$(ssh_remote id -un)
logs:    ssh $REMOTE tail -f $ROOT/logs/torann-sweep-${ARRAY_ID}_0.out
fetch:   cluster/fetch.sh $TAG
EOF

if [ "$WAIT" = 1 ]; then
    echo "waiting for $MERGE_ID ..."
    while ssh_remote "squeue -j $MERGE_ID -h" | grep -q .; do sleep 60; done
    "$(dirname "${BASH_SOURCE[0]}")/fetch.sh" "$TAG"
fi
